import os
import psutil

import torch
import torch.distributed as dist

def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2

def get_mem_info(prefix=''):
    return f'{prefix}CPU memory usage: {get_cpu_mem():.2f} MB'

def log_args(logger, args):
    logger.info("--------args----------")
    message = "\n".join([f"{k:<30}: {v}" for k, v in vars(args).items()])
    message += "\n"
    logger.info(message)
    logger.info("--------args----------\n")

def get_parameters_in_billions(model):
    gpus_per_model = 1 #torch.distributed.get_world_size(group=mpu.get_model_parallel_group())

    approx_parameters_in_billions = sum([sum([p.ds_numel if hasattr(p,'ds_id') else  p.nelement() for p in model_module.parameters()])
                                        for model_module in model])

    return approx_parameters_in_billions*gpus_per_model/(1e9)

def throughput_calculator(args, config, elapsed_time_per_iter, iteration_time=None, total_iterations=None, model=None):

    args.micro_batch_size = args.per_device_train_batch_size
    args.data_parallel_size = dist.get_world_size()
    args.world_size = args.data_parallel_size
    args.seq_length = args.block_size
    args.hidden_size = config.hidden_size
    args.num_layers = config.num_hidden_layers
    args.padded_vocab_size = config.vocab_size

    gpus_per_model = 1 #torch.distributed.get_world_size(group = mpu.get_model_parallel_group())
    batch_size = args.micro_batch_size * 1 * args.data_parallel_size
    samples_per_model = batch_size * args.seq_length
    model_replica_count = torch.distributed.get_world_size() / gpus_per_model
    approx_parameters_in_billions = None if (model is None) else get_parameters_in_billions(model)
    # elapsed_time_per_iter = iteration_time/total_iterations
    samples_per_second = samples_per_model / elapsed_time_per_iter

    #flops calculator
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    vocab_size = args.padded_vocab_size

    # General TFLOPs formula (borrowed from Equation 3 in Section 5.1 of
    # https://arxiv.org/pdf/2104.04473.pdf).
    # The factor of 4 is when used with activation check-pointing,
    # otherwise it will be 3.
    # checkpoint_activations_factor = 3
    checkpoint_activations_factor = 4
    if hasattr(args, 'checkpoint_activations') and args.checkpoint_activations:
        checkpoint_activations_factor = 4
    if hasattr(args, 'recompute_granularity') and (args.recompute_granularity == 'selective' or args.recompute_granularity == 'full'):
        checkpoint_activations_factor = 4
    seq_len = args.seq_length
    if hasattr(args, 'actual_seq_length'):
        seq_len = args.actual_seq_length
    flops_per_iteration = (24 * checkpoint_activations_factor * batch_size * seq_len * num_layers * (hidden_size**2)) * (1. + (seq_len / (6. * hidden_size)) + (vocab_size / (16. * num_layers * hidden_size)))
    tflops = flops_per_iteration / (elapsed_time_per_iter * args.world_size * (10**12))
    return samples_per_second, tflops, approx_parameters_in_billions