#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
import argparse
import json
import math
import os
import time
os.environ['MUSA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import random
from itertools import chain
from pathlib import Path

import torch
try:
    import torch_musa
except:
    pass
from utils import Logger, MultiTimer, get_mem_info, save, load, log_args, throughput_calculator
# from accelerate import Accelerator, DistributedType
# from accelerate.logging import get_logger
# from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.28.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examexples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

import colossalai
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam, CPUAdam
from colossalai.zero import ZeroOptimizer
from colossalai.zero.gemini import GeminiDDP
from colossalai.zero.gemini.colo_init_context import ColoInitContext
from colossalai.utils import get_current_device
from colossalai.tensor import ProcessGroup, ShardSpec, ColoParameter, ReplicaSpec, ComputePattern, ComputeSpec
from colossalai.booster.plugin.gemini_plugin import GeminiCheckpointIO
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator

from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

from arguments import parse_args
from data_provider import get_dataset


def main():
    args = parse_args()
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    try:
        torch_musa.set_device(local_rank)
    except:
        pass

    # colossalai.launch_from_torch(config={}, backend='mccl')
    colossalai.launch_from_torch(config={}, backend='mccl')
    coordinator = DistCoordinator()
    # colossalai.launch(config={},
    #                   rank=0,
    #                   world_size=1,
    #                 host='127.0.0.1',
    #                 port=random.randint(10023,45221),
    #                 backend='mccl')
    # rank = 0

    # ==============================
    # Initialize Booster
    # ==============================
    config = LlamaConfig.from_json_file(args.model_name_or_path + '/' + 'config_140b.json')
    print(config)
    if args.plugin == "gemini":
        PLACEMENT_POLICY = 'cpu'
        plugin = GeminiPlugin(precision=args.mixed_precision,
                              initial_scale=2**16,
                              max_norm=args.grad_clip,
                              device=get_current_device(),
                              placement_policy=PLACEMENT_POLICY,
                              pin_memory=True,
                              strict_ddp_mode=True,
                              hidden_dim=config.hidden_size)
    elif args.plugin == "gemini_auto":
        plugin = GeminiPlugin(
            precision=args.mixed_precision, placement_policy="auto", initial_scale=2**16, max_norm=args.grad_clip
        )
    elif args.plugin == "zero2":
        plugin = LowLevelZeroPlugin(
            stage=2, precision=args.mixed_precision, initial_scale=2**16, max_norm=args.grad_clip
        )
    elif args.plugin == "zero2_cpu":
        plugin = LowLevelZeroPlugin(
            stage=2, precision=args.mixed_precision, initial_scale=2**16, cpu_offload=True, max_norm=args.grad_clip
        )
    elif args.plugin == "hybrid_parallel":
        # modify the param accordingly, default configuration is for llama2-7b
        plugin = HybridParallelPlugin(
            tp_size=4,
            pp_size=2,
            num_microbatches=None,
            microbatch_size=1,
            enable_jit_fused=False,
            zero_stage=0,
            precision="fp32",
            initial_scale=1,
        )
    else:
        raise ValueError(f"Unknown plugin {args.plugin}")

    booster = Booster(plugin=plugin)


    device = get_current_device()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_clm_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    # device = torch.device(args.device)
    # accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, device=device, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    launch_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    logger = Logger(
        args.log_path, # os.path.join(args.log_path, launch_time),
        rank=rank,
    )
    logger.info(f'using {logger.log_file_path} to log')
    log_args(logger, args)

    if rank == 0: mlflow.log_params(vars(args))


    # If passed along, set the training seed now.
    # if args.seed is not None:
    #     set_seed(args.seed)

    # Handle the repository creation
    if args.push_to_hub:
        if args.hub_model_id is None:
            repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
        else:
            repo_name = args.hub_model_id
        create_repo(repo_name, exist_ok=True, token=args.hub_token)
        repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

        with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
            if "step_*" not in gitignore:
                gitignore.write("step_*\n")
            if "epoch_*" not in gitignore:
                gitignore.write("epoch_*\n")
    elif args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    # accelerator.wait_for_everyone()


    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    # if args.config_name:
    #     config = AutoConfig.from_pretrained(args.config_name)
    # elif args.model_name_or_path:
    #     config = AutoConfig.from_pretrained(args.model_name_or_path)
    # else:
    #     config = CONFIG_MAPPING[args.model_type]()
    #     logger.warning("You are instantiating a new config instance from scratch.")

    # if args.tokenizer_name:
    #     tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    # elif args.model_name_or_path:
    #     tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    # else:
    #     raise ValueError(
    #         "You are instantiating a new tokenizer from scratch. This is not supported by this script."
    #         "You can do it from another script, save it, and load it from here, using --tokenizer_name."
    #     )

    # print(args.model_name_or_path + '/config.json')
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    colo_device = torch.device('cpu')
    shard_pg = ProcessGroup(tp_degree=world_size) if args.shardinit else None
    default_dist_spec = ShardSpec([-1], [world_size]) if args.shardinit else None
    logger.info(get_mem_info(), [i for i in range(world_size)])
    # mixed_precision = torch.float16
    start = time.time()
    with ColoInitContext(device=colo_device,
                         default_dist_spec=default_dist_spec,
                         default_pg=shard_pg): #device=device
        model = LlamaForCausalLM(config)
    logger.info(f'coloinit has cost {(time.time() - start) / 60:.3f} mins', [i for i in range(world_size)])
    logger.info(get_mem_info(), [i for i in range(world_size)])

    model.tie_weights()
    numel = sum([p.numel() for p in model.parameters()])
    logger.info(f'shard init model parameter numel {numel}', [i for i in range(world_size)])

    # model_state_dict = torch.load('./chatglm6b.pt', map_location='cpu')
    # model.load_state_dict(model_state_dict, strict=False)
    model.gradient_checkpointing_enable()

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # start = time.time()
    # PLACEMENT_POLICY = 'cpu'
    # model = GeminiDDP(model,
    #                   device=get_current_device(),
    #                   placement_policy=PLACEMENT_POLICY,
    #                   pin_memory=True,
    #                   strict_ddp_mode=args.tp_degree == 1,
    #                   hidden_dim=config.hidden_size,
    #                   mixed_precision=mixed_precision)
    # logger.info(f'coloinit has cost {(time.time() - start) / 60:.3f} mins', [i for i in range(world_size)])

    train_dataset, eval_dataset, test_dataset = get_dataset(tokenizer, args, logger)
    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, sampler=DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()), shuffle=False, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, sampler=DistributedSampler(eval_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()), collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )
    test_dataloader = DataLoader(
        test_dataset, sampler=DistributedSampler(test_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()), collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layernorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # optimizer = HybridAdam(optimizer_grouped_parameters, lr=args.learning_rate)
    optimizer = CPUAdam(optimizer_grouped_parameters, lr=args.learning_rate, betas=[0.9, 0.95])
    # optimizer = ZeroOptimizer(optimizer, model)#2**14

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    start = time.time()
    model, optimizer, _, _, _ = booster.boost(
        model, optimizer
    )
    logger.info(f'gemini init has cost {(time.time() - start) / 60:.3f} mins', [i for i in range(world_size)])
    numel = sum([p.numel() for p in model.parameters()])
    logger.info(f'model parameter numel {numel}', [i for i in range(world_size)])

    if dist.get_rank() == 0:
        logger.info(f'chunk size: {model.unwrap().chunk_manager.dp_degree_chunk_size_dict}')
        for k, chunks in model.unwrap().chunk_manager.chunk_groups.items():
            logger.info(f'chunk len {k} : {len(chunks)}')
            for chunk_index, chunk in enumerate(chunks):
                logger.info(f'{chunk_index} : {chunk.chunk_size}')
                tensor_in_chunk = chunk.get_tensors()
                for tensor in tensor_in_chunk:
                    logger.info(' '.join([str(tensor.shape[i]) for i in range(len(tensor.shape))]))
                logger.info('-' * 50)
            break
    # Prepare everything with our `accelerator`.
    # model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
    #     model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler
    # )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        # accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=(dist.get_rank() != 0)) #, disable=not accelerator.is_local_main_process
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        load_start_epoch, load_start_step, sampler_start_idx = load(booster, model, optimizer, lr_scheduler, args.resume_from_checkpoint, logger)
        resume_step = load_start_step
        starting_epoch = load_start_epoch
        logger.info(f'resume from {starting_epoch}, {resume_step}', [i for i in range(dist.get_world_size())])

    #     if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
    #         # accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
    #         # accelerator.load_state(args.resume_from_checkpoint)
    #         path = os.path.basename(args.resume_from_checkpoint)
    #     else:
    #         # Get the most recent checkpoint
    #         dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
    #         dirs.sort(key=os.path.getctime)
    #         path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
    #     # Extract `epoch_{i}` or `step_{i}`
    #     training_difference = os.path.splitext(path)[0]

    #     if "epoch" in training_difference:
    #         starting_epoch = int(training_difference.replace("epoch_", "")) + 1
    #         resume_step = None
    #     else:
    #         # need to multiply `gradient_accumulation_steps` to reflect real steps
    #         resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
    #         starting_epoch = resume_step // len(train_dataloader)
    #         resume_step -= starting_epoch * len(train_dataloader)

    # # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    timer = MultiTimer()

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step <= resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue
            if dist.get_rank == 0:
                print(batch['input_ids'].size(), flush=True)
            batch['input_ids'] = batch['input_ids'].to(device)
            batch['attention_mask'] = batch['attention_mask'].to(device)
            batch['labels'] = batch['labels'].to(device)

            timer.start('forward')

            outputs = model(**batch)

            dist.barrier()
            timer.stop('forward', keep_in_history=True)
            # if dist.get_rank() == 0:
            forward_elapsed_time = timer.get_timer('forward').get_elapsed_time()
            logger.info(f'forward has cost {forward_elapsed_time:.3f}', [i for i in range(world_size)])

            loss = outputs.loss

            total_loss += loss.detach().float()
            # loss.backward()
            timer.start('backward')

            optimizer.backward(loss)

            dist.barrier()
            timer.stop('backward', keep_in_history=True)
            # if dist.get_rank() == 0:
            backward_elapsed_time = timer.get_timer('backward').get_elapsed_time()
            logger.info(f'backward has cost {backward_elapsed_time:.3f}', [i for i in range(world_size)])


            timer.start('optimizer')

            optimizer.step()

            dist.barrier()
            timer.stop('optimizer', keep_in_history=True)
            # if dist.get_rank() == 0:
            optimizer_elapsed_time = timer.get_timer('optimizer').get_elapsed_time()
            logger.info(f'optimizer has cost {optimizer_elapsed_time:.3f}', [i for i in range(world_size)])


            # We keep track of the loss at each epoch
            # if dist.get_rank() == 0:
            if completed_steps % args.log_interval == 0:
                elapsed_time_per_iteration = forward_elapsed_time + backward_elapsed_time + optimizer_elapsed_time
                samples_per_sec, tflops, approx_parameters_in_billions = throughput_calculator(
                    args, config, elapsed_time_per_iteration
                )
                current_lr = lr_scheduler.get_last_lr()[0]
                cur_loss = loss.cpu()
                log_str = f'| epoch: {epoch} | global step: {completed_steps} | step: {step} ' + \
                          f'| loss: {cur_loss.item():.5f} | ppl: {math.exp(cur_loss):.4f} | lr: {current_lr:.8f} '+ \
                          f'| second/batch: {elapsed_time_per_iteration :.3f} | token/second: {samples_per_sec:.2f} | tflops: {tflops:5f}'
                logger.info(log_str, [i for i in range(world_size)])
                if rank == 0: mlflow.log_metric("loss", float(cur_loss.item()), completed_steps)
                if rank == 0: mlflow.log_metric("second/batch", elapsed_time_per_iteration, completed_steps)
                if rank == 0: mlflow.log_metric("token/second", samples_per_sec, completed_steps)
                if rank == 0: mlflow.log_metric("tflops", tflops, completed_steps)

            lr_scheduler.step()
            optimizer.zero_grad()

            # evaluate(model, eval_dataloader, epoch, device, args)
            # test(model, test_dataloader, epoch, device, args)
            # dist.barrier()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if dist.get_rank() == 0:
                progress_bar.update(1)
            completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    save_dir = args.save_dir + '/'  + logger.latest_exp_file
                    save(
                        booster,
                        model,
                        optimizer,
                        lr_scheduler,
                        epoch,
                        step,
                        args.per_device_train_batch_size,
                        coordinator,
                        save_dir,
                        logger
                    )
                    # accelerator.save_state(output_dir)
            if completed_steps >= args.max_train_steps:
                break

        evaluate(model, eval_dataloader, epoch, device, args, logger)
        test(model, test_dataloader, epoch, device, args, logger)


        # if args.with_tracking:
        #     accelerator.log(
        #         {
        #             "perplexity": perplexity,
        #             "eval_loss": eval_loss,
        #             "train_loss": total_loss.item() / len(train_dataloader),
        #             "epoch": epoch,
        #             "step": completed_steps,
        #         },
        #         step=completed_steps,
        #     )

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

        if args.checkpointing_steps == "epoch":
            save_dir = args.save_dir + '/'  + logger.latest_exp_file
            save(
                booster,
                model,
                optimizer,
                lr_scheduler,
                epoch,
                step,
                args.per_device_train_batch_size,
                coordinator,
                save_dir,
                logger
            )
            # accelerator.save_state(output_dir)

    if args.with_tracking:
        pass
        # accelerator.end_training()

    # if args.output_dir is not None:
    #     # accelerator.wait_for_everyone()
    #     unwrapped_model = accelerator.unwrap_model(model)
    #     unwrapped_model.save_pretrained(
    #         args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
    #     )
    #     if accelerator.is_main_process:
    #         tokenizer.save_pretrained(args.output_dir)
    #         if args.push_to_hub:
    #             repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)
    logger.info('Congratulations, LLM Training fininsh.', [i for i in range(world_size)])


def evaluate(model, eval_dataloader, epoch, device, args, logger):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            batch['input_ids'] = batch['input_ids'].to(device)
            batch['attention_mask'] = batch['attention_mask'].to(device)
            batch['labels'] = batch['labels'].to(device)
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(loss.repeat(args.per_device_eval_batch_size))

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    if dist.get_rank() == 0:
        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")


def test(model, test_dataloader, epoch, device, args, logger):
    losses = []
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():

            batch['input_ids'] = batch['input_ids'].to(device)
            batch['attention_mask'] = batch['attention_mask'].to(device)
            batch['labels'] = batch['labels'].to(device)
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(loss.repeat(args.per_device_eval_batch_size))

    losses = torch.cat(losses)
    try:
        test_loss = torch.mean(losses)
        perplexity = math.exp(test_loss)
    except OverflowError:
        perplexity = float("inf")

    if dist.get_rank() == 0:
        logger.info(f"epoch {epoch}: perplexity: {perplexity} test_loss: {test_loss}")
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump({"test_perplexity": perplexity}, f)


# if __name__ == "__main__":
rank = int(os.environ['RANK'])
if rank == 0:
    import mlflow
    import sys
    mtags = {
        "python": sys.version,
        "pytorch": torch.__version__,
    }
    with mlflow.start_run(tags=mtags) as run:
      main()
else:
    main()
print('Congratulation!!')
