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
print('1' * 100)
import argparse
import json
import math
import os
import time
os.environ['MUSA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
try:
    import torch_musa
except:
    pass
from utils import Logger, MultiTimer, get_mem_info
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
from transformers import GPT2Config, GPT2LMHeadModel

import colossalai
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam, CPUAdam
from colossalai.zero import ZeroOptimizer
from colossalai.nn.parallel import GeminiDDP
from colossalai.zero.gemini.colo_init_context import ColoInitContext
from colossalai.utils import get_current_device
from colossalai.tensor import ProcessGroup, ShardSpec

from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Where to store the dataset.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        required=True
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default='./exp_log'
    )
    parser.add_argument(
        "--shardinit",
        action='store_true',
        help=
        "Shard the tensors when init the model to shrink peak memory size on the assigned device. Valid when using colossalai as dist plan.",
    )
    parser.add_argument(
        "--tp_degree",
        type=int,
        default=1,
        help="Tensor Parallelism Degree. Valid when using colossalai as dist plan.",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


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
    # colossalai.launch(config={},
    #                   rank=0,
    #                   world_size=1,
    #                 host='127.0.0.1',
    #                 port=random.randint(10023,45221), 
    #                 backend='mccl')
    # rank = 0
    
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

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        data_files = {}
        data_files["train"] = './wiki2/wiki.train.raw'
        data_files["validation"] = './wiki2/wiki.valid.raw'
        data_files["test"] = './wiki2/wiki.test.raw'

        raw_datasets = load_dataset('text', data_files=data_files)
        # raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir, verification_mode="no_checks")
        # if "validation" not in raw_datasets.keys():
        #     raw_datasets["validation"] = load_dataset(
        #         args.dataset_name,
        #         args.dataset_config_name,
        #         split=f"train[:{args.validation_split_percentage}%]",
        #     )
        #     raw_datasets["train"] = load_dataset(
        #         args.dataset_name,
        #         args.dataset_config_name,
        #         split=f"train[{args.validation_split_percentage}%:]",
        #     )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )

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
    config = LlamaConfig.from_json_file('./llama_config/config7b.json')
    print(config)
    # print(args.model_name_or_path + '/config.json')
    tokenizer = LlamaTokenizer.from_pretrained('./llama_config')
    colo_device = torch.device('cpu')
    shard_pg = ProcessGroup(tp_degree=world_size) if args.shardinit else None
    default_dist_spec = ShardSpec([-1], [world_size]) if args.shardinit else None
    print(get_mem_info())
    with ColoInitContext(device=colo_device, 
                         dtype=torch.float, 
                         default_dist_spec=default_dist_spec,
                         default_pg=shard_pg): #device=device
        model = LlamaForCausalLM(config)
    print(get_mem_info())

    model.tie_weights()
    numel = sum([p.numel() for p in model.parameters()])
    print(f'model parameter nueml {numel}')
    PLACEMENT_POLICY = 'cpu'
    # model_state_dict = torch.load('./chatglm6b.pt', map_location='cpu')
    # model.load_state_dict(model_state_dict, strict=False)
    model.gradient_checkpointing_enable()
    
    # model = model.to(device)
    
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    
    model = GeminiDDP(model, 
                      device=get_current_device(), 
                      placement_policy=PLACEMENT_POLICY, 
                      pin_memory=True, 
                      strict_ddp_mode=args.tp_degree == 1, 
                      hidden_dim=config.hidden_size,
                      mixed_precision=torch.float32)
                      
    if dist.get_rank() == 0:
        print('chunk size: ', model.chunk_manager.dp_degree_chunk_size_dict)

        for k, chunks in model.chunk_manager.chunk_groups.items():
            print(f'chunk len {k} : ', len(chunks))
            for chunk_index, chunk in enumerate(chunks):
                print(chunk_index, ': ', chunk.chunk_size)
                tensor_in_chunk = chunk.get_tensors()
                for tensor in tensor_in_chunk:
                    print(tensor.shape)
                print('-' * 50)
            break
        

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]
    test_dataset = lm_datasets["test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

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
    no_decay = ["bias", "layer_norm.weight"]
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
    # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    # optimizer = HybridAdam(optimizer_grouped_parameters, lr=args.learning_rate)
    optimizer = CPUAdam(optimizer_grouped_parameters, lr=args.learning_rate)
    optimizer = ZeroOptimizer(optimizer, model, initial_scale=1, growth_interval=1e7)#2**14

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
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            # accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            # accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    timer = MultiTimer()

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue

            batch['input_ids'] = batch['input_ids'].to(device)
            batch['attention_mask'] = batch['attention_mask'].to(device)
            batch['labels'] = batch['labels'].to(device)

            timer.start('forward')

            outputs = model(**batch)
        
            dist.barrier()
            timer.stop('forward', keep_in_history=True)
            if dist.get_rank() == 0:
                elapsed_time = timer.get_timer('forward').get_elapsed_time()
                logger.info(f'forward has cost {elapsed_time:.3f}')

            loss = outputs.loss
            # We keep track of the loss at each epoch
            if dist.get_rank() == 0:
                log_str = f'loss is {loss.cpu()}, step is {step}'
                logger.info(log_str, print_=False)
                
            total_loss += loss.detach().float()
            # loss.backward()
            timer.start('backward')

            optimizer.backward(loss)
        
            dist.barrier()
            timer.stop('backward', keep_in_history=True)
            if dist.get_rank() == 0:
                elapsed_time = timer.get_timer('backward').get_elapsed_time()
                logger.info(f'backward has cost {elapsed_time:.3f}')


            timer.start('optimizer')

            optimizer.step()
        
            dist.barrier()
            timer.stop('optimizer', keep_in_history=True)
            if dist.get_rank() == 0:
                elapsed_time = timer.get_timer('optimizer').get_elapsed_time()
                logger.info(f'optimizer has cost {elapsed_time:.3f}')

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
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
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
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
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
main()
print('Congratulation!!')
