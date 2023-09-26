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

import datasets
import torch
# import torch_musa
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
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
import subprocess
import socket
import torch.distributed as dist
from datetime import timedelta

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--torch",
        action='store_true',
        help=
        "Shard the tensors when init the model to shrink peak memory size on the assigned device. Valid when using colossalai as dist plan.",
    )
    parser.add_argument(
        "--mthreads",
        action='store_true',
        help=
        "Shard the tensors when init the model to shrink peak memory size on the assigned device. Valid when using colossalai as dist plan.",
    )
    parser.add_argument(
        "--cai",
        action='store_true',
        help=
        "Shard the tensors when init the model to shrink peak memory size on the assigned device. Valid when using colossalai as dist plan.",
    )
    parser.add_argument(
        "--ssh",
        action='store_true',
        help=
        "Shard the tensors when init the model to shrink peak memory size on the assigned device. Valid when using colossalai as dist plan.",
    )
    parser.add_argument(
        "--musa",
        action='store_true',
        help=
        "Shard the tensors when init the model to shrink peak memory size on the assigned device. Valid when using colossalai as dist plan.",
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        required=True
    )
    args = parser.parse_args()

    # Sanity checks

    return args

def whoami():
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    i_am = hostname + ' ' + ip
    return i_am

def main():
    args = parse_args()
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    # torch_musa.set_device(local_rank)
    # colossalai.launch_from_torch(config={}, backend='mccl')
    host_name = whoami()
    if args.torch:
        cmd = f'bash /home/dist/llm/cmd/torch_install.sh | tee /home/dist/llm/install_log/torch_musa/{rank}.txt'
    # cmd = 'bash /home/dist/llm/cmd/ls.sh'
    elif args.cai:
        
        cmd = f'bash ./cmd/colo_install.sh | tee ./install_log/colossalai/{host_name}.txt'
    elif args.mthreads:
        cmd = f'ifconfig > ./mthreads-gmi/{rank}.txt && /usr/bin/mthreads-gmi >> ./mthreads-gmi/{rank}.txt'
    elif args.ssh:
        cmd = f'bash ./cmd/sed_ssh.sh'
    elif args.musa:
        cmd = f'bash ./cmd/musa_install.sh | tee ./install_log/musatoolkit/{host_name}.txt'
    
    print(cmd)
    # cmd = 'bash /home/dist/llm/cmd/mpi.sh'
    # cmd = f'bash /home/dist/llm/cmd/lsmod.sh > ./musa_runtime/{rank}.txt'
    # cmd = 'bash /home/dist/llm/cmd/musa_install.sh'
    # cmd = 'bash /home/dist/llm/cmd/math_install.sh'
    
    # server_store = dist.TCPStore("10.79.254.116", 12346, world_size)
    # client_store = dist.TCPStore("10.79.254.116", 12345, world_size, False)
    # for i in range(world_size):
    #     if i == rank:
    #         res = subprocess.call(cmd, shell=True)
    #         # time.sleep(3)
    #         server_store.set("done", "true")        
    #     else:
    #         while True:
    #             flag = 1
    #             flag = server_store.get("done")
    #             print(flag)
    #             if flag == b'true':
    #                 flag = 1
    #                 break
    #     print(f'i am {rank}')

    print(f'i am {rank}')

    res = subprocess.call(cmd, shell=True)
    # res = 0
    print(f'complete {res} rank is {rank} ' + whoami())
    print(colossalai.__file__)
    

# if __name__ == "__main__":
main()
