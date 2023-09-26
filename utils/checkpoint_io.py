import os
import time
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# from transformers.models.llama.configuration_llama import LlamaConfig

from colossalai.cluster import DistCoordinator
from colossalai.booster import Booster
from .data_utils import load_json, prepare_dataloader, save_json

# MODEL_CONFIGS = {
#     "7b": LlamaConfig(max_position_embeddings=4096),
#     "13b": LlamaConfig(
#         hidden_size=5120,
#         intermediate_size=13824,
#         num_hidden_layers=40,
#         num_attention_heads=40,
#         max_position_embeddings=4096,
#     ),
#     "70b": LlamaConfig(
#         hidden_size=8192,
#         intermediate_size=28672,
#         num_hidden_layers=80,
#         num_attention_heads=64,
#         max_position_embeddings=4096,
#         num_key_value_heads=8,
#     ),
# }


def get_model_numel(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def format_numel_str(numel: int) -> str:
    B = 1024**3
    M = 1024**2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return tensor


def save(
    booster: Booster,
    model: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
    epoch: int,
    step: int,
    batch_size: int,
    coordinator: DistCoordinator,
    save_dir: str,
    logger
):
    save_dir = os.path.join(save_dir, f"epoch_{epoch}-step_{step}")
    os.makedirs(os.path.join(save_dir, "model"), exist_ok=True)

    start = time.time()
    booster.save_model(model, os.path.join(save_dir, "model"), shard=True)
    logger.info(f'save model has cost {(time.time() - start) / 60:.3f} mins', [i for i in range(dist.get_world_size())])

    start = time.time()
    # booster.save_optimizer(optimizer, os.path.join(save_dir, "optimizer"), shard=True)
    logger.info(f'save optimizer has cost {(time.time() - start) / 60:.3f} mins', [i for i in range(dist.get_world_size())])

    start = time.time()
    booster.save_lr_scheduler(lr_scheduler, os.path.join(save_dir, "lr_scheduler"))
    logger.info(f'save lr_scheduler has cost {(time.time() - start) / 60:.3f} mins', [i for i in range(dist.get_world_size())])

    running_states = {
        "epoch": epoch,
        "step": step,
        "sample_start_index": step * batch_size,
    }
    latest_file = {
        'lates_file_path': save_dir
    }
    if coordinator.is_master():
        save_json(latest_file, '/home/dist/cai/LM/checkpoint/llama2/latest_ckpt.json')
        save_json(running_states, os.path.join(save_dir, "running_states.json"))


def load(
    booster: Booster, model: nn.Module, optimizer: Optimizer, lr_scheduler: _LRScheduler, load_dir_from_specified: str,
    logger
) -> Tuple[int, int, int]:
    
    load_dir = load_json('/home/dist/cai/LM/checkpoint/llama2/latest_ckpt.json')['lates_file_path']
    logger.info(f'load from {load_dir}', [i for i in range(dist.get_world_size())])

    start = time.time()
    booster.load_model(model, os.path.join(load_dir, "model"))
    logger.info(f'load model has cost {(time.time() - start) / 60:.3f} mins', [i for i in range(dist.get_world_size())])

    start = time.time()
    # booster.load_optimizer(optimizer, os.path.join(load_dir, "optimizer"))
    logger.info(f'load optimizer has cost {(time.time() - start) / 60:.3f} mins', [i for i in range(dist.get_world_size())])

    start = time.time()
    booster.load_lr_scheduler(lr_scheduler, os.path.join(load_dir, "lr_scheduler"))
    logger.info(f'load optimzier has cost {(time.time() - start) / 60:.3f} mins', [i for i in range(dist.get_world_size())])

    running_states = load_json(os.path.join(load_dir, "running_states.json"))
    return running_states["epoch"], running_states["step"], running_states["sample_start_index"]