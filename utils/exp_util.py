import os
import psutil

def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2

def get_mem_info(prefix=''):
    return f'{prefix}CPU memory usage: {get_cpu_mem():.2f} MB'