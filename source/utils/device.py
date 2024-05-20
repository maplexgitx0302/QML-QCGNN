"""Getting system device information."""

import torch

def get_cpu_name():
    with open('/proc/cpuinfo', 'r') as f:
        for line in f:
            if line.strip() and line.rstrip('\n').startswith('model name'):
                return line.rstrip('\n').split(':')[1].strip()
    return 'Unknown CPU'

def get_gpu_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.cuda.current_device())
    else:
        return 'Unknown GPU'