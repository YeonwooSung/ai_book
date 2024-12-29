import torch
from torch.utils.data import DataLoader
from transformers import AdamW


def print_gpu_utilization():
    if torch.cuda.is_available():
        print("GPU Utilization")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.memory_reserved(i) / 1024**3:.2f}GB")
    else:
        print("No GPU available")


def estimate_memory_of_gradients(model):
    """
    Estimate the memory required to store the gradients of the model.
    """
    memory = 0
    for p in model.parameters():
        if p.grad is not None:
            memory += p.grad.element_size() * p.grad.nelement()
    return memory


def estimate_memory_of_optimizer(optimizer):
    """
    Estimate the memory required to store the optimizer state.
    """
    memory = 0
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                memory += v.element_size() * v.nelement()
    return memory
