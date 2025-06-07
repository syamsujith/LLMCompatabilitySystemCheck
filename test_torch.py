import torch

print("PyTorch version:", torch.__version__)
print("Is CUDA available?", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
