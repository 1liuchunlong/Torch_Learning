import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


# 输入张量，假设形状为 (batch_size, channels, height, width)
# 例如，1 个样本，3 个通道，空间大小为 4x4
input_tensor = torch.randn(1, 3, 4, 4)

# 定义 1x1 卷积层，将通道数从 3 变为 5
conv1x1 = torch.nn.Conv2d(in_channels=3, out_channels=5, kernel_size=1)

# 使用 1x1 卷积层对输入张量进行卷积
output_tensor = conv1x1(input_tensor)

# 输出结果的形状
print("Input shape:", input_tensor.shape)
print("Output shape:", output_tensor.shape)
print(output_tensor.data)