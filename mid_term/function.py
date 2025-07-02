import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.utils.data
import math
from copy import deepcopy
import numpy as np
import cv2 as cv
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim     # for constructing optimizer
import torchvision.models as models
from torch.autograd import Function
import os
def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size
    
    
import matplotlib.pyplot as plt
import numpy as np

def plot_weight_distribution(model):
    """
    绘制统计模型的权重分布

    参数:
        model: 统计模型对象，比如一个神经网络模型

    返回:
        None
    """
    # 获取模型的所有参数
    params = []
    for param in model.parameters():
        params.append(param.detach().cpu().numpy().flatten())  # 将参数转换为NumPy数组，并展平
    
    # 计算所有权重的最小值和最大值
    min_weight = np.min([np.min(param) for param in params])
    max_weight = np.max([np.max(param) for param in params])
    
    # 绘制权重分布
    plt.figure(figsize=(10, 6))
    plt.hist(params, bins=50, alpha=0.7, label=[f'Layer {i}' for i in range(len(params))])
    plt.title('Weight Distribution of Statistical Model')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    # 设置横坐标范围为权重的范围
    plt.xlim(min_weight, max_weight)
    
    plt.show()

import time
def run_benchmark(model, loader, device='cpu'):
    elapsed = 0
    model.eval()
    num_batches = 5
    # Run the scripted model on a few batches of images
    for i, (images, target) in enumerate(loader):
        images, target = images.to(device), target.to(device)
        if i < num_batches:
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = images.size()[0] * num_batches

    print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))
    return elapsed

class FakeQuantize(Function):

    @staticmethod
    def forward(ctx, x, qparam):
        x = qparam.quantize_tensor(x)
        x = qparam.dequantize_tensor(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
    
def interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    x_ = x.reshape(x.size(0), -1)
    xp = xp.unsqueeze(0)
    fp = fp.unsqueeze(0)
        
    m = (fp[:,1:] - fp[:,:-1]) / (xp[:,1:] - xp[:,:-1])  #slope
    b = fp[:, :-1] - (m.mul(xp[:, :-1]) )

    indicies = torch.sum(torch.ge(x_[:, :, None], xp[:, None, :]), -1) - 1  #torch.ge:  x[i] >= xp[i] ? true: false
    indicies = torch.clamp(indicies, 0, m.shape[-1] - 1)

    line_idx = torch.linspace(0, indicies.shape[0], 1, device=indicies.device).to(torch.long)
    line_idx = line_idx.expand(indicies.shape)
    # idx = torch.cat([line_idx, indicies] , 0)
    out = m[line_idx, indicies].mul(x_) + b[line_idx, indicies]
    out = out.reshape(x.shape)
    return out