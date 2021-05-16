# %%
import torch
import torch.utils.data as Data
from matplotlib import pyplot as plt
import numpy as np
import random
from torch import nn

# ---------生成数据集----------
n_samples, n_features = 1000, 2
true_w, true_b = [2, -3.4], 4.2

train_features = torch.tensor(np.random.normal(0, 1, (n_samples, n_features)), dtype=torch.float)
train_labels = true_w[0] * train_features[:, 0] + true_w[1] * train_features[:, 1] + true_b

# Pytorch读取数据
batch_size = 10
# 将训练数据的特征和标签组合起来
dataset = Data.TensorDataset(train_features, train_labels)

# 随机读取小批量数据w
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

# ----------定义模型------------
class LinearNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)


net = LinearNet(n_features)
