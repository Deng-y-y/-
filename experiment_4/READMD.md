# 基于 MLP 的 MNIST 手写数字分类实验报告

## 一、实验目的

本实验基于多层感知机（MLP）完成 MNIST 手写数字分类任务，采用控制变量法。固定网络结构、batch size、学习率、数据集与预处理，在训练环节设置三组对照分支，分别改变训练轮次 epoch、SGD 动量 momentum，观察两个超参对模型收敛速度、分类精度的影响，理解 SGD 动量机制的原理与作用。

## 二、实验环境、数据集与模型定义

### 2.1 依赖库导入

```
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
```

**代码分析**
- **torch**：PyTorch 核心库，提供张量运算与自动求导功能。
- **torch.nn**：神经网络模块，提供网络层、模型封装工具。
- **torch.nn.functional**：函数式神经网络接口，提供 ReLU、交叉熵等运算。
- **torch.optim**：优化器模块，实现 SGD、Adam 等参数更新算法。
- **numpy**：数值计算库，用于保存损失数据。
- **matplotlib.pyplot**：可视化库，绘制训练损失曲线。
- **torchvision**：视觉工具库

### 2.2 超参数与设备配置

```
batch_size = 2048
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**代码分析**
- `batch_size = 2048`：设置批次大小，每一轮迭代同时送入 2048 个样本参与训练。
- `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`：自动判断运行环境，优先使用 GPU 加速，无可用 GPU 时回退至 CPU 运算。

### 2.3 MNIST 数据集加载

```
transform = transforms.Compose([
    transforms.ToTensor(),
])
# 数据加载
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transform),
    batch_size=batch_size, shuffle=False
)

print("训练数据形状:", train_loader.dataset.data.shape)
print("测试数据形状:", test_loader.dataset.data.shape)
print("类别数量: 10 (数字0-9)")

```

**输出**：

```
训练数据形状: torch.Size([60000, 28, 28])
测试数据形状: torch.Size([10000, 28, 28])
类别数量: 10 (数字0‑9)
```
**代码分析**
- `datasets.MNIST`：加载 MNIST 手写数字数据集，MNIST 包含 60000 张训练样本、10000 张测试样本，28×28 灰度图。其中`train=True`为训练集，`train=False`为测试集。
- `transforms.ToTensor()`：将图像像素归一化至 [0,1]，转为张量格式。
- `DataLoader`：对数据集分批封装，`shuffle=True`打乱训练集样本顺序，避免顺序带来的学习偏差。

### 2.4 基础 MLP 模型定义

```
class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()
        self.l1 = nn.Linear(784, 128)
        self.l2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

model = mlp().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0)

print("模型结构:")
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n总参数数量: {total_params:,}")

```
**代码分析**
- `nn.Linear(784,128)`：第一层全连接层，将 28×28 展平后的 784 维向量映射至 128 维隐藏特征。
- `nn.Linear(128,10)`：输出层，将隐藏特征映射为 10 个类别的得分。
- `forward`：前向传播流程，隐藏层使用 ReLU 激活函数，输出层不添加激活，交由交叉熵损失处理。
- `optim.SGD(..., momentum=0)`：朴素 SGD 优化器，无动量，参数更新仅依赖当前批次梯度。

### 2.5 模型迭代训练

```
# 模型迭代训练
epochs = 30
train_losses = []
train_accuracies = []

print("开始训练...")
print("---")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.view(x.shape[0], -1).to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()
        total += y.size(0)

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total

    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
    print(f"Epoch [{epoch+1:2d}/{epochs}] - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")
print("---")
print("训练完成!")

```
**代码分析**
- `model.train()`：开启训练模式。
- `x.view()`：把二维图像张量展平为一维向量，适配全连接输入。
- `zero_grad()`：清空上一轮迭代的梯度，防止梯度累积。
- `loss.backward()`：反向传播，自动计算参数梯度。
- `optimizer.step()`：根据梯度更新网络权重。

### 2.6 测试模型

```
model.eval()
correct = 0
test_loss = 0
y_true = []
y_pred = []

print("开始测试...")
with torch.no_grad():
    for batch_idx, (x, y) in enumerate(test_loader):
        x, y = x.view(x.shape[0], -1).to(device), y.to(device)
        output = model(x)
        test_loss += F.cross_entropy(output, y)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()
        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

test_loss = test_loss / (batch_idx + 1)
acc = 100. * correct / len(test_loader.dataset)

print("---")
print('测试结果:')
print(f'测试损失: {test_loss:.4f}')
print(f'测试准确率: {acc:.2f}%')
print(f'正确样本数: {correct}/{len(test_loader.dataset)}')
print("---")

```
**测试结果**：
```
开始测试...
---
测试结果:
测试损失: 0.2454
测试准确率: 93.16%
正确样本数: 9316/10000
---
```
![training curve baseline](<../assets/training_curves (3).png>)

**代码分析**
- `model.eval()`：切换为评估模式。
- `torch.no_grad()`：关闭梯度计算，节约显存、加速推理。
- 循环遍历测试集，统计平均测试损失、测试准确率与正确样本数量。

## 三、尝试优化
### 3.1 增大轮次，其他数据不变
**改变部分代码**：
```
epochs = 50
train_losses = []
train_accuracies = []
model2 = mlp().to(device)
optimizer2 = optim.SGD(model2.parameters(), lr=0.1, momentum=0)

print("开始训练...")
print("---")
for epoch in range(epochs):
    model2.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.view(x.shape[0], -1).to(device), y.to(device)
        optimizer2.zero_grad()
        output = model2(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer2.step()

        running_loss += loss.item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()
        total += y.size(0)

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total

    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
    print(f"Epoch [{epoch+1:2d}/{epochs}] - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")
print("---")
print("训练完成!")
```
**最终测试结果**：
```
开始测试...
---
测试结果:
测试损失: 0.2065
测试准确率: 93.97%
正确样本数: 9397/10000
---
```
![training curve 50epoch](<../assets/training_curves (1).png>)

### 3.2 增大动量，其他数据不变
**改变部分代码**：
```
epochs = 30
train_losses = []
train_accuracies = []
model3 = mlp().to(device)
optimizer3 = optim.SGD(model3.parameters(), lr=0.1, momentum=0.9)

print("开始训练...")
print("---")
for epoch in range(epochs):
    model3.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.view(x.shape[0], -1).to(device), y.to(device)
        optimizer3.zero_grad()
        output = model3(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer3.step()

        running_loss += loss.item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()
        total += y.size(0)

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total

    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
    print(f"Epoch [{epoch+1:2d}/{epochs}] - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")
print("---")
print("训练完成!")
```
**最终测试结果**：
```
开始测试...
---
测试结果:
测试损失: 0.0859
测试准确率: 97.27%
正确样本数: 9727/10000
---
```
![training curve momentum0.9](<../assets/training_curves (2).png>)

## 四、实验结果对比与分析
![confusion matrix baseline](<../assets/confusion_matrix (3).png>)
![confusion matrix 50epoch](<../assets/confusion_matrix (1).png>)
![confusion matrix momentum0.9](<../assets/confusion_matrix (2).png>)
- 1. momentum 均为 0，epoch 分别 30、50的实验结果相对比：同为朴素 SGD，延长训练轮次到 50 轮，准确率会有小幅提升，但提升幅度不大。说明仅靠增加训练 epoch 的方式效率很低，朴素 SGD 到达损失平台后很难继续优化。
- 3. epoch 均为 30，momentum 分别 0、0.9的实验对比：训练轮次完全相同，加入动量之后收敛速度更快，同等训练轮数下测试准确率更高。验证 SGD 动量的作用：利用历史梯度的惯性加速下降、削弱震荡。动量带来的收益远大于单纯增加训练轮数。

## 六、实验总结与心得

本次实验设置三个训练对照分支，在固定网络结构的前提下，分别探究训练轮次、SGD 动量对 MLP 分类效果的影响。分支 1 作为基准，分支 2 观察延长训练轮次的效果，分支 3 观察动量机制的效果，全部训练结束之后统一在测试集上评估。
通过对照可以看出：朴素 SGD 收敛缓慢，单纯增加训练轮数性价比低；动量给梯度下降引入惯性，是高效的优化策略。本次实验也体现了控制变量的思路：只改动一个变量、其余条件全部固定，训练后统一测试，才可以准确判断超参数带来的影响。
