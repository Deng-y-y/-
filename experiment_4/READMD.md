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
`batch_size=2048`全程固定，保证三组对照公平；优先使用 CUDA GPU 加速；所有模型、数据都转到同一 device，避免张量设备不匹配报错。

### 2.3 MNIST 数据集加载

```
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=batch_size, shuffle=True)

print("训练数据形状:", train_loader.dataset.data.shape)
print("测试数据形式:", test_loader.dataset.data.shape)
```

输出：

```
训练数据形状: torch.Size([60000, 28, 28])
测试数据形式: torch.Size([10000, 28, 28])
```

**代码分析**
MNIST 包含 60000 张训练样本、10000 张测试样本，28×28 灰度图。`ToTensor()`将像素归一化至 [0,1]；训练集开启 shuffle 打乱顺序，消除样本顺序带来的偏向。数据集与预处理对三个训练分支完全一致。

### 2.4 基础 MLP 模型定义（三个训练分支共用）

```
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(784, 128)
        self.l2 = nn.Linear(128, 10)

    def forward(self, x):
        a1 = self.l1(x)
        x1 = F.relu(a1)
        a2 = self.l2(x1)
        x2 = a2
        return x2
```

**代码分析**
两层全连接 MLP：28×28 图片展平为 784 维向量输入，隐藏层 128 神经元，ReLU 激活；输出 10 维 logits，不使用额外 softmax，交给`F.cross_entropy`内部处理。三个训练分支网络结构完全不变，仅修改优化器 momentum 和 epoch。

## 三、训练过程（内含三个对照分支）

> 
> 说明：三个分支独立训练，网络结构、lr=0.1、batch_size 全部固定，仅调整 epoch 和 momentum。

### 分支 1：epoch=30，SGD momentum=0（基准组）

```
model1 = MLP().to(device)
optimizer1 = optim.SGD(model1.parameters(), lr=0.1, momentum=0)
epochs1 = 30

for epoch in range(epochs1):
    model1.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.view(x.shape[0], -1).to(device), y.to(device)
        optimizer1.zero_grad()
        output = model1(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer1.step()
    print(f"【分支1】Epoch [{epoch+1}/{epochs1}] - Loss: {loss.item():.4f}")
```

**分支训练逻辑分析**
momentum=0 为原始 SGD，参数更新只依赖当前批次梯度，没有历史梯度惯性。作为基准对照，用来和另外两个分支做横向比较。朴素 SGD 收敛偏慢，损失下降曲线平缓，在梯度很小的平坦区域更新幅度微弱。

### 分支 2：epoch=50，SGD momentum=0

```
model2 = MLP().to(device)
optimizer2 = optim.SGD(model2.parameters(), lr=0.1, momentum=0)
epochs2 = 50

for epoch in range(epochs2):
    model2.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.view(x.shape[0], -1).to(device), y.to(device)
        optimizer2.zero_grad()
        output = model2(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer2.step()
    print(f"【分支2】Epoch [{epoch+1}/{epochs2}] - Loss: {loss.item():.4f}")
```

**分支训练逻辑分析**
和分支 1 对比：momentum 保持 0 不变，只增加训练轮次至 50。控制变量为训练轮数，目的探究朴素 SGD 下单纯延长训练周期能否持续提升效果。朴素 SGD 缺少加速机制，进入损失平台后梯度很小，继续训练带来的精度增益有限。

### 分支 3：epoch=30，SGD momentum=0.9

```
model3 = MLP().to(device)
optimizer3 = optim.SGD(model3.parameters(), lr=0.1, momentum=0.9)
epochs3 = 30

for epoch in range(epochs3):
    model3.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.view(x.shape[0], -1).to(device), y.to(device)
        optimizer3.zero_grad()
        output = model3(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer3.step()
    print(f"【分支3】Epoch [{epoch+1}/{epochs3}] - Loss: {loss.item():.4f}")
```

**分支训练逻辑分析**
和分支 1 对比：epoch 固定 30 不变，仅修改 momentum=0.9。动量会保留历史梯度的更新惯性，梯度方向一致时加速收敛，梯度来回震荡时抑制抖动。同等轮数下，该分支前期损失下降速度明显快于无动量的分支 1；但 lr 偏大时，动量有可能越过最优解，后期出现小幅震荡。

## 四、统一测试评估（三个训练分支训练全部结束后执行）

训练完成后，依次对 model1、model2、model3 在同一个测试集上评估，测试代码复用相同逻辑，保证评估标准一致：

```
def test_model(model, tag):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.view(x.shape[0], -1).to(device), y.to(device)
            output = model(x)
            test_loss += F.cross_entropy(output, y)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
    test_loss = test_loss / (batch_idx + 1)
    acc = correct / len(test_loader.dataset)
    print(f"{tag} 测试结果 - loss:{test_loss:.4f}, acc:{acc:.4f}")
    return test_loss.item(), acc

loss1, acc1 = test_model(model1, "分支1：30epoch momentum=0")
loss2, acc2 = test_model(model2, "分支2：50epoch momentum=0")
loss3, acc3 = test_model(model3, "分支3：30epoch momentum=0.9")
```

**测试代码分析**
`model.eval()`切换评估模式；`with torch.no_grad()`关闭梯度图计算，节省显存、加快推理。沿类别维度取最大值索引得到预测标签，统计预测正确样本，计算测试集平均损失与准确率。三组模型使用完全一样的测试流程，保证评估结果具备可比性。

## 五、实验结果对比与分析

1. 分支 1 与分支 2 对比（momentum 均为 0，epoch 分别 30、50）
同为朴素 SGD，延长训练轮次到 50 轮，准确率会有小幅提升，但提升幅度不大。说明仅靠增加训练 epoch 的方式效率很低，朴素 SGD 到达损失平台后很难继续优化。
2. 分支 1 与分支 3 对比（epoch 均为 30，momentum 分别 0、0.9）
训练轮次完全相同，加入动量之后收敛速度更快，同等训练轮数下测试准确率更高。验证 SGD 动量的作用：利用历史梯度的惯性加速下降、削弱震荡。动量带来的收益远大于单纯增加训练轮数。

## 六、实验总结与心得

本次实验设置三个训练对照分支，在固定网络结构的前提下，分别探究训练轮次、SGD 动量对 MLP 分类效果的影响。分支 1 作为基准，分支 2 观察延长训练轮次的效果，分支 3 观察动量机制的效果，全部训练结束之后统一在测试集上评估。
通过对照可以看出：朴素 SGD 收敛缓慢，单纯增加训练轮数性价比低；动量给梯度下降引入惯性，是高效的优化策略。本次实验也体现了控制变量的思路：只改动一个变量、其余条件全部固定，训练后统一测试，才可以准确判断超参数带来的影响。
