# 实验一：计算机视觉开发环境搭建
## 一、实验目的
掌握Anaconda工具的基础使用方法，完成Python虚拟环境的构建；熟悉OpenCV计算机视觉库与CPU版PyTorch深度学习框架的部署流程，搭建可供后续计算机视觉实验使用的开发运行环境。

## 二、实验内容
### 1、Anaconda基础校验
Anaconda软件预先完成安装，在终端执行命令校验conda版本与基础配置参数，确认软件运行正常。

执行输出查看conda各项配置项，确认镜像、更新策略等参数无误。

### 2、虚拟环境创建与OpenCV库部署
1. 新建独立虚拟环境，指定Python解释器版本
```powershell
conda create -n cv python=3.9
```

2. 激活`cv`虚拟环境，使用国内清华镜像安装OpenCV‑Python
```powershell
conda activate cv
pip install opencv-python -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)
```

3. 查看环境已安装包，确认OpenCV安装结果
```powershell
(cv) PS C:\Users\25678\Desktop> pip list
```

> 关键包输出：

| Package | Version |
| ---- | ---- |
| numpy | 2.2.6 |
| opencv‑python | 4.12.0.88 |
| pip | 25.2 |

> 说明：本机使用Intel核显，无NVIDIA独立显卡，不支持CUDA硬件加速，不需要安装CUDA与cuDNN。

### 3、硬件信息查看
调用系统指令查看本机显卡硬件信息：
```powershell
wmic path win32_VideoController get name
```

### 4、PyTorch（CPU版本）框架安装
因国外源网络不稳定，使用清华镜像源安装CPU版PyTorch组件：
```powershell
pip install torch torchvision torchaudio -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple) --extra-index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
```

使用pip查询PyTorch相关包：
```powershell
(cv) PS C:\Users\25678\Desktop> pip list | findstr torch
```

### 5、PyTorch环境有效性验证
进入Python交互环境执行验证代码：
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```
> 输出版本号与`False`，CPU版本PyTorch验证通过。

## 三、实验总结
1. 本实验属于前置环境配置实验，完成计算机视觉实验平台搭建，为后续实验提供软件基础。
2. 掌握Anaconda虚拟环境创建、激活、包管理操作，理解虚拟环境实现项目依赖隔离的意义。
3. 在集成显卡设备上完成OpenCV、CPU版PyTorch部署，学会使用国内镜像源解决pip下载超时故障，完成整套开发环境有效性校验。

---

### 使用说明
1. 将上面全部内容复制，新建文本文档，粘贴后另存为 `README.md`，编码选择**UTF‑8**；
2. 如果要插入终端截图，在对应位置添加图片语法：`![环境验证截图](./xxx.png)`，图片和README放在同一个文件夹内；
3. 直接上传到GitHub对应实验文件夹即可。

需要我帮你再加一小节**实验遇到的问题与解决办法**吗，可以写你真实遇到的pip超时、conda list查不到pytorch这些情况。
