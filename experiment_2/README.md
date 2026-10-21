# 实验二：图像增强
> 数字图像处理实验，基于OpenCV、scikit‑image完成图像色彩转换、添加噪声、图像平滑滤波，手动实现彩色图像中值滤波。

## 一、实验目的
学会OpenCV的基本使用方法，利用OpenCV等计算机库对图像进行平滑、滤波等操作，实现图像增强。

## 二、实验环境
- Python 3.x
- 依赖库：
  - `opencv‑python`：图像处理
  - `scikit‑image`：`random_noise` 添加噪声
  - `numpy`：数值计算
  - `matplotlib`：绘图可视化
- 运行环境：conda虚拟环境 `cv`，Jupyter Notebook

### 环境安装
```bash
# 创建并进入虚拟环境
conda create -n cv python=3.10
conda activate cv

# 安装依赖
pip install opencv‑python scikit‑image numpy matplotlib

# jupyter内核注册
pip install ipykernel
python -m ipykernel install --user --name cv --display-name "Python (cv)"
