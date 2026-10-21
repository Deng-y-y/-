# 实验二：图像增强
> 数字图像处理实验，基于 OpenCV、scikit-image 完成图像色彩转换、添加噪声、图像平滑滤波，手动实现彩色图像中值滤波，代码存放于本目录下的 Untitled.ipynb。

## 一、实验目的
学会OpenCV的基本使用方法，利用OpenCV等计算机库对图像进行平滑、滤波等操作，实现图像增强，理解BGR/RGB色彩空间差异、噪声类型特点以及不同滤波算法的原理与适用场景。

## 二、实验内容
完整可运行代码见本目录：[Untitled (1).ipynb](./Untitled (1).ipynb)
### 2.1 导入依赖包
```python
import cv2
from skimage.util import random_noise
import numpy as np
from matplotlib import pyplot as plt
```

### 2.2 读取原始图像并进行色彩空间转换

读取本地图像，获取坐标`(100,100)`位置像素 BGR 数值，完成 BGR→RGB、BGR→GRAY 色彩空间转换。

```
img = cv2.imread('../assets/jp1.jpg')
(b, g, r) = img[100, 100]
print(b, g, r)

# BGR转RGB
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# BGR转灰度图
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

输出像素结果：`82 86 241`

### 2.3 添加噪声

对 RGB 图像分别添加椒盐噪声与高斯噪声：

- 椒盐噪声：`mode='s&p', amount=0.4`，40% 像素被噪声污染
- 高斯噪声：`mode='gaussian', mean=0.2, var=0.03`

```
sp_noise_img = random_noise(rgb_img, mode='s&p', amount=0.4)
gus_noise_img = random_noise(rgb_img, mode='gaussian', mean=0.2, var=0.03)
```

> 
> 色彩转换与噪声图像详情展示：[output_image.md](./output_image.md)

### 2.4 图像滤波

使用 OpenCV 的均值滤波、中值滤波、高斯滤波分别对两种噪声图像做去噪。

```
# 均值滤波
mean_sp = cv2.blur(sp_noise_img, (5, 5))
mean_gus = cv2.blur(gus_noise_img, (5, 5))

# 中值滤波
mid_sp = cv2.medianBlur((sp_noise_img*255).astype(np.uint8), 5)
mid_gus = cv2.medianBlur((gus_noise_img*255).astype(np.uint8), 5)

# 高斯滤波
gauss_sp = cv2.GaussianBlur((sp_noise_img*255).astype(np.uint8), (5, 5), 0)
gauss_gus = cv2.GaussianBlur((gus_noise_img*255).astype(np.uint8), (5, 5), 0)
```

### 2.5 手动实现彩色图像中值滤波

不调用 OpenCV 滤波接口，对 R、G、B 三通道分别进行滑动窗口处理，取窗口内像素中位数完成去噪。

```
def manual_median_filter_color(image, kernel_size=5):
    pad = kernel_size // 2
    filtered_img = np.zeros_like(image)
    for c in range(3):
        channel = image[:, :, c]
        padded_channel = np.pad(channel, pad_width=pad, mode='edge')
        for i in range(channel.shape[0]):
            for j in range(channel.shape[1]):
                region = padded_channel[i:i + kernel_size, j:j + kernel_size]
                filtered_img[i, j, c] = np.median(region)
    return filtered_img

manual_mid = manual_median_filter_color((sp_noise_img * 255).astype(np.uint8), kernel_size=5)
```

> 
> 滤波效果详情查看：[result.md](./result.md)


## 三、实验结果与分析

1. **原始图像与颜色空间转换**
OpenCV 默认读取图像为 BGR 格式，直接使用 matplotlib 展示会出现红蓝通道反转；转换为 RGB 后色彩符合人眼视觉；灰度图像舍弃彩色信息只保留亮度。
2. **噪声效果**

- 椒盐噪声：随机产生黑白孤立噪点，局部破坏明显。
- 高斯噪声：整张图像叠加服从高斯分布的随机扰动，画面布满细小颗粒。

3. **滤波器对比**

- 均值滤波：适合处理高斯噪声，平滑图像；对椒盐噪声去除效果差，容易模糊边缘。
- 中值滤波：对椒盐噪声去除效果最好，能够保留边缘细节；对高斯噪声改善有限。
- 高斯滤波：适合高斯噪声，平滑效果柔和；对椒盐噪声效果一般。

4. **手动中值滤波**
彩色图像需要分通道滤波，手写实现的中值滤波效果与 OpenCV 内置接口接近，验证滑动窗口取中值的算法原理。

> 
> 总结：
> 
> 
> - 椒盐噪声：优先选用**中值滤波**
> - 高斯噪声：优先选用**均值滤波 / 高斯滤波**

## 四、实验小结

本实验在 Jupyter Notebook 中完成图像读取、BGR-RGB 色彩转换、添加椒盐与高斯噪声，使用多种滤波器完成图像去噪，同时手动实现彩色图像中值滤波。通过实验直观观察不同噪声的特点以及各类滤波器的适用场景，加深了对图像平滑滤波算法原理的理解，掌握 OpenCV 基础图像读写与变换操作。

## 相关文档跳转

- 完整实验代码：[Untitled.ipynb](./Untitled (1))
- 色彩转换、噪声效果图详情：[output_image.md](./output_image.md)
- 滤波去噪结果详情：[result.md](./result.md)
