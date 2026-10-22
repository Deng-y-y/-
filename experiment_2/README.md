# 实验二：图像增强
> 数字图像处理实验，基于 OpenCV、scikit-image 完成图像色彩转换、添加噪声、图像平滑滤波，手动实现彩色图像中值滤波，代码存放于本目录下的 Untitled.ipynb。

## 一、实验目的
学会OpenCV的基本使用方法，利用OpenCV等计算机库对图像进行平滑、滤波等操作，实现图像增强，理解BGR/RGB色彩空间差异、噪声类型特点以及不同滤波算法的原理与适用场景。

## 二、实验内容
完整可运行代码见本目录：[Untitled (1).ipynb](./Untitled%20(1).ipynb)

### 2.1 导入依赖包
```python
import cv2
from skimage.util import random_noise
import numpy as np
from matplotlib import pyplot as plt
```
这里导入本次实验要用到的工具包，cv2 用来读取和处理图片，random_noise 用来给图片加噪声，numpy 用来做像素数值处理，matplotlib 用来画图展示图像。

### 3.2 读取原始图像并进行色彩空间转换

```
img = cv2.imread('../assets/jp1.jpg')
(b, g, r) = img[100, 100]
print(b, g, r)

# BGR转RGB
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# BGR转灰度图
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

读取本地图像文件，获取【100, 100】位置像素的 BGR 三个通道数值并打印输出。OpenCV 默认读出来的图片通道顺序是 BGR，不是我们常说的 RGB，所以要做转换。转换成 RGB 之后用 matplotlib 显示颜色才正常；转成灰度图之后图片就只剩下明暗信息，没有彩色。
输出像素结果：`82 86 241`

> 
> 色彩转换与噪声图像详情展示：[output_image.md](./output_image.md)

### 2.3 添加噪声

```
# 添加椒盐噪声
sp_noise_img = random_noise(rgb_img, mode='s&p', amount=0.4)
# 添加高斯噪声
gus_noise_img = random_noise(rgb_img, mode='gaussian', mean=0.2, var=0.03)
# 截断像素值，防止高斯噪声产生<0或>1的越界值，限制像素范围在[0,1]
sp_noise_img = np.clip(sp_noise_img,0,1)
gus_noise_img = np.clip(gus_noise_img,0,1)
```

给彩色原图加上两种常见噪声。椒盐噪声会随机在画面生成黑白小点，这里设置有 40% 的像素会被噪声污染；高斯噪声是铺满整张图的细碎颗粒噪点。高斯噪声计算的时候，像素值有可能跑出 0~1 的范围，所以用 np.clip 把数值限制在合法区间，避免画图的时候颜色错乱。

### 3.4 图像滤波

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

分别使用均值滤波、中值滤波、高斯滤波，对两种带噪声的图片做去噪处理，滤波窗口大小都设置为 5×5。均值滤波是取邻域像素平均值来平滑画面；中值滤波取邻域像素中间值，对付黑白噪点效果很好；高斯滤波给近处像素更高权重，画面会柔和地变平滑。

### 3.5 手动实现彩色图像中值滤波

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

不直接调用 OpenCV 自带的滤波函数，自己写代码实现中值滤波。彩色图有 R、G、B 三个通道，需要分开遍历每一个通道，用滑动窗口在图片上逐像素移动，取窗口里面像素的中位数作为新像素值，最后合成去噪后的彩色图，用来验证中值滤波的原理。

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

- 完整实验代码：[Untitled (1).ipynb](./Untitled%20(1).ipynb)
- 色彩转换、噪声效果图详情：[output_image.md](./output_image.md)
- 滤波去噪结果详情：[result.md](./result.md)
