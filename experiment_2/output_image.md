# 图像输出说明（色彩转换 & 噪声添加）
> 本文件为实验2图像结果说明，图像资源存放在项目 assets 目录

## 文件说明
1. `bgr_image.jpg`：OpenCV原生读取BGR格式图像，直接用matplotlib展示会红蓝通道颠倒
2. `rgb_image.jpg`：BGR转RGB后的彩色图像，色彩正常
3. `gray_image.jpg`：灰度转换结果，仅保留亮度信息
4. `noise_comparison.jpg`：原图、椒盐噪声、高斯噪声对比图

## 效果图展示
### BGR原始图像
![BGR原始图像](../assets/bgr_image.jpg)

### BGR转RGB图像
![RGB图像](../assets/rgb_image.jpg)

### 灰度图像
![灰度图像](../assets/gray_image.jpg)

### 噪声对比：原图｜椒盐噪声｜高斯噪声
![噪声对比图](../assets/noise_comparison.jpg)

## 简要原理
OpenCV默认读取顺序是BGR，matplotlib按照RGB解析，直接显示会颜色错乱。
- 椒盐噪声：随机将像素置0/255，适合用中值滤波处理
- 高斯噪声：像素叠加正态分布随机扰动，适合高斯/均值滤波平滑
