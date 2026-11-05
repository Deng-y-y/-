# 基于SIFT特征的图像拼接实验报告
## 一、实验简介
本实验基于SIFT尺度不变特征变换算法，完成两张存在重叠区域图像的特征提取、特征匹配、单应矩阵求解、透视变换与图像融合拼接。

## 二、实验内容
本次实验步骤包括：
1. 读取两张原图，转为灰度图；
2. 分支A：不改变尺寸，保留原始分辨率差异；分支B：resize对齐两张图分辨率；
3. SIFT检测关键点，计算特征描述子；
4. KNN匹配 + Lowe阈值筛选优质匹配点；
5. RANSAC求解单应矩阵H；
6. 透视变换，计算全景画布大小，图像融合；
7. 可视化原图、灰度图、匹配图、拼接结果，代码输出文件保存至 `experiment_3/result/`；
8. 拓展测试：图像缩小、亮度降低，观察匹配鲁棒性。

完整可运行代码见本目录：[Untitled.ipynb](./Untitled.ipynb)
> 报告展示图片统一存放于项目根目录 `assets/`

### 2.1 未对齐图片尺寸进行试验
#### 2.1.1 导入依赖包
```python
import cv2
from skimage.util import random_noise
import numpy as np
from matplotlib import pyplot as plt
```
这里导入本次实验要用到的工具包，cv2 用来读取和处理图片，random_noise 用来给图片加噪声，numpy 用来做像素数值处理，matplotlib 用来画图展示图像。

#### 2.1.2 原始输入图
```python
# 读取两张原图，不做尺寸对齐，转为灰度图
img1 = cv2.imread("p1.jpg")
img2 = cv2.imread("p2.jpg")
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
```

读取两张待拼接实景图像，保留两张图片原始分辨率，不进行 resize 操作，再转换为灰度图用于后续 SIFT 特征提取。

> ![原始彩色原图](../assets/input_rgb_raw.jpg)
> 图 1：原始彩色输入图，两张图像像素尺寸不一致

> ![原始灰度图](../assets/input_gray_raw.jpg)
> 图 2：原始灰度图，直接使用原图尺寸，未做 resize 对齐

#### 2.1.3 SIFT 特征匹配结果
```
# SIFT特征提取 + KNN匹配 + Lowe比率筛选
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good = []
for m,n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)
match_img = cv2.drawMatches(img1, kp1, img2, kp2, good, None)
```

使用 SIFT 算子检测关键点、计算 128 维描述子；采用 KNN 匹配（k=2），通过 Lowe 比率阈值 0.7 过滤错误匹配，绘制匹配连线。因为两张图尺寸不同，绘图函数会自动填充黑色背景补齐画布。

> ![特征匹配图](../assets/matched_keypoints_1.jpg)
> 图 5：SIFT 关键点匹配可视化，连线为经过 Lowe 筛选后的优质匹配对

#### 2.1.4 图像拼接结果
```
# RANSAC求解单应矩阵 + 透视变换 + 图像融合
src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
# 透视变换画布计算、平移、叠加融合
```

利用匹配点对，通过 RANSAC 鲁棒求解单应矩阵 H，剔除误匹配外点；对待拼接图像做透视变换，计算全景画布边界，平移坐标消除负边界，最后叠加两张图像完成拼接。

> ![标准拼接结果](../assets/stitched_result_1.jpg)
> 图 6：标准图像全景拼接融合结果

### 2.2 将图片尺寸对齐以后再进行实验

#### 2.1.1 导入依赖包
```python
import cv2
from skimage.util import random_noise
import numpy as np
from matplotlib import pyplot as plt
```
这里导入本次实验要用到的工具包，cv2 用来读取和处理图片，random_noise 用来给图片加噪声，numpy 用来做像素数值处理，matplotlib 用来画图展示图像。

#### 2.2.2 原始输入图
```
# 读取原图，将两张图resize到相同分辨率
img1 = cv2.imread("p1.jpg")
img2 = cv2.imread("p2.jpg")
h,w = img1.shape[:2]
img2_resize = cv2.resize(img2, (w,h))
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2_resize, cv2.COLOR_BGR2GRAY)
```
读取图像后，将第二张图像缩放至和第一张图像相同的宽高，统一分辨率之后再转为灰度图，消除两张图尺寸差异带来的画布黑边。

> ![对齐后彩色原图](../assets/input_rgb_aligned.jpg)
> 图 3：resize 对齐分辨率后的彩色图

> ![对齐后灰度图](../assets/input_gray_aligned.jpg)
> 图 4：resize 对齐后的灰度图

#### 2.2.3 SIFT 特征匹配结果
```
# 同样的SIFT、KNN、Lowe筛选流程
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good = []
for m,n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)
match_img = cv2.drawMatches(img1, kp1, img2_resize, kp2, good, None)
```

使用和上一组完全一致的 SIFT 匹配参数，仅前置增加 resize 对齐步骤。图像分辨率一致后，匹配可视化画布不会出现大块黑色填充区域，画面更加整洁。

> ![特征匹配图](../assets/matched_keypoints_2.jpg)
> 图 5：SIFT 关键点匹配可视化，连线为经过 Lowe 筛选后的优质匹配对

#### 2.2.4 图像拼接结果
```
# 单应矩阵求解、透视变换、图像融合流程不变
src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
```

算法逻辑和未对齐组保持完全相同，保证单一变量，用来对比**尺寸是否对齐**对拼接效果、匹配可视化的影响。

> ![标准拼接结果](../assets/stitched_result_2.jpg)
> 图 6：标准图像全景拼接融合结果
## 6 结果分析
1. **尺寸差异的影响**
两张图片分辨率不一致时，`cv2.drawMatches` 会自动填充黑色背景来对齐画布，不影响特征匹配本身；但透视变换后的画布更大，边缘会出现黑边。resize对齐后不会出现匹配图黑块，可视化更整洁。

2. **SIFT鲁棒性**
在图像缩放、亮度衰减的测试中，依然可以提取到足够多稳定特征点，成功求解单应矩阵并完成拼接，体现SIFT对尺度、光照变化的不变性。

3. **存在的局限**
本实验为简单直接叠加融合，没有做羽化、加权融合，重叠区域接缝处存在明显分界线；如果存在大视差、非平面场景，单应矩阵模型不再适用，拼接会出现明显畸变。

## 7 实验总结
本次实验完成了SIFT特征提取、筛选匹配、单应矩阵求解、透视变换与图像拼接完整流程。
对比了原图尺寸不一致和对齐尺寸两种方案的区别，验证了SIFT算法在光照、缩放扰动下的鲁棒性。理解了基于单应矩阵的平面图像拼接原理，同时认识到简单叠加融合方案的不足。
