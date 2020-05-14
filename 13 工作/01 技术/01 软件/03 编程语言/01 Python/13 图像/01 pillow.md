# Pillow

文档：


- [文档](https://pillow.readthedocs.io/en/3.1.x/index.html)
- [github](https://github.com/python-pillow/Pillow)


举例：


```py
from PIL import Image

im = Image.open('a.png')  # 打开一张 RGB 图像
print(im.getbands())  # 获取 RGB 三个波段
print(im.mode)
print(im.size)
print(im.info)

im2 = im.resize((256, 256), Image.BICUBIC)
im3 = im.thumbnail((100, 100), Image.BICUBIC)
```

输出：

```txt
('R', 'G', 'B', 'A')
RGBA
(80, 80)
{'srgb': 0}
```

说明：

- 通道 bands。即使图像的波段数，RGB图像，灰度图像
- 模式（mode）：定义了图像的类型和像素的位宽。共计 9 种模式：
  - 1：1位像素，表示黑和白，但是存储的时候每个像素存储为 8bit。
  - L：8位像素，表示黑和白。
  - P：8位像素，使用调色板映射到其他模式。
  - RGB：3x8位像素，为真彩色。
  - RGBA：4x8位像素，有透明通道的真彩色。
  - CMYK：4x8位像素，颜色分离。
  - YCbCr：3x8位像素，彩色视频格式。
  - I：32位整型像素。
  - F：32位浮点型像素。
- 尺寸（size）：获取图像水平和垂直方向上的像素数
- im.info 返回值为字典对象
- im.resize()和 im.thumbnail()用到了滤波器，有 4 种不同的采样滤波器：
  - NEAREST：（默认）最近滤波。从输入图像中选取最近的像素作为输出像素。
  - BILINEAR：双线性内插滤波。在输入图像的 2*2矩阵上进行线性插值。
  - BICUBIC：双立方滤波。在输入图像的 4*4矩阵上进行立方插值。
  - ANTIALIAS：平滑滤波。对所有可以影响输出像素的输入像素进行高质量的重采样滤波，以计算输出像素值。


注意：

- Pillow 使用笛卡尔像素坐标系统，坐标(0，0)位于左上角。位于坐标（0，0）处的像素的中心位于（0.5，0.5）。
- 调色板模式（"P"）适用一个颜色调色板为每一个像素定义具体的颜色值。（什么意思？）


问题解决：

- 在运行 `from PIL import Image` 的时候，提示说有个问题：PIL: DLL load failed: specified procedure could not be found 。把 pillow 4.1.0 卸载了重新安装了 4.0.0 就可以了。
  ```Python
  pip uninstall pillow
  pip install pillow==4.0.0
  ```
