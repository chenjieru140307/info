# OverFeat

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180922/I16hbC6dj1.png?imageslim">
</p>


目的：

- 进行定位


介绍：

- 用滑动窗口和规则块生成候选框
- 再利用多尺度滑动窗口增加检测结果，解决图像目标形状复杂、尺寸不一问题
- 最后利用卷积神经网络和回归模型分类、定位目标。

效果：

- 首次将分类、定位以及检测三个计算机视觉任务放在一起解决
