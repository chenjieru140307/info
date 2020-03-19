# YOLOv2 & YOLO9000


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180922/de04HBLHG7.png?imageslim">
</p>


目的:

- 解决 YOLO 召回率和定位精度方面的误差。


介绍：

- 使用 Darknet-19 作为特征提取网络
- 增加了批量归一化（Batch Normalization）的预处理
- 使用 224×224 和 448×448 两阶段训练 ImageNet 预训练模型后 fine-tuning。
- 相比于原来的 YOLO 是利用全连接层直接预测 bounding box的坐标，YOLOv2 借鉴了 Faster R-CNN 的思想，引入 anchor 机制，利用 K-Means 聚类的方式在训练集中聚类计算出更好的 anchor 模板，在卷积层使用 anchor boxes 操作，增加候选框的预测，同时采用较强约束的定位方法，大大提高算法召回率。结合图像细粒度特征，将浅层特征与深层特征相连，有助于对小尺寸目标的检测。


