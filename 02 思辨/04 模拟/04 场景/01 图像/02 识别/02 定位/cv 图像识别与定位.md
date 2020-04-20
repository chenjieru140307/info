
# CV 图像识别与定位


## 缘由：

对图像识别与定位进行总结，之前还有一篇，要合并到这里来。


# 1.两种思路：

* 思路 1：视作回归
* 思路 2：借助图像窗口


图像识别与定位：


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/8eGaaD9aCG.png?imageslim">
</p>

ImageNet

实际上有 识别+定位 两个任务


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/LdmE8F2F8E.png?imageslim">
</p>




# 思路 1：看作回归问题


4个数字，用 L2 loss/欧⽒氏距离损失?


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/ce03E2hGc1.png?imageslim">
</p>

步骤 1:

  * 先解决简单问题，搭一个识别图像的神经网络
  * 在 AlexNet VGG GoogleLenet ResNet上 fine-tune一下

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/70BH2lLeA5.png?imageslim">
</p>

步骤 2:




  * 在上述神经网络的尾部展开


  * 成为 classification + regression模式




<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/bEFg1FB1m5.png?imageslim">
</p>

步骤 3:




  * Regression(回归)部分用欧氏距离损失


  * 使用 SGD 训练




<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/BGBAKGdGEa.png?imageslim">
</p>

步骤 4:

* 预测阶段把 2 个“头部”模块拼上
* 完成不同的功能

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/dLGb4hfjbm.png?imageslim">
</p>

Regression(回归)的模块部分加在什么位置？




* (最后的)卷积层后
* 全连接层后




<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/AE1H3lbIkl.png?imageslim">
</p>

能否对主体有更细致的识别？




  * 提前规定好有 K 个组成部分


  * 做成 K 个部分的回归




<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/1I5mKmhaEm.png?imageslim">
</p>

应用：如何识别人的姿势？




  * 每个人的组成部分是固定的


  * 对 K 个组成部分(关节)做回归预测 => 首尾相接的线段




<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/kAa9l31bIL.png?imageslim">
</p>




# 思路 2: 图窗+识别与整合






  * 类似刚才的 classification + regression思路


  * 咱们取不同的大小的“框”


  * 让框出现在不同的位置


  * 判定得分


  * 按照得分高低对“结果框”做抽取和合并




<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/EkDCbBK8j3.png?imageslim">
</p>

实际应用时




  * 尝试各种大小窗口


  * 甚至会在窗口上再做一些“回归”的事情




<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/5HBDLF8g20.png?imageslim">
</p>

想办法克服一下过程中的“参数多”与“计算慢”




  * 最初的形式如下




<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/7i49Khi22j.png?imageslim">
</p>

想办法克服一下过程中的“参数多”与“计算慢”




  * 用多卷积核的卷积层 替换 全连接层


  * 降低参数量




<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/6FllDDGCK4.png?imageslim">
</p>

想办法克服一下过程中的“参数多”与“计算慢”




  * 测试/识别 阶段的计算是可复用的(小卷积)


  * 加速计算




<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/8hFD3ig7Gi.png?imageslim">
</p>




# 物体识别


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/KeJGm1bIl6.png?imageslim">
</p>

再次看做回归问题？


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/cEg2hb7lHa.png?imageslim">
</p>

其实你不知道图上有多少个物体…


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/E5h5e5JAm6.png?imageslim">
</p>

试着看做分类问题？


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/53da08g2f3.png?imageslim">
</p>

看做分类问题，难点是？




  * 你需要找“很多位置”，给“很多不同大小的框”


  * 你还需要对框内的图像分类(累计很多次)


  * 框的大小不一定对


  * …


  * 当然，如果你的 GPU 很强大，恩，那加油做吧…


看做分类问题，有没有办法优化下？


  * 为什么要先给定“框”，能不能找到“候选框”？


  * 想办法先找到“可能包含内容的图框”


关于“候选图框”识别，有什么办法？


  * 自下而上融合成“区域”


  * 将“区域”扩充为“图框”




<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/5K85KbaBC1.png?imageslim">
</p>

“图框”候选：其他方式？


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/gkJ5j5Jl36.png?imageslim">
</p>



<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/GLL82ajHKC.png?imageslim">
</p>




# R-CNN




<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/GkdAjg5dGl.png?imageslim">
</p>

Girschick et al, “Rich feature hierarchies for accurate object detection and semantic segmentation”, CVPR 2014


## 步骤 1：找一个预训练好的模型(Alexnet,VGG)针对你的场景做 fine-tune




<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/93dmJeJ46j.png?imageslim">
</p>




## 步骤 2：fine-tuning模型


比如 20 个物体类别+1个背景


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/7JJFFAEf7c.png?imageslim">
</p>




## 步骤 3：抽取图片特征






  * 用“图框候选算法”抠出图窗


  * Resize后用 CNN 做前向运算，取第 5 个池化层做特征


  * 存储抽取的特征到硬盘/数据库上




<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/0KfHdKg7D0.png?imageslim">
</p>




## 步骤 4：训练 SVM 识别是某个物体或者不是(2分类)




<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/mEaiDHG77m.png?imageslim">
</p>



<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/bIljiGcdBI.png?imageslim">
</p>




## 步骤 5：bbox regression






  * 微调图窗区域




<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/5e4I6beEG8.png?imageslim">
</p>




# R-CNN => Fast-rcnn




## 针对 R-CNN的改进 1






  * 共享图窗计算，从而加速




<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/aHjD5j7A50.png?imageslim">
</p>




## 针对 R-CNN的改进 2






  * 直接做成端到端的系统




<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/8Lm1AKAAc1.png?imageslim">
</p>

关于 RIP：Region of Interest Pooling


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/1F3JaGeBIl.png?imageslim">
</p>

维度不匹配怎么办：划分格子 grid => 下采样


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/mKc1K57Ga0.png?imageslim">
</p>

RIP：Region of Interest Pooling   映射关系显然是可以还原回去的


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/A2ddflj1d6.png?imageslim">
</p>




## 速度对比




<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/4fk90lc5LF.png?imageslim">
</p>




# Fast => Faster-rcnn


Region Proposal(候选图窗)一定要另外独立做吗？

一起用 RPN 做完得了！


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/C8L18ceefc.png?imageslim">
</p>

Ren et al, “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks”, NIPS 2015

关于 RPN：Region Proposal Network


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/Ilkfk6kDgi.png?imageslim">
</p>



<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/BCAmmbI4IL.png?imageslim">
</p>

关于 Faster R-CNN的整个训练过程


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/Jbm49m8hlf.png?imageslim">
</p>




## 速度/准度对比




<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/G9m6La96l0.png?imageslim">
</p>




# 新方法：R-FCN


Jifeng Dai,etc “R-FCN: Object Detection via Region-based Fully Convolutional Networks ”, 2016


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/2EhFB17gaE.png?imageslim">
</p>

每个颜色代表不同的位置选择区域。
The bank of kxk score maps correspond to a kxk spatial grid describing relative positions.

训练损失:


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/l5gEEC8L2A.png?imageslim">
</p>

分类损失:


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/6jFHi7aIc7.png?imageslim">
</p>

Region-sensitive score maps and ROI pooling


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/gF20KLk7C7.png?imageslim">
</p>



<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/BgeKb43aLF.png?imageslim">
</p>






# COMMENT：

**与之前的 http://106.15.37.116/2018/04/02/cv-object-detection/ 有部分是重复的，因此要自己过滤之后进行拆分和梳理。**





R-CNN
(Cafffe + MATLAB): https://github.com/rbgirshick/rcnn (非常慢，看看就好)
Fast R-CNN
(Caffe + MATLAB): https://github.com/rbgirshick/fast-rcnn (非端到端)
Faster R-CNN
(Caffe + MATLAB): https://github.com/ShaoqingRen/faster_rcnn
(Caffe + python): https://github.com/rbgirshick/py-faster-rcnn
SSD
(Caffe + python)https://github.com/weiliu89/caffe/tree/ssd
R-FCN
(Caffe + Matlab) https://github.com/daijifeng001/R-FCN
(Caffe + python) https://github.com/Orpine/py-R-FCN


# 相关：

1. 七月在线 深度学习
