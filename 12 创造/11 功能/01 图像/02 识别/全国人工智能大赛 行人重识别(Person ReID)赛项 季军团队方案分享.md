### 赛题任务

给定一张含有某个行人的查询图片，行人重识别算法需要在行人图像库中查找并返回特定数量的含有该行人的图片。

评测方法：50%首位准确率（Rank-1 Accuracy）+50% mAP（mean Average Precision）

和鲸科技赛事主页👇

首届“全国人工智能大赛”（行人重识别 Person ReID 赛项）​www.kesci.com

## 团队信息

团队名称：DMT

团队成员： 

[@罗浩.ZJU](https://www.zhihu.com/people/90ded85705d56c296e072bb4605aa8d3)

 

[@何shuting](https://www.zhihu.com/people/02bac87ffc7218c12a4677028b0cc63a)

 

 

[@古有志](https://www.zhihu.com/people/904db3f455564a0ae98bc0c7fb1255eb)

## 方案分享

## 初赛

在初赛阶段，我们团队以之前发表在TMM和CVPRW2019的ReID strong baseline作为baseline（已开源，补充信息1），针对初赛数据集进行了适应和改进。

![img](https://pic4.zhimg.com/80/v2-0baa1bd23cb80e5a352765e184885acb_1440w.jpg)

这篇ReID strong baseline主要在之前的baseline上，创新地提出了BNNeck的结构，解决了ID loss和triplet loss不会同步收敛的问题，并配合一些常见的训练技巧，使得模型的分数在Market1501可以达到94.5%的Rank1和85.9%的mAP。（参考资料1，2，补充信息1）

![img](https://pic4.zhimg.com/80/v2-f27cd55f54a069290a9dddac4849042b_1440w.jpg)

在ReID strong baseline基础上，针对初赛的数据集，进行针对性的改进，如下：

1. 数据预处理：适应比赛数据集

- 光照增广、随机擦除、随机Crop、随机翻转等
- 处理长尾数据、增大Image Size

\2. ID Loss：增强generalization

- - Cross Entropy → Arcface

\3. 模型替换：增强capacity

- - ResNet50 → ResNet101-IBN-a, ResNet101-IBN-b

![img](https://pic1.zhimg.com/80/v2-0f64be6458fdbebcc9b5b52ae06e6038_1440w.jpg)

\4. 优化器：更快的收敛

- Adam → Ranger

\5. 重排序：提高准确度

- K-reciprocal Re-ranking & Query Expansion（参考资料3）
- Test set augmentation

## **复赛**

### 问题描述

复赛阶段，我们着重解决以下3个问题问题：

1. 第一个问题，也是此次比赛的核心：Unsupervised Domain Adaptation (UDA)，即训练集和测试集分布不一致

\2. 大规模：训练集85K图像，测试集166K图像。由于需要引入测试集进行无监督训练，所以后处理需要大量计算资源

\3. 可复现：复赛赛题中，对于复现有时间和空间复杂度要求：48h，1GPU，90G内存，需要考虑如何充分利用时间和计算力

基于数据集的情况，我们统计了三通道像素值的均值和方差，将数据分成了3个部分（ABC域)，其中测试集中的C域在训练中从未出现，因为是一个UDA的问题。

![img](https://pic4.zhimg.com/80/v2-7186b74df45a1acbec954337a5284d77_1440w.jpg)

UDA问题，有两大类的方法解决，一类是GAN-Based对应的是source domain，另一类是Clustering-Based对应的是target domain。GAN-Based缺点是训练稳定性较差，质量难以保证，Clustering-Based缺点是通常需要迭代式聚类整个Target Domain。考虑到Clustering-Based方法在性能上更加鲁棒，且GAN模块训练需要消耗时间，它所带来的涨点是否值得也需要考虑，所以我们聚类这种思路。学术界定义的UDA问题中测试集只有target domain，而比赛的测试集既有source domain也有target domain，并没有一个很好的方法可以直接应用在现有的数据集上。

![img](https://pic1.zhimg.com/80/v2-e7447131e740580ff71d3afce684a644_1440w.jpg)

### JT-PC框架

为了UDA的问题，我们设计了渐进式无监督聚类联合训练算法（Joint Training and Progressive Clustering ，JT-PC)，主要结构如下

![img](https://pic3.zhimg.com/80/v2-a3275c0ca8829e364060ae9b45850106_1440w.jpg)

1. 使用全部训练数据训练模型1，模型1使用Resnet101-ibn-a，在训练集上直接训练。
2. 对图像进行均值和方差统计，将训练集分为A域和B域，将测试集分为A域和C域。利用模型1对测试集C域进行无监督聚类构造伪标签。
3. 训练模型2，模型2使用Resnet101-ibn-b，训练集包括两部分，一部分为去除单张图像的无长尾训练集，一部分为步骤2中标注的C域伪标签测试集。
4. 使用模型2对测试集C域再次构造伪标签。
5. 训练模型3，模型3采用SeResnet-ibn-a，训练集包括两部分，一部分为无长尾训练集，一部分为步骤4中挑选的伪标签测试集。
6. 对三个模型进行测试，并使用reranking重排，然后ensemble三个模型的结果。

其中构造伪标签的部分步骤为：

1. 以Query为中心，根据距离阈值进行聚类得到伪标签
2. 针对被打上多个伪标签的Outlier，取距离最小的中心为伪标签

![img](https://pic3.zhimg.com/80/v2-d1ee30851eeee64aa381ce9ce8b0beee_1440w.jpg)

确定模型之后，我们对其需要消耗的资源和性能进行评估。我们只使用Backbone的Global Feature，最终将训练+推理耗时压缩在了42.5h。可以看到我们的JT-PC框架使得模型在测试集的分数越来越高，并且这种方式非常适合业界的产品版本迭代，随着训练数据量的增多不断对模型进行无监督更新。

![img](https://pic2.zhimg.com/80/v2-38c7e19b516feec38b6bda93fb172f69_1440w.jpg)

### 工程优化

由于复赛对于程序运行时间和内存都有要求，目前开源的Re-ranking代码的资源消耗无法满足比赛要求，我们对Re-ranking的代码进行了无精度损失的工程优化。为了适合不同的使用场合，我们分为快速版和低耗版，快速版速度提升将近20倍，低耗版内存消耗减小接近7倍。最终Re-ranking算法将分数提升1.9%，16.6W测试集耗时为30分钟，内存消耗控制在30G。（参考资料3）

![img](https://pic4.zhimg.com/80/v2-952821afaf05bd0dec002ed16c4d147f_1440w.jpg)

### 复赛总结

复赛部分总结为以下几点：

1. 使用了一个性能与速度均不错的基准模型，相关论文已经被顶级期刊TMM和CVPRW会议接受。
2. 使用JT-PC无监督聚类的方式对测试集进行伪标签标注，并依据标签置信度将少量测试样本加入到训练集来训练模型，以增强模型的cross domain能力。
3. 优化reranking代码，最终实现速度提升16倍，内存使用减少3倍以上

## 决赛

我们分析了决赛使用的数据集，总结出了主要的四个难点：光线变化、跨模态互搜、属性标签的使用和姿态变化。因为比赛保密协议，所以细节无法透露更多，以下示例均挑自公开的学术数据集。

![img](https://pic3.zhimg.com/80/v2-61151ea03b7fe7b28fa48ae165d5768a_1440w.jpg)

针对上面四个问题，我们设计了模块化的解决方案，并将每个模块融合成为我们最终使用的模型框架，由于决赛保密协议的要求，决赛方案不能进行详细介绍：

![img](https://pic1.zhimg.com/80/v2-d61ef6a3895ac0480a2973caefcc6404_1440w.jpg)

### 跨模态(Cross-Modality)

CDP，Cross-Spectrum Dual-Subspace Pairing

![img](https://pic3.zhimg.com/80/v2-3181c0d9a0952529e47feea92542b9aa_1440w.jpg)

- Motivation：打乱颜色通道，让模型更加关注轮廓信息，比起只用灰图更加多样性
- Inference：只对灰图的Query进行排序，彩图由正常训练的ReID网络进行排序
- Performance：对于灰图Query（占比约1/6），性能提升约8%（提升~1%）
- Visualization：模型对于颜色信息不太敏感

（参考资料4，5）

### 属性标签

Two-Stage Attribute-Refined Network (TAN)

![img](https://pic2.zhimg.com/80/v2-bf3cc5083c3903f181f1bc132c3af7d5_1440w.jpg)

### 姿态变化（半身图）

（比赛禁止使用姿态点、语义分割模型）

Dynamically Matching Local Information (DMLI)

![img](https://pic3.zhimg.com/80/v2-bd7f3ac395c3512b2fd91269958f42ce_1440w.jpg)

（参考资料6，7）

### 决赛总结

决赛部分总结为以下几点：

1. 针对主要难点，以我们开源的Baseline为基础，设计模块化的解决方案
2. 针对特定数据和光线问题，进行了数据清洗与数据增强
3. 针对跨模态问题，提出了CDP方法
4. 针对属性标签，设计了TAN方法
5. 针对姿态变化，使用了我们的DMLI方法（参考资料4）
6. 相比于Baseline，性能提升明显

不同于其他队伍使用PCB、MGN等局部特征方案，我们队伍整个比赛过程大多只使用了backbone的global feature，backbone的模型压缩在业界已经比较成熟，所以非常适合产品的部署。该次比赛初赛和复赛阶段的代码后续将整理开源，敬请期待！

## 补充信息：

1. 一个更加强力的ReID Baseline

Github：[https://github.com/michuanhaohao/reid-strong-baseline](https://link.zhihu.com/?target=https%3A//github.com/michuanhaohao/reid-strong-baseline)

知乎专栏：[https://zhuanlan.zhihu.com/p/61831669](https://zhuanlan.zhihu.com/p/61831669)

\2. 队长罗浩个人主页：[http://luohao.site/](https://link.zhihu.com/?target=http%3A//luohao.site/)

## 参考资料：

1. Luo Hao et al. A Strong Baseline and Batch Normalization Neck for Deep Person Re-identification. TMM, 2019.
2. Luo Hao et al. Bag of Tricks and A Strong Baseline for Deep Person Re-identification. CVPRW, 2019, Oral.
3. Zhong Zhun et al. Re-ranking person re-identification with k-reciprocal encoding. CVPR, 2017.
4. Wu A, Zheng W S, Yu H X, et al. Rgb-infrared cross-modality person re-identification. ICCV, 2017.
5. Fan X, Luo Hao, et al.Cross-Spectrum Dual-Subspace Pairing for RGB-infrared Cross-Modality Person Re-Identification.
6. Luo Hao, et al. AlignedReID++: Dynamically matching local information for person re-identification. PR, 2019.
7. Luo Hao, et al. STNReID: Deep Convolutional Networks with Spatial Transformer Networks for Partial Person Re-Identification. TMM, 2020.