AI造出的假图片恐怕很难再骗过AI了。

连英伟达本月刚上线的StyleGAN2也被攻破了。即使是人眼都分辨看不出来假脸图片，还是可以被AI正确鉴别。

![img](https://pic2.zhimg.com/80/v2-2c184e8b81a974684c7e06fa11af6ba1_1440w.jpg)

最新研究发现，只要用让AI学会鉴别某一只GAN生成的假图片，它就掌握了鉴别各种假图的能力。

不论是GAN生成的，Deepfake的，超分辨率的，还是怎样得来的，只要是AI合成图片，都可以拿一个通用的模型检测出来。

尽管各种CNN的原理架构完全不同，但是并不影响检测器发现造假的通病。

只要做好适当的预处理和后处理，以及适当的数据扩增，便可以鉴定图片是真是假，不论训练集里有没有那只AI的作品。

这就是Adobe和UC伯克利的科学家们发表的新成果。

有网友表示，如果他们把这项研究用来参加Kaggle的假脸识别大赛，那么将有可能获得最高50万美元奖金。

![img](https://pic2.zhimg.com/80/v2-bd3e94225cc968456c7c1dacb1cef99d_1440w.jpg)

![img](https://pic3.zhimg.com/80/v2-b1de24beff894fbcd634944ffe6293aa_1440w.jpg)

然而他们并没有，而是先在ArXiv公布了预印本，并且还被CVPR 2020收录。

最近，他们甚至将论文代码在GitHub上开源，还提供了训练后的权重供读者下载。

## 造出7万多张假图

要考验AI鉴别假货的能力，论文的第一作者、来自伯克利的学生Wang Sheng-Yu用11种模型生成了不同的图片，涵盖了各种CNN架构、数据集和损失。

所有这些模型都具有上采样卷积结构，通过一系列卷积运算和放大操作来生成图像，这是CNN生成图像最常见的设计。

![img](https://pic4.zhimg.com/80/v2-be90127c36aa780c1c10ef014eecdd13_1440w.jpg)

有ProGAN、StyleGAN、BigGAN、BigGAN、GauGAN等等，这些GAN各有特色。

ProGAN和StyleGAN为每个类别训练不同的网络；StyleGAN将较大的像素噪声注入模型，引入高频细节；BigGAN具有整体式的类条件结构；进行图像转换的GauGAN、CycleGAN、StarGAN。

除了GAN以外，还有其他处理图片的神经网络：

- 直接优化感知损失 ，无需对抗训练的级联细化网络（CRN）；
- 条件图像转换模型隐式最大似然估计（IMLE）；
- 改善低光照曝光不足的SITD模型；
- 超分辨率模型，即二阶注意力网络（SAN）；
- 用于换脸的的开源DeepFake工具faceswap。

![img](https://pic1.zhimg.com/80/v2-7b4ba368f9d80ea5d72eb58cf959d47c_1440w.jpg)

主流图片处理CNN模型应有尽有。他们总共造出了7万多张“假图”。

虽然生成这些图片所用的算法大相径庭、风格迥异，但是总有会有一些固有缺陷，这里面既有CNN本身的问题，也有GAN的局限性。

这是因为常见的CNN生成的内容降低了图片的表征能力，而这些工作大部分集中在网络执行上采样和下采样的方式上。下采样是将图像压缩，上采样是将图像插值到更大的分辨率上。

之前，Azulay和Weiss等人的研究表明，表明卷积网络忽略了经典的采样定理，而跨步卷积（strided convolutions）操作减少了平移不变性，导致很小的偏移也会造成输出的极大波动。

![img](https://pic3.zhimg.com/80/v2-d0dd006115d66fb525b2bf6d6860b14a_1440w.jpg)

另外，朱俊彦团队发表在ICCV 2019上的论文表明，GAN的生成能力有限，并分析了预训练GAN无法生成的图像结构。

今年7月，哥伦比亚大学的Zhang Xu等人进一步发现了GAN的“通病”，常见GAN中包含的上采样组件会引起伪像。

他们从理论上证明了，这些伪像在频域中表现为频谱的复制，这在频谱图上表现十分明显。

比如同样是一张马的图片，真实照片的信号主要集中在中心区域，而GAN生成的图像，频谱图上出现了四个小点。

![img](https://pic3.zhimg.com/80/v2-c9fdfc3ef0706cbfa6999e8e2bebb6fe_1440w.jpg)

因此他们提出了一种基于频谱而不是像素的分类器模型，在分辨假图像上达到了最先进的性能。

而Wang同学发现，不仅是GAN，其他的CNN在生成图像时，也会在频谱图中观察到周期性的图案。

![img](https://pic4.zhimg.com/80/v2-6d60f3fb38ac6cf89aff73cf07e3fbc3_1440w.jpg)

## 训练AI辨别真伪

刚才生成的数据集，包含了11个模型生成的假图。

不过，真假分类器并不是用这个大合集来训练的。

真正的训练集里，只有英伟达**ProGAN**这一个模型的作品，这是关键。

![img](https://pic2.zhimg.com/80/v2-f2782d444ed5c71fc4a31790e2b069a5_1440w.jpg)

### **△** ProGAN过往作品展

团队说，只选一个模型的作品用来训练，是因为这样的做法更能适应现实任务：

现实世界里，数据多样性永远是未知的，你不知道自己训练出的AI需要泛化到怎样的数据上。所以，干脆就用一种模型生成的图像来训练，专注于帮AI提升泛化能力。

而其他模型生成的作品，都是测试泛化能力用的。

(如果用很多模型的假图来训练，泛化任务就变得简单了，很难观察出泛化能力有多强。)

具体说来，真假分类器是个基于ResNet-50的网络，先在ImageNet上做了预训练，然后用ProGAN的作品做二分类训练。

![img](https://pic1.zhimg.com/80/v2-7770101eb5709c14a3c8263be1567820_1440w.jpg)

**△** ProGAN原理

不过，**训练集**不是一只ProGAN的作品。团队用了20只ProGAN，每只负责生成LSUN数据集里的一个类别。一只ProGAN得到3.6万张训练用图，200张验证用图，一半是生成的假图，一半是真图。

把20只ProGAN的成果加在一起，训练集有**72万张**，验证集有**4000张**。

为了把单一数据集的训练成果，推广到其他的数据集上，团队用了自己的方法：

最重要的就是**数据扩增**。先把所有图像左右翻转，然后用高斯模糊，JPEG压缩，以及模糊+JPEG这些手段来处理图像。

扩增手段并不特别，重点是让数据扩增以**后处理**的形式出现。团队说，这种做法带来了惊人的泛化效果 (详见后文) 。

训练好了就来看看成果吧。

## 明辨真伪

研究人员主要是用平均精度 (Average Precision) 这个指标，来衡量分类器的表现。

在多个不同的CNN模型生成的图片集里，ProGAN训练出的分类器都得到了不错的泛化：

![img](https://pic1.zhimg.com/80/v2-0fa6f87efc181cc52bdd3cd62dc88d48_1440w.jpg)

几乎所有测试集，AP分值都在90以上。只在StyleGAN的分值略低，是88.2。

不论是GAN，还是不用对抗训练、只优化感知损失的模型、还是超分辨率模型，还是Deepfake的作品，全部能够泛化。

团队还分别测试了不同因素对泛化能力产生的影响：

一是，数据扩增对泛化能力有所提升。比如，StyleGAN从96.3提升到99.6，BigGAN从72.2提升到88.2，GauGAN从67.0提升到98.1等等。更直观的表格如下，左边是没有扩增：

![img](https://pic1.zhimg.com/80/v2-84e057de24fb60c575d84c27e5912108_1440w.jpg)

另外，数据扩增也让分类器更加鲁棒了。

二是，数据多样性也对泛化能力有提升。还记得当时ProGAN生成了LSUN数据集里20个类别的图片吧。大体上看，用越多类别的图像来训练，得到的成绩就越好：

![img](https://pic2.zhimg.com/80/v2-93736f196c468d143b1373ed39f17115_1440w.jpg)

然后，再来试想一下，这时候如果突然有个新模型被开发出来，AI也能适应么？

这里，团队用了刚出炉没多久的英伟达**StyleGAN2**，发现分类器依然可以良好地泛化：

![img](https://pic3.zhimg.com/80/v2-5e0ce0bf9f4dbd5df848a1fe049beed2_1440w.jpg)

最后，还有一个问题。

AI识别假图，和人类用肉眼判断的机制一样么？

团队用了一个“Fakeness (假度) ”分值，来表示AI眼里一张图有多假。AI觉得越假，分值越高。

![img](https://pic3.zhimg.com/80/v2-c370bfc386aa9278e599f8394efbf416_1440w.jpg)

实验结果是，在大部分数据集里，AI眼里的假度，和人类眼里的假度，并没有明显的相关性。

只在BigGAN和StarGAN两个数据集上，假度分值越高时，能看到越明显的瑕疵。

更多数据集上没有这样的表现，说明分类器很有可能更倾向于学习**低层**的缺陷，而肉眼看到的瑕疵可能更偏向于**高层**。

## 安装使用

说完了论文，下面我们就可以去GitHub上体验一下这个模型的厉害了。

论文源代码基于PyTorch框架，需要安装NVIDIA GPU才能运行，因为项目依赖于CUDA。

首先将项目克隆到本地，安装依赖项。

```
pip install -r requirements.txt
```

考虑到训练成本巨大，作者还提供权重和测试集下载，由于这些文件存放在Dropbox上不便国内用户下载，在我们公众号中回复**CNN**即可获得国内网盘地址。

下载完成后将这两个文件移动到weights目录下。

然后我们就可以用来判别图像的真假了：

```
# Model weights need to be downloaded.
python demo.py examples/real.png weights/blur_jpg_prob0.1.pth
python demo.py examples/fake.png weights/blur_jpg_prob0.1.pth
```

如果你有能力造出一个自己的GAN，还可以用它来检测你模型的造假能力。

```
# Run evaluation script. Model weights need to be downloaded.
python eval.py
```

作者就用它鉴别了13种CNN模型制造的图片，证明了它的泛化能力。

![img](https://pic3.zhimg.com/80/v2-e0a6e469b05bfdd14ae5a7d7e1811da6_1440w.jpg)

## 闪闪发光作者团

这篇文章的第一作者是来自加州大学伯克利分校的**Wang Sheng-Yu**，他现在是伯克利人工智能研究实验室（BAIR）的一名研究生，在鉴别假图上是个好手。


今年他和Adobe合作的另一篇论文Detecting Photoshopped Faces by Scripting Photoshop，可以发现照片是否经过PS瘦脸美颜的操作，而且还能恢复“照骗”之前的模样。

![img](https://pic1.zhimg.com/80/v2-4535db266bf3ec182e55686b6f8541a4_1440w.jpg)

这篇的另一名作者Richard Zhang与Wang同学在上面的文章中也有合作，2018年之前他在伯克利攻读博士学位，毕业后进入Adobe工作。


这篇文章的通讯作者Alexei Efros，他曾是朱俊彦的导师，本文提到的CycleGAN正是出自朱俊彦博士之手。Alexei现在是加州大学伯克利分校计算机系教授，此前曾在CMU机器人学院任教9年。


## 传送门

论文地址：

CNN-generated images are surprisingly easy to spot... for now​arxiv.org

源代码：

https://github.com/peterwang512/CNNDetection​github.com
