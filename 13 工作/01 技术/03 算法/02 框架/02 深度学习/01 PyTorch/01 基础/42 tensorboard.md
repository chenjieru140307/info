
## 可视化工具

在训练神经网络时，我们希望能更直观地了解训练情况，包括损失曲线、输入图片、输出图片、卷积核的参数分布等信息。这些信息能帮助我们更好地监督网络的训练过程，并为参数优化提供方向和依据。最简单的办法就是打印输出，但其只能打印数值信息，不够直观，同时无法查看分布、图片、声音等。


在本节，我们将介绍两个深度学习中常用的可视化工具：Tensorboard和Visdom。

### Tensorboard

Tensorboard最初是作为TensorFlow的可视化工具迅速流行开来。作为和TensorFlow深度集成的工具，Tensorboard能够展现你的TensorFlow网络计算图，绘制图像生成的定量指标图以及附加数据。但同时Tensorboard也是一个相对独立的工具，只要用户保存的数据遵循相应的格式，tensorboard就能读取这些数据并进行可视化。这里我们将主要介绍如何在PyTorch中使用tensorboardX[^1]进行训练损失的可视化。

TensorboardX是将Tensorboard的功能抽取出来，使得非TensorFlow用户也能使用它进行可视化，几乎支持原生TensorBoard的全部功能。


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200525/w10VJq9CWOX1.png?imageslim">
</p>


tensorboard的安装主要分为以下两步：

- 安装TensorFlow：如果电脑中已经安装完TensorFlow 可以跳过这一步，具体安装教程参见[TensorFlow官网](https://www.tensorflow.org/install/)，或使用pip直接安装，推荐使用[清华的软件源](https://mirrors.tuna.tsinghua.edu.cn/help/tensorflow/)。
- 安装tensorboard: `pip install tensorboard`
- 安装tensorboardX：可通过 `pip install tensorboardX`命令直接安装。

tensorboardX的使用：


- 首先用如下命令启动tensorboard：
  - `tensorboard --logdir <your/running/dir> --port <your_bind_port>`

举例：

```py
from tensorboardX import SummaryWriter

# 构建logger对象，logdir用来指定log文件的保存路径
# flush_secs用来指定刷新同步间隔
logger = SummaryWriter(log_dir='logs', flush_secs=2)

for ii in range(100):
    logger.add_scalar('data/loss', 10 - ii ** 0.5)
    logger.add_scalar('data/accuracy', ii ** 0.5 / 10)
```

说明：

- 打开浏览器输入`http://localhost:6006`（其中6006应改成你的tensorboard所绑定的端口），即可看到如图2所示的结果。
- 左侧的Horizontal Axis下有三个选项，分别是：
  - Step：根据步长来记录，log_value时如果有步长，则将其作为x轴坐标描点画线。
  - Relative：用前后相对顺序描点画线，可认为logger自己维护了一个`step`属性，每调用一次log_value就自动加１。
  - Wall：按时间排序描点画线。
- 左侧的Smoothing条可以左右拖动，用来调节平滑的幅度。点击右上角的刷新按钮可立即刷新结果，默认是每30s自动刷新数据。可见tensorboard_logger的使用十分简单，但它只能统计简单的数值信息，不支持其它功能。

感兴趣的读者可以从github项目主页获取更多信息，本节将把更多的内容留给另一个可视化工具：Visdom。