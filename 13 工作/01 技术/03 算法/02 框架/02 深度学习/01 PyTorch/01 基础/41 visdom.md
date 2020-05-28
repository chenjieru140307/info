# visdom






## Visdom打不开及其解决方案

**新版的visdom已经解决了这个问题,只需要升级即可**
```
pip install --upgrade visdom
```
之前的[解决方案](https://github.com/chenyuntc/pytorch-book/blob/2c8366137b691aaa8fbeeea478cc1611c09e15f5/README.md#visdom%E6%89%93%E4%B8%8D%E5%BC%80%E5%8F%8A%E5%85%B6%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88) 不再需要，已删除。



### Visdom

[Visdom](https://github.com/facebookresearch/visdom)是Facebook专门为PyTorch开发的一款可视化工具，其开源于2017年3月。Visdom十分轻量级，但却支持非常丰富的功能，能胜任大多数的科学运算可视化任务。


Visdom可以创造、组织和共享多种数据的可视化，包括数值、图像、文本，甚至是视频，其支持PyTorch、Torch及Numpy。用户可通过编程组织可视化空间，或通过用户接口为生动数据打造仪表板，检查实验结果或调试代码。

Visdom中有两个重要概念：

- env：环境。不同环境的可视化结果相互隔离，互不影响，在使用时如果不指定env，默认使用`main`。不同用户、不同程序一般使用不同的env。
- pane：窗格。窗格可用于可视化图像、数值或打印文本等，其可以拖动、缩放、保存和关闭。一个程序中可使用同一个env中的不同pane，每个pane可视化或记录某一信息。

如图所示，当前env共有两个pane，一个用于打印log，另一个用于记录损失函数的变化。点击 clear 按钮可以清空当前env的所有 pane，点击save按钮可将当前env保存成json 文件，保存路径位于 `~/.visdom/` 目录下。也可修改env的名字后点击fork，保存当前env的状态至更名后的env。


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200525/k7vWHKOUGjiz.png?imageslim">
</p>

安装：

- `pip install visdom`

启动：

- 通过 `python -m visdom.server`命令启动visdom服务，或通过`nohup python -m visdom.server &` 命令将服务放至后台运行。
- Visdom服务是一个web server服务，默认绑定8097端口，客户端与服务器间通过tornado 进行非阻塞交互。


使用注意：

- 需手动指定保存 env，可在 web 界面点击 save 按钮或在程序中调用 save 方法，否则 visdom 服务重启后，env 等信息会丢失。
- 客户端与服务器之间的交互采用 tornado异步框架，可视化操作不会阻塞当前程序，网络异常也不会导致程序退出。

Visdom 以 Plotly 为基础，支持丰富的可视化操作，下面举例说明一些最常用的操作。


```py
import torch as t
import visdom


# 新建一个连接客户端
# 指定env = u'test1'，默认端口为8097，host是‘localhost'
vis = visdom.Visdom(env=u'test1',use_incoming_socket=False)

x = t.arange(1, 30, 0.01)
y = t.sin(x)
vis.line(X=x, Y=y, win='sinx', opts={'title': 'y=sin(x)'})
```


输出：

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200525/Etx5sYDIak8U.png?imageslim">
</p>


说明：

- vis = visdom.Visdom(env=u'test1')，用于构建一个客户端，客户端除指定env之外，还可以指定host、port等参数。
- vis作为一个客户端对象，可以使用常见的画图函数，包括：
    - line：类似Matlab中的`plot`操作，用于记录某些标量的变化，如损失、准确率等
    - image：可视化图片，可以是输入的图片，也可以是GAN生成的图片，还可以是卷积核的信息
    - text：用于记录日志等文字信息，支持html格式
    - histgram：可视化分布，主要是查看数据、参数的分布
    - scatter：绘制散点图
    - bar：绘制柱状图
    - pie：绘制饼状图
    - 更多操作可参考visdom的github主页
    
这里主要介绍深度学习中常见的line、image和text操作。

Visdom同时支持PyTorch的tensor和Numpy的ndarray两种数据结构，但不支持Python的int、float等类型，因此每次传入时都需先将数据转成ndarray或tensor。上述操作的参数一般不同，但有两个参数是绝大多数操作都具备的：

- win：用于指定pane的名字，如果不指定，visdom将自动分配一个新的pane。如果两次操作指定的win名字一样，新的操作将覆盖当前pane的内容，因此建议每次操作都重新指定win。
- opts：选项，接收一个字典，常见的option包括`title`、`xlabel`、`ylabel`、`width`等，主要用于设置pane的显示格式。

之前提到过，每次操作都会覆盖之前的数值，但往往我们在训练网络的过程中需不断更新数值，如损失值等，这时就需要指定参数`update='append'`来避免覆盖之前的数值。而除了使用update参数以外，还可以使用`vis.updateTrace`方法来更新图，但`updateTrace`不仅能在指定pane上新增一个和已有数据相互独立的Trace，还能像`update='append'`那样在同一条trace上追加数据。

举例：

```py
import torch as t
import visdom

# 新建一个连接客户端
# 指定env = u'test1'，默认端口为8097，host是‘localhost'
vis = visdom.Visdom(env=u'test1', use_incoming_socket=False)

# append 追加数据
for ii in range(0, 10):
    y = x
    x = t.Tensor([ii])
    y = x
    vis.line(X=x, Y=y, win='polynomial', update='append' if ii > 0 else None)

# updateTrace 新增一条线
x = t.arange(0, 9, 0.1)
y = (x ** 2) / 9
vis.line(X=x, Y=y, win='polynomial', name='this is a new Trace', update='new')
```


输出：

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200525/wsm2PIBDdGyF.svg">
</p>

image的画图功能可分为如下两类：

- `image`接收一个二维或三维向量，$H\times W$ 或 $3 \times H\times W$，前者是黑白图像，后者是彩色图像。
- `images` 接收一个四维向量 $N\times C\times H\times W$，$C$ 可以是1或3，分别代表黑白和彩色图像。可实现类似torchvision中make_grid的功能，将多张图片拼接在一起。`images`也可以接收一个二维或三维的向量，此时它所实现的功能与image一致。

举例：

```py
import torch as t
import visdom

# 新建一个连接客户端
# 指定env = u'test1'，默认端口为8097，host是‘localhost'
vis = visdom.Visdom(env=u'test1', use_incoming_socket=False)

# 可视化一个随机的黑白图片
vis.image(t.randn(64, 64).numpy())

# 随机可视化一张彩色图片
vis.image(t.randn(3, 64, 64).numpy(), win='random2')

# 可视化36张随机的彩色图片，每一行6张
vis.images(t.randn(36, 3, 64, 64).numpy(), nrow=6, win='random3', opts={'title': 'random_imgs'})
```



输出：

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200525/3UKH9f74dMpH.png?imageslim">
</p>


`vis.text` 用于可视化文本，支持所有的html标签，同时也遵循着html的语法标准。例如，换行需使用`<br>`标签，`\r\n`无法实现换行。下面举例说明。

举例：

```py
import torch as t
import visdom

# 新建一个连接客户端
# 指定env = u'test1'，默认端口为8097，host是‘localhost'
vis = visdom.Visdom(env=u'test1', use_incoming_socket=False)

vis.text(u'''<h1>Hello Visdom</h1><br>Visdom是Facebook专门为<b>PyTorch</b>开发的一个可视化工具，
         在内部使用了很久，在2017年3月份开源了它。

         Visdom十分轻量级，但是却有十分强大的功能，支持几乎所有的科学运算可视化任务''',
         win='visdom',
         opts={'title': u'visdom简介'}
         )

```

输出：

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200525/6RNMxsc42zjS.png?imageslim">
</p>


## 在启动visdom服务时遇到如下情况，长时间无反应

问题：

```powershell
Downloading scripts, this may take a little while
```

解决办法：

1. 找到visdom模块安装位置
   其位置为`python`或`anaconda`安装目录下`\Lib\site-packages\visdon`
   ```
   ├─static
   │ ├─css
   │ ├─fonts
   │ └─js
   ├─__pycache__
   ├─__init__.py
   ├─__init__.pyi
   ├─py.typed
   ├─server.py
   └─VERSION
   ```
   可在`python`或`anaconda`安装目录下搜索找到
2. 修改文件`server.py`
   修改函数`download_scripts_and_run`，将`download_scripts()`注释掉
   该函数位于全篇末尾，1917行

```python
def download_scripts_and_run():
    # download_scripts()
    main()

if __name__ == "__main__":
    download_scripts_and_run()
```

1. 替换文件
   将文件覆盖到`\visdom\static`文件夹下

下载：

- [百度网盘](https://pan.baidu.com/s/1gAV4NCe8Mf4ukorny7GKNw) 
提取码：pb9y


至此，该问题解决完毕。
使用命令`python -m visdom.server`开启服务
