## Docker

Docker 是一个开源的应用容器引擎，基于 [Go 语言](https://link.zhihu.com/?target=https%3A//www.runoob.com/go/go-tutorial.html) 并遵从 Apache2.0 协议开源。

Docker 可以让开发者打包他们的应用以及依赖包到一个轻量级、可移植的容器中，然后发布到任何流行的 Linux 机器上，也可以实现虚拟化。

容器是完全使用沙箱机制，相互之间不会有任何接口（类似 iPhone 的 app）,更重要的是容器性能开销极低。

**在Ubuntu上安装Docker**

[https://mirror.tuna.tsinghua.edu.cn/help/docker-ce/](https://link.zhihu.com/?target=https%3A//mirror.tuna.tsinghua.edu.cn/help/docker-ce/)

如果你过去安装过 docker，先删掉:

```text
sudo apt-get remove docker docker-engine docker.io
```

首先安装依赖:

```text
sudo apt-get install apt-transport-https ca-certificates curl gnupg2 software-properties-common
```

根据你的发行版，下面的内容有所不同。你使用的发行版：

信任 Docker 的 GPG 公钥:

```text
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```

对于 amd64 架构的计算机，添加软件仓库:

```bash
sudo add-apt-repository \
   "deb [arch=amd64] https://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
```

如果你是树莓派或其它ARM架构计算机，请运行:

```text
echo "deb [arch=armhf] https://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/ubuntu \
     $(lsb_release -cs) stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list
```

最后安装

```text
sudo apt-get update
sudo apt-get install docker-ce
```

**也可以使用安装包进行安装**

下载安装包

```text
wget https://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/ubuntu/dists/xenial/pool/stable/amd64/docker-ce_18.06.3~ce~3-0~ubuntu_amd64.deb
```

使用dpkg命令安装

```text
dpkg -i docker-ce_18.06.3~ce~3-0~ubuntu_amd64.deb
```

验证是否安装成功

```text
docker --version 
# 显示版本相关信息
# 安装成功
```

更多知识

[Docker 教程 | 菜鸟教程www.runoob.com![图标](https://pic4.zhimg.com/v2-c6ffee657c4358d50ff048456fd94ffb_180x120.jpg)](https://link.zhihu.com/?target=https%3A//www.runoob.com/docker/docker-tutorial.html)

推荐书目[《深入浅出Docker》([英\],Nigel,Poulton（奈吉尔·波尔顿）)【摘要 书评 试读】- 京东图书](https://link.zhihu.com/?target=https%3A//item.jd.com/12564378.html%3Fcu%3Dtrue)推荐书目

[《深入浅出Docker》([英\],Nigel,Poulton（奈吉尔·波尔顿）)【摘要 书评 试读】- 京东图书item.jd.com](https://link.zhihu.com/?target=https%3A//item.jd.com/12564378.html%3Fcu%3Dtrue)

## Pycharm

PyCharm是一种[Python](https://link.zhihu.com/?target=https%3A//baike.baidu.com/item/Python/407313) IDE，带有一整套可以帮助用户在使用Python语言开发时提高其效率的工具，比如调试、语法高亮、Project管理、代码跳转、智能提示、自动完成、单元测试、版本控制。此外，该IDE提供了一些高级功能，以用于支持Django框架下的专业Web开发。

下载地址：

[下载地址www.jetbrains.com](https://link.zhihu.com/?target=https%3A//www.jetbrains.com/pycharm/download/%23section%3Dlinux)

可以选择安装专业版还是免费版

在校大学生可以申请教育优惠，免费使用专业版

[申请地址www.jetbrains.com](https://link.zhihu.com/?target=https%3A//www.jetbrains.com/community/education/)

申请地址：

[JetBrains Products for Learningwww.jetbrains.com](https://link.zhihu.com/?target=https%3A//www.jetbrains.com/shop/eform/students)

方法一： 利用校园邮箱申请（.edu）

![img](https://pic4.zhimg.com/80/v2-debb6a02d02f2896f78964a3da8dcf43_hd.jpg)

如果没有教育邮箱就看看方式二吧

方式二（上传学生证件）：

![img](https://pic3.zhimg.com/80/v2-94ee5c2620940be855b1d1e484036616_hd.jpg)

![img](https://pic4.zhimg.com/80/v2-c6d0a6f31344cac7d3faed764a0c4a8f_hd.jpg)

然后等着邮件就可以了。

## Tensorflow安装

[Docker](https://link.zhihu.com/?target=https%3A//docs.docker.com/install/) 使用容器创建虚拟环境，以便将 TensorFlow 安装与系统的其余部分隔离开来。TensorFlow 程序在此虚拟环境中运行，该环境能够与其主机共享资源（访问目录、使用 GPU、连接到互联网等）。系统会针对每个版本测试 [TensorFlow Docker 映像](https://link.zhihu.com/?target=https%3A//hub.docker.com/r/tensorflow/tensorflow/)。

Docker 是在 Linux 上启用 TensorFlow [GPU 支持](https://link.zhihu.com/?target=https%3A//tensorflow.google.cn/install/gpu)的最简单方法，因为只需在主机上安装 [NVIDIA® GPU 驱动程序](https://link.zhihu.com/?target=https%3A//github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions%23how-do-i-install-the-nvidia-driver)（无需安装 NVIDIA® CUDA® 工具包）。注意不需要安装工具包。

### GPU 支持

Docker 是在 GPU 上运行 TensorFlow 的最简单方法，因为主机只需安装 [NVIDIA® 驱动程序](https://link.zhihu.com/?target=https%3A//github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions%23how-do-i-install-the-nvidia-driver)（无需安装 NVIDIA® CUDA® 工具包）。

安装 [nvidia-docker](https://link.zhihu.com/?target=https%3A//github.com/NVIDIA/nvidia-docker) 可启动支持 NVIDIA® GPU 的 Docker 容器。`nvidia-docker` 仅适用于 Linux，详情请参阅对应的[平台支持常见问题解答](https://link.zhihu.com/?target=https%3A//github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions%23platform-support)。

检查 GPU 是否可用：

```bash
intbjw@ubuntuos:~$ lspci | grep -i nvidia
01:00.0 3D controller: NVIDIA Corporation GP107M [GeForce GTX 1050 Mobile] (rev a1)
```

安装nvidia-docker：

```bash
# Add the package repositories
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
$ sudo systemctl restart docker
```

验证 `nvidia-docker` 安装：

```bash
docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
intbjw@ubuntuos:~$ docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
Unable to find image 'nvidia/cuda:latest' locally
latest: Pulling from nvidia/cuda
7ddbc47eeb70: Pull complete 
c1bbdc448b72: Pull complete 
8c3b70e39044: Pull complete 
45d437916d57: Pull complete 
d8f1569ddae6: Pull complete 
85386706b020: Pull complete 
ee9b457b77d0: Pull complete 
be4f3343ecd3: Pull complete 
30b4effda4fd: Pull complete 
Digest: sha256:e69c21509c9857dfd9fab762a53d5be6f9434eb75a6eb6a8e7e550e4ff56e045
Status: Downloaded newer image for nvidia/cuda:latest
Fri Jan  3 04:03:33 2020       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 435.21       Driver Version: 435.21       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 1050    Off  | 00000000:01:00.0 Off |                  N/A |
| N/A   46C    P5    N/A /  N/A |    437MiB /  2002MiB |      3%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```

### 使用支持 GPU 的映像的示例

下载并运行支持 GPU 的 TensorFlow 映像（可能需要几分钟的时间）：

```bash
docker run --runtime=nvidia -it --rm tensorflow/tensorflow:latest-gpu \
       python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"
```

设置支持 GPU 的映像可能需要一段时间。如果重复运行基于 GPU 的脚本，您可以使用 `docker exec` 重用容器。

使用最新的 TensorFlow GPU 映像在容器中启动 `bash` shell 会话：

```bash
docker run --runtime=nvidia -it tensorflow/tensorflow:latest-gpu bash
```

到这里你的Tensorflow就安装成功了。

## 在Pycharm用使用Tensorflow

在Settings中设置我们刚刚安装好的Tensorflow

点击这里，选择Add

![img](https://pic3.zhimg.com/80/v2-f62bef570eee178a3d6166c350b5cbae_hd.jpg)

选择Docker

![img](https://pic4.zhimg.com/80/v2-d12ce758583fd00c4a667d44247dc487_hd.jpg)

这里我已经创建过了，点击NEW。

![img](https://pic2.zhimg.com/80/v2-d850b1fbff53ab0fafe7850717733af5_hd.jpg)

显示Connection successful

在Image name中选中你刚刚安装好的Docker镜像。

## Hello World！！！

```python
from __future__ import print_function

import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')

# Start tf session
sess = tf.Session()

# Run the op
print(sess.run(hello))
```

![img](https://pic1.zhimg.com/80/v2-2997ca1f6436dc6fe7890173ad68afe0_hd.jpg)

end.

如果有错误欢迎指出，遇到问题也可以评论，或私聊我。觉得写的还可以给个赞再走。


# 相关

- [Ubuntu 18.04 + Docker + Pycharm打造机器学习平台Tensorflow](https://zhuanlan.zhihu.com/p/100841363)