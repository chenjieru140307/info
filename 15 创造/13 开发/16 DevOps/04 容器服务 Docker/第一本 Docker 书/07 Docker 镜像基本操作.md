# Docker 镜像基本操作

## 什么是Docker镜像

Docker镜像是由文件系统叠加而成。

<p align="center">
    <img width="50%" height="50%" src="http://images.iterate.site/blog/image/20200204/WSs8oVdutWTb.png?imageslim">
</p>



- 最底端是一个引导文件系统，即bootfs，这很像典型的 Linux/Unix的引导文件系统。Docker用户几乎永远不会和引导文件系统有什么交互。实际上，当一个容器启动后，它将会被移到内存中，而引导文件系统则会被卸载（unmount），以留出更多的内存供initrd磁盘镜像使用。到目前为止，Docker看起来还很像一个典型的Linux虚拟化栈。
- Docker镜像的第二层是root文件系统rootfs，它位于引导文件系统之上。rootfs可以是一种或多种操作系统（如Debian或者Ubuntu文件系统）。在传统的Linux引导过程中，root文件系统会最先以只读的方式加载，当引导结束并完成了完整性检查之后，它才会被切换为读写模式。但是在Docker里，root文件系统永远只能是只读状态，并且Docker利用联合加载（union mount）技术又会在root文件系统层上加载更多的只读文件系统。联合加载指的是一次同时加载多个文件系统，但是在外面看起来只能看到一个文件系统。联合加载会将各层文件系统叠加到一起，这样最终的文件系统会包含所有底层的文件和目录。

Docker将这样的文件系统称为镜像。一个镜像可以放到另一个镜像的顶部。位于下面的镜像称为父镜像（parent image），可以依次类推，直到镜像栈的最底部，最底部的镜像称为基础镜像（base image）。最后，当从一个镜像启动容器时，Docker会在该镜像的最顶层加载一个读写文件系统。我们想在Docker中运行的程序就是在这个读写层中执行的。



Docker文件系统层：


当Docker第一次启动一个容器时，初始的读写层是空的。当文件系统发生变化时，这些变化都会应用到这一层上。比如，如果想修改一个文件，这个文件首先会从该读写层下面的只读层复制到该读写层。该文件的只读版本依然存在，但是已经被读写层中的该文件副本所隐藏。通常这种机制被称为写时复制（copy on write），这也是使Docker如此强大的技术之一。每个只读镜像层都是只读的，并且以后永远不会变化。

当创建一个新容器时，Docker会构建出一个镜像栈，并在栈的最顶端添加一个读写层。这个读写层再加上其下面的镜像层以及一些配置数据，就构成了一个容器。

在上一章我们已经知道，容器是可以修改的，它们都有自己的状态，并且是可以启动和停止的。容器的这种特点加上镜像分层框架（image-layering framework），使我们可以快速构建镜像并运行包含我们自己的应用程序和服务的容器。


## 列出镜像

我们先从如何列出Docker主机上可用的镜像来开始Docker镜像之旅。

```py
root@iZuf66eabunrloh2og4jgsZ:~# docker images
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
ubuntu              latest              ccc6e87d482b        2 weeks ago         64.2MB
root@iZuf66eabunrloh2og4jgsZ:~# docker images ubuntu
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
ubuntu              latest              ccc6e87d482b        2 weeks ago         64.2MB
```

说明：

- 可以看到，我们已经获得了一个镜像列表，它们都来源于一个名为ubuntu的仓库。镜像来自之前我们执行的 docker run 命令，它在执行时同时进行了镜像下载。
- 通过 `docker images ubuntu` 来指定对应仓库的镜像。


注意：

- 本地镜像都保存在 Docker 宿主机的`/var/lib/docker`目录下。
- 每个镜像都保存在 Docker 所采用的存储驱动目录下面，如 aufs 或者 devicemapper。也可以在`/var/lib/docker/containers` 目录下面看到所有的容器。

镜像从仓库下载下来。镜像保存在仓库中，而仓库存在于Registry中。默认的Registry是由Docker公司运营的公共Registry服务，即Docker Hub：

<p align="center">
    <img width="50%" height="50%" src="http://images.iterate.site/blog/image/20200204/4J6ca4w5ongq.png?imageslim">
</p>


Docker Registry的代码是开源的，因此，可以运行自己的私有Registry。同时，Docker公司也提供了一个商业版的Docker Hub，即Docker Trusted Registry，这是一个可以运行在公司防火墙内部的产品，之前被称为Docker Enterprise Hub。

在Docker Hub（或者用户自己运营的Registry）中，镜像是保存在仓库中的。可以将镜像仓库想象为类似Git仓库的东西。它包括镜像、层以及关于镜像的元数据（metadata）。

每个镜像仓库都可以存放很多镜像（比如，ubuntu仓库包含了 Ubuntu 12.04、12.10、13.04、13.10和14.04的镜像）。让我们看一下ubuntu仓库的另一个镜像：

```sh
root@iZuf66eabunrloh2og4jgsZ:~# docker pull ubuntu:16.04
16.04: Pulling from library/ubuntu
0a01a72a686c: Pull complete 
cc899a5544da: Pull complete 
19197c550755: Pull complete 
716d454e56b6: Pull complete 
Digest: sha256:3f3ee50cb89bc12028bab7d1e187ae57f12b957135b91648702e835c37c6c971
Status: Downloaded newer image for ubuntu:16.04
root@iZuf66eabunrloh2og4jgsZ:~# docker images
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
ubuntu              16.04               96da9143fb18        2 weeks ago         124MB
ubuntu              latest              ccc6e87d482b        2 weeks ago         64.2MB
```

可见：

- 我们已经得到了Ubuntu的latest镜像和16.04镜像。这表明ubuntu镜像实际上是聚集在一个仓库下的一系列镜像。

注意：

- <span style="color:red;">我们虽然称其为Ubuntu操作系统，但实际上它并不是一个完整的操作系统。它只是一个裁剪版，只包含最低限度的支持系统运行的组件。</span>

为了区分同一个仓库中的不同镜像，Docker提供了一种称为标签（tag）的功能。每个镜像在列出来时都带有一个标签，如12.04、12.10、quantal或者precise等。每个标签对组成特定镜像的一些镜像层进行标记（比如，标签12.04就是对所有Ubuntu 12.04镜像的层的标记）。这种机制使得在同一个仓库中可以存储多个镜像。

我们可以通过在仓库名后面加上一个冒号和标签名来指定该仓库中的某一镜像：

```sh
$ sudo docker run -t -i --name new_container ubuntu:12.04 /bin/bash
root@79e36bff89b4:/#
```

在构建容器时指定仓库的标签也是一个很好的习惯。这样便可以准确地指定容器来源于哪里。不同标签的镜像会有不同，比如Ubutnu 12.04和14.04就不一样，指定镜像的标签会让我们确切知道自己使用的是ubuntu:12.04，这样我们就能准确知道自己在干什么。

Docker Hub中有两种类型的仓库：

- 用户仓库（user repository）
- 顶层仓库（top-level repository）

用户仓库的镜像都是由Docker用户创建的，而顶层仓库则是由Docker内部的人来管理的。

用户仓库的命名由用户名和仓库名两部分组成，如aaa/puppet。用户名：aaa。仓库名： puppet。

与之相对，顶层仓库只包含仓库名部分，如ubuntu仓库。顶层仓库由Docker公司和由选定的能提供优质基础镜像的厂商（如Fedora团队提供了fedora镜像）管理，用户可以基于这些基础镜像构建自己的镜像。同时顶层仓库也代表了各厂商和Docker公司的一种承诺，即顶层仓库中的镜像是架构良好、安全且最新的。


注意：

- 用户贡献的镜像都是由Docker社区用户提供的，这些镜像并没有经过Docker公司的确认和验证，在使用这些镜像时需要自己承担相应的风险。




## 拉取镜像

用docker run命令从镜像启动一个容器时，如果该镜像不在本地，Docker会先从Docker Hub下载该镜像。

如果没有指定具体的镜像标签，那么Docker会自动下载latest标签的镜像。

通常使用 docker pull 来拉取镜像

```py
$ sudo docker pull fedora:20
fedora:latest: The image you are pulling has been verified
782cf93a8f16: Pull complete
7d3f07f8de5f: Pull complete
511136ea3c5a: Already exists
Status: Downloaded newer image for fedora:20
```



## 查找镜像

我们也可以通过命令来查找所有Docker Hub上公共的可用镜像，如

```sh
root@iZuf66eabunrloh2og4jgsZ:~# docker search ubuntu
NAME                                                      DESCRIPTION                                     STARS     OFFICIAL   AUTOMATED
ubuntu                                                    Ubuntu is a Debian-based Linux operating s...   10445     [OK]       
dorowu/ubuntu-desktop-lxde-vnc                            Docker image to provide HTML5 VNC interfac...   387                  [OK]
rastasheep/ubuntu-sshd                                    Dockerized SSH service, built on top of of...   240                  [OK]
consol/ubuntu-xfce-vnc                                    Ubuntu container with "headless" VNC sessi...   208                  [OK]
ubuntu-upstart                                            Upstart is an event-based replacement for ...   105       [OK]       
...
root@iZuf66eabunrloh2og4jgsZ:~# 
root@iZuf66eabunrloh2og4jgsZ:~# docker pull darksheer/ubuntu
Using default tag: latest
latest: Pulling from darksheer/ubuntu
6b98dfc16071: Pull complete 
4001a1209541: Pull complete 
6319fc68c576: Pull complete 
b24603670dc3: Pull complete 
97f170c87c6f: Pull complete 
26645cab468d: Pull complete 
Digest: sha256:6767630851d0af4f26df67e0082abbb967307c140b2536c71a852a8f61f3080c
Status: Downloaded newer image for darksheer/ubuntu:latest
root@iZuf66eabunrloh2og4jgsZ:~# 
root@iZuf66eabunrloh2og4jgsZ:~# docker run -t -i darksheer/ubuntu /bin/bash


```

说明：

- docker search 可以查找所有 Docker Hub 上公共的可用镜像。
- 反馈内容为：
  - 仓库名；
  - 镜像描述；
  - 用户评价（Stars）—反应出一个镜像的受欢迎程度；
  - 是否官方（Official）—由上游开发者管理的镜像（如fedora镜像由Fedora 团队管理）；
  - 自动构建（Automated）—表示这个镜像是由Docker Hub的自动构建（Automated Build）流程创建的。

<span style="color:red;">什么是自动构建？</span>



## 删除镜像

如果不再需要一个镜像了，也可以将它删除。

```sh
$ sudo docker rmi aaa/static_web
Untagged: 06c6c1f81534
Deleted: 06c6c1f81534
Deleted: 9f551a68e60f
Deleted: 997485f46ec4
Deleted: a101d806d694
Deleted: 85130977028d
$ sudo docker rmi aaa/apache2 aaa/puppetmaster
$ sudo docker rmi 'docker images -a -q'
```

可见：

- 这里我们删除了aaa/static_web镜像。在这里也可以看到Docker的分层文件系统：每一个Deleted:行都代表一个镜像层被删除。
- 可以在命令行中指定一个镜像名列表来删除多个镜像。
- 可以使用 `'docker images -a -q'` 列出所有镜像后删除。


注意：

- 该操作只会将本地的镜像删除。如果之前已经将该镜像推送到Docker Hub上，那么它在Docker Hub上将依然存在。


如果想删除一个Docker Hub上的镜像仓库，需要在登录Docker Hub后使用Delete repository链接来删除：

<p align="center">
    <img width="50%" height="50%" src="http://images.iterate.site/blog/image/20200205/SaL9RzdcIMkD.png?imageslim">
</p>
