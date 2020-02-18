# Docker 安装

## Ubuntu 安装

### 检查前提条件


1．内核

```sh
$ uname -a
Linux iZuf66eabunrloh2og4jgsZ 4.4.0-117-generic #141-Ubuntu SMP Tue Mar 13 11:58:07 UTC 2018 x86_64 x86_64 x86_64 GNU/Linux
```

说明：

- 是 4.4.0 x86_64 内核，是 Ubuntu 16.04 的。

2．检查Device Mapper

这里将使用Device Mapper作为存储驱动。任何Ubuntu 12.04或更高版本的宿主机应该都已经安装了Device Mapper。


代码清单2-5　检查Device Mapper

```sh
$ ls -l /sys/class/misc/device-mapper
lrwxrwxrwx 1 root root 0 Oct 22  2018 /sys/class/misc/device-mapper -> ../../devices/virtual/misc/device-mapper
```

如果没有出现device-mapper的相关信息，也可以尝试加载dm_mod模块：

```
$ sudo modprobe dm_mod
```

###　安装 Docker

将使用Docker团队提供的DEB软件包来安装Docker。

1. 添加Docker的APT仓库并自动将仓库的GPG公钥添加到宿主机中。

```sh
$ sudo lsb_release --codename | cut -f2
xenial
$ sudo sh -c "echo deb https://apt.dockerproject.org/repo ubuntu-xxxxx main > /etc/apt/sources.list.d/docker.list"
$ sudo apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys 58118E89F3A912897C070ADBF76221572C52609D
$ sudo apt-get update
```

说明：

1. 通过 lsb_release 获得 Ubuntu 发行版本。此处为 xenial。
1. 添加 Docker 的 APT 仓库。注：将 `ubuntu-xxxxx` 用 `ubuntu-xenial` 代替。
3. 要添加 Docker 仓库的 GPG 密钥。即添加软件源的 KEY。
4. 更新 APT 源。



2. 安装Docker软件包

```sh
$ sudo apt-get install docker-engine
$ docker info
Containers: 0
 Running: 0
 Paused: 0
 Stopped: 0
Images: 0
Server Version: 17.05.0-ce
Storage Driver: aufs
 Root Dir: /var/lib/docker/aufs
 Backing Filesystem: extfs
 Dirs: 0
 Dirperm1 Supported: true
Logging Driver: json-file
Cgroup Driver: cgroupfs
Plugins: 
 Volume: local
 Network: bridge host macvlan null overlay
Swarm: inactive
Runtimes: runc
Default Runtime: runc
Init Binary: docker-init
containerd version: 9048e5e50717ea4497b757314bad98ea3763c145
runc version: 9c2d8d184e5da67c95d601382adf14862e4f2228
init version: 949e6fa
Security Options:
 apparmor
 seccomp
  Profile: default
Kernel Version: 4.4.0-117-generic
Operating System: Ubuntu 16.04.5 LTS
OSType: linux
Architecture: x86_64
CPUs: 1
Total Memory: 1.953GiB
Name: iZuf66eabunrloh2og4jgsZ
ID: VK6M:67FI:HXPJ:NEOH:YGXF:AY2C:SIHS:Y5ND:PKP6:E2T4:7R3P:JCHG
Docker Root Dir: /var/lib/docker
Debug Mode (client): false
Debug Mode (server): false
Registry: https://index.docker.io/v1/
Experimental: false
Insecure Registries:
 127.0.0.0/8
Live Restore Enabled: false

WARNING: No swap limit support
```

说明：

1. 安装Docker软件包以及一些必需的软件包。
2. 安装完毕，用 docker info 命令查看 Docker 是否已经正常安装并运行。

安装完成之后, 有一个 `docker` 命令可供使用. 同时, `docker` 的服务默认监听在一个 **sock** 文件上(这样除了命令行工具, 各语言的 API 都很容易实现了).

权限方面, `docker` 的功能限制于 root 用户, docker 用户组. 所以, 你要么带着 `sudo` 用, 要么把当前用户加入到 docker 组:

```sh
$ sudo groupadd docker
$ sudo gpasswd -a zys docker
```

###　使用 UFW

在Ubuntu中，如果使用UFW[19]，即Uncomplicated Firewall，那么还需对其做一点儿改动才能让Docker工作。Docker使用一个网桥来管理容器中的网络。默认情况下，UFW会丢弃所有转发的数据包（也称分组）。因此，需要在UFW中启用数据包的转发，这样才能让Docker正常运行。

我们只需要对 `/etc/default/ufw` 文件做一些改动即可。

```sh
DEFAULT_FORWARD_POLICY="DROP"  原始的UFW转发策略
修改为：
DEFAULT_FORWARD_POLICY="ACCEPT" 新的UFW转发策略
```

保存修改内容并重新加载 UFW 防火墙：

```sh
$ sudo ufw reload
```



## 在 Windows 中安装 Docker Toolbox

注意：

Docker for Windows requires Windows 10 Pro or Enterprise version 14393, or Windows server 2016 RTM to run

如果使用的是Microsoft Windows系统，也可以使用Docker Toolbox工具快速上手Docker。Docker Toolbox是一个Docker组件的集合，还包括一个极小的虚拟机，在Windows宿主机上安装了一个支持命令行工具，并提供了一个Docker环境。

Docker Toolbox自带了很多组件，包括：

- VirtualBox；
- Docker客户端；
- Docker Compose（参见第7章）；
- Kitematic——一个Docker和Docker Hub的GUI客户端；
- Docker Machine——用于帮助用户创建Docker主机。

提示 也可以通过使用包管理器Chocolatey[24]来安装Docker客户端。

2.5.1　在Windows中安装Docker Toolbox

要在Windows中安装Docker Toolbox，需要从GitHub上下载相应的安装程序，可

以在https://www.docker.com/toolbox 找到。

首先也需要下载最新版本的Docker Toolbox，如代码清单2-36所示。

代码清单2-36　下载Docker Toolbox的. exe文件

$ wget https://github.com/docker/toolbox/releases/download/v1.9.1/DockerToolbox-1.9.1.exe





运行下载的安装文件，并根据提示安装Docker Toolbox，如图2-3所示。

图2-3 在Windows中安装Docker Toolbox

注意 只能在运行Windows 7.1、8/8.1或者更新版本上安装Docker Toolbox。

2.5.2　在Windows中启动Docker Toolbox

安装完Docker Toolbox后，就可以从桌面或者Applications文件夹运行Docker

CLI应用，如图2-4所示。

图2-4 在Windows中运行Docker Toolbox

2.5.3　测试Docker Toolbox

现在，就可以尝试使用将本机的Docker客户端连接虚拟机中运行的Docker守护进

程，来测试Docker Toolbox是否已经正常安装，如代码清单2-37所示。

代码清单2-37　在Windows中测试Docker Toolbox

$ docker info

Containers: 0

Images: 0

Driver: aufs

Root Dir: /mnt/sda1/var/lib/docker/aufs

Dirs: 0

. . .

Kernel Version: 3.13.3-tinycore64

太棒了！现在，Windows宿主机也可以运行Docker了！




## 使用本书的 Docker Toolbox示例

本书中的一些示例可能会要求通过网络接口或网络端口连接到某个容器，通常这个

地址是Docker服务器的localhost或IP地址。因为Docker Toolbox创建了一个本

地虚拟机，它拥有自己的网络接口和IP地址，所以我们需要连接的是Docker

Toolbox的地址，而不是你的localhost或你的宿主机的IP地址。

要想得到Docker Toolbox的IP地址，可以查看DOCKER_HOST环境变量的值。当在

OS X或者Windows上运行Docker CLI命令时，Docker Toolbox会设置这个变量的

值。

此外，也可以运行docker-machine ip命令来查看Docker Toolbox的IP地址，如

代码清单2-38所示。

代码清单2-38　获取Docker Toolbox的虚拟机的IP地址

$ docker-machine ip

The VM's Host only interface IP address is: 192.168.59.103

那么，来看一个要求连接localhost上容器的示例，比如使用curl命令，只需

将localhost替换成相应的IP地址即可。

因此，代码清单2-39所示的curl命令就变成了代码清单2-40所示的形式。

代码清单2-39　初始curl命令

$ curl localhost:49155

代码清单2-40　更新后的curl命令

$ curl 192.168.59.103:49155

另外，很重要的一点是，任何使用卷或带有-v选项的docker run命令挂载到

Docker容器的示例都不能在Windows上工作。用户无法将宿主机上的本地目录挂接

到运行在Docker Toolbox虚拟机内的Docker宿主机上，因为它们无法共享文件系

统。如果要使用任何带有卷的示例，如本书第5章和第6章中的示例，建议用户在基

于Linux的宿主机上运行Docker。

本书由「ePUBw.COM」整理，ePUBw.COM 提供最新最全的优质

电子书下载！！！





## Docker 安装脚本

我们可以使用远程安装脚本在相应的宿主机上安装Docker。


注意：

- 该脚本目前只支持在Ubuntu、Fedora、Debian和Gentoo中安装Docker，不久的未来可能会支持更多的系统。


可以


首先，需要确认curl命令已经安装，如代码清单2-41所示。

代码清单2-41　测试curl
```sh
$ whereis curl
curl: /usr/bin/curl /usr/share/man/man1/curl.1.gz
$ sudo apt-get -y install curl
$ curl https://get.docker.com/ | sudo sh
```

说明：

1. 确认 curl 命令已经安装
2. 如 curl 没有安装，进行安装。
3. 从 get.docker.com 网站获取安装脚本，使用安装脚本来安装Docker。这个脚本会自动安装Docker所需的依赖，并且检查当前系统的内核版本是否满足要求，以及是否支持所需的存储驱动，最后会安装Docker并启动Docker守护进程。

