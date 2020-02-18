# Docker 介绍


## Docker资源

- Docker官方主页（http://www.docker.com/）。
- Docker Hub（http://hub.docker.com）。
- Docker官方文档（http://docs.docker.com/）。
- Docker快速入门指南（http://www.docker.com/tryit/）。
- Docker Forge（https://github.com/dockerforge）：收集了各种Docker 工具、组件和服务。


## Docker 的应用场景

使用 Docker，可以快速构建一个应用程序服务器、一个消息总线、一套实用工具、一个持续集成（continuous integration，CI）测试环境或者任意一种应用程序、服务或工具。可以在本地构建一个完整的测试环境，也可以为生产或开发快速复制一套复杂的应用程序栈。可以说，Docker的应用场景相当广泛。

Docker的一些应用场景如下：

- 加速本地开发和构建流程，使其更加高效、更加轻量化。本地开发人员可以构建、运行并分享Docker容器。容器可以在开发环境中构建，然后轻松地提交到测试环境中，并最终进入生产环境。
- 能够让独立服务或应用程序在不同的环境中，得到相同的运行结果。这一点在面向服务的架构和重度依赖微型服务的部署中尤其实用。
- 用Docker创建隔离的环境来进行测试。例如，用Jenkins CI这样的持续集成工具启动一个用于测试的容器。<span style="color:red;">如何做到？</span>
- Docker可以让开发者先在本机上构建一个复杂的程序或架构来进行测试，而不是一开始就在生产环境部署、测试。
- 构建一个多用户的平台即服务（PaaS）基础设施。<span style="color:red;">如何做到？</span>
- 为开发、测试提供一个轻量级的独立沙盒环境，或者将独立的沙盒环境用于技术教学，如Unix shell的使用、编程语言教学。
- 提供软件即服务（SaaS）应用程序。<span style="color:red;">如何做到？</span>
- 高性能、超大规模的宿主机部署。




## Docker 包括以下几个部分。

<span style="color:red;">这个是不是比较旧了？</span>

- 一个原生的Linux容器格式，Docker中称为libcontainer。
- Linxu内核的命名空间（namespace）[9]，用于隔离文件系统、进程和网络。
- 文件系统隔离：每个容器都有自己的root文件系统。
- 进程隔离：每个容器都运行在自己的进程环境中。
- 网络隔离：容器间的虚拟网络接口和IP地址都是分开的。
- 资源隔离和分组：使用cgroups[10]（即control group，Linux的内核特性之一）将CPU和内存之类的资源独立分配给每个Docker容器。
- 写时复制[11]：文件系统都是通过写时复制创建的，这就意味着文件系统是分层的、快速的，而且占用的磁盘空间更小。
- 日志：容器产生的STDOUT、STDERR和STDIN这些IO流都会被收集并记入日志，用来进行日志分析和故障排错。
- 交互式shell：用户可以创建一个伪tty终端，将其连接到STDIN，为容器提供一个交互式的shell。

