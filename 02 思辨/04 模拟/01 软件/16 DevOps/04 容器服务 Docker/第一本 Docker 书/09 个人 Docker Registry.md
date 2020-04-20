# 个人 Docker Registry

显然，拥有Docker镜像的一个公共的Registry非常有用。但是，有时候我们可能希望构建和存储包含不想被公开的信息或数据的镜像。

这时候我们有以下两种选择：

- 利用Docker Hub上的私有仓库。
- 在防火墙后面运行你自己的Registry。

感谢Docker公司的团队开源了他们用于运行Docker Registry的代码[5]，这样我们就可以基于此代码在内部运行自己的Registry。目前Registry还不支持用户界面，只能以API服务的方式来运行。

提示 如果在代理或者公司防火墙之后运行Docker，也可以使用HTTPS_PROXY、HTTP_PROXY和NO_PROXY等选项来控制Docker如何互连。

## 从容器运行 Registry

从Docker容器安装一个 Registry 非常简单：

```sh
$ sudo docker run -p 5000:5000 registry:2
```

说明：

- 该命令将会启动一个运行 Registry 应用 2.0 版本的容器，并将5000端口绑定到本地宿主机。

注意：

- 从Docker 1.3.1 开始，需要在启动Docker守护进程的命令中添加 --insecure-registry localhost:5000 标志，并重启守护进程，才能使用本地Registry。<span style="color:red;">没明白？</span>


## 测试新 Registry

<span style="color:red;">这个例子没跑通，`docker.example.com:5000` 是哪里来的？在哪里设置的？</span>

那么如何使用新的Registry呢？

让我们先来看看是否能将本地已经存在的镜像 aaa/static_web上传到我们的新Registry上去。


```sh
$ sudo docker images aaa/static_web
REPOSITORY　　　　　　 TAG　　 ID　　　　　　　CREATED　　　　　 SIZE
aaa/static_web　latest　22d47c8cb6e5　24 seconds ago　12.29 kB (virtual 326 MB)
$ sudo docker tag 22d47c8cb6e5 docker.example.com:5000/aaa/static_web
$ sudo docker push docker.example.com:5000/aaa/static_web
The push refers to a repository [docker.example.com:5000/aaa/static_web] (len: 1)
Processing checksums
Sending image list
Pushing repository docker.example.com:5000/aaa/static_web (1 tags) Pushing 22d47c8cb6e556420e5d58ca5cc376ef18e2de93b5cc90e868a1bbc8318c1c Buffering to disk 58375952/? (n/a)
Pushing 58.38 MB/58.38 MB (100%)
. . .
$ sudo docker run -t -i docker.example.com:5000/aaa/static_web /bin/bash
```

说明：

- 通过 docker images命令来找到这个镜像的ID。即22d47c8cb6e5。
- 使用新的Registry给该镜像打上标签。为了指定新的Registry目的地址，需要在镜像名前加上主机名和端口前缀。在这个例子里，我们的 Registry 主机名为docker.example.com。
- 为镜像打完标签之后，就能通过docker push命令将它推送到新的Registry中去了。
- 可以使用docker run命令构建新容器。


这是在防火墙后面部署自己的Docker Registry的最简单的方式。我们并没有解释如何配置或者管理Registry。如果想深入了解如何配置认证和管理后端镜像存储方式，以及如何管理Registry等详细信息，可以在Docker Registry部署文档查看完整的配置和部署说明。


