# 构建 Docker 镜像

前面我们已经看到了如何拉取已经构建好的带有定制内容的Docker镜像，那么我们如何修改自己的镜像，并且更新和管理这些镜像呢？

注意：

- 一般来说，我们不是真正“创建”新镜像，而是基于一个已有的基础镜像，如ubuntu或fedora等，构建新镜像而已。如果真的想从零构建一个全新的镜像，也可以参考 <https://docs.docker.com/articles/baseimages/>。


构建Docker镜像有以下两种方法。

- 使用docker commit命令。
- 使用docker build命令和Dockerfile文件。

现在我们并不推荐使用docker commit命令，而应该使用更灵活、更强大的Dockerfile来构建Docker镜像。

我们先介绍下 docker commit 命令：

还是会先介绍一下如何使用docker commit构建Docker镜像。之后，我们将重点介

绍Docker所推荐的镜像构建方法：编写Dockerfile之后使用docker build命令。


## 创建Docker Hub账号

<span style="color:red;">此处无法连接到网络，因此没有真正使用。</span>

构建镜像中很重要的一环就是如何共享和发布镜像。可以将镜像推送到Docker Hub 或者用户自己的私有Registry中。为了完成这项工作，需要在Docker Hub上创建一个账号，可以从 <https://hub.docker.com/account/signup/> 加入Docker Hub。

登录 Docker Hub：

```sh
$ sudo docker login
Username: aaa
Password:
Email: james@lovedthanlost.net
Login Succeeded
```

说明：

- 这条命令将会完成登录到Docker Hub的工作，并将认证信息保存起来以供后面使用。
- 可以使用docker logout命令从一个Registry服务器退出。

注意：

- 用户的个人认证信息将会保存到 $HOME/.docker/config.json。

## 用 Docker 的 commit 命令创建镜像

创建Docker镜像的第一种方法是使用docker commit命令。

可以将此想象为我们是在往版本控制系统里提交变更。我们先创建一个容器，并在容器里做出修改，就像修改代码一样，最后再将修改提交为一个新镜像。

先从创建一个新容器开始，这个容器基于我们前面已经见过的ubuntu镜像。

```sh
$ sudo docker run -i -t ubuntu /bin/bash
root@4aab3ce3cb76:/#
root@4aab3ce3cb76:/# apt-get -yqq update
root@4aab3ce3cb76:/# apt-get -y install apache2
root@4aab3ce3cb76:/# exit
$ sudo docker commit 4aab3ce3cb76 aaa/apache2
8ce0ea7a1528
$ sudo docker images aaa/apache2
aaa/apache2 latest 8ce0ea7a1528 13 seconds ago 90.63 MB
$ sudo docker commit -m "A new custom image" -a "James Turnbull" 4aab3ce3cb76 aaa/apache2:webserver
f99ebb6fed1f559258840505a0f5d5b6173177623946815366f3e3acff01adef
$ sudo docker inspect aaa/apache2:webserver
[{
　　"Architecture": "amd64",
　　"Author": "James Turnbull",
　　"Comment": "A new custom image",
　　. . .
}]
$ sudo docker run -t -i aaa/apache2:webserver /bin/bash
```



说明：

- 创建一个要进行修改的定制容器。
- 在容器中安装Apache。我们会将这个容器作为一个Web服务器来运行，所以我们想把它的当前状态保存下来。这样就不必每次都创建一个新容器并再次在里面安装Apache了。为了完成此项工作，需要先使用exit命令从容器里退出，之后再运行docker commit命令。
- docker commit命令中，指定了要提交的修改过的容器的ID（可以通过docker ps -l -q命令得到刚创建的容器的ID），以及一个目标镜像仓库和镜像名，这里是aaa/apache2。需要注意的是，docker commit提交的只是创建容器的镜像与容器的当前状态之间有差异的部分，这使得该更新非常轻量。
- 我们可以看到我们提交的镜像，这个镜像现在存放在 aaa/apache2 仓库里，在本地。
- 可以在提交镜像时指定更多的数据（包括标签）来详细描述所做的修改。在这条命令里，我们指定了更多的信息选项。
  - 首先 `-m` 选项用来指定新创建的镜像的提交信息。
  - 同时还指定了`--a`选项，用来列出该镜像的作者信息。
  - 接着指定了想要提交的容器的ID。
  - 最后的 aaa/apache2 指定了镜像的用户名和仓库名，并为该镜像增加了一个 webserver 标签。
- 可以用docker inspect命令来查看新创建的镜像的详细信息。
- 可以使用 docker run 来用刚创建的镜像建立一个容器。用了完整标签aaa/apache2:webserver来指定这个镜像。


## 用 Dockerfile 构建镜像

推荐。

Dockerfile使用基本的基于DSL（Domain Specific Language)）语法的指令来构建一个Docker镜像，我们推荐使用Dockerfile方法来代替docker commit，因为通过前者来构建镜像更具备可重复性、透明性以及幂等性。

一旦有了Dockerfile，我们就可以使用docker build命令基于该Dockerfile中的指令构建一个新的镜像。

现在就让我们创建一个目录并在里面创建初始的Dockerfile。我们将创建一个包含简单Web服务器的Docker镜像：

```sh
$ mkdir static_web
$ cd static_web
$ touch Dockerfile
```

说明：

- 我们创建了一个名为static_web的目录用来保存Dockerfile，这个目录就是我们的构建环境（build environment），Docker则称此环境为上下文（context）或者构建上下文（build context）。Docker会在构建镜像时将构建上下文和该上下文中的文件和目录上传到Docker守护进程。这样Docker守护进程就能直接访问用户想在镜像中存储的任何代码、文件或者其他数据。
- 作为开始，我们还创建了一个空Dockerfile


编写 Dockerfile：


```docker
# Version: 0.0.1
FROM ubuntu:14.04
MAINTAINER James Turnbull "james@example.com"
RUN apt-get update && apt-get install -y nginx
RUN echo 'Hi, I am in your container' \
　　 >/usr/share/nginx/html/index.html
EXPOSE 80
```

说明：

- Dockerfile由一系列指令和参数组成。每条指令，如FROM，都必须为大写字母，且后面要跟随一个参数：FROM ubuntu:14.04。Dockerfile中的指令会按顺序从上到下执行，所以应该根据需要合理安排指令的顺序。
- 以 `#` 开头的行都会被认为是注释。
- 每个Dockerfile的第一条指令必须是FROM。FROM指令指定一个已经存在的镜像，后续指令都将基于该镜像进行，这个镜像被称为基础镜像（base iamge）。
- 接着指定了MAINTAINER指令，这条指令会告诉Docker该镜像的作者是谁，以及作者的电子邮件地址。这有助于标识镜像的所有者和联系方式。
- 在这些指令之后，我们指定了两条RUN指令。RUN指令会在当前镜像中运行指定的命令。
  - 在这个例子里，我们通过RUN指令更新了已经安装的APT仓库，安装了nginx 包，之后创建了/usr/share/nginx/html/index.html文件，该文件有一些简单的示例文本。像前面说的那样，每条RUN指令都会创建一个新的镜像层，如果该指令执行成功，就会将此镜像层提交，之后继续执行Dockerfile中的下一条指令。
  - 默认情况下，RUN指令会在shell里使用命令包装器/bin/sh -c来执行。如果是在一个不支持shell的平台上运行或者不希望在shell中运行（比如避免shell字符串篡改），也可以使用exec格式的RUN指令，如下所示：
    ```
    RUN [ "apt-get", " install", "-y", "nginx" ]
    ```
    在这种方式中，我们使用一个数组来指定要运行的命令和传递给该命令的每个参数。
- 接着设置了EXPOSE指令，这条指令告诉Docker该容器内的应用程序将会使用容器的指定端口。这并不意味着可以自动访问任意容器运行中服务的端口（这里是80）。出于安全的原因，Docker并不会自动打开该端口，而是需要用户在使用docker run运行容器时来指定需要打开哪些端口。一会儿我们将会看到如何从这一镜像创建一个新容器。
  - 可以指定多个EXPOSE指令来向外部公开多个端口。
  - Docker也使用EXPOSE指令来帮助将多个容器链接。用户可以在运行时以docker run命令通过--expose选项来指定对外部公开的端口。





每条指令都会创建一个新的镜像层并对镜像进行提交。

Docker大体上按照如下流程执行Dockerfile中的指令：

- Docker从基础镜像运行一个容器。
- 执行一条指令，对容器做出修改。
- 执行类似docker commit的操作，提交一个新的镜像层。
- Docker再基于刚提交的镜像运行一个新容器。
- 执行Dockerfile中的下一条指令，直到所有指令都执行完毕。

可见：

- 如果用户的Dockerfile由于某些原因（如某条指令失败了）没有正常结束，那么用户将得到了一个可以使用的镜像。这对调试非常有帮助：可以基于该镜像运行一个具备交互功能的容器，使用最后创建的镜像对为什么用户的指令会失败进行调试。



### 基于 Dockerfile 构建新镜像

执行docker build命令时，Dockerfile中的所有指令都会被执行并且提交，并且在该命令成功结束后返回一个新镜像。

```sh
root@iZuf66eabunrloh2og4jgsZ:/home/docker_t# cd static_web
root@iZuf66eabunrloh2og4jgsZ:/home/docker_t/static_web# docker build -t="aaa/static_web" .
Sending build context to Docker daemon  2.048kB
Step 1/5 : FROM ubuntu:14.04
14.04: Pulling from library/ubuntu
2e6e20c8e2e6: Pull complete 
30bb187ac3fc: Pull complete 
b7a5bcc4a58a: Pull complete 
Digest: sha256:ffc76f71dd8be8c9e222d420dc96901a07b61616689a44c7b3ef6a10b7213de4
Status: Downloaded newer image for ubuntu:14.04
 ---> 6e4f1fe62ff1
Step 2/5 : MAINTAINER James Turnbull "james@example.com"
 ---> Running in 93a2a517bf70
 ---> 3670aa2b5f03
Removing intermediate container 93a2a517bf70
Step 3/5 : RUN apt-get update && apt-get install -y nginx
 ---> Running in 59c3ba6de367
Get:1 http://security.ubuntu.com trusty-security InRelease [65.9 kB]
Ign http://archive.ubuntu.com trusty InRelease
Get:2 http://security.ubuntu.com trusty-security/main amd64 Packages [1032 kB]
Get:3 http://archive.ubuntu.com trusty-updates InRelease [65.9 kB]
Get:4 http://archive.ubuntu.com trusty-backports InRelease [65.9 kB]
Hit http://archive.ubuntu.com trusty Release.gpg
Get:5 http://archive.ubuntu.com trusty-updates/main amd64 Packages [1460 kB]
Get:6 https://esm.ubuntu.com trusty-infra-security InRelease
略...
Processing triggers for sgml-base (1.26+nmu4ubuntu1) ...
 ---> 2db3eac53057
Removing intermediate container 59c3ba6de367
Step 4/5 : RUN echo 'Hi, I am in your contaier'     >/usr/share/nginx/html/index.html
 ---> Running in f73b8aa9da83
 ---> 8dce2369cab5
Removing intermediate container f73b8aa9da83
Step 5/5 : EXPOSE 80
 ---> Running in 44d3b6741068
 ---> f17222e4f144
Removing intermediate container 44d3b6741068
Successfully built f17222e4f144
Successfully tagged aaa/static_web:latest
root@iZuf66eabunrloh2og4jgsZ:/home/docker_t/static_web# 
```

说明：

- 我们使用了docker build命令来构建新镜像。我们通过指定-t选项为新镜像设置了仓库和名称，本例中仓库为 aaa，镜像名为static_web。强烈建议各位为自己的镜像设置合适的名字以方便追踪和管理。也可以在构建镜像的过程中为镜像设置一个标签，其使用方法为“镜像名:标签”，
    ```sh
    $ sudo docker build -t="aaa/static_web:v1" .
    ```
- 上面命令中最后的 `.` 告诉Docker到本地目录中去找Dockerfile文件。也可以指定一个Git仓库的源地址来指定Dockerfile的位置，这里Docker假设在这个Git仓库的根目录下存在Dockerfile文件。
    ```sh
    $ sudo docker build -t="aaa/static_web:v1" git@github.com:aaa/docker-static_web
    ```
- `Sending build context to Docker daemon 2.56 kB` 可以看到构建上下文已经上传到了Docker守护进程。
- 之后，可以看到Dockerfile中的每条指令会被顺序执行，而且作为构建过程的最终结果，返回了新镜像的ID，即22d47c8cb6e5。构建的每一步及其对应指令都会独立运行，并且在输出最终镜像ID之前，Docker会提交每步的构建结果。


注意：

- 如果没有制定任何标签，Docker将会自动为镜像设置一个latest标签。
- 可以通过`-f`标志指定一个区别于标准Dockerfile的构建源的位置。例如，`docker build -t "aaa/static_- web" -f path/to/file`，这个文件可以不必命名为Dockerfile，但是必须要位于构建上下文之中。<span style="color:red;">什么是必须要位于构建上下文中？</span>
- 如果在构建上下文的根目录下存在以.dockerignore命名的文件的话，那么该文件内容会被按行进行分割，每一行都是一条文件过滤匹配模式。这非常像.gitignore文件，该文件用来设置哪些文件不会被当作构建上下文的一部分，因此可以防止它们被上传到Docker守护进程中去。该文件中模式的匹配规则采用了Go语言中的filepath[2]。


### 指令失败时会怎样

前面介绍了一个指令失败时将会怎样。下面来看一个例子：假设我们在第4步中将软

件包的名字弄错了，比如写成了ngin。

再来运行一遍构建过程并看看当指令失败时会怎样，如代码清单4-28所示。


```sh
root@iZuf66eabunrloh2og4jgsZ:/home/docker_t/static_web# docker build -t="aab/static_web" .
Sending build context to Docker daemon  2.048kB
Step 1/5 : FROM ubuntu:14.04
 ---> 6e4f1fe62ff1
Step 2/5 : MAINTAINER James Turnbull "james@example.com"
 ---> Using cache
 ---> 3670aa2b5f03
Step 3/5 : RUN apt-get update && apt-get install -y ngin
 ---> Running in c2eb341a51aa
Ign http://archive.ubuntu.com trusty InRelease
Get:1 http://security.ubuntu.com trusty-security InRelease [65.9 kB]
Get:2 http://archive.ubuntu.com trusty-updates InRelease [65.9 kB]
Get:3 http://security.ubuntu.com trusty-security/main amd64 Packages [1032 kB]
Get:4 http://archive.ubuntu.com trusty-backports InRelease [65.9 kB]
Get:5 https://esm.ubuntu.com trusty-infra-security InRelease
Get:6 https://esm.ubuntu.com trusty-infra-updates InRelease
Hit http://archive.ubuntu.com trusty Release.gpg
Get:7 https://esm.ubuntu.com trusty-infra-security/main amd64 Packages
Get:8 http://archive.ubuntu.com trusty-updates/main amd64 Packages [1460 kB]
Get:9 http://security.ubuntu.com trusty-security/restricted amd64 Packages [18.1 kB]
Get:10 http://security.ubuntu.com trusty-security/universe amd64 Packages [377 kB]
Get:11 https://esm.ubuntu.com trusty-infra-updates/main amd64 Packages
Get:12 http://security.ubuntu.com trusty-security/multiverse amd64 Packages [4730 B]
Get:13 http://archive.ubuntu.com trusty-updates/restricted amd64 Packages [21.4 kB]
Get:14 http://archive.ubuntu.com trusty-updates/universe amd64 Packages [671 kB]
Get:15 http://archive.ubuntu.com trusty-updates/multiverse amd64 Packages [16.1 kB]
Get:16 http://archive.ubuntu.com trusty-backports/main amd64 Packages [14.7 kB]
Get:17 http://archive.ubuntu.com trusty-backports/restricted amd64 Packages [40 B]
Get:18 http://archive.ubuntu.com trusty-backports/universe amd64 Packages [52.5 kB]
Get:19 http://archive.ubuntu.com trusty-backports/multiverse amd64 Packages [1392 B]
Hit http://archive.ubuntu.com trusty Release
Get:20 http://archive.ubuntu.com trusty/main amd64 Packages [1743 kB]
Get:21 http://archive.ubuntu.com trusty/restricted amd64 Packages [16.0 kB]
Get:22 http://archive.ubuntu.com trusty/universe amd64 Packages [7589 kB]
Get:23 http://archive.ubuntu.com trusty/multiverse amd64 Packages [169 kB]
Fetched 13.7 MB in 7s (1810 kB/s)
Reading package lists...
Reading package lists...
Building dependency tree...
Reading state information...
E: Unable to locate package ngin
The command '/bin/sh -c apt-get update && apt-get install -y ngin' returned a non-zero code: 100
root@iZuf66eabunrloh2og4jgsZ:/home/docker_t/static_web# docker run -t -i 3670aa2b5f03 /bin/bash
root@7817cae469f3:/# apt-get install -y nginx
```


说明：

- 我可以用docker run命令来基于这次构建到目前为止已经成功的最后一步创建一个容器，在这个例子里，使用的镜像ID是3670aa2b5f03。使用这个镜像来创建容器来进行调试。
- 在容器中再次运行 `apt-get install -y ngin`，并指定正确的包名，或者通过进一步调试来找出到底是哪里出错了。一旦解决了这个问题，就可以退出容器，使用正确的包名修改Dockerfile文件，之后再尝试进行构建。

### Dockerfile 和构建缓存

由于每一步的构建过程都会将结果提交为镜像，所以Docker的构建镜像过程就显得非常聪明。它会将之前的镜像层看作缓存。比如，在我们的调试例子里，我们不需要在第1步到第3步之间进行任何修改，因此Docker会将之前构建时创建的镜像当做缓存并作为新的开始点。实际上，当再次进行构建时，Docker会直接从第4步开始。当之前的构建步骤没有变化时，这会节省大量的时间。如果真的在第1步到第3步之间做了什么修改，Docker则会从第一条发生了变化的指令开始。

然而，有些时候需要确保构建过程不会使用缓存。比如，如果已经缓存了前面的第3步，即apt-get update，那么Docker将不会再次刷新APT包的缓存。这时用户可能需要取得每个包的最新版本。要想略过缓存功能，可以使用docker build的--no-cache标志：

```sh
$ sudo docker build --no-cache -t="aaa/static_web" .
```

### 基于构建缓存的Dockerfile模板

构建缓存带来的一个好处就是，我们可以实现简单的Dockerfile模板（比如在Dockerfile文件顶部增加包仓库或者更新包，从而尽可能确保缓存命中）。我一般都会在自己的Dockerfile文件顶部使用相同的指令集模板，比如对Ubuntu，使用代码清单4-31所示的模版。
Ubuntu系统的Dockerfile模板：

```docker
FROM ubuntu:14.04
MAINTAINER James Turnbull "james@example.com"
ENV REFRESHED_AT 2014-07-01
RUN apt-get -qq update
```

说明：

- 首先，我通过FROM指令为新镜像设置了一个基础镜像ubuntu:14.04。
- 接着，我又使用 MAINTAINER 指令添加了自己的详细联系信息。之后我又使用了一条新出现的指令ENV来在镜像中设置环境变量。在这个例子里，我通过ENV指令来设置了一个名为REFRESHED_AT的环境变量，这个环境变量用来表明该镜像模板最后的更新时间。有了这个模板，如果想刷新一个构建，只需修改ENV指令中的日期。这使Docker在命中ENV指令时开始重置这个缓存，并运行后续指令而无须依赖该缓存。也就是说，RUN apt-get update这条指令将会被再次执行，包缓存也将会被刷新为最新内容。<span style="color:red;">什么是命中这个时间？</span>
- 最后，我使用了RUN指令来运行 `apt-get -qq update` 命令。该指令运行时将会刷新APT包的缓存，用来确保我们能将要安装的每个软件包都更新到最新版本。


可以扩展此模板，比如适配到不同的平台或者添加额外的需求。比如，可以这样来支持一个fedora镜像：

```sh
FROM fedora:20
MAINTAINER James Turnbull "james@example.com"
ENV REFRESHED_AT 2014-07-01
RUN yum -q makecache
```

在 Fedora 中使用 Yum 实现了与上面的 Ubuntu 例子中非常类似的功能。

### 查看新镜像

现在来看一下新构建的镜像。

```sh
root@iZuf66eabunrloh2og4jgsZ:/home/docker_t/static_web# docker images aaa/static_web
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
aaa/static_web      latest              f17222e4f144        18 hours ago        232MB
root@iZuf66eabunrloh2og4jgsZ:/home/docker_t/static_web# docker history aaa/static_web
IMAGE               CREATED             CREATED BY                                      SIZE                COMMENT
f17222e4f144        18 hours ago        /bin/sh -c #(nop)  EXPOSE 80/tcp                0B                  
8dce2369cab5        18 hours ago        /bin/sh -c echo 'Hi, I am in your contaier...   26B                 
2db3eac53057        18 hours ago        /bin/sh -c apt-get update && apt-get insta...   35.2MB              
3670aa2b5f03        18 hours ago        /bin/sh -c #(nop)  MAINTAINER James Turnbu...   0B                  
6e4f1fe62ff1        6 weeks ago         /bin/sh -c #(nop)  CMD ["/bin/bash"]            0B                  
<missing>           6 weeks ago         /bin/sh -c mkdir -p /run/systemd && echo '...   7B                  
<missing>           6 weeks ago         /bin/sh -c set -xe   && echo '#!/bin/sh' >...   195kB               
<missing>           6 weeks ago         /bin/sh -c [ -z "$(apt-get indextargets)" ]     0B                  
<missing>           6 weeks ago         /bin/sh -c #(nop) ADD file:276b5d943a4d284...   196MB 
```

说明：

- docker history 命令可以探求镜像是如何构建出来的。从结果可以看到新构建的 `aaa/static_web` 镜像的每一层，以及创建这些层的Dockerfile指令。

### 从新镜像启动容器

我们也可以基于新构建的镜像启动一个新容器，来检查一下我们的构建工作是否一切正常。

```sh
root@iZuf66eabunrloh2og4jgsZ:/home/docker_t/static_web# docker run -d -p 80 --name static_web aaa/static_web nginx -g "daemon off;"
d86b6f82e406ab399b42f7e6f8a98827f2153f809c8da74bb3ae42a8577abd77
```

说明：

- 这里，我使用 `docker run` 命令，基于刚才构建的镜像的名字，启动了一个名为 static_web 的新容器。我们同时指定了`-d`选项，告诉 Docker 以分离（detached）的方式在后台运行。这种方式非常适合运行类似Nginx守护进程这样的需要长时间运行的进程。我们也指定了需要在容器中运行的命令：`nginx -g "daemon off;"`。这将以前台运行的方式启动Nginx，来作为我们的Web服务器。
- 我们这里也使用了一个新的`-p`标志，该标志用来控制Docker在运行时应该公开哪些网络端口给外部（宿主机）。这里 `docker run`命令将在Docker宿主机上随机打开一个端口，这个端口会连接到容器中的80端口上。运行一个容器时，Docker可以通过两种方法来在宿主机上分配端口：
  - Docker可以在宿主机上随机选择一个位于32768~61000的一个比较大的端口号来映射到容器中的80端口上。
  - 可以在Docker宿主机中指定一个具体的端口号来映射到容器中的80端口上。

使用docker ps命令来看一下容器的端口分配情况：


```sh
root@iZuf66eabunrloh2og4jgsZ:/home/docker_t/static_web# docker ps -l
CONTAINER ID        IMAGE               COMMAND                  CREATED             STATUS              PORTS                   NAMES
d86b6f82e406        aaa/static_web      "nginx -g 'daemon ..."   5 minutes ago       Up 5 minutes        0.0.0.0:32768->80/tcp   static_web
root@iZuf66eabunrloh2og4jgsZ:/home/docker_t/static_web# docker port static_web
80/tcp -> 0.0.0.0:32768
root@iZuf66eabunrloh2og4jgsZ:/home/docker_t/static_web# docker port static_web 80
0.0.0.0:32768
```

说明：

- 可以看到，容器中的80端口被映射到了宿主机的32768上。
- 我们也可以通过docker port来查看容器的端口映射情况。命令中我们指定了想要查看映射情况的容器的ID和容器的端口号，这里是80。该命令返回了宿主机中映射的端口，即32768。

`-p` 选项还为我们在将容器端口向宿主机公开时提供了一定的灵活性。比如，可以指定将容器中的端口映射到Docker宿主机的某一特定端口上。


```sh
$ sudo docker run -d -p 8090:80 --name static_web aaa/static_web nginx -g "daemon off;"
$ sudo docker run -d -p 127.0.0.1:80:80 --name static_web aaa/static_web nginx -g "daemon off;"
$ sudo docker run -d -p 127.0.0.1::80 --name static_web aaa/static_web nginx -g "daemon off;"
$ sudo docker run -d -P --name static_web aaa/static_web nginx -g "daemon off;"
```

说明：

- 上面的命令会将容器内的80端口绑定到本地宿主机的8090端口上。很明显，我们必须非常小心地使用这种直接绑定的做法：如果要运行多个容器，只有一个容器能成功地将端口绑定到本地宿主机上。这将会限制Docker的灵活性。
- 我们也可以将端口绑定限制在特定的网络接口（即IP地址）上，这里，我们将容器内的80端口绑定到了本地宿主机的127.0.0.1这个IP的80端口上。
- 我们也可以使用类似的方式将容器内的80端口绑定到一个宿主机的随机端口上。这里，我们并没有指定具体要绑定的宿主机上的端口号，只指定了一个IP地址127.0.0.1，这时我们可以使用docker inspect或者docker port命令来查看容器内的80端口具体被绑定到了宿主机的哪个端口上。
- -P参数，该参数可以用来对外公开在Dockerfile中通过EXPOSE指令公开的所有端口。该命令会将容器内的80端口对本地宿主机公开，并且绑定到宿主机的一个随机端口上。该命令会将用来构建该镜像的Dockerfile文件中EXPOSE指令指定的其他端口也一并公开。<span style="color:red;">哦，这个也可以。</span>


注意：

- 也可以通过在端口绑定时使用`/udp`后缀来指定UDP端口。
- 可以从http://docs.docker.com/userguide/dockerlinks/#network-port-mapping-refresher 获得更多关于端口重定向的信息。

有了这个端口号，就可以使用本地宿主机的IP地址或者127.0.0.1的localhost连接到运行中的容器，查看Web服务器内容了：

```sh
root@iZuf66eabunrloh2og4jgsZ:/home/docker_t/static_web# curl localhost:32768
Hi, I am in your contaier
```

这是就得到了一个非常简单的基于Docker的Web服务器。

注意：

- 可以通过 ifconfig 或者 ip addr 命令来查看本地宿主机的IP地址。<span style="color:red;">暂时不是很了解。</span>


### Dockerfile指令

我们已经看过了一些Dockerfile中可用的指令，如RUN和EXPOSE。但是，实际上还可以在Dockerfile中放入很多其他指令，这些指令包括CMD、ENTRYPOINT、ADD、COPY、VOLUME、WORKDIR、USER、ONBUILD、LABEL、STOPSIGNAL 和ENV等。可以在 http://docs.docker.com/reference /builder/ 查看Dockerfile中可以使用的全部指令的清单。

在后面的几章中我们还会学到更多关于Dockerfile的知识，并了解如何将非常酷的应用程序打包到Docker容器中去。

1．CMD

CMD指令用于指定一个容器启动时要运行的命令。这有点儿类似于RUN指令，只是RUN指令是指定镜像被构建时要运行的命令，而CMD是指定容器被启动时要运行的命令。这和使用docker run命令启动容器时指定要运行的命令非常类似。

比如：

```
$ sudo docker run -i -t aaa/static_web /bin/true
```

可以认为上述所示中的 `/bin/true` 的命令和在Dockerfile中使用如下 CMD 指令是等效的。

```
CMD ["/bin/true"]
```

当然也可以为要运行的命令指定参数：

```
CMD ["/bin/bash", "-l"]
```

这里我们将-l标志传递给了 `/bin/bash` 命令。

注意：

- 需要注意的是，要运行的命令是存放在一个数组结构中的。这将告诉Docker按指定的原样来运行该命令。当然也可以不使用数组而是指定CMD指令，这时候Docker会在指定的命令前加上/bin/sh -c。这在执行该命令的时候可能会导致意料之外的行为，所以Docker推荐一直使用以数组语法来设置要执行的命令。
- 而且，还需注意，使用docker run命令可以覆盖CMD指令。如果我们在Dockerfile 里指定了CMD指令，而同时在docker run命令行中也指定了要运行的命令，命令行中指定的命令会覆盖Dockerfile中的CMD指令。
- 深刻理解CMD和ENTRYPOINT之间的相互作用关系也非常重要，我们将在后面对此进行更详细的说明。

让我们来更贴近一步来看看这一过程。假设我们的Dockerfile文件中有上述的 CMD 指令 `CMD [ "/bin/bash" ]` ，可以使用docker build命令构建一个新镜像（假设镜像名为aaa/test），并基于此镜像启动一个新容器，如代码清单4-49所示。

```sh
$ sudo docker run -t -i aaa/test
root@e643e6218589:/_#_
```

注意到有什么不一样的地方了吗？在docker run命令的末尾我们并未指定要运行什么命令。实际上，Docker使用了CMD指令中指定的命令。

如果我指定了要运行的命令会怎样呢？

```sh
$ sudo docker run -i -t aaa/test /bin/ps
PID TTY 　　　 TIME CMD
1 ? 00:00:00 ps
$
```

可以看到，在这里我们指定了想要运行的命令 `/bin/ps`，该命令会列出所有正在运行的进程。在这个例子里，容器并没有启动shell，而是通过命令行参数覆盖了CMD指令中指定的命令，容器运行后列出了正在运行的进程的列表，之后停止了容器。

注意：

- 在Dockerfile中只能指定一条CMD指令。如果指定了多条CMD指令，也只有最后一条CMD指令会被使用。如果想在启动容器时运行多个进程或者多条命令，可以考虑使用类似Supervisor这样的服务管理工具。

2．ENTRYPOINT

ENTRYPOINT指令与CMD指令非常类似，也很容易和CMD指令弄混。这两个指令到底有什么区别呢？为什么要同时保留这两条指令？

正如我们已经了解到的那样，我们可以在docker run命令行中覆盖CMD指令。有时候，我们希望容器会按照我们想象的那样去工作，这时候CMD就不太合适了。而ENTRYPOINT指令提供的命令则不容易在启动容器时被覆盖。实际上，docker run命令行中指定的任何参数都会被当做参数再次传递给ENTRYPOINT指令中指定的命令。

让我们来看一个ENTRYPOINT指令的例子：

```docker
ENTRYPOINT ["/usr/sbin/nginx"]
或：
ENTRYPOINT ["/usr/sbin/nginx", "-g", "daemon off;"]
```

说明：

- 类似于CMD指令，我们也可以在该指令中通过数组的方式为命令指定相应的参数。

注意：

- 从上面看到的CMD指令可以看到，我们通过以数组的方式指定ENTRYPOINT在想运行的命令前加入/bin/sh -c来避免各种问题。

现在重新构建我们的镜像，并将ENTRYPOINT设置为`ENTRYPOINT ["/usr/sbin/ nginx"]`：

```sh
$ sudo docker build -t="aaa/static_web" .
$ sudo docker run –t -i aaa/static_web -g "daemon off;"
```

说明：

- 从上面可以看到，我们重新构建了镜像，并且启动了一个交互的容器。我们指定了 `-g "daemon off;"` 参数，这个参数会传递给用ENTRYPOINT指定的命令，在这里该命令为 `/usr/sbin/nginx -g "daemon off;"`。该命令会以前台运行的方式启动 Nginx守护进程，此时这个容器就会作为一台Web服务器来运行。

我们也可以组合使用ENTRYPOINT和CMD指令来完成一些巧妙的工作。比如，我们可能想在Dockerfile里指定代码清单4-55所示的内容。


```docker
ENTRYPOINT ["/usr/sbin/nginx"]
CMD ["-h"]
```

此时当我们启动一个容器时，任何在命令行中指定的参数都会被传递给Nginx守护进程。比如，我们可以指定-g "daemon off";参数让Nginx守护进程以前台方式运行。如果在启动容器时不指定任何参数，则在CMD指令中指定的-h参数会被传递给Nginx守护进程，即Nginx服务器会以/usr/sbin/nginx -h的方式启动，该命令用来显示Nginx的帮助信息。

这使我们可以构建一个镜像，该镜像既可以运行一个默认的命令，同时它也支持通过docker run命令行为该命令指定可覆盖的选项或者标志。

<span style="color:red;">不错。</span>

注意：

- 如果确实需要，用户也可以在运行时通过`docker run` 的 `--entrypoint` 标志覆盖 ENTRYPOINT 指令。

3．WORKDIR

WORKDIR指令用来在从镜像创建一个新容器时，在容器内部设置一个工作目录，`ENTRYPOINT`和`/`或`CMD`指定的程序会在这个目录下执行。

我们可以使用该指令为Dockerfile中后续的一系列指令设置工作目录，也可以为最终的容器设置工作目录。比如，我们可以如下所示这样为特定的指令设置不同的工作目录。

```docker
WORKDIR /opt/webapp/db
RUN bundle install
WORKDIR /opt/webapp
ENTRYPOINT [ "rackup" ]
```

说明：

- 这里，我们将工作目录切换为/opt/webapp/db后运行了bundle install命令。
- 之后又将工作目录设置为/opt/webapp，最后设置了ENTRYPOINT指令来启动rackup命令。

可以通过 `-w` 标志在运行时覆盖工作目录。

```sh
$ sudo docker run -ti -w /var/log ubuntu pwd
/var/log
```

该命令会将容器内的工作目录设置为 `/var/log`。

4．ENV

ENV指令用来在镜像构建过程中设置环境变量：

```docker
ENV RVM_PATH /home/rvm/
```

这个新的环境变量可以在后续的任何RUN指令中使用，这就如同在命令前面指定了环境变量前缀一样：

```docker
RUN gem install unicorn
```

该指令会以如下所示的方式执行：

```docker
RVM_PATH=/home/rvm/ gem install unicorn
```

可以在ENV指令中指定单个或多个环境变量。

```docker
ENV RVM_PATH=/home/rvm RVM_ARCHFLAGS="-arch i386"
```

也可以在其他指令中使用这些环境变量：

```docker
ENV TARGET_DIR /opt/app
WORKDIR $TARGET_DIR
```

在这里我们设定了一个新的环境变量TARGET_DIR，并在WORKDIR中使用了它的值。因此实际上WORKDIR指令的值会被设为/opt/app。

注意：

- 如果需要，可以通过在环境变量前加上一个反斜线来进行转义。<span style="color:red;">为什么要进行转义？</span>

这些环境变量也会被持久保存到从我们的镜像创建的任何容器中。所以，如果我们在使用`ENV RVM_PATH /home/rvm/`指令构建的容器中运行env命令，将会看到如下所示的结果。

```sh
root@bf42aadc7f09:~# env
. . .
RVM_PATH=/home/rvm/
. . .
```

也可以使用docker run命令行的`-e`标志来传递环境变量。这些变量将只会在运行时有效：

```sh
$ sudo docker run -ti -e "WEB_PORT=8080" ubuntu env
HOME=/
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOSTNAME=792b171c5e9f
TERM=xterm
WEB_PORT=8080
```

我们可以看到，在容器中 WEB_PORT 环境变量被设为了8080。

5．USER

USER 指令用来指定该镜像会以什么样的用户去运行。

```docker
USER nginx
```

基于该镜像启动的容器会以nginx用户的身份来运行。

我们可以指定用户名或UID以及组或GID，甚至是两者的组合：


```sh
USER user
或：
USER user:group
或：
USER uid
或：
USER uid:gid
或：
USER user:gid
或：
USER uid:group
```

也可以在docker run命令中通过`-u`标志来覆盖该指令指定的值。

如果不通过USER指令指定用户，默认用户为root。

6．VOLUME

<span style="color:red;">为什么需要这个卷？</span>

VOLUME指令用来向基于镜像创建的容器添加卷。一个卷是可以存在于一个或者多个容器内的特定的目录，这个目录可以绕过联合文件系统，并提供如下共享数据或者对数据进行持久化的功能。

- 卷可以在容器间共享和重用。
- 一个容器可以不是必须和其他容器共享卷。
- 对卷的修改是立时生效的。
- 对卷的修改不会对更新镜像产生影响。
- 卷会一直存在直到没有任何容器再使用它。

卷功能让我们可以将数据（如源代码）、数据库或者其他内容添加到镜像中而不是将这些内容提交到镜像中，并且允许我们在多个容器间共享这些内容。我们可以利用此功能来测试容器和内部的应用程序代码，管理日志，或者处理容器内部的数据库。我们将在第5章和第6章看到相关的例子。

```docker
VOLUME ["/opt/project"]
或
VOLUME ["/opt/project", "/data" ]
```

说明：

- 这条指令将会为基于此镜像创建的任何容器创建一个名为/opt/project的挂载点。
- 可以通过指定数组的方式指定多个卷。


提示 docker cp是和VOLUME指令相关并且也是很实用的命令。该命令允许从容器复制文件和复制文件到容器上。可以从Docker命令行文档（https://docs.docker.com/engine/reference/commandline/cp/）中获得更多信息。

如果现在就对卷功能很好奇，也可以在 http://docs.docker.com/userguide/dockervolumes/ 读到更多关于卷的信息。

7．ADD

ADD指令用来将构建环境下的文件和目录复制到镜像中。

比如，在安装一个应用程序时。ADD指令需要源文件位置和目的文件位置两个参数：

```docker
ADD software.lic /opt/application/software.lic
或：
ADD http://wordpress.org/latest.zip /root/wordpress.zip
或：
ADD latest.tar.gz /var/www/wordpress/
```

说明：

- 这里的ADD指令将会将构建目录下的software.lic文件复制到镜像中的 `/opt/application/software.lic`。指向源文件的位置参数可以是一个URL，或者构建上下文或环境中文件名或者目录。不能对构建目录或者上下文之外的文件进行ADD操作。
  - 在ADD文件时，Docker通过目的地址参数末尾的字符来判断文件源是目录还是文件。如果目标地址以/结尾，那么Docker就认为源位置指向的是一个目录。如果目的地址不是以/结尾，那么Docker就认为源位置指向的是文件。
- 文件源也可以使用URL的格式。
- ADD在处理本地归档文件（tar archive）时还有一些小魔法。
  - 如果将一个归档文件（合法的归档文件包括gzip、bzip2、xz）指定为源文件，Docker会自动将归档文件解开（unpack）。
  - 这条命令会将归档文件latest.tar.gz解开到/var/www/wordpress/目录下。
  - Docker解开归档文件的行为和使用带-x选项的tar命令一样：该指令执行后的输出是原目的目录已经存在的内容加上归档文件中的内容。如果目的位置的目录下已经存在了和归档文件同名的文件或者目录，那么目的位置中的文件或者目录不会被覆盖。

注意：

- 如果目的位置不存在的话，Docker将会为我们创建这个全路径，包括路径中的任何目录。新创建的文件和目录的模式为0755，并且UID和GID都是0。
- 目前Docker还不支持以URL方式指定的源位置中使用归档文件。这种行为稍显得有点儿不统一，在以后的版本中应该会有所变化。
- ADD指令会使得构建缓存变得无效，这一点也非常重要。如果通过ADD指令向镜像添加一个文件或者目录，那么这将使Dockerfile中的后续指令都不能继续使用之前的构建缓存。<span style="color:red;">那么这个 ADD 会是重复添加吗？嗯，应该是，因为如果重复，并不会覆盖。</span>

8．COPY

COPY指令非常类似于ADD，它们根本的不同是COPY只关心在构建上下文中复制本地文件，而不会去做文件提取（extraction）和解压（decompression）的工作。

```docker
COPY conf.d/ /etc/apache2/
```

说明：

- 这条指令将会把本地conf.d目录中的文件复制到/etc/apache2/目录中。
- 文件源路径必须是一个与当前构建环境相对的文件或者目录，本地文件都放到和Dockerfile同一个目录下。不能复制该目录之外的任何文件，因为构建环境将会上传到Docker守护进程，而复制是在Docker守护进程中进行的。任何位于构建环境之外的东西都是不可用的。COPY指令的目的位置则必须是容器内部的一个绝对路径。
- 任何由该指令创建的文件或者目录的UID和GID都会设置为0。
- 如果源路径是一个目录，那么这个目录将整个被复制到容器中，包括文件系统元数据；如果源文件为任何类型的文件，则该文件会随同元数据一起被复制。在这个例子里，源路径以/结尾，所以Docker会认为它是目录，并将它复制到目的目录中。
- 如果目的位置不存在，Docker将会自动创建所有需要的目录结构，就像mkdir -p命令那样。

9．LABEL

LABEL指令用于为Docker镜像添加元数据。元数据以键值对的形式展现。

```docker
LABEL version="1.0"
LABEL location="New York" type="Data Center" role="Web Server"
```

LABEL 指令以 label="value" 的形式出现。可以在每一条指令中指定一个元数据，或者指定多个元数据，不同的元数据之间用空格分隔。推荐将所有的元数据都放到一条LABEL指令中，以防止不同的元数据指令创建过多镜像层。可以通过docker inspect命令来查看Docker镜像中的标签信息。<span style="color:red;">嗯，要注意指令的行数不能过多，因为会创建过多镜像层。</span>

```sh
$ sudo docker inspect aaa/apache2
...
"Labels": {
　　 "version": "1.0",
　　 "location"="New York",
　　 "type"="Data Center",
　　 "role"="Web Server"
},
...
```


10．STOPSIGNAL

STOPSIGNAL指令用来设置停止容器时发送什么系统调用信号给容器。

这个信号必须是内核系统调用表中合法的数，如9，或者SIGNAME格式中的信号名称，如SIGKILL。

11．ARG

ARG指令用来定义可以在docker build命令运行时传递给构建运行时的变量，我们只需要在构建时使用--build-arg标志即可。用户只能在构建时指定在Dockerfile文件中定义过的参数。

```docker
ARG build
ARG webapp_user=user
```

说明：

- 第二条ARG指令设置了一个默认值，如果构建时没有为该参数指定值，就会使用这个默认值。
  
下面我们就来看看如何在docker build中使用这些参数。


```sh
$ docker build --build-arg build=1234 -t aaa/webapp .
```

说明：

- 这里构建aaa/webapp镜像时，build变量将会设置为1234，而webapp_user变量则会继承设置的默认值user。

注意：

- 读到这里，也许你会认为使用ARG来传递证书或者秘钥之类的信息是一个不错的想法。但是，请千万不要这么做。你的机密信息在构建过程中以及镜像的构建历史中会被暴露。<span style="color:red;">注意。</span>

Docker预定义了一组ARG变量，可以在构建时直接使用，而不必再到Dockerfile中自行定义。

```
HTTP_PROXY
http_proxy
HTTPS_PROXY
https_proxy
FTP_PROXY
ftp_proxy
NO_PROXY
no_proxy
```

要想使用这些预定义的变量，只需要给docker build命令传递 `--build-arg <variable>=<value>`标志就可以了。

12．ONBUILD

ONBUILD指令能为镜像添加触发器（trigger）。当一个镜像被用做其他镜像的基础镜像时（比如用户的镜像需要从某未准备好的位置添加源代码，或者用户需要执行特定于构建镜像的环境的构建脚本），该镜像中的触发器将会被执行。

触发器会在构建过程中插入新指令，我们可以认为这些指令是紧跟在FROM之后指定的。

触发器可以是任何构建指令：

```docker
ONBUILD ADD . /app/src
ONBUILD RUN cd /app/src && make
```

说明：

- 上面的代码将会在创建的镜像中加入ONBUILD触发器。

ONBUILD指令可以在镜像上运行docker inspect命令来查看：

```sh
$ sudo docker inspect 508efa4e4bf8
. . .
"OnBuild": [
"ADD . /app/src",
"RUN cd /app/src/ && make"
]
. . .
```


比如，我们为Apache2镜像构建一个全新的Dockerfile，该镜像名为aaa/apache2：

```docker
FROM ubuntu:14.04
MAINTAINER James Turnbull "james@example.com"
RUN apt-get update && apt-get install -y apache2
ENV APACHE_RUN_USER www-data
ENV APACHE_RUN_GROUP www-data
ENV APACHE_LOG_DIR /var/log/apache2
ONBUILD ADD . /var/www/
EXPOSE 80
ENTRYPOINT ["/usr/sbin/apache2"]
CMD ["-D", "FOREGROUND"]
```


现在我们就来构建该镜像，

```sh
$ sudo docker build -t="aaa/apache2" .
. . .
Step 7 : ONBUILD ADD . /var/www/
---> Running in 0e117f6ea4ba
---> a79983575b86
. . .
Successfully built a79983575b86
```


在新构建的镜像中包含一条ONBUILD指令，该指令会使用ADD指令将构建环境所在的目录下的内容全部添加到镜像中的`/var/www/`目录下。我们可以轻而易举地将这个Dockerfile作为一个通用的Web应用程序的模板，可以基于这个模板来构建Web应用程序。

我们可以通过构建一个名为webapp的镜像来看看如何使用镜像模板功能：

```sh
FROM aaa/apache2
MAINTAINER James Turnbull "james@example.com"
ENV APPLICATION_NAME webapp
ENV ENVIRONMENT development
```


让我们看看构建这个镜像时将会发生什么事情：

```sh
$ sudo docker build -t="aaa/webapp" .
. . .
Step 0 : FROM aaa/apache2
# Executing 1 build triggers
Step onbuild-0 : ADD . /var/www/
---> 1a018213a59d
---> 1a018213a59d
Step 1 : MAINTAINER James Turnbull "james@example.com"
. . .
Successfully built 04829a360d86
```

可见：

- 在FROM指令之后，Docker插入了一条ADD指令，这条ADD指令就是在ONBUILD触发器中指定的。执行完该ADD指令后，Docker才会继续执行构建文件中的后续指令。
- 这种机制使我每次都会将本地源代码添加到镜像，就像上面我们做到的那样，也支持我为不同的应用程序进行一些特定的配置或者设置构建信息。这时，可以将aaa/apache2当做一个镜像模板。

注意：

- ONBUILD触发器会按照在父镜像中指定的顺序执行，并且只能被继承一次（也就是说只能在子镜像中执行，而不会在孙子镜像中执行）。如果我们再基于 aaa/webapp构建一个镜像，则新镜像是aaa/apache2的孙子镜像，因此在该镜像的构建过程中，ONBUILD触发器是不会被执行的。
- 有好几条指令是不能用在ONBUILD指令中的，包括FROM、MAINTAINER和ONBUILD本身。之所以这么规定是为了防止在 Dockerfile构建过程中产生递归调用的问题。



## 将镜像推送到 Docker Hub

镜像构建完毕之后，我们也可以将它上传到Docker Hub上面去，这样其他人就能使用这个镜像了。

比如，我们可以在组织内共享这个镜像，或者完全公开这个镜像。

Docker Hub也提供了对私有仓库的支持，这是一个需要付费的功能，用户可以将镜像存储到私有仓库中，这样只有用户或者任何与用户共享这个私有仓库的人才能访问该镜像。这样用户就可以将机密信息或者代码放到私有镜像中，不必担心被公开访问了。

我们可以通过 docker push 命令将镜像推送到 Docker Hub。现在就让我们来试一试如何推送：


```sh
$ sudo docker push static_web
2013/07/01 18:34:47 Impossible to push a "root" repository.
Please rename your repository in <user>/<repo> (ex: aaa/static_web)

$ sudo docker push aaa/static_web
The push refers to a repository [aaa/static_web] (len: 1)
Processing checksums
Sending image list
Pushing repository aaa/static_web to registry-1.docker.io (1 tags)
. . .
```

说明：

- 直接使用 `push static_web` 出什么问题了？我们尝试将镜像推送到远程仓库static_web，但是Docker认为这是一个root仓库。root仓库是由Docker公司的团队管理的，因此会拒绝我们的推送请求。让我们再换一种方式试一下。
- 这次我们使用了一个名为aaa/static_web的用户仓库，成功地将镜像推送到了Docker Hub。我们将会使用自己的用户ID，这个ID也是我们前面创建的，并选择了一个合法的镜像名（如youruser/yourimage）。

我们可以在Docker Hub看到我们上传的镜像：

<p align="center">
    <img width="50%" height="50%" src="http://images.iterate.site/blog/image/20200205/oTpnLXJG7RuX.png?imageslim">
</p>


## 自动构建

除了从命令行构建和推送镜像，Docker Hub还允许我们定义自动构建（Automated Builds）。为了使用自动构建，我们只需要将GitHub 或 BitBucket 中含有Dockerfile文件的仓库连接到Docker Hub即可。向这个代码仓库推送代码时，将会触发一次镜像构建活动并创建一个新镜像。在之前该工作机制也被称为可信构建（Trusted Build）。

自动构建同样支持私有GitHub和BitBucket仓库。

在Docker Hub中添加自动构建任务的第一步是将GitHub或者BitBucket账号连接到Docker Hub。具体操作是，打开Docker Hub，登录后单击个人信息链接，之后单击Add Repository ->Automated Build按钮：

<p align="center">
    <img width="50%" height="50%" src="http://images.iterate.site/blog/image/20200205/ywsJNbdGBj8Y.png?imageslim">
</p>

你将会在此页面看到关于链接到GitHub或者BitBucket账号的选项。单击GitHub logo下面的Select按钮开始账号链接。你将会转到GitHub页面并看到Docker Hub的账号链接授权请求。

在GitHub上有两个选项：Public and Private (recommended)和Limited。选择 Public and Private (recommended)并单击Allow Access完成授权操作。有可能会被要求输入GitHub的密码来确认访问请求。

之后，系统将提示你选择用来进行自动构建的组织和仓库：

<p align="center">
    <img width="50%" height="50%" src="http://images.iterate.site/blog/image/20200205/MRLlOjVIw8es.png?imageslim">
</p>

单击想用来进行自动构建的仓库后面的Select按钮，之后开始对自动构建进行配置：

<p align="center">
    <img width="50%" height="50%" src="http://images.iterate.site/blog/image/20200205/vaHPSrk4REIL.png?imageslim">
</p>


指定想使用的默认的分支名，并确认仓库名。为每次自动构建过程创建的镜像指定一个标签，并指定Dockerfile的位置。默认的位置为代码仓库的根目录下，但是也可以随意设置该路径。

最后，单击Create Repository按钮来将你的自动构建添加到Docker Hub中：

<p align="center">
    <img width="50%" height="50%" src="http://images.iterate.site/blog/image/20200205/gDX6t7jGGOp3.png?imageslim">
</p>


你会看到你的自动构建已经被提交了。单击Build Status链接可以查看最近一次构建的状态，包括标准输出的日志，里面记录了构建过程以及任何的错误。如果该构建状态为Done，则表示该自动构建为最新状态。Error状态则表示构建过程出现错误。你可以单击查看详细的日志输出。

注意：

- 不能通过docker push命令推送一个自动构建，只能通过更新你的GitHub或者BitBucket仓库来更新你的自动构建。


