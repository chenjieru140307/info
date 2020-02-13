

## 运行一个 Docker 容器

运行容器：

```sh
root@iZuf66eabunrloh2og4jgsZ:/home/docker_t# docker run -i -t ubuntu /bin/bash
Unable to find image 'ubuntu:latest' locally
latest: Pulling from library/ubuntu
5c939e3a4d10: Pull complete 
c63719cdbe7a: Pull complete 
19a861ea6baf: Pull complete 
651c9d2d6c4f: Pull complete 
Digest: sha256:8d31dad0c58f552e890d68bbfb735588b6b820a46e459672d96e585871acc110
Status: Downloaded newer image for ubuntu:latest
root@a046a6e23893:/#
```

说明：

- `-i` 标志保证容器中STDIN是开启的，尽管我们并没有附着到容器中。持久的标准输入是交互式shell的“半边天”。
- `-t` 标志则是另外“半边天”，它告诉Docker为要创建的容器分配一个伪tty终端。这样，新创建的容器才能提供一个交互式shell。
- `ubuntu` 即使用的 ubuntu 镜像来创建容器。ubuntu镜像是一个常备镜像，也可以称为“基础”（base）镜像，它由Docker 公司提供，保存在Docker Hub[3]Registry上。可以以基础镜像为基础，在选择的操作系统上构建自己的镜像。
- `/bin/bash`是我们告诉Docker在新容器中要运行什么命令，在本例中我们在容器中运行 `/bin/bash` 命令启动了一个Bash shell。当容器创建完毕之后，Docker就会执行容器中的 `/bin/bash` 命令，这时就可以看到容器内的shell了，也就是 `root@a046a6e23893:/#`。这时我们已经以root用户登录到了新容器中，容器的ID a046a6e23893。
- 命令执行时，首先Docker会检查本地是否存在ubuntu镜像，如果本地还没有该镜像的话，那么Docker就会连接官方维护的Docker HubRegistry，查看Docker Hub中是否有该镜像。Docker一旦找到该镜像，就会下载该镜像并将其保存到本地宿主机中。随后，Docker在文件系统内部用这个镜像创建了一个新容器。该容器拥有自己的网络、IP地址，以及一个用来和宿主机进行通信的桥接网络接口。

注意：

- 若要在命令行下创建一个我们能与之进行交互的容器，而不是一个运行后台服务的容器，则 `-i -t` 两个参数已经是最基本的参数了。
- 类似地，Docker也提供了docker create命令来创建一个容器，但是并不运行它。这让我们可以在自己的容器工作流中对其进行细粒度的控制。

## 在容器里执行命令


现在，我们已经以root用户登录到了新容器中，容器的ID a046a6e23893，乍看起来有些令人迷惑的字符串`。这是一个完整的Ubuntu系统，可以用它来做任何事情。

下面就来研究一下这个容器。

首先，我们可以获取该容器的主机名：

```sh
root@a046a6e23893:/# hostname
a046a6e23893
root@a046a6e23893:/# cat /etc/hosts
127.0.0.1	localhost
::1	localhost ip6-localhost ip6-loopback
fe00::0	ip6-localnet
ff00::0	ip6-mcastprefix
ff02::1	ip6-allnodes
ff02::2	ip6-allrouters
192.168.0.2	a046a6e23893
root@a046a6e23893:/# apt update && apt install -y iproute2
root@a046a6e23893:/# ip a
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
5: eth0@if6: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP group default 
    link/ether 02:42:c0:a8:00:02 brd ff:ff:ff:ff:ff:ff link-netnsid 0
    inet 192.168.0.2/20 scope global eth0
       valid_lft forever preferred_lft forever
root@a046a6e23893:/# ps -aux
USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
root         1  0.0  0.1  18508  3420 ?        Ss   11:29   0:00 /bin/bash
root       305  0.0  0.1  34400  2820 ?        R+   13:35   0:00 ps -aux
root@a046a6e23893:/# exit
exit
root@iZuf66eabunrloh2og4jgsZ:/home/docker_t# docker ps -a
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS                     PORTS               NAMES
a046a6e23893        ubuntu              "/bin/bash"         2 hours ago         Exited (0) 2 minutes ago                       amazing_darwin

```

说明：

- `hostname` 可以获取该容器的主机名，可见，容器的主机名就是该容器的ID。
- `cat /etc/hosts` 查看 `/etc/hosts` 文件，可见，Docker已在hosts文件中为该容器的IP地址添加了一条主机配置项 `192.168.0.2	a046a6e23893`。
- `apt update && apt install -y iproute2` 在容器中安装 ip 指令。
- `ip a` 可以查看容器的网络配置情况。可以看到，这里有lo的环回接口，还有IP为 192.168.0.2 的标准eth0网络接口，和普通宿主机是完全一样的。
- `ps -aux` 查看容器中运行的进程。
- `exit` 当所有工作都结束时，输入exit，就可以返回到Ubuntu宿主机的命令行提示符了。
- 退出后，这个时候容器就已经停止运行了！只有在指定的/bin/bash命令处于运行状态的时候，我们的容器也才会相应地处于运行状态。一旦退出容器，/bin/bash命令也就结束了，这时容器也随之停止了运行。但容器仍然是存在的。
- `docker ps -a` 可以查看当前系统中容器的列表。
  - 默认情况下，当执行docker ps命令时，只能看到正在运行的容器。
  - 如果指定-a标志的话，那么docker ps命令会列出所有容器，包括正在运行的和已经停止的。
  - 也可以为docker ps命令指定-l标志，列出最后一个运行的容器，无论其正在运行还是已经停止。
  - 也可以通过--format标志，进一步控制显示哪些信息，以及如何显示这些信息。
- 从 `docker ps -a` 命令的输出结果中我们可以看到关于这个容器的很多有用信息：ID、用于创建该容器的镜像、容器最后执行的命令、创建时间以及容器的退出状态（在上面的例子中，退出状态是0，因为容器是通过正常的exit命令退出的）。我们还可以看到，每个容器都有一个名称。

注意：

- 有3种方式可以唯一指代容器：短UUID（如a046a6e23893）、长UUID（如f7cbdac 22a02e03c9438c729345e54db9d20cfa2ac1fc3494b6eb60872e74778）或者名称（如gray_cat）。<span style="color:red;">长 UUID 在哪里看？</span>



## 为容器命名

Docker会为我们创建的每一个容器自动生成一个随机的名称。例如，上面我们刚刚创建的容器就被命名为gray_cat。如果想为容器指定一个名称，而不是使用自动生成的名称，则可以用`--name`标志来实现：

```sh
$ sudo docker run --name bob -i -t ubuntu /bin/bash
root@aa3f365f0f4e:/# exit
```

说明：

- 上述命令将会创建一个名为bob_the_container的容器。

一个合法的容器名称只能包含以下字符：小写字母a~z、大写字母A~Z、数字0~9、下划线、圆点、横线（如果用正则表达式来表示这些符号，就是[a-zA-Z0-9_.-]）。

在很多Docker命令中，都可以用容器的名称来替代容器ID，后面我们将会看到。容器名称有助于分辨容器，当构建容器和应用程序之间的逻辑连接时，容器的名称也有助于从逻辑上理解连接关系。具体的名称（如web、db）比容器ID和随机容器名好记多了。我推荐大家都使用容器名称，以更加方便地管理容器。

注意：

- 容器的命名必须是唯一的。如果试图创建两个名称相同的容器，则命令将会失败。
- 如果要使用的容器名称已经存在，可以先用docker rm命令删除已有的同名容器后，再来创建新的容器。



## 重新启动已经停止的容器

```sh
$ sudo docker start bob
或：
$ sudo docker start aa3f365f0f4e
或：
$ sudo docker restart bob

root@iZuf66eabunrloh2og4jgsZ:~# docker ps
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
186b69c71976        ubuntu              "/bin/bash"         12 hours ago        Up 4 seconds                            bob
```

说明：

- 重新启动一个已经停止的容器。
- 除了容器名称，也可以用容器ID来指定容器。
- 也可以使用 docker restart 命令来重新启动一个容器。
- 这时运行不带-a标志的docker ps命令，就应该看到我们的容器已经开始运行了。


## 附着到容器上

Docker 容器重新启动的时候，会沿用docker run命令时指定的参数来运行，因此我们的容器重新启动后会运行一个交互式会话shell。此外，也可以用docker attach命令，重新附着到该容器的会话上。

如下：

```sh
$ sudo docker attach bob
或：
$ sudo docker attach 186b69c71976

root@186b69c71976:/#
```

说明：

- 现在，又重新回到了容器的Bash提示符。如果退出容器的shell，容器会再次停止运行。




## 创建守护式容器

除了这些交互式运行的容器（interactive container），也可以创建长期运行的容器。守护式容器（daemonized container）没有交互式会话，非常适合运行应用程序和服务。大多数时候我们都需要以守护式来运行我们的容器。

下面就来启动一个守护式容器：


```py
root@iZuf66eabunrloh2og4jgsZ:~# docker run --name daemon -d ubuntu /bin/sh -c "while true; do echo hello world; sleep 1; done"
d07d5d29637a2e574a67ee1046e9f65350dcd5f40798a11ab2109c4436d06325
```

说明：

- `-d` 参数，因此Docker会将容器放到后台运行。
- 我们还在容器要运行的命令里使用了一个while循环，该循环会一直打印 hello world，直到容器或其进程停止运行。


## 容器内部都在干些什么

现在我们已经有了一个在后台运行while循环的守护型容器。

为了探究该容器内部都在干些什么，可以用docker logs命令来获取容器的日志。

```sh
$ sudo docker logs daemon_dave
hello world
hello world
hello world
hello world
hello world
hello world
hello world
...
$ sudo docker logs -f daemon_dave
hello world
hello world
hello world
hello world
hello world
hello world
hello world
...
$ sudo docker logs --tail 10 daemon_dave
hello world
hello world
...
$ sudo docker logs --tail 0 -f daemon_dave
hello world
hello world
...
$ sudo docker logs -ft daemon_dave
2020-02-04T02:35:09.172284569Z hello world
2020-02-04T02:35:10.173296087Z hello world
2020-02-04T02:35:11.174278626Z hello world
2020-02-04T02:35:12.175231311Z hello world
2020-02-04T02:35:13.176282763Z hello world
2020-02-04T02:35:14.177235255Z hello world
2020-02-04T02:35:15.178207248Z hello world
2020-02-04T02:35:16.179180301Z hello world
...
```


可见：

- while 循环正在向日志里打印 hello world。Docker会输出最后几条日志项并返回。
- 我们也可以在命令后使用 `-f` 参数来监控Docker的日志，这与tail -f命令非常相似。可以通过Ctr+C退出日志跟踪。
- 可以用 `docker logs --tail 10 daemon_dave` 获取日志的最后10行内容。
- 可以用 `docker logs --tail 0 -f daemon_dave` 命令来跟踪某个容器的最新日志而不必读取整个日志文件。
- 可以使用 `-t` 标志为每条日志项加上时间戳。

不明白的：

- <span style="color:red;">为什么这个 `-ft` 指令打出来的时间戳并不是严格的 `1s` 为分隔的呢？</span>






## Docker 日志设定


```py
root@iZuf66eabunrloh2og4jgsZ:~# docker run --log-driver="syslog" --name daemon -d ubuntu /bin/sh -c "while true; do echo hello world; sleep 1; done"
0735f7b1ac5594109577364289a69f37603d793f0178f00fe598aa9914603a1e
```

说明：

- `--log-driver` 可以用来设定日志流向。可以在启动Docker守护进程时指定该选项，将所有容器的日志都输出到某处，或者通过docker run对个别的容器进行日志重定向输出。
  - 默认的是 `json-file`
  - `syslog` 将禁用 docker logs 命令，并且将所有容器的日志输出都重定向到 Syslog。
  - 还有一个可用的选项是none，这个选项将会禁用所有容器中的日志，导致docker logs命令也被禁用。
  - 新的日志驱动也在不断地增加，如 Graylog GELF协议、Fluentd 以及日志轮转驱动等。

注意：

- 如果是在Docker Toolbox中运行Docker，应该在虚拟机中启动Syslog守护进程。可以先通过docker-machine ssh命令连接到Docker Toolbox虚拟机，再在其中运行syslogd命令来启动Syslog守护进程。



## 查看容器内的进程

除了容器的日志，也可以查看容器内部运行的进程。

要做到这一点，要使用docker

top命令，如代码清单3-23所示。

代码清单3-23　查看守护式容器的进程


```sh
root@iZuf66eabunrloh2og4jgsZ:~# docker top daemon
UID                 PID                 PPID                C                   STIME               TTY                 TIME                CMD
root                10243               10226               0                   10:46               ?                   00:00:00            /bin/sh -c while true; do echo hello world; sleep 1; done
root                10749               10243               0                   10:54               ?                   00:00:00            sleep 1
```

说明：

- top 可以看到容器内的所有进程（主要还是我们的while循环）、运行进程的用户及进程ID



## Docker 统计信息

除了docker top命令，还可以使用docker stats命令，它用来显示一个或多个容器的统计信息。让我们来看看它的输出是什么样的。

```sh
root@iZuf66eabunrloh2og4jgsZ:~# docker stats daemon bob
CONTAINER           CPU %               MEM USAGE / LIMIT   MEM %               NET I/O             BLOCK I/O           PIDS
daemon              0.10%               284KiB / 1.953GiB   0.01%               0B / 0B             94.2kB / 0B         2
bob                 0.00%               0B / 0B             0.00%               0B / 0B             0B / 0B             0

CONTAINER           CPU %               MEM USAGE / LIMIT   MEM %               NET I/O             BLOCK I/O           PIDS
daemon              0.10%               284KiB / 1.953GiB   0.01%               0B / 0B             94.2kB / 0B         2
bob                 0.00%               0B / 0B             0.00%               0B / 0B             0B / 0B             0
```


说明：

- stats 可以用来显示一个或多个容器的统计信息。
- 统计信息为它们的CPU、内存、网络I/O及存储I/O的性能和指标。这对快速监控一台主机上的一组容器非常有用。



## 在容器内部运行进程

可以通过docker exec命令在容器内部额外启动新进程。


可以在容器内运行的进程有两种类型：后台任务和交互式任务。后台任务在容器内运行且没有交互需求，而交互式任务则保持在前台运行。对于需要在容器内部打开shell的任务，交互式任务是很实用的。

一个后台任务的例子：

$ sudo docker exec -d daemon touch /etc/new_config_file

说明：

- -d 表明需要运行一个后台进程，-d标志之后，指定的是要在内部执行这个命令的容器的名字以及要执行的命令。
- 上面例子中的命令会在daemon_dave容器内创建了一个空文件，文件名为`/etc/new_config_file`。

通过docker exec后台命令，可以在正在运行的容器中进行维护、监控及管理任务。

可以对docker exec启动的进程使用–u标志为新启动的进程指定一个用户属主。我们也可以在daemon_dave容器中启动一个诸如打开shell的交互式任务，如下：

```sh
$ sudo docker exec -t -i daemon_dave /bin/bash
```


说明：

- 和运行交互容器时一样，这里的-t和-i标志为我们执行的进程创建了TTY并捕捉STDIN。接着我们指定了要在内部执行这个命令的容器的名字以及要执行的命令。
- 在上面的例子中，这条命令会在daemon_dave容器内创建一个新的bash会话，有了这个会话，我们就可以在该容器中运行其他命令了。





## 停止守护式容器

要停止守护式容器，只需要执行docker stop命令：

```sh
$ sudo docker stop daemon_dave
或：
$ sudo docker stop c2c4e57c12c4
```

注意：

- docker stop命令会向Docker容器进程发送SIGTERM信号。如果想快速停止某个容器，也可以使用docker kill命令来向容器进程发送SIGKILL信号。
- 要想查看已经停止的容器的状态，则可以使用docker ps命令。还有一个很实用的命令docker ps -n x，该命令会显示最后x个容器，不论这些容器正在运行还是已经停止。



## 自动重启容器

如果由于某种错误而导致容器停止运行，还可以通过--restart标志，让Docker自动重新启动该容器。

```sh
$ sudo docker run --restart=always --name daemon_dave -d ubuntu /bin/sh -c "while true; do echo hello world; sleep 1; done"
```

说明：

- `--restart` 标志会检查容器的退出代码，并据此来决定是否要重启容器。默认的行为是Docker不会重启容器。
    - `--restart` 标志被设置为always。无论容器的退出代码是什么，Docker都会自动重启该容器。
    - 除了always，还可以将这个标志设为on-failure，这样，只有当容器的退出代码为非0值的时候，才会自动重启。另外，on-failure还接受一个可选的重启次数参数，如 `--restart=on-failure:5`，这样，当容器退出代码为非0时，Docker会尝试自动重启该容器，最多重启5次。



## 深入容器

除了通过docker ps命令获取容器的信息，还可以使用docker inspect来获得更多的容器信息：

```py
root@iZuf66eabunrloh2og4jgsZ:~# docker inspect daemon
[
    {
        "Id": "0735f7b1ac5594109577364289a69f37603d793f0178f00fe598aa9914603a1e",
        "Created": "2020-02-04T02:46:04.229914672Z",
        "Path": "/bin/sh",
        "Args": [
            "-c",
            "while true; do echo hello world; sleep 1; done"
        ],
        "State": {
            "Status": "running",
            "Running": true,
            "Paused": false,
            "Restarting": false,
            "OOMKilled": false,
            "Dead": false,
            "Pid": 11492,
            "ExitCode": 0,
            "Error": "",
            "StartedAt": "2020-02-04T03:03:02.581124782Z",
            "FinishedAt": "2020-02-04T03:01:24.531646632Z"
        },
        "Image": "sha256:ccc6e87d482b79dd1645affd958479139486e47191dfe7a997c862d89cd8b4c0",
        "ResolvConfPath": "/var/lib/docker/containers/0735f7b1ac5594109577364289a69f37603d793f0178f00fe598aa9914603a1e/resolv.conf",
        "HostnamePath": "/var/lib/docker/containers/
        ...
root@iZuf66eabunrloh2og4jgsZ:~# docker inspect --format='{{ .State.Running }}' daemon
true
root@iZuf66eabunrloh2og4jgsZ:~# docker inspect --format='{{ .NetworkSettings.IPAddress }}' daemon
192.168.0.2
root@iZuf66eabunrloh2og4jgsZ:~# docker inspect --format '{{.Name}} {{.State.Running}}' daemon bob
/daemon true
/bob false
```

说明：

- docker inspect命令会对容器进行详细的检查，然后返回其配置信息，包括名称、命令、网络配置以及很多有用的数据。
- 可以用-f或者--format标志来选定查看结果。上面这条命令会返回容器的运行状态，示例中该状态为false。
- 我们还能获取其他有用的信息，如容器IP地址。
- 可以同时指定多个容器，并显示每个容器的输出结果。

注意：

- `--format` 或者 `-f` 标志远非表面看上去那么简单。该标志实际上支持完整的Go语言模板。用它进行查询时，可以充分利用Go语言模板的优势。可以为该参数指定要查询和返回的查看散列（inspect hash）中的任意部分。
- 除了查看容器，还可以通过浏览`/var/lib/docker`目录来深入了解 Docker的工作原理。该目录存放着 Docker 镜像、容器以及容器的配置。所有的容器都保存在`/var/lib/docker/containers`目录下。


## 删除容器

如果容器已经不再使用，可以使用docker rm命令来删除它们。


```sh
$ sudo docker rm 80430f8d0921
$ sudo docker rm -f 80430f8d0921

$ sudo docker rm 'sudo docker ps -a -q'
```

说明：

- 以通过给docker rm命令传递`-f`标志来删除运行中的Docker容器。
- 目前，还没有办法一次删除所有容器，不过可以通过 `sudo docker rm 'sudo docker ps -a -q'` 来删除全部容器。`docker ps` 命令会列出现有的全部容器，`-a`标志代表列出所有容器，而`-q`标志则表示只需要返回容器的ID而不会返回容器的其他信息。这样我们就得到了容器ID的列表，并传给了`docker rm`命令，从而达到删除所有容器的目的。