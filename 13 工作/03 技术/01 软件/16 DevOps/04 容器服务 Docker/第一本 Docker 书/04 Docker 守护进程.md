# Docker 守护进程

Docker 以 root权限运行它的守护进程，来处理普通用户无法完成的操作（如挂载文件系统）。当Docker软件包安装完毕后，默认会立即启动Docker守护进程。

docker程序是Docker守护进程的客户端程序，同样也需要以root身份运行。用户可以使用docker daemon命令控制Docker守护进程。


守护进程监听 `/var/run/docker.sock` 这个 Unix 套接字文件，来获取来自客户端的Docker请求。如果系统中存在名为docker的用户组的话，Docker则会将该套接字文件的所有者设置为该用户组。这样，docker用户组的所有用户都可以直接运行Docker，而无需再使用sudo命令了。

注意：

- 尽管docker用户组方便了Docker的使用，但它毕竟是一个安全隐患。因为docker用户组对Docker具有与root用户相同的权限，所以docker用户组中应该只能添加那些确实需要使用Docker的用户和程序。


## 配置 Docker 守护进程

<span style="color:red;">为什么要配置监听接口？</span>

运行Docker守护进程时，可以用 `-H` 标志调整守护进程绑定监听接口的方式。

可以使用 `-H` 标志指定不同的网络接口和端口配置。

举例，要想绑定到网络接口：

```sh
$ sudo docker daemon -H tcp://0.0.0.0:2375
```

说明：

- 这条命令会将Docker守护进程绑定到宿主机上的所有网络接口。

Docker客户端不会自动监测到网络的变化，需要通过`-H`选项来指定服务器的地址。例如，如果把守护进程端口改成4200，那么运行客户端时就必须指定docker -H :4200。

如果不想每次运行客户端时都加上`-H`标志，可以通过设置DOCKER_HOST环境变量来省略此步骤。如下：

```sh
$ export DOCKER_HOST="tcp://0.0.0.0:2375"
```

注意：

- 默认情况下，Docker的客户端-服务器通信是不经认证的。这就意味着，如果把Docker绑定到对外公开的网络接口上，那么任何人都可以连接到该Docker守护进程。Docker 0.9及更高版本提供了TLS认证。介绍Docker API时会详细了解如何启用TLS认证。

也能通过`-H`标志指定一个Unix套接字路径，例如，指定unix://home/docker/docker.socket。如下，将Docker守护进程绑定到非默认套接字

```sh
$ sudo docker daemon -H unix://home/docker/docker.sock
```

当然，也可以同时指定多个绑定地址，将Docker守护进程绑定到多个地址：

```sh
$ sudo docker daemon -H tcp://0.0.0.0:2375 -H unix://home/docker/docker.sock
```

如果你的 Docker 运行在代理或者公司防火墙之后，也可以使用 HTTPS_PROXY、HTTP_ PROXY 和 NO_PROXY 选项来控制守护进程如何连接。

还可以使用-D标志来输出Docker守护进程的更详细的信息，开启Docker守护进程的调试模式：

```sh
$ sudo docker daemon -D
```


要想让这些改动永久生效，需要编辑启动配置项。在Ubuntu中，需要编辑`/etc/default/docker`文件，并修改`DOCKER_OPTS`变量。

注意：

- 在其他平台中，可以通过适当的init系统来管理和更新Docker守护进程的启动配置。

##　检查 Docker 守护进程是否正在运行

在Ubuntu中，如果Docker是通过软件包安装的话，可以运行Upstart的status命令来检查Docker守护进程是否正在运行：

```sh
$ sudo status docker
docker start/running, process 18147
```

此外，还可以用Upstart的start和stop命令来启动和停止Docker守护进程：


```sh
$ sudo stop docker
docker stop/waiting
$ sudo start docker
docker start/running, process 18192
```

如果守护进程没有运行，执行docker客户端命令时就会出现如下的错误：

```sh
2014/05/18 20:08:32 Cannot connect to the Docker daemon. Is 'docker -d' running on this host?
```