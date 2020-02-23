# 补充

- 这个没有看完。

# 在测试中使用Docker

接下来让我们试着在实际开发和测试过程中使用Docker。

首先来看看Docker如何使开发和测试更加流程化，效率更高。

为了演示，我们将会看到下面3个使用场景。

- 使用Docker测试一个静态网站。
- 使用Docker创建并测试一个Web应用。
- 将Docker用于持续集成。

注意 作者使用持续集成环境的经验大都基于Jenkins，因此本书里使用Jenkins作为持续集成环境的例子。

读者可以把这几节所讲的思想应用到任何持续集成平台中。




在前两个使用场景中，我们将主要关注以本地开发者为主的开发和测试，而在最后一个使用场景里，我们会看到如何在更广泛的多人开发中将Docker用于构建和测试。

本章将介绍如何将使用Docker作为每日生活和工作流程的一部分，包括如何连接不同的容器等有用的概念。本章会包含很多有用的信息，告诉读者通常如何运行和管理Docker。所以，即便读者并不关心上述使用场景，作者也推荐读者能阅读本章。


## 使用Docker测试静态网站

将Docker作为本地Web开发环境是Docker的一个最简单的应用场景。这样的环境可以完全复制生产环境，并确保用户开发的东西在生产环境中也能运行。下面从将Nginx Web服务器安装到容器来架构一个简单的网站开始。这个网站暂且命名为Sample。

### Sample网站的初始Dockerfile

先创建一个文件夹作为上下文：

```sh
$ mkdir sample
$ cd sample
$ touch Dockerfile
```

在 sample 中创建 nginx 文件夹，并放入一些 Nginx 配置文件，用来复制到容器里面给 nginx 使用：

```sh
$ mkdir nginx && cd nginx
$ wget https://raw.githubusercontent.com/jamtur01/dockerbook-code/master/code/5/sample/nginx/global.conf
$ wget https://raw.githubusercontent.com/jamtur01/dockerbook-code/master/code/5/sample/nginx/nginx.conf
$ cd ..
```

global.conf 内容为：

```conf
server {
        listen          0.0.0.0:80;
        server_name     _;

        root            /var/www/html/website;
        index           index.html index.htm;

        access_log      /var/log/nginx/default_access.log;
        error_log       /var/log/nginx/default_error.log;
}
```

可见：

- nginx 在监听 80 端口。
- 网络服务的根路径设置为 `/var/www/html/website`

nginx.conf 内容为：

```conf
user www-data;
worker_processes 4;
pid /run/nginx.pid;
daemon off;

events {  }

http {
  sendfile on;
  tcp_nopush on;
  tcp_nodelay on;
  keepalive_timeout 65;
  types_hash_max_size 2048;
  include /etc/nginx/mime.types;
  default_type application/octet-stream;
  access_log /var/log/nginx/access.log;
  error_log /var/log/nginx/error.log;
  gzip on;
  gzip_disable "msie6";
  include /etc/nginx/conf.d/*.conf;
}
```

说明：

- 在这个配置文件里，`daemon off;` 选项阻止Nginx进入后台，强制其在前台运行。这是因为要想保持Docker容器的活跃状态，需要其中运行的进程不能中断。默认情况下，Nginx 会以守护进程的方式启动，这会导致容器只是短暂运行，在守护进程被fork启动后，发起守护进程的原始进程就会退出，这时容器就停止运行了。<span style="color:red;">不是特别理解。</span>




现在看一下我们将要为Sample网站创建的Dockerfile：

```docker
FROM ubuntu:14.04
MAINTAINER James Turnbull "james@example.com"
ENV REFRESHED_AT 2014-06-01
RUN apt-get -yqq update && apt-get -yqq install nginx
RUN mkdir -p /var/www/html/website
ADD nginx/global.conf /etc/nginx/conf.d/
ADD nginx/nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
```

说明：

- 这个简单的Dockerfile内容包括以下几项：
  - 安装 Nginx。
  - 在容器中创建一个目录 `/var/www/html/website/`。这个目录就是上面 global.conf 文件中的根路径。
  - 将来自我们下载的本地文件的Nginx配置文件添加到镜像中。这个Nginx配置文件是为了运行Sample网站而配置的。
  - 公开镜像的80端口。
  - `nginx/global.conf` 复制到 `/etc/nginx/conf.d/`目录中。
  - `nginx/nginx.conf` 文件中的 `daemon off;` 选项使得 Nginx 为非守护进程的模式，这样可以让Nginx在Docker容器里工作。


### 构建 Sample 网站和 Nginx 镜像

利用之前的Dockerfile，可以用docker build命令构建出新的镜像，并将这个镜像命名为 aaa/nginx：

```sh
root@iZuf66eabunrloh2og4jgsZ:/home/docker_t/sample# docker build -t aaa/nginx .
root@iZuf66eabunrloh2og4jgsZ:/home/docker_t/sample# docker history aaa/nginx
IMAGE               CREATED             CREATED BY                                      SIZE                COMMENT
b5f65327fdd4        45 seconds ago      /bin/sh -c #(nop)  EXPOSE 80/tcp                0B                  
9e7d87efef81        45 seconds ago      /bin/sh -c #(nop) ADD file:d6698a182fafaf3...   415B                
a95509a2e5e1        45 seconds ago      /bin/sh -c #(nop) ADD file:9778ae1b4389601...   286B                
cb86b1eac389        46 seconds ago      /bin/sh -c mkdir -p /var/www/html/website       0B                  
59348ed7fac8        46 seconds ago      /bin/sh -c apt-get -yqq update && apt-get ...   35.2MB              
807bd5cb018f        3 minutes ago       /bin/sh -c #(nop)  ENV REFRESHED_AT=2014-0...   0B                  
699bcb8b3f99        3 minutes ago       /bin/sh -c #(nop)  MAINTAINER yd li "aaa@e...   0B                  
6e4f1fe62ff1        6 weeks ago         /bin/sh -c #(nop)  CMD ["/bin/bash"]            0B                  
<missing>           6 weeks ago         /bin/sh -c mkdir -p /run/systemd && echo '...   7B                  
<missing>           6 weeks ago         /bin/sh -c set -xe   && echo '#!/bin/sh' >...   195kB               
<missing>           6 weeks ago         /bin/sh -c [ -z "$(apt-get indextargets)" ]     0B                  
<missing>           6 weeks ago         /bin/sh -c #(nop) ADD file:276b5d943a4d284...   196MB   
```


说明：

- 使用docker history命令查看构建新镜像的步骤和层级。history 命令从新构建的 jamtur01/nginx镜像的最后一层开始，追溯到最开始的父镜像ubuntu:14.04。这个命令也展示了每步之间创建的新层，以及创建这个层所使用的Dockerfile里的指令。

### 从 Sample 网站和 Nginx 镜像构建容器

现在可以使用jamtur01/nginx镜像，并开始从这个镜像构建可以用来测试Sample网站的容器。

为此，需要添加Sample网站的代码。现在下载这段代码到sample目录，


```sh
$ mkdir website && cd website
$ wget https://raw.githubusercontent.com/jamtur01/dockerbook-code/master/code/5/sample/website/index.html
$ cd..
```

这将在sample目录中创建一个名为website的目录，然后为Sample网站下载index.html文件，放到website目录中。

运行一个容器：

```sh
root@iZuf66eabunrloh2og4jgsZ:/home/docker_t/sample# docker run -d -p 80 --name website \
> -v $PWD/website:/var/www/html/website \
> aaa/nginx nginx
7b736dbcfa82421fa1fa04f9d4a23d4f80b4b66761b04e04d8446622c5e6b33a
root@iZuf66eabunrloh2og4jgsZ:/home/docker_t/sample# 
```


注意：

- 可以看到，在执行docker run时传入了nginx作为容器的启动命令。一般情况下，这个命令无法让Nginx以交互的方式运行。我们已经在提供给Docker的配置里加入了指令daemon off，这个指令让Nginx启动后以交互的方式在前台运行。

说明：

- 可以看到，我们使用docker run命令从aaa/nginx镜像创建了一个名为website的容器。读者已经见过了大部分选项，不过-v选项是新的。-v这个选项允许我们将宿主机的目录作为卷，挂载到容器里。

现在稍微偏题一下，我们来关注一下卷这个概念。卷在Docker里非常重要，也很有用。卷是在一个或者多个容器内被选定的目录，可以绕过分层的联合文件系统（Union File System），为Docker提供持久数据或者共享数据。这意味着对卷的修改会直接生效，并绕过镜像。当提交或者创建镜像时，卷不被包含在镜像里。

注意：

- 卷可以在容器间共享。即便容器停止，卷里的内容依旧存在。在后面的章节会看到如何使用卷来管理数据。

回到刚才的例子。当我们因为某些原因不想把应用或者代码构建到镜像中时，就体现出卷的价值了。例如：

- 希望同时对代码做开发和测试；
- 代码改动很频繁，不想在开发过程中重构镜像；
- 希望在多个容器间共享代码。

`-v` 选项通过指定一个目录或者登上与容器上与该目录分离的本地宿主机来工作，这两个目录用`:`分隔。如果容器目录不存在，Docker会自动创建一个。

也可以通过在目录后面加上`rw`或者`ro`来指定容器内目录的读写状态：

```sh
$ sudo docker run -d -p 80 --name website \
-v $PWD/website:/var/www/html/website:ro \
aaa/nginx nginx
```

说明：

- 这将使目的目录 `/var/www/html/website` 变成只读状态。
- 在 Nginx 网站容器里，我们通过卷将 `$PWD/website` 挂载到容器的`/var/www/html/website`目录，顺利挂载了正在开发的本地网站。在Nginx配置里（在配置文件`/etc/nginx/conf.d/global.conf`中），已经指定了这个目录为Nginx服务器的工作目录。




```sh
root@iZuf66eabunrloh2og4jgsZ:~# docker ps –l
CONTAINER ID　IMAGE　　　　　　　　　　 ... PORTS　　　　　　　　　　 NAMES
6751b94bb5c0　jamtur01/nginx:latest ... 0.0.0.0:49161->80/tcp　website
root@iZuf66eabunrloh2og4jgsZ:~# curl 0.0.0.0:32772
<head>

<title>Test website</title>

</head>

<body>

<h1>This is a test website</h1>

</body>
```

说明：

- `docker ps` 可以看到名为website的容器正处于活跃状态，容器的80端口被映射到宿主机的49161端口。
- `curl` 可以访问对应的端口。

由于在服务器上尝试此程序，因此，我从本地电脑也可以访问：

<p align="center">
    <img width="50%" height="50%" src="http://images.iterate.site/blog/image/20200206/fhclRSmP5AcI.png?imageslim">
</p>


或者可以在Docker的宿主机上浏览49161端口：

<p align="center">
    <img width="50%" height="50%" src="http://images.iterate.site/blog/image/20200206/VBbGBqiaFPPd.png?imageslim">
</p>



注意：

- 如果用户在使用BootDocker或者Docker Toolbox，需要注意这两个工具都会在本地创建一个虚拟机，这个虚拟机具有自己独立的网络接口和IP地址。需要连接到虚拟机的地址，而不是localhost或者用户的本地主机的IP地址。

### 修改网站

我们已经得到了一个可以工作的网站！现在，如果要修改网站，该怎么办？可以直接打开本地宿主机的website目录下的index.html文件并修改：

```sh
$ vi $PWD/website/index.html
```

改为：

```
<head>

<title>Test website</title>

</head>

<body>

<h1>This is a test website for Docker</h1>

</body>
```


刷新一下浏览器，看看现在的网站是什么样的：

<p align="center">
    <img width="50%" height="50%" src="http://images.iterate.site/blog/image/20200206/deUvWw0EToir.png?imageslim">
</p>

<span style="color:red;">为什么这个地方可以直接刷新呢？是因为 `-v`这个选项允许我们将宿主机的目录作为卷，挂载到容器里吗？</span>


可以看到，Sample网站已经更新了。显然这个修改太简单了，不过可以看出，更复杂的修改也并不困难。更重要的是，正在测试网站的运行环境，完全是生产环境里的真实状态。


现在可以给每个用于生产的网站服务环境（如Apache、Nginx）配置一个容器，给不同开发框架的运行环境（如PHP或者Ruby on Rails）配置一个容器，或者给后端数据库配置一个容器，等等。

<span style="color:red;">这个地方有点不是很清楚，到底一个项目中，要配置多少类似的容器，按什么来划分？</span>




## 使用Docker构建并测试Web应用程序

现在来看一个更复杂的例子，测试一个更大的Web应用程序。我们将要测试一个基于Sinatra的Web应用程序，而不是静态网站，然后我们将基于Docker来对这个应用进行测试。

Sinatra是一个基于Ruby的Web应用框架，它包含一个Web应用库，以及简单的领域专用语言（即DSL）来构建Web应用程序。与其他复杂的Web应用框架（如Ruby on Rails）不同，Sinatra并不遵循MVC模式，而关注于让开发者创建快速、简单的Web应用。

因此，Sinatra非常适合用来创建一个小型的示例应用进行测试。在这个例子里，我们将创建一个应用程序，它接收输入的URL参数，并以JSON散列的结构输出到客户端。通过这个例子，我们也将展示一下如何将Docker容器链接起来。

### 构建Sinatra应用程序

我们先来创建一个sinatra目录，用来存放应用程序的代码，以及构建时我们所需的所有相关文件：

```sh
$ mkdir -p sinatra
$ cd sinatra
```

在sinatra目录下，让我们从Dockerfile开始，构建一个基础镜像，并用这个镜像来开发Sinatra Web应用程序：

```docker
FROM ubuntu:14.04
MAINTAINER James Turnbull "james@example.com"
ENV REFRESHED_AT 2014-06-01
RUN apt-get update -yqq && apt-get -yqq install ruby ruby-dev build-essential redis-tools
RUN gem install --no-rdoc --no-ri sinatra json redis
RUN mkdir -p /opt/webapp
EXPOSE 4567
CMD [ "/opt/webapp/bin/webapp" ]
```


可见：

- 我们已经创建了另一个基于Ubuntu的镜像，安装了Ruby和RubyGem，并且使用gem命令安装了sinatra、json和redis gem。sinatra是Sinatra的库，json用来提供对JSON的支持。redis gem在后面会用到，用来和Redis数据库进行集成。
- 我们已经创建了一个目录来存放新的Web应用程序，并公开了WEBrick的默认端口4567。
- 最后，使用CMD指定/opt/webapp/bin/webapp作为Web应用程序的启动文件。

现在使用docker build命令来构建新的镜像：

```sh
$ sudo docker build -t jamtur01/sinatra .
```

<span style="color:red;">这里有问题，build 到 redis 的时候，说版本不支持。</span>

### 创建Sinatra容器

我们已经创建了镜像，现在让我们下载Sinatra Web应用程序的源代码。这份代码可以在本书的官网[5]或Docker Book网站[6]找到。这个应用程序在webapp目录下，由bin和lib两个目录组成。

现在将其下载到sinatra目录中：


```sh
$ cd sinatra
$ wget --cut-dirs=3 -nH -r --reject Dockerfile,index.html –-no-parent http://dockerbook.com/code/5/sinatra/webapp/
$ ls -l webapp
...
```


下面我们就来快速浏览一下webapp源代码的核心，其源代码保存在 `sinatra/webapp/lib/app.rb` 文件中：


```rb
require "rubygems"
require "sinatra"
require "json"
class App < Sinatra::Application
set :bind, '0.0.0.0'
get '/' do
"<h1>DockerBook Test Sinatra app</h1>"
end
post '/json/?' do
params.to_json
end
end
```


可以看到，这个程序很简单，所有访问/json端点的POST请求参数都会被转换为JSON的格式后输出。

这里还要使用chmod命令保证webapp/bin/webapp 这个文件可以执行：

```
$ chmod +x webapp/bin/webapp
```


现在我们就可以基于我们的镜像，通过docker run命令启动一个新容器。要启动容器，我们需要在sinatra目录下，因为我们需要将这个目录下的源代码通过卷挂载到容器中去：

```sh
$ sudo docker run -d -p 4567 --name webapp \
-v $PWD/webapp:/opt/webapp jamtur01/sinatra
```


这里从jamtur01/sinatra镜像创建了一个新的名为webapp的容器。指定了一个新卷，使用存放新Sinatra Web应用程序的webapp目录，并将这个卷挂载到在Dockerfile里创建的目录/opt/webapp。

我们没有在命令行中指定要运行的命令，而是使用在镜像的Dockerfile中CMD指令设置的命令：

```docker
. . .
CMD [ "/opt/webapp/bin/webapp" ]
. . .
```


从这个镜像启动容器时，将会执行这一命令。

也可以使用`docker logs`命令查看被执行的命令都输出了什么：

```sh
$ sudo docker logs webapp
[2013-08-05 02:22:14] INFO　WEBrick 1.3.1
[2013-08-05 02:22:14] INFO　ruby 1.8.7 (2011-06-30) [x86_64-linux]
== Sinatra/1.4.3 has taken the stage on 4567 for development with backup from WEBrick
[2013-08-05 02:22:14] INFO　WEBrick::HTTPServer#start: pid=1 port=4567
```

运行docker logs命令时加上`-f`标志可以达到与执行`tail -f`命令一样的效果—持续输出容器的STDERR和STDOUT里的内容：

```sh
$ sudo docker logs -f webapp
. . .
```

可以使用`docker top`命令查看Docker容器里正在运行的进程：

```sh
$ sudo docker top webapp
UID　PID　　PPID　　C　STIME　TTY　TIME　　　CMD
root 21506　15332　 0　20:26　?　　00:00:00　/usr/bin/ruby /opt/webapp/bin/webapp
```


从这一日志可以看出，容器中已经启动了Sinatra，而且WEBrick服务进程正在监听4567端口，等待测试。先查看一下这个端口映射到本地宿主机的哪个端口：

```sh
$ sudo docker port webapp 4567
0.0.0.0:49160
```


目前，Sinatra应用还很基础，没做什么。它只是接收输入参数，并将输入转化为JSON输出。现在可以使用curl命令来测试这个应用程序了：


```sh
$ curl -i -H 'Accept: application/json' \
-d 'name=Foo&status=Bar' http://localhost:49160/json
HTTP/1.1 200 OK
X-Content-Type-Options: nosniff
Content-Length: 29
X-Frame-Options: SAMEORIGIN
Connection: Keep-Alive
Date: Mon, 05 Aug 2013 02:22:21 GMT
Content-Type: text/html;charset=utf-8
Server: WEBrick/1.3.1 (Ruby/1.8.7/2011-06-30)
X-Xss-Protection: 1; mode=block
{"name":"Foo","status":"Bar"}
```


可以看到，我们给Sinatra应用程序传入了一些URL参数，并看到这些参数转化成JSON散列后的输出：{"name":"Foo","status":"Bar"}。

成功！然后试试看，我们能不能通过连接到运行在另一个容器里的服务，把当前的示例应用程序容器扩展为真正的应用程序栈。

### 扩展Sinatra应用程序来使用Redis

现在我们将要扩展Sinatra应用程序，加入Redis后端数据库，并在Redis数据库中存储输入的URL参数。为了达到这个目的，我们要下载一个新版本的Sinatra应用程序。我们还将创建一个运行Redis数据库的镜像和容器。之后，要利用Docker的特性来关联两个容器。

1．升级我们的Sinatra应用程序

让我们从下载一个升级版的Sinatra应用程序开始，这个升级版中增加了连接Redis的配置。在sinatra目录中，我们下载了我们这个应用的启用了Redis的版本，并保存到一个新目录webapp_redis中：

```sh
$ cd sinatra
$ wget --cut-dirs=3 -nH -r --reject Dockerfile,index.html -–no-
　parent http://dockerbook.com/code/5/sinatra/webapp_redis/
$ ls -l webapp_redis
. . .
```


我们看到新应用程序已经下载，现在让我们看一下`lib/app.rb`文件中的核心代码：

```rb
require "rubygems"
require "sinatra"
require "json"
require "redis"
class App < Sinatra::Application
redis = Redis.new(:host => 'db', :port => '6379')
set :bind, '0.0.0.0'
get '/' do
"<h1>DockerBook Test Redis-enabled Sinatra app</h1>"
end
get '/json' do
params = redis.get "params"
params.to_json
end
post '/json/?' do
redis.set "params", [params].to_json
params.to_json
end
end
```


注意 可以在http://dockerbook.com/code/5/sinatra/webapp_redis/ 或者Docker Book网站（https:// github.com/jamtur01/dockerbook-code）上获取升级版的启用了Redis的Sinatra应用程序的完整代码。

我们可以看到新版本的代码和前面的代码几乎一样，只是增加了对Redis的支持。我们创建了一个到Redis的连接，用来连接名为db的宿主机上的Redis数据库，端口为6379。我们在POST请求处理中，将URL参数保存到了Redis数据库中，并在需要的时候通过GET请求从中取回这个值。

我们同样需要确保webapp_redis/bin/webapp文件在使用之前具备可执行权限，这可以通过chmod命令来实现：

```sh
$ chmod +x webapp_redis/bin/webapp
```


2．构建Redis数据库镜像

为了构建Redis数据库，要创建一个新的镜像。我们需要在sinatra目录下创建一个redis目录，用来保存构建Redis容器所需的所有相关文件：

```sh
$ mkdir -p sinatra/redis
$ cd sinatra/redis
```



在`sinatra/redis` 目录中，让我们从Redis镜像的另一个Dockerfile开始：

```docker
FROM ubuntu:14.04
MAINTAINER James Turnbull "james@example.com"
ENV REFRESHED_AT 2014-06-01
RUN apt-get -yyq update && apt-get -yqq install redis-server redis-tools
EXPOSE 6379
ENTRYPOINT [ "/usr/bin/redis-server" ]
CMD []
```


我们在Dockerfile里指定了安装Redis服务器，公开6379端口，并指定了启动Redis服务器的ENTRYPOINT。现在来构建这个镜像，命名为jamtur01/redis：

```sh
$ sudo docker build -t jamtur01/redis .
```

现在从这个新镜像构建容器。

```sh
$ sudo docker run -d -p 6379 --name redis jamtur01/redis
0a206261f079
```


可以看到，我们从jamtur01/redis镜像启动了一个新的容器，名字是redis。注意，我们指定了-p标志来公开6379端口。

看看这个端口映射到宿主机的哪个端口：

```sh
$ sudo docker port redis 6379
0.0.0.0:49161
```


Redis的端口映射到了49161端口。试着连接到这个Redis实例。

我们需要在本地安装Redis客户端做测试。在Ubuntu系统上，客户端程序一般在redis-tools包里：

```sh
$ sudo apt-get -y install redis-tools
```


而在Red Hat及相关系统上，包名则为redis：

```sh
$ sudo yum install -y -q redis
```

然后，可以使用redis-cli命令来确认Redis服务器工作是否正常：


```sh
$ redis-cli -h 127.0.0.1 -p 49161
redis 127.0.0.1:49161>
```

这里使用Redis客户端连接到127.0.0.1的49161端口，验证了Redis服务器正在正常工作。可以使用quit命令来退出Redis CLI 接口。

### 将Sinatra应用程序连接到Redis容器

现在来更新Sinatra应用程序，让其连接到Redis并存储传入的参数。为此，需要能够与Redis服务器对话。要做到这一点，可以用以下几种方法。

Docker的内部网络。

从Docker 1.9及之后的版本开始，可以使用Docker Networking以及docker network命令。

Docker链接。一个可以将具体容器链接到一起来进行通信的抽象层。

那么，我们应该选择哪种方法呢？第一种方法，Docker的内部网络这种解决方案并不是灵活、强大。我们针对这种方式的讨论，也只是为了介绍Docker网络是如何工作的。我们不推荐采用这种方式来连接Docker容器。

两种比较现实的连接Docker容器的方式是Docker Networking和Docker链接（Docker link）。具体应该选择哪种方式取决于用户运行的Docker的版本。如果





用户正在使用Docker 1.9或者更新的版本，推荐使用Docker Networking，如果使用的是Docker 1.9之前的版本，应该选择Docker链接。

在Docker Networking和Docker链接之间也有一些区别。这也是我们推荐使用Docker Networking而不是链接的原因。

Docker Networking可以将容器连接到不同宿主机上的容器。通过Docker Networking连接的容器可以在无需更新连接的情况下，对停止、启动或者重启容器。而使用Docker链接，则可能需要更新一些配置，或者重启相应的容器来维护Docker容器之间的链接。

使用Docker Networking，不必事先创建容器再去连接它。同样，也不必关心容器的运行顺序，读者可以在网络内部获得容器名解析和发现。

在后面几节中，我们将会看到将Docker容器连接起来的各种解决方案。

### Docker内部连网

第一种方法涉及Docker自己的网络栈。到目前为止， 我们看到的Docker容器都是公开端口并绑定到本地网络接口的，这样可以把容器里的服务在本地Docker宿主机所在的外部网络上（比如，把容器里的80端口绑到本地宿主机的更高端口上）公开。除了这种用法，Docker这个特性还有种用法我们没有见过，那就是内部网络。在安装Docker时，会创建一个新的网络接口，名字是docker0。每个Docker容器都会在这个接口上分配一个IP地址。来看看目前Docker宿主机上这个网络接口的信息，如代码清单5-39所示。

提示 Docker自1.5.0版本开始支持IPv6，要启动这一功能，可以在运行Docker守护进程时加上--ipv6标志。

代码清单5-39　docker0网络接口

```sh
$ ip a show docker0
4: docker0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP
　　link/ether 06:41:69:71:00:ba brd ff:ff:ff:ff:ff:ff
　　inet 172.17.42.1/16 scope global docker0
　　inet6 fe80::1cb3:6eff:fee2:2df1/64 scope link
　　valid_lft forever preferred_lft forever
. . .
```


可以看到，docker0接口有符合RFC1918的私有IP地址，范围是172.16~172.30。接口本身的地址172.17.42.1是这个Docker网络的网关地址，也是所有Docker容器的网关地址。

提示 Docker会默认使用172.17.x.x作为子网地址，除非已经有别人占用了这个子网。如果这个子网被占用了，Docker会在172.16~172.30这个范围内尝试创建子网。

接口docker0是一个虚拟的以太网桥，用于连接容器和本地宿主网络。如果进一步查看Docker宿主机的其他网络接口，会发现一系列名字以veth开头的接口，如代码

清单5-40所示。

代码清单5-40　veth接口

vethec6a　Link encap:Ethernet　HWaddr 86:e1:95:da:e2:5a
　　　　　 inet6 addr: fe80::84e1:95ff:feda:e25a/64 Scope:Link
. . .

Docker每创建一个容器就会创建一组互联的网络接口。这组接口就像管道的两端（就是说，从一端发送的数据会在另一端接收到）。这组接口其中一端作为容器里的eth0接口，而另一端统一命名为类似vethec6a这种名字，作为宿主机的一个端口。可以把veth接口认为是虚拟网线的一端。这个虚拟网线一端插在名为docker0的网桥上，另一端插到容器里。通过把每个veth*接口绑定到docker0网桥，Docker创建了一个虚拟子网，这个子网由宿主机和所有的Docker容器共享。

进入容器里面，看看这个子网管道的另一端，如代码清单5-41所示。

代码清单5-41　容器内的eth0接口

```sh
$ sudo docker run -t -i ubuntu /bin/bash
root@b9107458f16a:/# ip a show eth0
1483: eth0: <BROADCAST,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast state UP group default qlen 1000
　　link/ether f2:1f:28:de:ee:a7 brd ff:ff:ff:ff:ff:ff
　　inet 172.17.0.29/16 scope global eth0
　　inet6 fe80::f01f:28ff:fede:eea7/64 scope link
　　valid_lft forever preferred_lft forever
```


可以看到，Docker给容器分配了IP地址172.17.0.29作为宿主虚拟接口的另一端。

这样就能够让宿主网络和容器互相通信了。

让我们从容器内跟踪对外通信的路由，看看是如何建立连接的，如代码清单5-42所示。

代码清单5-42　在容器内跟踪对外的路由

```sh
root@b9107458f16a:/# apt-get -yqq update && apt-get install -yqq traceroute
. . .
root@b9107458f16a:/# traceroute google.com
traceroute to google.com (74.125.228.78), 30 hops max, 60 byte packets 1　172.17.42.1 (172.17.42.1)　0.078 ms　0.026 ms　0.024 ms
. . .
15　iad23s07-in-f14.1e100.net (74.125.228.78)　32.272 ms　28.050 ms　25.662 ms
```



可以看到，容器地址后的下一跳是宿主网络上docker0接口的网关 IP172.17.42.1。

不过Docker网络还有另一个部分配置才能允许建立连接：防火墙规则和NAT配置。

这些配置允许Docker在宿主网络和容器间路由。现在来查看一下宿主机上的

IPTables NAT配置，如代码清单5-43所示。

代码清单5-43　Docker的iptables和NAT配置

$ sudo iptables -t nat -L -n

Chain PREROUTING (policy ACCEPT)

target　prot opt source　　　 destination

DOCKER　all　 --　0.0.0.0/0　 0.0.0.0/0　　 ADDRTYPE match dst-type LOCAL

Chain OUTPUT (policy ACCEPT)

target　prot opt source　　　 destination

DOCKER　all　 --　0.0.0.0/0　 !127.0.0.0/8　ADDRTYPE match dst-type LOCAL

Chain POSTROUTING (policy ACCEPT)

target　　 prot opt source　　　　　destination

MASQUERADE all　--　172.17.0.0/16　!172.17.0.0/16

Chain DOCKER (2 references)

target　prot opt source　　　destination

DNAT　　tcp　 --　0.0.0.0/0　0.0.0.0/0　　　tcp dpt:49161 to:172.17.0.18:6379

这里有几个值得注意的IPTables规则。首先，我们注意到，容器默认是无法访问

的。从宿主网络与容器通信时，必须明确指定打开的端口。下面我们以DNAT（即目

标NAT）这个规则为例，这个规则把容器里的访问路由到Docker宿主机的49161端

口。

提示 想了解更多关于Docker的高级网络配置，有一篇文章[7]很有用。

Redis容器的网络

下面我们用docker`ìnspect命令来查看新的Redis容器的网络配置，如代码清单

5-44所示。

代码清单5-44　Redis容器的网络配置

$ sudo docker inspect redis

. . .

　　"NetworkSettings": {

　　　　 "Bridge": "docker0",

　　　　 "Gateway": "172.17.42.1",

　　　　 "IPAddress": "172.17.0.18",

　　　　 "IPPrefixLen": 16,

　　　　 "PortMapping": null,

　　　　 "Ports": {

　　　　　　 "6379/tcp": [

　　　　　　　　{

　　　　　　　　　　"HostIp": "0.0.0.0",

　　　　　　　　　　"HostPort": "49161"

　　　　　　　　}

　　　　　　]

　　　　}

　　},

. . .

docker inspect命令展示了Docker容器的细节，这些细节包括配置信息和网络状

况。为了清晰，这个例子去掉了大部分信息，只展示了网络配置。也可以在命令里

使用-f标志，只获取IP地址，如代码清单5-45所示。

代码清单5-45　查看Redis容器的IP地址

$ sudo docker inspect -f '{{ .NetworkSettings.IPAddress }}' redis 172.17.0.18

通过运行docker inspect命令可以看到，容器的IP地址为172.17.0.18，并使用

了docker0接口作为网关地址。还可以看到6379端口被映射到本地宿主机的49161端

口。只是，因为运行在本地的Docker宿主机上，所以不是一定要用映射后的端口，

也可以直接使用172.17.0.18地址与Redis服务器的6379端口通信，如代码清单5-

46所示。

代码清单5-46　直接与Redis容器通信

$ redis-cli -h 172.17.0.18

redis 172.17.0.18:6379>

在确认完可以连接到Redis服务之后，可以使用quit命令退出Redis接口。

注意 Docker默认会把公开的端口绑定到所有的网络接口上。因此，也可以通过localhost或者127.0.0.1来

访问Redis服务器。

因此，虽然第一眼看上去这是让容器互联的一个好方案，但可惜的是，这种方法有

两个大问题：第一，要在应用程序里对Redis容器的IP地址做硬编码；第二，如果

重启容器，Docker会改变容器的IP地址。现在用docker restart命令来看看地址

的变化，如代码清单5-47所示。（如果使用docker kill命令杀死容器再重启，也

会得到同样的结果。）

代码清单5-47　重启Redis容器

$ sudo docker restart redis





让我们查看一下容器的IP地址，如代码清单5-48所示。

代码清单5-48　查找重启后Redis容器的IP地址

$ sudo docker inspect -f '{{ .NetworkSettings.IPAddress }}' redis 172.17.0.19

可以看到，Redis容器有了新的IP地址172.17.0.19，这就意味着，如果在Sinatra

应用程序里硬编码了原来的地址，那么现在就无法让应用程序连接到Redis数据库

了。这可不那么好用。

谢天谢地，从Docker 1.9开始，Docker连网已经灵活得多。让我们来看一下，如何

用新的连网框架连接容器。

5.2.6　Docker Networking

容器之间的连接用网络创建，这被称为Docker Networking，也是Docker 1.9发

布版本中的一个新特性。Docker Networking允许用户创建自己的网络，容器可以

通过这个网上互相通信。实质上，Docker Networking以新的用户管理的网络补充

了现有的docker0。更重要的是，现在容器可以跨越不同的宿主机来通信，并且网

络配置可以更灵活地定制。Docker Networking也和Docker Compose以及Swarm进

行了集成，第7章将对Docker Compose和Swarm进行介绍。

注意 Docker Networking支持也是可插拔的，也就是说可以增加网络驱动以支持来自不同网络设备提供商

（如Cisco和VMware）的特定拓扑和网络框架。

下面我们就来看一个简单的例子，启动前面的Docker链接例子中使用的Web应用程

序以及Redis容器。要想使用Docker网络，需要先创建一个网络，然后在这个网络

下启动容器，如代码清单5-49所示。

代码清单5-49　创建Docker网络

$ sudo docker network create app

ec8bc3a70094a1ac3179b232bc185fcda120dad85dec394e6b5b01f7006476d4

这里用docker network命令创建了一个桥接网络，命名为app，这个命令返回新创

建的网络的网络ID。

然后可以用docker network inspect命令查看新创建的这个网络，如代码清单5-

50所示。

代码清单5-50　查看app网络

$ sudo docker network inspect app

[

　　{

　　　　"Name": "app",

　　　　"Id": "

　　　　 ec8bc3a70094a1ac3179b232bc185fcda120dad85dec394e6b5b01f7006476d4

　　　　 ",

　　　　"Scope": "local",

　　　　"Driver": "bridge",

　　　　"IPAM": {

　　　　　　"Driver": "default",

　　　　　　"Config": [

　　　　　　　　{}

　　　　　　]

　　　　},

　　　　"Containers": {},

　　　　"Options": {}

　　}

]

我们可以看到这个新网络是一个本地的桥接网络（这非常像docker0网络），而且

现在还没有容器在这个网络中运行。

提示 除了运行于单个主机之上的桥接网络，我们也可以创建一个overlay网络，overlay网络允许我们跨多

台宿主机进行通信。可以在Docker多宿主机网络文档[8]中获取更多关于overlay网络的信息。

可以使用docker network ls命令列出当前系统中的所有网络，如代码清单5-51所

示。

代码清单5-51　docker``network``ls命令

$ sudo docker network ls

NETWORK ID　　　　　　NAME　　　　　　　　　　DRIVER

a74047bace7e　　　　　bridge　　　　　　　　 bridge

ec8bc3a70094　　　　　app　　　　　　　　　　 bridge

8f0d4282ca79　　　　　none　　　　　　　　　　null

7c8cd5d23ad5　　　　　host　　　　　　　　　　host

也可以使用docker network rm命令删除一个Docker网络。下面我们先从启动

Redis容器开始，在之前创建的app网络中添加一些容器，如代码清单5-52所示。

代码清单5-52　在Docker网络中创建Redis容器

$ sudo docker run -d --net=app --name db jamtur01/redis

这里我们基于jamtur01/redis镜像创建了一个名为db的新容器。我们同时指定了一

个新的标志--net，--net标志指定了新容器将会在哪个网络中运行。

这时，如果再次运行docker network inspect命令，将会看到这个网络更详细的

信息，如代码清单5-53所示。

代码清单5-53　更新后的app网络

$ sudo docker network inspect app

[

　　{

　　　　"Name": "app",

　　　　"Id": "

　　　　 ec8bc3a70094a1ac3179b232bc185fcda120dad85dec394e6b5b01f7006476d4

　　　　 ",

　　　　"Scope": "local",

　　　　"Driver": "bridge",

　　　　"IPAM": {

　　　　　　"Driver": "default",

　　　　　　"Config": [

　　　　　　　　{}

　　　　　　]

　　　　},

　　　　"Containers": {

　　　　　　 "9

　　　　　　　a5ac1aa39d84a1678b51c26525bda2b89fb9a837f03c871441aec645958fe73

　　　　　　　": {

　　　　　　　　"EndpointID": "21

　　　　　　　　 a90395cb5a2c2868aaa77e05f0dd06a4ad161e13e99ed666741dc0219174ef

　　　　　　　　 ",

　　　　　　　　"MacAddress": "02:42:ac:12:00:02",

　　　　　　　　"IPv4Address": "172.18.0.2/16",

　　　　　　　　"IPv6Address": ""

　　　　　　}

　　　　},

　　　　"Options": {}

　　}

]

现在在这个网络中，我们可以看到一个容器，它有一个MAC地址，并且IP地址

为172.18.0.2。

接着，我们再在我们创建的网络下增加一个运行启用了Redis的Sinatra应用程序的

容器，要做到这一点，需要先回到sinatra/webapp目录下，如代码清单5-54所

示。

代码清单5-54　链接Redis容器

$ cd sinatra/webapp

$ sudo docker run -p 4567 \

--net=app --name webapp -t -i \

-v $PWD/webapp:/opt/webapp jamtur01/sinatra \

/bin/bash

root@305c5f27dbd1:/#

注意 这是启用了Redis的Sinatra应用程序，我们在前面Docker链接的例子中用过。其代码可以从

http://dockerbook.com/code/5/sinatra/webapp_redis/或者Docker Book网站[9]获取。

我们在app网络下启动了一个名为webapp的容器。我们以交互的方式启动了这个容

器，以便我们可以进入里面看看它内部发生了什么。

由于这个容器是在app网络内部启动的，因此Docker将会感知到所有在这个网络下

运行的容器，并且通过/etc/hosts文件将这些容器的地址保存到本地DNS中。我们

就在webapp容器中看看这些信息，如代码清单5-55所示。

代码清单5-55　webapp容器的/etc/hosts文件

cat /etc/hosts

172.18.0.3　　　305c5f27dbd1

127.0.0.1　　　 localhost

. . .

172.18.0.2　　　db

172.18.0.2　　　db.app

我们可以看到/etc/hosts文件包含了webapp容器的IP地址，以及一条localhost

记录。同时，该文件还包含两条关于db容器的记录。第一条是db容器的主机名和IP

地址172.18.0.2。第二条记录则将app网络名作为域名后缀添加到主机名后面，app

网络内部的任何主机都可以使用hostname.app的形式来被解析，这个例子里

是db.app。下面我们就来试试，如代码清单5-56所示。

代码清单5-56　Pinging db.ap``p

$ ping db.app

PING db.app (172.18.0.2) 56(84) bytes of data.

64 bytes from db (172.18.0.2): icmp_seq=1 ttl=64 time=0.290 ms

64 bytes from db (172.18.0.2): icmp_seq=2 ttl=64 time=0.082 ms

64 bytes from db (172.18.0.2): icmp_seq=3 ttl=64 time=0.111 ms

. . .

但是，在这个例子里，我们只需要db条目就可以让我们的应用程序正常工作了，我

们的Redis连接代码里使用的也是db这个主机名，如代码清单5-57所示。

代码清单5-57　代码中指定的Redis DB主机名

redis = Redis.new(:host => 'db', :port => '6379')

现在，就可以启动我们的应用程序，并且让Sinatra应用程序通过db和webapp两个

容器间的连接，将接收到的参数写入Redis中，db和webapp容器间的连接也是通过

app网络建立的。重要的是，如果任何一个容器重启了，那么它们的IP地址信息则会

自动在/etc/hosts文件中更新。也就是说，对底层容器的修改并不会对我们的应用

程序正常工作产生影响。

让我们在容器内启动我们的应用程序，如代码清单5-58所示。

代码清单5-58　启动启用了Redis的Sinatra应用程序

root@305c5f27dbd1:/# _nohup /opt/webapp/bin/webapp &_

nohup: ignoring input and appending output to 'nohup.out'

这里我们以后台运行的方式启动了这个Sinatra应用程序，下面我们就来检查一下

我们的Sinatra容器为这个应用程序绑定了哪个端口，如代码清单5-59所示。

代码清单5-59　检查Sinatra容器的端口映射情况

$ sudo docker port webapp 4567

0.0.0.0:49161

很好，我们看到容器中的4567端口被绑定到了宿主机上的49161端口。让我们利用

这些信息在Docker宿主机上，通过curl命令来测试一下我们的应用程序，如代码清

单5-60所示。

代码清单5-60　测试启用了Redis的Sinatra应用程序

$ curl -i -H 'Accept: application/json' \

-d 'name=Foo&status=Bar' http://localhost:49161/json

HTTP/1.1 200 OK

X-Content-Type-Options: nosniff

Content-Length: 29

X-Frame-Options: SAMEORIGIN

Connection: Keep-Alive

Date: Mon, 01 Jun 2014 02:22:21 GMT

Content-Type: text/html;charset=utf-8

Server: WEBrick/1.3.1 (Ruby/1.8.7/2011-06-30)

X-Xss-Protection: 1; mode=block

{"name":"Foo","status":"Bar"}

接着我们再来确认一下Redis实例是否已经接收到了这次更新，如代码清单5-61所

示。

代码清单5-61　确认Redis容器数据

$ curl -i http://localhost:49161/json

"[{\"name\":\"Foo\",\"status\":\"Bar\"}]"

我们连接到了已经连接到Redis的应用程序，然后检查了一下是否存在一个名

为params的键，并查询这个键，看我们的参数（name=Foò`和``status=Bar）是

否已经保存到Redis中。一切工作正常。

1．将已有容器连接到Docker网络

也可以将正在运行的容器通过docker network connect命令添加到已有的网络

中。因此，我们可以将已经存在的容器添加到app网络中。假设已经存在的容器名

为db2，这个容器里也运行着Redis，让我们将这个容器添加到app网络中去，如代

码清单5-62所示。

代码清单5-62　添加已有容器到app网络

$ sudo docker network connect app db2

现在如果查看app网络的详细信息，应该会看到3个容器，如代码清单5-63所示。

代码清单5-63　添加db2容器后的app网络

$ sudo docker network inspect app

. . .

"Containers": {

"2

fa7477c58d7707ea14d147f0f12311bb1f77104e49db55ac346d0ae961ac401

": {

"EndpointID": "

c510c78af496fb88f1b455573d4c4d7fdfc024d364689a057b98ea20287bfc0d

",

"MacAddress": "02:42:ac:12:00:02",

"IPv4Address": "172.18.0.2/16",

"IPv6Address": ""

},

"305

c5f27dbd11773378f93aa58e86b2f710dbfca9867320f82983fc6ba79e779

": {

"EndpointID": "37

be9b06f031fcc389e98d495c71c7ab31eb57706ac8b26d4210b81d5c687282

",

"MacAddress": "02:42:ac:12:00:03",

"IPv4Address": "172.18.0.3/16",

"IPv6Address": ""

},

"70

df5744df3b46276672fb49f1ebad5e0e95364737334e188a474ef4140ae56b

": {

"EndpointID": "47

faec311dfac22f2ee8c1b874b87ce8987ee65505251366d4b9db422a749a1e

",

"MacAddress": "02:42:ac:12:00:04",

"IPv4Address": "172.18.0.4/16",

"IPv6Address": ""

}

},

. . .

所有这3个容器的/etc/hosts文件都将会包含webapp、db和db2容器的DNS信息。

我们也可以通过docker network disconnect命令断开一个容器与指定网络的链

接，如代码清单5-64所示。

代码清单5-64　从网络中断开一个容器

$ sudo docker network disconnect app db2

这条命令会从app网络中断开db2容器。

一个容器可以同时隶属于多个Dcoker网络，所以可以创建非常复杂的网络模型。

提示 Docker官方文档[10]有中很多关于Docker Networking的详细信息。

2．通过Docker链接连接容器

连接容器的另一种选择就是使用Docker链接。在Docker 1.9之前，这是首选的容器

连接方式，并且只有在运行1.9之前版本的情况下才推荐这种方式。让一个容器链接

到另一个容器是一个简单的过程，这个过程要引用容器的名字。

考虑到还在使用低于Docker 1.9版本的用户，我们来看看Docker链接是如何工作

的。让我们从新建一个Redis容器开始（或者也可以重用之前创建的那个容器），如

代码清单5-65所示。

代码清单5-65　启动另一个Redis容器

$ sudo docker run -d --name redis jamtur01/redis

提示 还记得容器的名字是唯一的吗？如果要重建一个容器，在创建另一个名叫redis的容器之前，需要先

用docker``rm命令删掉旧的redis容器。

现在我们已经在新容器里启动了一个Redis实例，并使用--name标志将新容器命名

为redis。

注意 读者也应该注意到了，这里没有公开容器的任何端口。一会儿就能看到这么做的原因。

现在让我们启动Web应用程序容器，并把它链接到新的Redis容器上去，如代码清单

5-66所示。

代码清单5-66　链接Redis容器

$ sudo docker run -p 4567 \

--name webapp --link redis:db -t -i \

-v $PWD/webapp_redis:/opt/webapp jamtur01/sinatra \

/bin/bash

root@811bd6d588cb:/#

提示 还需要使用docker rm命令停止并删除之前的webapp容器。

这个命令做了不少事情，我们要逐一解释。首先，我们使用-p标志公开了4567端

口，这样就能从外面访问Web应用程序。

我们还使用了--name标志给容器命名为webapp，并使用了-v标志把Web应用程序目

录作为卷挂载到了容器里。

然而，这次我们使用了一个新标志--link。--link标志创建了两个容器间的客户-

服务链接。这个标志需要两个参数：一个是要链接的容器的名字，另一个是链接的

别名。这个例子中，我们创建了客户联系，webapp容器是客户，redis容器是“服

务”，并且为这个服务增加了db作为别名。这个别名让我们可以一致地访问容器公

开的信息，而无须关注底层容器的名字。链接让服务容器有能力与客户容器通信，

并且能分享一些连接细节，这些细节有助于在应用程序中配置并使用这个链接。

连接也能得到一些安全上的好处。注意，启动Redis容器时，并没有使用-p标志公

开Redis的端口。因为不需要这么做。通过把容器链接在一起，可以让客户容器直接

访问任意服务容器的公开端口（即客户webapp容器可以连接到服务redis容器的

6379端口）。更妙的是，只有使用--link标志链接到这个容器的容器才能连接到这

个端口。容器的端口不需要对本地宿主机公开，现在���们已经拥有一个非常安全的

模型。通过这个安全模型，就可以限制容器化应用程序被攻击面，减少应用暴露的

网络。

提示 如果用户希望，出于安全原因（或者其他原因），可以强制Docker只允许有链接的容器之间互相通信。

为此，可以在启动Docker守护进程时加上--icc=false标志，关闭所有没有链接的容器间的通信。

也可以把多个容器链接在一起。比如，如果想让这个Redis实例服务于多个Web应用

程序，可以把每个Web应用程序的容器和同一个redis容器链接在一起，如代码清单

5-67所示。

代码清单5-67　链接Redis容器

$ sudo docker run -p 4567 --name webapp2 --link redis:db ...

. . .

$ sudo docker run -p 4567 --name webapp3 --link redis:db ...

. . .

我们也能够指定多次--link标志来连接到多个容器。

提示 容器链接目前只能工作于同一台Docker宿主机中，不能链接位于不同Docker宿主机上的容器。对于多宿

主机网络环境，需要使用Docker Networking，或者使用我们将在第7章讨论的Docker Swarm。Docker

Swarm可以用于完成多台宿主机上的Docker守护进程之间的编排。

最后，让容器启动时加载shell，而不是服务守护进程，这样可以查看容器是如何链

接在一起的。Docker在父容器里的以下两个地方写入了链接信息。

/etc/hosts文件中。

包含连接信息的环境变量中。

先来看看/etc/hosts文件，如代码清单5-68所示。

代码清单5-68　webapp的/etc/hosts文件

root@811bd6d588cb:/_# cat /etc/hosts_

172.17.0.33 811bd6d588cb

. . .

172.17.0.31 db b9107458f16a redis

这里可以看到一些有用的项。第一项是容器自己的IP地址和主机名（主机名是容器

ID的一部分）。第二项是由该连接指令创建的，它是redis容器的IP地址、名字、

容器ID和从该连接的别名衍生的主机名db。现在试着ping一下db容器，如代码清单

5-69所示。

提示 容器的主机名也可以不是其ID的一部分。可以在执行docker run命令时使用-h或者--hostname标志来

为容器设定主机名。

代码清单5-69　ping一下db容器

root@811bd6d588cb:/# ping db

PING db (172.17.0.31) 56(84) bytes of data.

64 bytes from db (172.17.0.31): icmp_seq=1 ttl=64 time=0.623 ms 64 bytes from db (172.17.0.31): icmp_seq=2 ttl=64 time=0.132 ms 64 bytes from db (172.17.0.31): icmp_seq=3 ttl=64 time=0.095 ms 64 bytes from db (172.17.0.31): icmp_seq=4 ttl=64 time=0.155 ms

. . .

如果在运行容器时指定--add-host选项，也可以在/etc/hosts文件中添加相应的

记录。例如，我们可能想添加运行Docker的主机的主机名和IP地址到容器中，如代

码清单5-70所示。

代码清单5-70　在容器内添加/etc/hosts记录

$ sudo docker run -p 4567 --add-host=docker:10.0.0.1 --name

　webapp2 --link redis:db ...

这将会在容器的/etc/hosts文件中添加一个名为docker、IP地址为10.0.0.1的宿

主机记录。

提示 还记得之前提到过，重启容器时，容器的IP地址会发生变化的事情么？从Docker 1.3开始，如果被连接

的容器重启了，/etc/host文件中的IP地址会用新的IP地址更新。

我们已经连到了Redis数据库，不过在真的利用这个连接之前，我们先来看看环境变

量里包含的其他连接信息。

让我们运行env命令来查看环境变量，如代码清单5-71所示。

代码清单5-71　显示用于连接的环境变量





root@811bd6d588cb:/# env

HOSTNAME=811bd6d588cb

DB_NAME=/webapp/db

DB_PORT_6379_TCP_PORT=6379

DB_PORT=tcp://172.17.0.31:6379

DB_PORT_6379_TCP=tcp://172.17.0.31:6379

DB_ENV_REFRESHED_AT=2014-06-01

DB_PORT_6379_TCP_ADDR=172.17.0.31

DB_PORT_6379_TCP_PROTO=tcp

PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin REFRESHED_AT=2014-06-01

. . .

可以看到不少环境变量，其中一些以DB开头。Docker在连接webapp和redis容器

时，自动创建了这些以DB开头的环境变量。以DB开头是因为DB是创建连接时使用的

别名。

这些自动创建的环境变量包含以下信息：

子容器的名字；

容器里运行的服务所使用的协议、IP和端口号；

容器里运行的不同服务所指定的协议、IP和端口号；

容器里由Docker设置的环境变量的值。

具体的变量会因容器的配置不同而有所不同（如容器的Dockerfil中由ENV和

EXPOSE指令定义的内容）。重要的是，这些变量包含一些我们可以在应用程序中用

来进行持久的容器间链接的信息。

5.2.7　使用容器连接来通信

那么如何使用这个连接呢？有以下两种方法可以让应用程序连接到Redis。

使用环境变量里的一些连接信息。

使用DNS和/etc/hosts信息。

先试试第一种方法，看看Web应用程序的lib/app.rb文件是如何利用这些新的环境

变量的，如代码清单5-72所示。

代码清单5-72　通过环境变量建立到Redis的连接

require 'uri'

. . .

uri=URI.parse(ENV['DB_PORT'])

redis = Redis.new(:host => uri.host, :port => uri.port)

. . .

这里使用Ruby的URI模块来解析DB_PORT环境变量，让后我们使用解析后的宿主机和

端口数出来配置Redis的连接信息。我们的应用程序现在就可以使用该连接信息来找

到在已链接容器中的Redis了。这种抽象模式避免了我们在代码中对Redis的IP地址

和端口进行硬编码，但是它仍然是一种简陋的服务发现方式。

还有一种方法，就是更灵活的本地DNS，这也是我们将要选用的解决方案，如代码清

单5-73所示。

提示 也可以在docker run命令中加入--dns或者--dns-search标志来为某个容器单独配置DNS。你可以设置

本地DNS解析的路径和搜索域。在https://docs.docker.com/articles/ networking/上可以找到更详细的

配置信息。如果没有这两个标志，Docker会根据宿主机的信息来配置DNS解析。可以在/etc/resolv.conf文

件中查看DNS解析的配置情况。

代码清单5-73　使用主机名连接Redis





redis = Redis.new(:host => 'db', :port => '6379') 我们的应用程序会在本地查找名叫db的宿主机，找到/etc/hosts文件里的相关项并

解析宿主机到正确的IP地址。这也解决了硬编码IP地址的问题。

我们现在就能像在5.2.7节中那样测试我们的容器连接是否能够正常工作了。

5.2.8　连接容器小结

我们已经了解了所有能让Docker容器互相连接的方式。在Docker 1.9及之后版本中

我们推荐使用Docker Networking，而在Docker 1.9之前的版本中则建议使用

Docker链接。无论采用哪种方式，读者都已经看到，我们可以轻而易举地创建一个

包含以下组件的Web应用程序栈：

一个运行Sinatra的Web服务器容器；

一个Redis数据库容器；

这两个容器间的一个安全连接。

读者应该也能看出，基于这个概念，我们可以轻易地扩展出任意数量的应用程序

栈，并由此来管理复杂的本地开发环境，比如：

Wordpress、HTML、CSS和JavaScript；

Ruby on Rails；

Django和Flask；

Node.js；

Play！；

用户喜欢的其他框架。

这样就可以在本地环境构建、复制、迭代开发用于生产的应用程序，甚至很复杂的

多层应用程序。







## Docker用于持续集成

到目前为止，所有的测试例子都是本地的、围绕单个开发者的（就是说，如何让本地开发者使用Docker来测试本地网站或者应用程序）。现在来看看在多开发者的持续集成[11]测试场景中如何使用Docker。

Docker很擅长快速创建和处理一个或多个容器。这个能力显然可以为持续集成测试这个概念提供帮助。在测试场景里，用户需要频繁安装软件，或者部署到多台宿主机上，运行测试，再清理宿主机为下一次运行做准备。

在持续集成环境里，每天要执行好几次安装并分发到宿主机的过程。这为测试生命周期增加了构建和配置开销。打包和安装也消耗了很多时间，而且这个过程很恼人，尤其是需求变化频繁或者需要复杂、耗时的处理步骤进行清理的情况下。

Docker让部署以及这些步骤和宿主机的清理变得开销很低。为了演示这一点，我们将使用Jenkins CI构建一个测试流水线：首先，构建一个运行Docker的Jenkins服务器。为了更有意思些，我们会让Docker递归地运行在Docker内部。这就和套娃一样！

提示 可以在https://github.com/jpetazzo/dind 读到更多关于在Docker中运行Docker的细节。

一旦Jenkins运行起来，将展示最基础的单容器测试运行，最后将展示多容器的测试场景。

提示 除了Jenkins，还有许多其他的持续集成工具，包括Strider[12]和Drone.io[13]这种直接利用Docker 的工具，这些工具都是真正基于Docker的。另外，Jenkins也提供了一个插件，这样就可以不用使用我们将要看到的Docker-in-Docker这种方式了。使用Docker插件可能更简单，但我觉得使用Docker-in-Docker这种方式很有趣。

### 构建Jenkins和Docker服务器

为了提供一个Jenkins服务器，从Dockerfile开始构建一个安装了Jenkins和Docker的Ubuntu 14.04镜像。我们先创建一个jenkins目录，来存放构建所需的所有相关文件：


```sh
$ mkdir jenkins
$ cd jenkins
```

在jenkins目录中，我们从Dockerfile开始：

```docker
FROM ubuntu:14.04
MAINTAINER james@example.com
ENV REFRESHED_AT 2014-06-01
RUN apt-get update -qq && apt-get install -qqy curl apt-transport-https
RUN apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys 58118E89F3A912897C070ADBF76221572C52609D
RUN echo deb https://apt.dockerproject.org/repo ubuntu-trusty main > /etc/apt/sources.list.d/docker.list
RUN apt-get update -qq && apt-get install -qqy iptables cacertificates openjdk-7-jdk git-core docker-engine
ENV JENKINS_HOME /opt/jenkins/data
ENV JENKINS_MIRROR http://mirrors.jenkins-ci.org
RUN mkdir -p $JENKINS_HOME/plugins
RUN curl -sf -o /opt/jenkins/jenkins.war -L $JENKINS_MIRROR/warstable/latest/jenkins.war
RUN for plugin in chucknorris greenballs scm-api git-client git ws-cleanup ;\
do curl -sf -o $JENKINS_HOME/plugins/${plugin}.hpi \
-L $JENKINS_MIRROR/plugins/${plugin}/latest/${plugin}.hpi; done
ADD ./dockerjenkins.sh /usr/local/bin/dockerjenkins.sh
RUN chmod +x /usr/local/bin/dockerjenkins.sh
VOLUME /var/lib/docker
EXPOSE 8080
ENTRYPOINT [ "/usr/local/bin/dockerjenkins.sh" ]
```

说明：

- 首先，它设置了Ubuntu环境，加入了需要的Docker APT仓库，并加入了对应的GPGkey。
- 之后更新了包列表，并安装执行Docker和Jenkins所需要的包。我们使用与第2章相同的指令，加入了一些Jenkins需要的包。
- 然后，我们创建了/opt/jenkins目录，并把最新稳定版本的Jenkins下载到这个目录。还需要一些Jenkins插件，给Jenkins提供额外的功能（比如支持Git版本控制）。
- 我们还使用ENV指令把JENKINS_HOME和JENKINS_MIRROR环境变量设置为Jenkins的数据目录和镜像站点。
- 然后我们指定了VOLUME指令。还记得吧，VOLUME指令从容器运行的宿主机上挂载一个卷。在这里，为了“骗过”Docker，指定/var/lib/docker作为卷。这是因为/var/lib/docker目录是Docker用来存储其容器的目录。这个位置必须是真实的文件系统，而不能是像Docker镜像层那种挂载点。那么，我们使用VOLUME指令告诉Docker进程，在容器运行内部使用宿主机的文件系统作为容器的存储。这样，容器内嵌Docker的/var/lib/docker目录将保存在宿主机系统的/var/lib/docker/volumes目录下的某个位置。
- 我们已经公开了Jenkins默认的8080端口。
- 最后，我们指定了一个要运行的shell脚本（可以在http://dockerbook.com/code/5/jenkins/dockerjenkins.sh 找到）作为容器的启动命令。这个shell脚本（由ENTRYPOINT指令指定）帮助在宿主机上配置 Docker，允许在Docker里运行Docker，开启Docker守护进程，并且启动Jenkins。在 https://github.com/jpetazzo/dind 可以看到更多关于为什么需要一个shell脚本来允许Docker中运行Docker的信息。

我们继续在jenkins目录下工作，刚刚我们在这个目录下创建了Dockerfile文件，现在让我们来获取这个shell脚本：


```sh
$ cd jenkins
$ wget https://raw.githubusercontent.com/jamtur01/dockerbook-code/master/code/5/jenkins/dockerjenkins.sh
$ chmod 0755 dockerjenkins.sh
```

dockerjenkins.sh 内容如下：

```sh
#!/bin/bash

# First, make sure that cgroups are mounted correctly.
CGROUP=/sys/fs/cgroup

[ -d $CGROUP ] ||
  mkdir $CGROUP

mountpoint -q $CGROUP ||
  mount -n -t tmpfs -o uid=0,gid=0,mode=0755 cgroup $CGROUP || {
    echo "Could not make a tmpfs mount. Did you use -privileged?"
    exit 1
  }

# Mount the cgroup hierarchies exactly as they are in the parent system.
for SUBSYS in $(cut -d: -f2 /proc/1/cgroup)
do
  [ -d $CGROUP/$SUBSYS ] || mkdir $CGROUP/$SUBSYS
  mountpoint -q $CGROUP/$SUBSYS ||
    mount -n -t cgroup -o $SUBSYS cgroup $CGROUP/$SUBSYS
done

# Now, close extraneous file descriptors.
pushd /proc/self/fd
for FD in *
do
  case "$FD" in
  # Keep stdin/stdout/stderr
  [012])
    ;;
  # Nuke everything else
  *)
    eval exec "$FD>&-"
    ;;
  esac
done
popd

docker daemon &
exec java -jar /opt/jenkins/jenkins.war
```


已经有了Dockerfile，用docker build命令来构建一个新的镜像，并创建容器：

```sh
$ sudo docker build -t jamtur01/dockerjenkins.
$ sudo docker run -p 8080:8080 --name jenkins --privileged \
-d jamtur01/dockerjenkins
190f5c6333576f017257b3348cf64dfcd370ac10721c1150986ab1db3e3221ff8
$ sudo docker logs jenkins
Running from: /opt/jenkins/jenkins.war
webroot: EnvVars.masterEnvVars.get("JENKINS_HOME")
Sep 8, 2013 12:53:01 AM winstone.Logger logInternal
INFO: Beginning extraction from war file
. . .
INFO: HTTP Listener started: port=8080
. . .
INFO: Jenkins is fully up and running
```

说明：

- 可以看到，这里使用了一个新标志`--privileged`来运行容器。--privileged标志很特别，可以启动Docker的特权模式，这种模式允许我们以其宿主机具有的（几乎）所有能力来运行容器，包括一些内核特性和设备访问。这是让我们可以在Docker中运行Docker必要的魔法。
- 要么不断地检查日志，要么使用-f标志运行docker logs命令。


注意：

- 让Docker运行在--privileged特权模式会有一些安全风险。在这种模式下运行容器对Docker宿主机拥有root访问权限。确保已经对Docker宿主机进行了恰当的安全保护，并且只在确实可信的域里使用特权访问Docker宿主机，或者仅在有类似信任的情况下运行容器。
- 还可以看到，我们使用了-p标志在本地宿主机的8080端口上公开8080端口。一般来说，这不是一种好的做法，不过足以让一台Jenkins服务器运行起来。


太好了。现在Jenkins服务器应该可以通过8080端口在浏览器中访问了：

<p align="center">
    <img width="50%" height="50%" src="http://images.iterate.site/blog/image/20200206/RmybR4aTfdoJ.png?imageslim">
</p>


### 创建新的Jenkins作业

现在Jenkins服务器已经运行，让我们来创建一个Jenkin作业吧。单击create new jobs（创建新作业）链接，打开了创建新作业的向导：

<p align="center">
    <img width="50%" height="50%" src="http://images.iterate.site/blog/image/20200206/DKG59Fvtg1Qq.png?imageslim">
</p>


把新作业命名为Docker_test_job，选择作业类型为Freestyle project，并单击OK继续到下一个页面。

图5-4　创建新的Jenkins作业

现在把这些区域都填好。先填好作业描述，然后单击Advanced Project Options（高级项目选项）下面的Advanced...（高级）按钮，单击Use Custom workspace（使用自定义工作空间）的单选按钮，并指定/tmp/jenkins-buildenv/${JOB_NAME}/ workspace作为Directory（目录）。这个目录是运行Jenkins的工作空间。

在Source Code Management（源代码管理）区域里，选择Git并指定测试仓库https:// github.com/jamtur01/docker-jenkins-sample.git。图5-5所示是一个简单的仓库，它包含了一些基于Ruby的RSpec测试。

<p align="center">
    <img width="50%" height="50%" src="http://images.iterate.site/blog/image/20200206/6ALoVCLsgvdd.png?imageslim">
</p>


图5-5　Jenkins作业细节1

现在往下滚动页面，更新另外一些区域。首先，单击Add Build Step（增加构建步骤）按钮增加一个构建的步骤，选择Execute shell（执行shell脚本）。之后使用定义的脚本来启动测试和Docker，如代码清单5-81所示。

代码清单5-81　用于Jenkins作业的Docker shell脚本

```sh
# 构建用于此作业的镜像
IMAGE=$(docker build . | tail -1 | awk '{ print $NF }')
# 构建挂载到Docker的目录
MNT="$WORKSPACE/.."
# 在Docker里执行编译测试
CONTAINER=$(docker run -d -v "$MNT:/opt/project" $IMAGE /bin/bash -c 'cd /opt/project/workspace && rake spec')
# 进入容器，这样可以看到输出的内容
docker attach $CONTAINER
# 等待程序退出，得到返回码
RC=$(docker wait $CONTAINER)
# 删除刚刚用到的容器
docker rm $CONTAINER
# 使用刚才的返回码退出整个脚本
exit $RC
```


这个脚本都做了什么呢？首先，它将使用包含刚刚指定的Git仓库的Dockerfile创建一个新的Docker镜像。这个Dockerfile提供了想要执行的测试环境。让我们来看

一下这个Dockerfile，如代码清单5-82所示。

代码清单5-82　用于测试作业的Dockerfile

```docker
FROM ubuntu:14.04
MAINTAINER James Turnbull "james@example.com"
ENV REFRESHED_AT 2014-06-01
RUN apt-get update
RUN apt-get -y install ruby rake
RUN gem install --no-rdoc --no-ri rspec ci_reporter_rspec
```


提示 如果用户的测试依赖或者需要别的包，只需要根据新的需求更新Dockerfile，然后在运行测试时会重新构建镜像。

可以看到，Dockerfile构建了一个Ubuntu宿主机，安装了Ruby和RubyGems，之后安装了两个gem：rspec和ci_reporter_rspec。这样构建的镜像可以用于测试典型的基于Ruby且使用RSpec测试框架的应用程序。ci_reporter_rspec gem会把RSpec的输出转换为JUnit格式的XML输出，并交给Jenkins做解析。一会儿就能看到这个转换的结果。

回到之前的脚本。从Dockerfile构建镜像。接下来，创建一个包含Jenkins工作空间（就是签出Git仓库的地方）的目录，会把这个目录挂载到Docker容器，并在这个目录里执行测试。

然后，我们从这个镜像创建了容器，并且运行了测试。在容器里，把工作空间挂载到/opt/project目录。之后执行命令切换到这个目录，并执行rake spec来运行RSpec测试。

现在容器启动了，我们拿到了容器的ID。

提示 Docker在启动容器时支持--cidfile选项，这个选项会让Docker截获容器ID并将其存到--cidfile选项指定的文件里，如--cidfile=/tmp/containerid.txt。

现在使用docker attach命令进入容器，得到容器执行时输出的内容，然后使用docker wait命令。docker wait命令会一直阻塞，直到容器里的命令执行完成才会返回容器退出时的返回码。变量RC捕捉到容器退出时的返回码。

最后，清理环境，删除刚刚创建的容器，并使用容器的返回码退出。这个返回码应该就是测试执行结果的返回码。Jenkins依赖这个返回码得知作业的测试结果是成功还是失败。

接下来，单击Add post-build action（加入构建后的动作），加入一个Publish JUint test result report（公布JUint测试结果报告）的动作。在Test report XML``s（测试报告的XML文件）域，需要指定spec/reports/*.xml。这个目录是ci_reporter gem的XML输出的位置，找到这个目录会让Jenkins处理测试的历史结果和输出结果。

最后，必须单击Save按钮保存新的作业，如图5-6所示。



<p align="center">
    <img width="50%" height="50%" src="http://images.iterate.site/blog/image/20200206/4zbhw55Kqlkh.png?imageslim">
</p>


图5-6　Jenkins作业细节2

### 运行Jenkins作业

现在我们来运行Jenkins作业。单击Build Now（现在构建）按钮，就会看到有个作业出现在Build History（构建历史）方框里，如图5-7所示。


<p align="center">
    <img width="50%" height="50%" src="http://images.iterate.site/blog/image/20200206/PucnWpNKn8jX.png?imageslim">
</p>

图5-7　运行Jenkins作业

注意 第一次运行测试时，可能会因为构建新的镜像而等待较长一段时间。但是，下次运行测试时，因为Docker已经准备好了镜像，执行速度就会比第一次快多了。

单击这个作业，看看正在执行的测试运行的详细信息，如图5-8所示。

<p align="center">
    <img width="50%" height="50%" src="http://images.iterate.site/blog/image/20200206/gTzWLh9vVn7s.png?imageslim">
</p>

图5-8　Jenkins作业的细节

单击Console Output（控制台输出），查看测试作业已经执行的命令，如图5-9所示。

<p align="center">
    <img width="50%" height="50%" src="http://images.iterate.site/blog/image/20200206/pljXcz5H20B1.png?imageslim">
</p>

图5-9　Jenkins作业的控制台输出

可以看到，Jenkins正在将Git仓库下载到工作空间。然后会执行shell脚本并使用docker build构建Docker镜像。然后我们捕获镜像的ID并用docker run创建一个新容器。正在运行的这个新容器内会执行RSpec测试并且捕获测试结果和返回码。如果这个作业使用返回码0退出，这个作业就会被标识为测试成功。


单击Test Result（测试结果）链接，可以看到详细的测试报告。这个报告是从测试的RSpec结果转换为JUnit格式后得到的。这个转换由ci_reporter gem完成，并在“构建后的步骤”里被捕获。

### 与Jenkins作业有关的下一步

可以通过启用SCM轮询，让Jenkins作业自动执行。它会在有新的改动签入Git仓库后，触发自动构建。类似的自动化还可以通过提交后的钩子或者GitHub或者Bitbucket仓库的钩子来完成。

### Jenkins设置小结

到现在为止，我们已经做了不少事情：安装并运行了Jenkins，创建了第一个作业。这个Jenkins作业使用Docker创建了一个镜像，而这个镜像使用仓库里的Dockerfile管理和更新。这种情况下，不但架构配置和代码可以同步更新，管理配置的过程也变得很简单。然后我们通过镜像创建了运行测试的容器。测试完成后，可以丢弃这个容器。整个测试过程轻量且快速。将这个例子适配到其他不同的测试平台或者其他语言的测试框架也很容易。

提示 也可以使用参数化构建[16]来让作业和shell脚本更加通用，方便应用到更多框架和语言。

## 多配置的Jenkins

之前我们已经见过使用Jenkins构建的简单的单个容器。如果要测试的应用依赖多

个平台怎么办？假设要在Ubuntu、Debian和CentOS上测试这个程序。要在多平台

测试，可以利用Jenkins里叫“多配置作业”的作业类型的特性。多配置作业允许

运行一系列的测试作业。当Jenkins多配置作业运行时，会运行多个配置不同的子

作业。

5.4.1　创建多配置作业

现在来创建一个新的多配置作业。从Jenkins控制台里单击New Item（新项目），

将新作业命名为Docker_matrix_job，选择Multi-configuration project（创

建多配置项目），并单击OK按钮，如图5-10所示。

图5-10　创建多配置作业

这个页面与之前看到的创建作业时的页面非常类似。给作业加上描述，选择Git作为

仓库类型，并指定之前那个示例应用的仓

库：https://github.com/jamtur01/docker-jenkins-sample. git。具体如图



5-11所示。

图5-11　设置多配置作业1

接下来，向下滚动，开始设置多配置的维度（axis）。维度是指作为作业的一部分

执行的一系列元素。单击Add Axis（添加维度）按钮，并选择User-defined

Axis（用户自定义维度）。指定这个维度的名字为OS（OS是Operating System的

缩写），并设置3个值，即centos、debian和ubuntu。当执行多配置作业时，

Jenkins会查找这个维度，并生成3个作业：维度上的每个值对应一个作业。

还要注意，在Build Environment（构建环境）部分我们单击了Delete

workspace before build starts（构建前删除工作空间）。这个选项会在一系

列新作业初始化之前，通过删除已经签出的仓库，清理构建环境。具体如图5-12所

示。



图5-12　设置多配置作业2

最后，我们通过一个简单的shell脚本指定了另一个shell构建步骤。这个脚本是在

之前使用的shell脚本的基础上修改而成的，如代码清单5-83所示。

代码清单5-83　Jenkins多配置shell脚本

# 构建此次运行需要的镜像

cd $OS && IMAGE=$(docker build . | tail -1 | awk '{ print $NF }')

# 构建挂载到Docker的目录

MNT="$WORKSPACE/.."

# 在Docker内执行构建过程

CONTAINER=$(docker run -d -v "$MNT:/opt/project" $IMAGE /bin/bash

-c "cd/opt/project/$OS && rake spec")

# 进入容器，以便可以看到输出的内容

docker attach $CONTAINER

# 进程退出后，得到返回值

RC=$(docker wait $CONTAINER)

# 删除刚刚使用的容器

docker rm $CONTAINER

# 使用刚才的返回值退出脚本

exit $RC

来看看这个脚本有哪些改动：每次执行作业都会进入不同的以操作系统为名的目

录。在我们的测试仓库里有3个目录：centos、debian和ubuntu。每个目录里的

Dockerfile都不同，分别包含构建CentOS、Debian和Ubuntu镜像的指令。这意味

着每个被启动的作业都会进入对应的目录，构建对应的操作系统的镜像，安装相应

的环境需求，并启动基于这个镜像的容器，最后在容器里运行测试。

我们来看这些新的Dockerfile中的一个，如代码清单5-84所示。

代码清单5-84　基于CentOS的Dockerfile

FROM centos:latest

MAINTAINER James Turnbull "james@example.com"

ENV REFRESHED_AT 2014-06-01

RUN yum -y install ruby rubygems rubygem-rake

RUN gem install --no-rdoc --no-ri rspec ci_reporter_rspec

这是一个基于以前的作业针对CentOS修改过的Dockerfile。这个Dockerfile和之

前做的事情一样，只是改为使用适合CentOS的命令，比如使用yum来安装包。

加入一个构建后的动作Publish JUnit test result report（发布JUnit测试结

果）并指定XML输出的位置为spec/reports/*.xml。这样可以检查测试输出的结

果。

最后，单击Save来创建新作业，并保存配置。

现在可以看到刚刚创建的作业，并且注意到这个作业包含一个叫





作Configurations（配置）的区域，包含了该作业的各维度上每个元素的子作业，

如图5-13所示。

图5-13　Jenkins多配置作业

5.4.2　测试多配置作业

现在我们来测试这个新作业。单击Build Now按钮启动多配置作业。当Jenkins开始

运行时，会先创建一个主作业。之后，这个主作业会创建3个子作业。每个子作业会

使用选定的3个平台中的一个来执行测试。

注意 和之前的作业一样，第一次运行作业时也需要一些时间来构建测试所需的镜像。一旦镜像构建好后，下

一次运行就会快很多。Docker只会在更新了Dockerfile之后修改镜像。

可以看到，主作业会先执行，然后执行每个子作业。其中新的centos子作业的输出

如图5-14所示。





图5-14　centos子作业

可以看到，centos作业已经执行了：绿球图标表示这个测试执行成功。可以更深入

地看一下执行细节。单击Build History里第一个条，如图5-15所示。

图5-15　centos子作业细节

在这里可以看到更多centos作业的执行细节。可以看到这个作业Started by

upstream project Docker_matrix_job，构建编号为1。要看执行时的精确细

节，可以单击Console Output链接来查看控制台的输出内容，具体如图5-16所

示。





图5-16　centos子作业的控制台输出

可以看到，这个作业复制了仓库，构建了需要的Docker镜像，从镜像启动了容器，

最后运行了所有的测试。所有测试都成功了（如果有需要，可以单击Test Result

链接来检查测试上传的JUnit结果）。

现在这个简单又强大的多平台测试应用程序的例子就成功演示完了。

5.4.3　Jenkins多配置作业小结

这些例子展示了在Jenkins持续集成中使用Docker的简单实现。读者可以对这些例

子进行扩展，加入从自动化触发构建到包含多平台、多架构、多版本的多级作业矩

阵等功能。那个简单的构建脚本也可以写得更加严谨，或者支持执行多个容器（比

如，为网页、数据库和应用程序层提供分离的容器，以模拟更加真实的多层生产环

境）。

本书由「ePUBw.COM」整理，ePUBw.COM 提供最新最全的优质

电子书下载！！！





5.5　其他选择

在Docker的生态环境中，持续集成和持续部署（CI/CD）是很有意思的一部分。除

了与现有的Jenkins这种工具集成，也有很多人直接使用Docker来构建这类工具。

5.5.1　Drone

Drone是著名的基于Docker开发的CI/CD工具之一。它是一个SaaS持续集成平台，

可以与GitHub、Bitbucket和Google Code仓库关联，支持各种语言，包括

Python、Node.js、Ruby、Go等。Drone在一个Docker容器内运行加到其中的仓库

的测试集。

5.5.2　Shippable

Shippable是免费的持续集成和部署服务，基于GitHub和Bitbucket。它非常快，

也很轻量，原生支持Docker。

本书由「ePUBw.COM」整理，ePUBw.COM 提供最新最全的优质

电子书下载！！！





5.6　小结

本章演示了如何在开发和测试流程中使用Docker。我们看到如何在本地工作站或者

虚拟机里以一个开发者为中心使用Docker做测试，也探讨了如何使用Jenkins CI这

种持续集成工具配合Docker进行可扩展的测试。我们已经了解了如何使用Docker构

建单功能的测试，以及如何构建分布式矩阵作业。

下一章我们将开始了解如何使用Docker在生产环境中提供容器化、可堆叠、可扩展的弹性服务。
