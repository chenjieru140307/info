




第6章　使用Docker构建服务

第5章介绍了如何利用Docker来使用容器在本地开发工作流和持续集成环境中方便

快捷地进行测试。本章继续探索如何利用Docker来运行生产环境的服务。

本章首先会构建简单的应用，然后会构建一个更复杂的多容器应用。这些应用会展

示，如何利用链接和卷之类的Docker特性来组合并管理运行于Docker中的应用。

本书由「ePUBw.COM」整理，ePUBw.COM 提供最新最全的优质

电子书下载！！！





6.1　构建第一个应用

要构建的第一个应用是使用Jekyll 框架[1]的自定义网站。我们会构建以下两个镜

像。

一个镜像安装了Jekyll及其他用于构建Jekyll网站的必要的软件包。

一个镜像通过Apache来让Jekyll网站工作起来。

我们打算在启动容器时，通过创建一个新的Jekyll网站来实现自服务。工作流程如

下。

创建Jekyll基础镜像和Apache镜像（只需要构建一次）。

从Jekyll镜像创建一个容器，这个容器存放通过卷挂载的网站源代码。

从Apache镜像创建一个容器，这个容器利用包含编译后的网站的卷，并为其服

务。

在网站需要更新时，清理并重复上面的步骤。

可以把这个例子看作是创建一个多主机站点最简单的方法。实现很简单，本章后半

部分会以这个例子为基础做更多扩展。

6.1.1　Jekyll基础镜像

让我们开始为第一个镜像（Jekyll基础镜像）创建Dockerfile。我们先创建一个新

目录和一个空的Dockerfile，如代码清单6-1所示。

代码清单6-1　创建Jekyll Dockerfile





$ mkdir jekyll

$ cd jekyll

$ vi Dockerfile

现在我们来看看Dockerfile文件的内容，如代码清单6-2所示。

代码清单6-2　Jekyll Dockerfile

FROM ubuntu:14.04

MAINTAINER James Turnbull <james@example.com>

ENV REFRESHED_AT 2014-06-01

RUN apt-get -yqq update

RUN apt-get -yqq install ruby ruby-dev make nodejs

RUN gem install --no-rdoc --no-ri jekyll –v 2.5.3

VOLUME /data

VOLUME /var/www/html

WORKDIR /data

ENTRYPOINT [ "jekyll", "build", "--destination=/var/www/html" ]

这个Dockerfile使用了第3章里的模板作为基础。镜像基于Ubuntu 14.04，并且安

装了Ruby和用于支持Jekyll的包。然后我们使用VOLUME指令创建了以下两个卷。

/data/，用来存放网站的源代码。

/var/www/html/，用来存放编译后的Jekyll网站码。

然后我们需要将工作目录设置到/data/，并通过ENTRYPOINT指令指定自动构建的命

令，这个命令会将工作目录/data/中的所有的Jekyll网站代码构建

到/var/www/html/目录中。

6.1.2　构建Jekyll基础镜像





通过这个Dockerfile，可以使用docker build命令构建出可以启动容器的镜像，

如代码清单6-3所示。

代码清单6-3　构建Jekyll镜像

$ sudo docker build -t jamtur01/jekyll .

Sending build context to Docker daemon　2.56 kB

Sending build context to Docker daemon

Step 0 : FROM ubuntu:14.04

---> 99ec81b80c55

Step 1 : MAINTAINER James Turnbull <james@example.com>

...

Step 7 : ENTRYPOINT [ "jekyll", "build" "--destination=/var/www/html" ]

---> Running in 542e2de2029d

---> 79009691f408

Removing intermediate container 542e2de2029d

Successfully built 79009691f408

这样就构建了名为jamtur01/jekyll、ID为79009691f408的新镜像。这就是将要

使用的新的Jekyll镜像。可以使用docker images命令来查看这个新镜像，如代码

清单6-4所示。

代码清单6-4　查看新的Jekyll基础镜像

$ sudo docker images

REPOSITORY　　　 TAG　　ID　　　　　　　CREATED　　　　SIZE

jamtur01/jekyll latest 79009691f408　6 seconds ago　12.29 kB (virtual 671 MB)

...

6.1.3　Apache镜像

接下来，我们来构建第二个镜像，一个用来架构新网站的Apache服务器。我们先创

建一个新目录和一个空的Dockerfile，如代码清单6-5所示。

代码清单6-5　创建Apache Dockerfile

$ mkdir apache

$ cd apache

$ vi Dockerfile

现在我们来看看这个Dockerfile的内容，如代码清单6-6所示。

代码清单6-6　Jekyll Apache的Dockerfile

FROM ubuntu:14.04

MAINTAINER James Turnbull <james@example.com>

ENV REFRESHED_AT 2014-06-01

RUN apt-get -yqq update

RUN apt-get -yqq install apache2

VOLUME [ "/var/www/html" ]

WORKDIR /var/www/html

ENV APACHE_RUN_USER www-data

ENV APACHE_RUN_GROUP www-data

ENV APACHE_LOG_DIR /var/log/apache2

ENV APACHE_PID_FILE /var/run/apache2.pid

ENV APACHE_RUN_DIR /var/run/apache2

ENV APACHE_LOCK_DIR /var/lock/apache2

RUN mkdir -p $APACHE_RUN_DIR $APACHE_LOCK_DIR $APACHE_LOG_DIR

EXPOSE 80

ENTRYPOINT [ "/usr/sbin/apache2" ]

CMD ["-D", "FOREGROUND"]

这个镜像也是基于Ubuntu 14.04的，并安装了Apache。然后我们使用VOLUME指令





创建了一个卷，即/var/www/html/，用来存放编译后的Jekyll网站。然后

将/var/www/html设为工作目录。

然后我们使用ENV指令设置了一些必要的环境变量，创建了必要的目录，并且使

用EXPOSE公开了80端口。最后指定了ENTRYPOINT和CMD指令组合来在容器启动时默

认运行Apache。

6.1.4　构建Jekyll Apache镜像

有了这个Dockerfile，可以使用docker build命令来构建可以启动容器的镜像，

如代码清单6-7所示。

代码清单6-7　构建Jekyll Apache镜像

$ sudo docker build -t jamtur01/apache .

Sending build context to Docker daemon　2.56 kB

Sending build context to Docker daemon

Step 0 : FROM ubuntu:14.04

---> 99ec81b80c55

Step 1 : MAINTAINER James Turnbull <james@example.com>

---> Using cache

---> c444e8ee0058

. . .

Step 11 : CMD ["-D", "FOREGROUND"]

---> Running in 7aa5c127b41e

---> fc8e9135212d

Removing intermediate container 7aa5c127b41e

Successfully built fc8e9135212d

这样就构建了名为jamtur01/apache、ID为fc8e9135212d的新镜像。这就是将要





使用的Apache镜像。可以使用docker images命令来查看这个新镜像，如代码清单

6-8所示。

代码清单6-8　查看新的Jekyll Apache镜像

$ sudo docker images

REPOSITORY　　　 TAG　　ID　　　　　　　CREATED　　　　SIZE

jamtur01/apache latest fc8e9135212d　6 seconds ago　12.29 kB (virtual 671 MB)

...

6.1.5　启动Jekyll网站

现在有了以下两个镜像。

Jekyll：安装了Ruby及其他必备软件包的Jekyll镜像。

Apache：通过Apache Web服务器来让Jekyll网站工作起来的镜像。

我们从使用docker run命令来创建一个新的Jekyll容器开始我们的网站。我们将启

动容器，并构建我们的网站。

然后我们需要一些我的博客的源代码。先把示例Jekyll博客复制到$HOME目录（在

这个例子里是/home/james）中，如代码清单6-9所示。

代码清单6-9　创建示例Jekyll博客

$ cd $HOME

$ git clone https://github.com/jamtur01/james_blog.git

在这个目录下可以看到一个启用了Twitter Bootstrap[2]的最基础的Jekyll博

客。如果你也想使用这个博客，可以修改_config.yml文件和主题，以符合你的要

求。

现在在Jekyll容器里使用这个示例数据，如代码清单6-10所示。

代码清单6-10　创建Jekyll容器

$ sudo docker run -v /home/james/james_blog:/data/ \

--name james_blog jamtur01/jekyll

Configuration file: /data/_config.yml

　　　　　　　Source: /data

　　　　Destination: /var/www/html

　　　Generating...

　　　　　　　　　　　 done.

我们启动了一个叫作james_blog的新容器，把本地的james_blog目录作为/data/

卷挂载到容器里。容器已经拿到网站的源代码，并将其构建到已编译的网站，存放

到/var/www/html/目录。

卷是在一个或多个容器中特殊指定的目录，卷会绕过联合文件系统，为持久化数据

和共享数据提供几个有用的特性。

卷可以在容器间共享和重用。

共享卷时不一定要运行相应的容器。

对卷的修改会直接在卷上反映出来。

更新镜像时不会包含对卷的修改。

卷会一直存在，直到没有容器使用它们。

利用卷，可以在不用提交镜像修改的情况下，向镜像里加入数据（如源代码、数据

或者其他内容），并且可以在容器间共享这些数据。

卷在Docker宿主机的/var/lib/docker/volumes目录中。可以通过docker inspect命令查看某个卷的具体位置，如docker inspect -f "{{ range

.Mounts }}`` ``{{.}}{{end}}"。

提示 在Docker 1.9中，卷功能已经得到扩展，能通过插件的方式支持第三方存储系统，如Ceph、Flocker和

EMC等。可以在卷插件文档[3]和docker volume create命令文档[4]中获得更详细的解释。

所以，如果想在另一个容器里使用/var/www/html/卷里编译好的网站，可以创建一

个新的链接到这个卷的容器，如代码清单6-11所示。

代码清单6-11　创建Apache容器

$ sudo docker run -d -P --volumes-from james_blog jamtur01/apache 09a570cc2267019352525079fbba9927806f782acb88213bd38dde7e2795407d 这看上去和典型的docker run很像，只是使用了一个新标志--volumes-from。标

志--volumes-from把指定容器里的所有卷都加入新创建的容器里。这意味着，

Apache容器可以访问之前创建的james_blog容器里/var/www/html卷中存放的编

译后的Jekyll网站。即便james_blog容器没有运行，Apache容器也可以访问这个

卷。想想，这只是卷的特性之一。不过，容器本身必须存在。

注意 即使删除了使用了卷的最后一个容器，卷中的数据也会持久保存。

构建Jekyll网站的最后一步是什么？来查看一下容器把已公开的80端口映射到了哪

个端口，如代码清单6-12所示。

代码清单6-12　解析Apache容器的端口

$ sudo docker port 09a570cc2267 80

0.0.0.0:49160





现在在Docker宿主机上浏览该网站，如图6-1所示。

图6-1　Jekyll网站

现在终于把Jekyll网站运行起来了！

6.1.6　更新Jekyll网站

如果要更新网站的数据，就更有意思了。假设要修改Jekyll网站。我们将通过编辑

james_blog/_config.yml文件来修改博客的名字，如代码清单6-13所示。

代码清单6-13　编辑Jekyll博客

$ vi james_blog/_config.yml

并将title域改为James' Dynamic Docker-driven Blog。

那么如何才能更新博客网站呢？只需要再次使用docker start命令启动Docker容

器即可，如代码清单6-14所示。

代码清单6-14　再次启动james_blog容器



$ sudo docker start james_blog

james_blog

看上去什么都没发生。我们来查看一下容器的日志，如代码清单6-15所示。

代码清单6-15　查看james_blog容器的日志

$ sudo docker logs　 james_blog

Configuration file:　/data/_config.yml

　　　　　　　 Source:　/data

　　　　 Destination:　/var/www/html

　　　 Generating...

　　　　　　　　　 　done.

Configuration file:　/data/_config.yml

　　　　　　　Source:　/data

　　　　Destination:　/var/www/html

　　　 Generating...

　　　　　　　　　　　　 done.

可以看到，Jekyll编译过程第二次被运行，并且网站已经被更新。这次更新已经写

入了对应的卷。现在浏览Jekyll网站，就能看到变化了，如图6-2所示。





图6-2　更新后的Jekyll网站

由于共享的卷会自动更新，这一切都不需要更新或者重启Apache容器。这个流程非

常简单，可以将其扩展到更复杂的部署环境。

6.1.7　备份Jekyll卷

你可能会担心万一不小心删除卷（尽管能使用已有的步骤轻松重建这个卷）。由于

卷的优点之一就是可以挂载到任意容器，因此可以轻松备份它们。现在创建一个新

容器，用来备份/var/www/html卷，如代码清单6-16所示。

代码清单6-16　备份/var/www/html卷

$ sudo docker run --rm --volumes-from james_blog \

-v $(pwd):/backup ubuntu \

tar cvf /backup/james_blog_backup.tar /var/www/html

tar: Removing leading '/' from member names

/var/www/html/

/var/www/html/assets/

/var/www/html/assets/themes/

. . .

$ ls james_blog_backup.tar

james_blog_backup.tar

这里我们运行了一个已有的Ubuntu容器，并把james_blog的卷挂载到该容器里。这

会在该容器里创建/var/www/html目录。然后我们使用-v标志把当前目录（通过

$(pwd)命令获得）挂载到容器的/backup目录。最后我们的容器运行这一备份命

令，如代码清单6-17所示。

提示 我们还指定了--rm标志，这个标志对于只用一次的容器，或者说用完即扔的容器，很有用。这个标志会





在容器的进程运行完毕后，自动删除容器。对于只用一次的容器来说，这是一种很方便的清理方法。

代码清单6-17　备份命令

tar cvf /backup/james_blog_backup.tar /var/www/html

这个命令会创建一个名为jams_blog_backup.tar的tar文件（该文件包括

了/var/www/html目录里的所有内容），然后退出。这个过程创建了卷的备份。

这显然只是一个最简单的备份过程。用户可以扩展这个命令，备份到本地存储或者

云端（如Amazon S3[5]或者更传统的类似Amanda[6]的备份软件）。

提示 这个例子对卷中存储的数据库或者其他类似的数据也适用。只要简单地把卷挂载到新容器，完成备份，

然后废弃这个用于备份的容器就可以了。

6.1.8　扩展Jekyll示例网站

下面是几种扩展Jekyll网站的方法。

运行多个Apache容器，这些容器都使用来自james_blog容器的卷。在这些

Apache容器前面加一个负载均衡器，我们就拥有了一个Web集群。

进一步构建一个镜像，这个镜像把用户提供的源数据复制（如通过git

clone）到卷里。再把这个卷挂载到从jamtur01/jekyll镜像创建的容器。这

就是一个可迁移的通用方案，而且不需要宿主机本地包含任何源代码。

在上一个扩展基础上可以很容易为我们的服务构建一个Web前端，这个服务用于

从指定的源自动构建和部署网站。这样用户就有一个完全属于自己的GitHub

Pages了。

本书由「ePUBw.COM」整理，ePUBw.COM 提供最新最全的优质

电子书下载！！！





6.2　使用Docker构建一个Java应用服务

现在我们来试一些稍微不同的方法，考虑把Docker作为应用服务器和编译管道。这

次做一个更加“企业化”且用于传统工作负载的服务：获取Tomcat服务器上的WAR

文件，并运行一个Java应用程序。为了做到这一点，构建一个有两个步骤的Docker

管道。

一个镜像从URL拉取指定的WAR文件并将其保存到卷里。

一个含有Tomcat服务器的镜像运行这些下载的WAR文件。

6.2.1　WAR文件的获取程序

我们从构建一个镜像开始，这个镜像会下载WAR文件并将其挂载在卷里，如代码清单

6-18所示。

代码清单6-18　创建获取程序（fetcher）的Dockerfile

$ mkdir fetcher

$ cd fetcher

$ touch Dockerfile

现在我们来看看这个Dockerfile的内容，如代码清单6-19所示。

代码清单6-19　WAR文件的获取程序

FROM ubuntu:14.04

MAINTAINER James Turnbull <james@example.com>

ENV REFRESHED_AT 2014-06-01

RUN apt-get -yqq update





RUN apt-get -yqq install wget

VOLUME [ "/var/lib/tomcat7/webapps/" ]

WORKDIR /var/lib/tomcat7/webapps/

ENTRYPOINT [ "wget" ]

CMD [ "-?" ]

这个非常简单的镜像只做了一件事：容器执行时，使用wget从指定的URL获取文件

并把文件保存在/var/lib/tomcat7/webapps/目录。这个目录也是一个卷，并且是

所有容器的工作目录。然后我们会把这个卷共享给Tomcat服务器并且运行里面的内

容。

最后，如果没有指定URL，ENTRYPOINT和CMD指令会让容器运行，在容器不带URL运

行的时候，这两条指令通过返回wget帮助来做到这一点。

现在我们来构建这个镜像，如代码清单6-20所示。

代码清单6-20　构建获取程序的镜像

$ sudo docker build -t jamtur01/fetcher .

6.2.2　获取WAR文件

现在让我们获取一个示例文件来启动新镜像。从https://tomcat.

apache.org/tomcat-7.0-doc/ appdev/sample/下载Apache Tomcat示例应用，

如代码清单6-21所示。

代码清单6-21　获取WAR文件

$ sudo docker run -t -i --name sample jamtur01/fetcher \

https://tomcat.apache.org/tomcat-7.0-doc/appdev/sample/sample.war

--2014-06-21 06:05:19--　https://tomcat.apache.org/tomcat-7.0-doc/appdev/

　sample/sample.war

Resolving tomcat.apache.org (tomcat.apache.org)...

　140.211.11.131, 192.87.106.229, 2001:610:1:80bc:192:87:106:229

Connecting to tomcat.apache.org (tomcat.apache.org)

　|140.211.11.131|:443...connected.

HTTP request sent, awaiting response... 200 OK

Length: 4606 (4.5K)

Saving to: 'sample.war'

100%[=================================>] 4,606　　　 --.-K/s　 in 0s 2014-06-21 06:05:19 (14.4 MB/s) - 'sample.war' saved [4606/4606]

可以看到，容器通过提供的URL下载了sample.war文件。从输出结果看不出最终的

保存路径，但是因为设置了容器的工作目录，sample.war文件最终会保存

到/var/lib/tomcat7/webapps/目录中。

可以在/var/lib/docker目录找到这个WAR文件。我们先用docker inspect命令查

找卷的存储位置，如代码清单6-22所示。

代码清单6-22　查看示例里的卷

$ sudo docker inspect -f "{{ range .Mounts }}{{.}}{{end}}" sample

{c20a0567145677ed46938825f285402566e821462632e1842e82bc51b47fe4dc

　　/var/lib/docker/volumes/

　c20a0567145677ed46938825f285402566e821462632e1842e82bc51b47fe4dc

　　/_data /var/lib/tomcat7/webapps local true}

然后我们可以查看这个目录，如代码清单6-23所示。

代码清单6-23　查看卷所在的目录





$ ls -l /var/lib/docker/volumes/

　c20a0567145677ed46938825f285402566e821462632e1842e82bc51b47fe4dc

　/_data

total 8

-rw-r--r-- 1 root root 4606 Mar 31 2012 sample.war

6.2.3　Tomecat7应用服务器

现在我们已经有了一个可以获取WAR文件的镜像，并已经将示例WAR文件下载到了容

器中。接下来我们构建Tomcat应用服务器的镜像来运行这个WAR文件，如代码清单

6-24所示。

代码清单6-24　创建Tomcat 7 Dockerfile

$ mkdir tomcat7

$ cd tomcat7

$ touch Dockerfile

现在我们来看看这个Dockerfile，如代码清单6-25所示。

代码清单6-25　Tomcat 7应用服务器

FROM ubuntu:14.04

MAINTAINER James Turnbull <james@example.com>

ENV REFRESHED_AT 2014-06-01

RUN apt-get -yqq update

RUN apt-get -yqq install tomcat7 default-jdk

ENV CATALINA_HOME /usr/share/tomcat7

ENV CATALINA_BASE /var/lib/tomcat7

ENV CATALINA_PID /var/run/tomcat7.pid

ENV CATALINA_SH /usr/share/tomcat7/bin/catalina.sh





ENV CATALINA_TMPDIR /tmp/tomcat7-tomcat7-tmp

RUN mkdir -p $CATALINA_TMPDIR

VOLUME [ "/var/lib/tomcat7/webapps/" ]

EXPOSE 8080

ENTRYPOINT [ "/usr/share/tomcat7/bin/catalina.sh", "run" ]

这个镜像很简单。我们需要安装Java JDK和Tomcat服务器。我们首先指定一些启动

Tomcat需要的环境变量，然后我们创建一个临时目录，还创建

了/var/lib/tomcat7/`` ``webapps/卷，公开了Tomcat默认的8080端口，最后

使用ENTRYPOINT指令来启动Tomcat。

现在我们来构建Tomcat 7镜像，如代码清单6-26所示。

代码清单6-26　构建Tomcat 7镜像

$ sudo docker build -t jamtur01/tomcat7 .

6.2.4　运行WAR文件

现在，让我们创建一个新的Tomcat实例，运行示例应用，如代码清单6-27所示。

代码清单6-27　创建第一个Tomcat实例

$ sudo docker run --name sample_app --volumes-from sample \

-d -P jamtur01/tomcat7

这会创建一个名为sample_app的容器，这个容器会复用sample容器里的卷。这意味

着存储在/var/lib/tomcat7/webapps/卷里的WAR文件会从sample容器挂载

到sample_app容器，最终被Tomcat加载并执行。





让我们在Web浏览器里看看这个示例程序。首先，我们必须使用docker port命令找

出被公开的端口，如代码清单6-28所示。

代码清单6-28　查找Tomcat应用的端口

sudo docker port sample_app 8080

0.0.0.0:49154

现在我们来浏览这个应用（使用URL和端口，并在最后加上/sample）看看都有什

么，如图6-3所示。

图6-3　我们的Tomcat示例应用

应该能看到正在运行的Tomcat应用。

6.2.5　基于Tomcat应用服务器的构建服务

现在有了自服务Web服务的基础模块，让我们来看看怎么基于这些基础模块做扩展。

为了做到这一点，我们已经构建好了一个简单的基于Sinatra的Web应用，这个应用

可以通过网页自动展示Tomcat应用。这个应用叫TProv。可以在本书官网[7]或者

GitHub[8]找到其源代码。

然后我们使用这个程序来演示如何扩展之前的示例。首先，要保证已经安装了

Ruby，如代码清单6-29所示。TProv应用会直接安装在Docker宿主机上，因为这个

应用会直接和Docker守护进程交互。这也正是要安装Ruby的地方。

注意 也可以把TProv应用安装在Docker容器里。

代码清单6-29　安装Ruby

$ sudo apt-get -qqy install ruby make ruby-dev

然后可以通过Ruby gem安装这个应用，如代码清单6-30所示。

代码清单6-30　安装TProv应用

$ sudo gem install --no-rdoc --no-ri tprov

. . .

Successfully installed tprov-0.0.4

这个命令会安装TProv应用及相关的支撑gem。

然后可以使用tprov命令来启动应用，如代码清单6-31所示。

代码清单6-31　启动TProv应用

$ sudo tprov

[2014-06-21 16:17:24] INFO　WEBrick 1.3.1

[2014-06-21 16:17:24] INFO　ruby 1.8.7 (2011-06-30) [x86_64-linux]

== Sinatra/1.4.5 has taken the stage on 4567 for development with backup from WEBrick

[2014-06-21 16:17:24] INFO　WEBrick::HTTPServer_#start_: _pid=14209 port=4567_

这个命令会启动应用。现在我们可以在Docker宿主机上通过端口4567浏览TProv网

站，如图6-4所示。





图6-4　TProv网络应用

如我们所见，我们可以指定Tomcat应用的名字和指向Tomcat WAR文件的URL。从

https://gwt-examples.googlecode.com/files/Calendar.war下载示例日历应

用程序，并将其称为Calendar，如图6-5所示。

图6-5　下载示例应用程序

单击Submit按钮下载WAR文件，将其放入卷里，运行Tomcat服务器，加载卷里的



WAR文件。可以点击List instances（展示实例）链接来查看实例的运行状态，如

图6-6所示。

图6-6　展示Tomcat实例

这展示了：

容器的ID；

容器的内部IP地址；

服务映射到的接口和端口。

利用这些信息我们可以通过浏览映射的端口来查看应用的运行状态，还可以使

用Delete?（是否删除）单选框来删除正在运行的实例。

可以查看TProv应用的源代码[9]，看看程序是如何实现这些功能的。这个应用很简

单，只是通过shell执行docker程序，再捕获输出，来运行或者删除容器。

可以随意使用TProv代码，在之上做扩展，或者干脆重新写一份自己的代码[10]。本

文的应用主要用于展示，使用Docker构建一个应用程序部署管道是很容易的事情。

警告 TProv应用确实太简单了，缺少某些错误处理和测试。这个应用的开发过程很快：只写了一个小时，用

于展示在构建应用和服务时Docker是一个多么强大的工具。如果你在这个应用里找到了bug（或者想把它写得

更好），可以通过在https://github.com/jamtur01/docker book-code提交issue或者PR来告诉我。

本书由「ePUBw.COM」整理，ePUBw.COM 提供最新最全的优质

电子书下载！！！





6.3　多容器的应用栈

在最后一个服务应用的示例中我们把一个使用Express框架的、带有Redis后端的

Node.js应用完全Docker化了。这里要继续演示如何把之前两章学到的Docker特性

结合起来使用，包括链接和卷。

在这个例子中，我们会构建一系列的镜像来支持部署多容器的应用。

一个Node容器，用来服务于Node应用，这个容器会链接到。

一个Redis主容器，用于保存和集群化应用状态，这个容器会链接到。

两个Redis副本容器，用于集群化应用状态。

一个日志容器，用于捕获应用日志。

我们的Node应用程序会运行在一个容器中，它后面会有一个配置为“主-副本”模式

运行在多个容器中的Redsi集群。

6.3.1　Node.js镜像

先从构建一个安装了Node.js的镜像开始，这个镜像有Express应用和相应的必要的

软件包，如代码清单6-32所示。

代码清单6-32　创建Node.js Dockerfile

$ mkdir nodejs

$ cd nodejs

$ mkdir -p nodeapp

$ cd nodeapp

$ wget https://raw.githubusercontent.com/jamtur01/dockerbook-code

　/master/code/6/node/nodejs/nodeapp/package.json $ wget https://raw.githubusercontent.com/jamtur01/dockerbook-code

　/master/code/6/node/nodejs/nodeapp/server.js

$ cd ..

$ vi Dockerfile

我们已经创建了一个叫nodejs的新目录，然后创建了子目录nodeapp来保存应用代

码。然后我们进入这个目录，并下载了Node.js应用的源代码。

注意 可以从本书官网[11]或者GitHub仓库[12]下载Node应用的源代码。

最后我们回到了nodejs目录。现在我们来看看这个Dockerfile的内容，如代码清单

6-33所示。

代码清单6-33　Node.js镜像

FROM ubuntu:14.04

MAINTAINER James Turnbull <james@example.com>

ENV REFRESHED_AT 2014-06-01

RUN apt-get -yqq update

RUN apt-get -yqq install nodejs npm

RUN ln -s /usr/bin/nodejs /usr/bin/node

RUN mkdir -p /var/log/nodeapp

ADD nodeapp /opt/nodeapp/

WORKDIR /opt/nodeapp

RUN npm install

VOLUME [ "/var/log/nodeapp" ]

EXPOSE 3000

ENTRYPOINT [ "nodejs", "server.js" ]

Node.js镜像安装了Node，然后我们用了一个简单的技巧把二进制文件nodejs链接

到node，解决了Ubuntu上原有的一些无法向后兼容的问题。

然后我们将nodeapp的源代码通过ADD指令添加到/opt/nodeapp目录。这个

Node.js应用是一个简单的Express服务器，包括一个存放应用依赖信息的

package.json文件和包含实际应用代码的server.js文件，我们来看一下该应用的

部分代码，如代码清单6-34所示。

代码清单6-34　Node.js应用的server.js文件

. . .

var logFile = fs.createWriteStream('/var/log/nodeapp/nodeapp.log',

{flags: 'a'});

app.configure(function() {

. . .

　app.use(express.session({

　　　　store: new RedisStore({

　　　　　　host: process.env.REDIS_HOST || 'redis_primary',

　　　　　　port: process.env.REDIS_PORT || 6379,

　　　　　　db: process.env.REDIS_DB || 0

　　　　}),

　　　　cookie: {

. . .

app.get('/', function(req, res) {

　res.json({

　　status: "ok"

　});

});

. . .

var port = process.env.HTTP_PORT || 3000;

server.listen(port);

console.log('Listening on port ' + port);





server.js文件引入了所有的依赖，并启动了Express应用。Express应用把会话

（session）信息保存到Redis里，并创建了一个以JSON格式返回状态信息的节点。

这个应用默认使用redis_primary作为主机名去连接Redis，如果有必要，可以通

过环境变量覆盖这个默认的主机名。

这个应用会把日志记录到/var/log/nodeapp/nodeapp.log文件里，并监听3000端

口。

注意 可以从本书官网[13]或者GitHub[14]得到这个Node应用的源代码。

接着我们将工作目录设置为/opt/nodeapp，并且安装了Node应用的必要软件包，还

创建了用于存放Node应用日志的卷/var/log/nodeapp。

最后我们公开了3000端口，并使用ENTRYPOINT指定了运行Node应用的命令nodejs

server.js。

现在我们来构建镜像，如代码清单6-35所示。

代码清单6-35　构建Node.js镜像

$ sudo docker build -t jamtur01/nodejs .

6.3.2　Redis基础镜像

现在我们继续构建第一个Redis镜像：安装Redis的基础镜像（如代码清单6-36所

示）。然后我们会使用这个镜像构建Redis主镜像和副本镜像。

代码清单6-36　创建Redis基础镜像的Dockerfile

$ mkdir redis_base

$ cd redis_base

$ vi Dockerfile

现在我们来看看这个Dockerfile的内容，如代码清单6-37所示。

代码清单6-37　基础Redis镜像

FROM ubuntu:14.04

MAINTAINER James Turnbull <james@example.com>

ENV REFRESHED_AT 2014-06-01

RUN apt-get -yqq update

RUN apt-get install -yqq software-properties-common python-software-properties

RUN add-apt-repository ppa:chris-lea/redis-server

RUN apt-get -yqq update

RUN apt-get -yqq install redis-server redis-tools

VOLUME [ "/var/lib/redis", "/var/log/redis/" ]

EXPOSE 6379

CMD[]

这个Redis基础镜像安装了最新版本的Redis（从PPA库安装，而不是使用Ubuntu自

带的较老的Redis包），指定了两个VOLUME（/var/lib/redis

和/var/log/redis），公开了Redis的默认端口6379。因为不会执行这个镜像，所

以没有包含ENTRYPOINT或者CMD指令。然后我们将只是基于这个镜像构建别的镜

像。

现在我们来构建Redis基础镜像，如代码清单6-38所示。

代码清单6-38　构建Redis基础镜像

$ sudo docker build -t jamtur01/redis .





6.3.3　Redis主镜像

我们继续构建第一个Redis镜像，即Redis主服务器，如代码清单6-39所示。

代码清单6-39　创建Redis主服务器的Dockerfile

$ mkdir redis_primary

$ cd redis_primary

$ vi Dockerfile

我们来看看这个Dockerfile的内容，如代码清单6-40所示。

代码清单6-40　Redis主镜像

FROM jamtur01/redis

MAINTAINER James Turnbull <james@example.com>

ENV REFRESHED_AT 2014-06-01

ENTRYPOINT [ "redis-server", "--logfile /var/log/redis/redis-server.log" ]

Redis主镜像基于之前的jamtur01/redis镜像，并通过ENTRYPOINT指令指定了

Redis服务启动命令，Redis服务的日志文件保存到/var/log/redis/redis-

server.log。

现在我们来构建Redis主镜像，如代码清单6-41所示。

代码清单6-41　构建Redis主镜像

$ sudo docker build -t jamtur01/redis_primary .

6.3.4　Redis副本镜像





为了配合Redis主镜像，我们会创建Redis副本镜像，保证为Node.js应用提供

Redis服务的冗余度，如代码清单6-42所示。

代码清单6-42　创建Redis副本镜像的Dockerfile

$ mkdir redis_replica

$ cd redis_replica

$ touch Dockerfile

现在我们来看看对应的Dockerfile，如代码清单6-43所示。

代码清单6-43　Redis副本镜像

FROM jamtur01/redis

MAINTAINER James Turnbull <james@example.com>

ENV REFRESHED_AT 2014-06-01

ENTRYPOINT [ "redis-server", "--logfile /var/log/redis/redis-replica.log",

　"--slaveof redis_primary 6379" ]

Redis副本镜像也是基于jamtur01/redis构建的，并且通过ENTRYPOINT指令指定

了运行Redis服务器的命令，设置了日志文件和slaveof选项。这就把Redis配置为

主-副本模式，从这个镜像构建的任何容器都会将redis_primary主机的Redis作为

主服务，连接其6379端口，成为其对应的副本服务器。

现在我们来构建Redis副本镜像，如代码清单6-44所示。

代码清单6-44　构建Redis副本镜像

$ sudo docker build -t jamtur01/redis_replica .

6.3.5　创建Redis后端集群

现在我们已经有了Redis主镜像和副本镜像，已经可以构建我们自己的Redis复制环

境了。首先我们创建一个用来运行我们的Express应用程序的网络，我们称其

为express���如代码清单6-45所示。

代码清单6-45　创建express网络

$ sudo docker network create express

dfe9fe7ee5c9bfa035b7cf10266f29a701634442903ed9732dfdba2b509680c2

现在让我们在这个网络中运行Redis主容器，如代码清单6-46所示。

代码清单6-46　运行Redis主容器

$ sudo docker run -d -h redis_primary \

--net express --name redis_primary jamtur01/redis_primary

d21659697baf56346cc5bbe8d4631f670364ffddf4863ec32ab0576e85a73d27

这里使用docker run命令从jamtur01/redis_primary镜像创建了一个容器。这里

使用了一个以前没有见过的新标志-h，这个标志用来设置容器的主机名。这会覆盖

默认的行为（默认将容器的主机名设置为容器ID）并允许我们指定自己的主机名。

使用这个标志可以确保容器使用redis_primary作为主机名，并被本地的DNS服务正

确解析。

我们已经指定了--``name标志，确保容器的名字是redis_primary，我们还指定

了--``net标志，确保``该容器``在express网络中运行。稍后我们会看到，我们

将使用这个网络来保证容器连通性。

让我们使用docker logs命令来查看Redis主容器的运行状况，如代码清单6-47所

示。

代码清单6-47　Redis主容器的日志

$ sudo docker logs redis_primary

什么日志都没有？这是怎么回事？原来Redis服务会将日志记录到一个文件而不是记

录到标准输出，所以使用Docker查看不到任何日志。那怎么能知道Redis服务器的

运行情况呢？为了做到这一点，可以使用之前创建的/var/log/redis卷。现在我们

来看看这个卷，读取一些日志文件的内容，如代码清单6-48所示。

代码清单6-48　读取Redis主日志

$ sudo docker run -ti --rm --volumes-from redis_primary \

ubuntu cat /var/log/redis/redis-server.log

...

[1] 25 Jun 21:45:03.074 # Server started, Redis version 2.8.12

[1] 25 Jun 21:45:03.074 # WARNING overcommit_memory is set to 0!

Background save may fail under low memory condition. To fix

this issue add 'vm.overcommit_memory = 1' to /etc/sysctl.conf

and then reboot or run the command 'sysctl vm.overcommit_memory

=1' for this to take effect.

[1] 25 Jun 21:45:03.074 * The server is now ready to accept

　connections on port 6379

这里以交互方式运行了另一个容器。这个命令指定了--rm标志，它会在进程运行完

后自动删除容器。我们还指定了--volumes-from标志，告诉它从redis_primary

容器挂载了所有的卷。然后我们指定了一个ubuntu基础镜像，并告诉它执行cat

var/log/`` ``redis/redis-server.log来展示日志文件。这种方法利用了卷的

优点，可以直接从redis_primary容器挂载/var/log/redis目录并读取里面的日

志文件。一会儿我们将会看到更多使用这个命令的情况。

查看Redis日志，可以看到一些常规警告，不过一切看上去都没什么问题。Redis服

务器已经准备好从6379端口接收数据了。

那么下一步，我们创建一个Redis副本容器，如代码清单6-49所示。

代码清单6-49　运行第一个Redis副本容器

$ sudo docker run -d -h redis_replica1 \

--name redis_replica1 \

--net express \

jamtur01/redis_replica

0ae440b5c56f48f3190332b4151c40f775615016bf781fc817f631db5af34ef8

这里我们运行了另一个容器：这个容器来自jamtur01/redis_replica镜像。和之

前一样，命令里指定了主机名（通过-h标志）和容器名（通过--name标志）都

是redis_`` ``replica1。我们还使用了--net标志在express网络中运行Redis副

本容器。

提示 在Docker 1.9之前的版本中，不能使用Docker Networking，只能使用Docker链接来连接Redis主容

器和副本容器。

现在我们来检查一下这个新容器的日志，如代码清单6-50所示。

代码清单6-50　读取Redis副本容器的日志

$ sudo docker run -ti --rm --volumes-from redis_replica1 \

ubuntu cat /var/log/redis/redis-replica.log

. . .

[1] 25 Jun 22:10:04.240 # Server started, Redis version 2.8.12

[1] 25 Jun 22:10:04.240 # WARNING overcommit_memory is set to 0!

Background save may fail under low memory condition. To fix

this issue add 'vm.overcommit_memory = 1' to /etc/sysctl.conf and then reboot or run the command 'sysctl vm.overcommit_memory

=1' for this to take effect.

[1] 25 Jun 22:10:04.240 * The server is now ready to accept

connections on port 6379

[1] 25 Jun 22:10:04.242 * Connecting to MASTER redis_primary:6379

[1] 25 Jun 22:10:04.244 * MASTER <-> SLAVE sync started

[1] 25 Jun 22:10:04.244 * Non blocking connect for SYNC fired the event.

[1] 25 Jun 22:10:04.244 * Master replied to PING, replication can continue...

[1] 25 Jun 22:10:04.245 * Partial resynchronization not possible (no cached master)

[1] 25 Jun 22:10:04.246 * Full resync from master: 24

　a790df6bf4786a0e886be4b34868743f6145cc:1485

[1] 25 Jun 22:10:04.274 * MASTER <-> SLAVE sync: receiving 18 bytes from master

[1] 25 Jun 22:10:04.274 * MASTER <-> SLAVE sync: Flushing old data

[1] 25 Jun 22:10:04.274 * MASTER <-> SLAVE sync: Loading DB in memory

[1] 25 Jun 22:10:04.275 * MASTER <-> SLAVE sync: Finished with success 这里通过交互的方式运行了一个新容器来查询日志。和之前一样，我们又使用了--

rm标志，它在命令执行完毕后自动删除容器。我们还指定了--volumes-from标志，

挂载了redis_replica1容器的所有卷。然后我们指定了ubuntu基础镜像，并让它

cat日志文件/var/log/ redis/redis-replica.log。

到这里我们已经成功启动了redis_primary和redis_replica1容器，并让这两个

容器进行主从复制。

现在我们来加入另一个副本容器redis_replica2，确保万无一失，如代码清单6-

51所示。

代码清单6-51　运行第二个Redis副本容器

$ sudo docker run -d -h redis_replica2 \

--name redis_replica2 \

--net express \

jamtur01/redis_replica

72267cd74c412c7b168d87bba70f3aaa3b96d17d6e9682663095a492bc260357

我们来看看新容器的日志，如代码清单6-52所示。

代码清单6-52　第二个Redis副本容器的日志

$ sudo docker run -ti --rm --volumes-from redis_replica2 ubuntu \

cat /var/log/redis/redis-replica.log

. . .

[1] 25 Jun 22:11:39.417 # Server started, Redis version 2.8.12

[1] 25 Jun 22:11:39.417 # WARNING overcommit_memory is set to 0!

Background save may fail under low memory condition. To fix

this issue add 'vm.overcommit_memory = 1' to /etc/sysctl.conf

and then reboot or run the command 'sysctl vm.overcommit_memory

=1' for this to take effect.

[1] 25 Jun 22:11:39.417 * The server is now ready to accept connections on port 6379

[1] 25 Jun 22:11:39.417 * Connecting to MASTER redis_primary:6379

[1] 25 Jun 22:11:39.422 * MASTER <-> SLAVE sync started

[1] 25 Jun 22:11:39.422 * Non blocking connect for SYNC fired the event.

[1] 25 Jun 22:11:39.422 * Master replied to PING, replication can continue...

[1] 25 Jun 22:11:39.423 * Partial resynchronization not possible (no cached master)

[1] 25 Jun 22:11:39.424 * Full resync from master: 24

　a790df6bf4786a0e886be4b34868743f6145cc:1625

[1] 25 Jun 22:11:39.476 * MASTER <-> SLAVE sync: receiving 18 bytes from





　master

[1] 25 Jun 22:11:39.476 * MASTER <-> SLAVE sync: Flushing old data

[1] 25 Jun 22:11:39.476 * MASTER <-> SLAVE sync: Loading DB in memory 现在可以确保Redis服务万无一失了！

6.3.6　创建Node容器

现在我们已经让Redis集群运行了，我们可以为启动Node.js应用启动一个容器，如

代码清单6-53所示。

代码清单6-53　运行Node.js容器

$ sudo docker run -d \

--name nodeapp -p 3000:3000 \

--net express \

jamtur01/nodejs

9a9dd33957c136e98295de7405386ed2c452e8ad263a6ec1a2a08b24f80fd175

提示 在Docker 1.9之前的版本中，不能使用Docker Networking，只能使用Docker链接来连接Node和

Redis容器。

我们从jamtur01/nodejs镜像创建了一个新容器，命名为nodeapp，并将容器内的

3000端口映射到宿主机的3000端口。同样我们的新nodeapp容器也是运行

在express网络中。

可以使用docker logs命令来看看nodeapp容器在做什么，如代码清单6-54所示。

代码清单6-54　nodeapp容器的控制台日志

$ sudo docker logs nodeapp





Listening on port 3000

从这个日志可以看到Node应用程序监听了3000端口。

现在我们在Docker宿主机上打开相应的网页，看看应用工作的样子，如图6-7所

示。

图6-7　Node应用程序

可以看到Node应用只是简单地返回了OK状态，如代码清单6-55所示。

代码清单6-55　Node应用的输出

{

　"status": "ok"

}

这个输出表明应用正在工作。浏览器的会话状态会先被记录到Redis主容

器redis_primary，然后复制到两个Redis副本容器redis_replica1和

redis_replica2。

6.3.7　捕获应用日志

现在应用已经可以运行了，需要把这个应用放到生产环境中。在生产环境里需要确

保可以捕获日志并将日志保存到日志服务器。我们将使用Logstash[15]来完成这件

事。我们先来创建一个Logstash镜像，如代码清单6-56所示。

代码清单6-56　创建Logstash的Dockerfile

$ mkdir logstash

$ cd logstash

$ touch Dockerfile

现在我们来看看这个Dockerfile的内容，如代码清单6-57所示。

代码清单6-57　Logstash镜像

FROM ubuntu:14.04

MAINTAINER James Turnbull <james@example.com>

ENV REFRESHED_AT 2014-06-01

RUN apt-get -yqq update

RUN apt-get -yqq install wget

RUN wget -O - http://packages.elasticsearch.org/GPG-KEY-elasticsearch |

　apt-key add -

RUN echo 'deb <a>http://packages.elasticsearch.org/logstash/1.4/debian</a> stable main' > /etc/apt/sources.list.d/logstash.list

RUN apt-get -yqq update

RUN apt-get -yqq install logstash

ADD logstash.conf /etc/

WORKDIR /opt/logstash

ENTRYPOINT [ "bin/logstash" ]

CMD [ "--config=/etc/logstash.conf" ]

我们已经创建了镜像并安装了Logstash，然后将logstash.conf文件使用ADD指令

添加到/etc/目录。现在我们来看看logstash.conf文件的内容，如代码清单6-58

所示。

代码清单6-58　Logstash配置文件

input {

　file {

　　type => "syslog"

　　path => ["/var/log/nodeapp/nodeapp.log", "/var/log/redis/

　　　redis-server.log"]

　}

}

output {

　stdout {

　　codec => rubydebug

　}

}

这个Logstash配置很简单，它监控两个文件，即/var/log/nodeapp/nodeapp.log

和/var/log/redis/redis-server.log。Logstash会一直监视这两个文件，将其

中新的内容发送给Logstash。配置文件的第二部分是output部分，接受所有

Logstash输入的内容并将其输出到标准输出上。现实中，一般会将Logstash配置为

输出到Elasticsearch集群或者其他的目的地，不过这里只使用标准输出做演示，

所以忽略了现实的细节。

注意 如果不太了解Logstash，想要深入学习可以参考作者的书[16]或者Logstash文档[17]。

我们指定了工作目录为/opt/logstash。最后，我们指定了`ÈNTRYPOINT为

bin/`` ``logstash，并且指定了CMD为--config=/etc/logstash.conf。这样

容器启动时会启动Logstash并加载/etc/logstash.conf配置文件。

现在我们来构建Logstash镜像，如代码清单6-59所示。

代码清单6-59　构建Logstash镜像

$ sudo docker build -t jamtur01/logstash .

构建好镜像后，可以从这个镜像启动一个容器，如代码清单6-60所示。

代码清单6-60　启动Logstash容器

$ sudo docker run -d --name logstash \

--volumes-from redis_primary \

--volumes-from nodeapp \

jamtur01/logstash

我们成功地启动了一个名为logstash的新容器，并指定了两次--volumes-from标

志，分别挂载了redis_primary和nodeapp容器的卷，这样就可以访问Redis和

Node的日志文件了。任何加到这些日志文件里的内容都会反映在logstash容器的卷

里，并传给Logstash做后续处理。

现在我们使用-f标志来查看logstash容器的日志，如代码清单6-61所示。

代码清单6-61　logstash容器的日志

$ sudo docker logs -f logstash

{:timestamp=>"2014-06-26T00:41:53.273000+0000", :message=>"Using milestone 2 input plugin 'file'. This plugin should be stable, but if you see strange behavior, please let us know! For more

　information on plugin milestones, see http://logstash.net/docs

　/1.4.2-modified/plugin-milestones", :level=>:warn}

现在再在浏览器里刷新Web应用，产生一个新的日志事件。这样应该能在logstash

容器的输出中看到这个事件，如代码清单6-62所示。

代码清单6-62　Logstash中的Node事件





{

　　　　"message" => "63.239.94.10 - - [Thu, 26 Jun 2014 01:28:42

　　　　　GMT] \"GET /hello/frank HTTP/1.1\" 200 22 \"-\" \"

　　　　　Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4)

　　　　　AppleWebKit/537.36 (KHTML, like Gecko) Chrome

　　　　　/35.0.1916.153 Safari/537.36\"",

　　　　 "@version" => "1",

　　　　"@timestamp" => "2014-06-26T01:28:42.593Z",

　　　　　　 "type" => "syslog",

　　　　　　 "host" => "cfa96519ba54",

　　　　　　 "path" => "/var/log/nodeapp/nodeapp.log"

}

现在Node和Redis容器都将日志输出到了Logstash。在生产环境中，这些事件会发

到Logstash服务器并存储在Elasticsearch里。如果要加入新的Redis副本容器或

者其他组件，也可以很容易地将其日志输出到日志容器里。

注意 如果需要，也可以通过卷对Redis做备份。

6.3.8　Node程序栈的小结

现在我们已经演示过了如何使用多个容器组成应用程序栈，演示了如何使用Docker

链接来将应用容器连在一起，还演示了如何使用Docker卷来管理应用中各种数据。

这些技术可以很容易地用来构建更加复杂的应用程序和架构。

本书由「ePUBw.COM」整理，ePUBw.COM 提供最新最全的优质

电子书下载！！！





6.4　不使用SSH管理Docker容器

最后，在结束关于使用Docker运行服务的话题之前，了解一些管理Docker容器的方

法以及这些方法与传统管理方法的区别是很重要的。

传统上讲，通过SSH登入运行环境或者虚拟机里来管理服务。在Docker的世界里，

大部分容器都只运行一个进程，所以不能使用这种访问方法。不过就像之前多次看

到的，其实不需要这种访问：可以使用卷或者链接完成大部分同样的管理操作。比

如说，如果服务通过某个网络接口做管理，就可以在启动容器时公开这个接口；如

果服务通过Unix套接字（socket）来管理，就可以通过卷公开这个套接字。如果需

要给容器发送信号，就可以像代码清单6-63所示那样使用docker kill命令发送信

号。

代码清单6-63　使用docker kill发送信号

$ sudo docker kill -s <signal> <container>

这个操作会发送指定的信号（如HUP信号）给容器，而不是杀掉容器。

然而，有时候确实需要登入容器。即便如此，也不需要在容器里执行SSH服务或者打

开任何不必要的访问。需要登入容器时，可以使用一个叫nsenter的小工具。

注意 nsenter一般适用于Docker 1.2 或者更早的版本。docker exec命令是在Docker 1.3中引入的，替换

了它的大部分功能。

工具nsenter让我们可以进入Docker用来构成容器的内核命名空间。从技术上说，

这个工具可以进入一个已经存在的命名空间，或者在新的一组命名空间里执行一个

进程。简单来说，使用nsenter可以进入一个已经存在的容器的shell，即便这个容

器没有运行SSH或者任何类似目的的守护进程。可以通过Docker容器安装nsenter，

如代码清单6-64所示。

代码清单6-64　安装nsenter

$ sudo docker run -v /usr/local/bin:/target jpetazzo/nsenter

这会把nsenter安装到/usr/local/bin目录下，然后立刻就可以使用这个命令。

提示 工具nsenter也可能由所使用的Linux发行版（在util-linux包里）提供。

为了使用nsenter，首先要拿到要进入的容器的进程ID（PID）。可以使用docker

inspect命令获得PID，如代码清单6-65所示。

代码清单6-65　获取容器的进程ID

PID=$(sudo docker inspect --format '{{.State.Pid}}' <container>) 然后就可以进入容器，如代码清单6-66所示。

代码清单6-66　使用nsenter进入容器

$ sudo nsenter --target $PID --mount --uts --ipc --net --pid

这会在容器里启动一个shell，而不需要SSH或者其他类似的守护进程或者进程。

我们还可以将想在容器内执行的命令添加在nsenter命令行的后面，如代码清单6-

67所示。

代码清单6-67　使用nsenter在容器内执行命令

$ sudo nsenter --target $PID --mount --uts --ipc --net --pid ls bin　boot　dev　etc　home　lib　lib64　media　mnt　opt　proc . . .

这会在目标容器内执行ls命令。

本书由「ePUBw.COM」整理，ePUBw.COM 提供最新最全的优质

电子书下载！！！





6.5　小结

在本章中我们演示了如何使用Docker容器构建一些生产用的服务程序，还进一步演

示了如何构建多容器服务并管理应用栈。本章的例子将Docker链接和卷融合在一

起，并使用这些特性提供一些扩展的功能，比如记录日志和备份。

在下一章中我们会演示如何使用Docker Compose、Docker Swarm和Consul工具来

对Docker进行编配。

[1] 　 http://jekyllrb.com/

[2] 　 http://getbootstrap.com

[3] 　 http://docs.docker.com/engine/extend/plugins_volume/

[4]

https://docs.docker.com/engine/reference/commandline/volume_create/

[5] 　 http://aws.amazon.com/s3/

[6] 　 http://www.amanda.org

[7] 　 http://dockerbook.com/code/6/tomcat/tprov/

[8] 　 https://github.com/jamtur01/dockerbook-code/tree/master/code/6/tomcat/tprov

[9] 　 https://github.com/jamtur01/dockerbook-

code/blob/master/code/6/tomcat/tprov/lib/tprov/app.rb

[10] 　 完全是你自己写的代码——我很喜欢自己写代码，而不是直接用别人的。

[11] 　 http://dockerbook.com/code/6/node/

[12] 　 https://github.com/jamtur01/dockerbook-code/tree/master/code/6/node/

[13] 　 http://dockerbook.com/code/6/node/

[14] 　 https://github.com/jamtur01/dockerbook-code/tree/master/code/6/node/

[15] 　 http://logstash.net/

[16] 　 http://www.logstashbook.com

[17] 　 http://logstash.net

本书由「ePUBw.COM」整理，ePUBw.COM 提供最新最全的优质

电子书下载！！！





第7章　Docker编配和服务发现

编配（orchestration）是一个没有严格定义的概念。这个概念大概描述了自动配

置、协作和管理服务的过程。在Docker的世界里，编配用来描述一组实践过程，这

个过程会管理运行在多个Docker容器里的应用，而这些Docker容器有可能运行在多

个宿主机上。Docker对编配的原生支持非常弱，不过整个社区围绕编配开发和集成

了很多很棒的工具。

在现在的生态环境里，已经围绕Docker构建和集成了很多的工具。一些工具只是简

单地将多个容器快捷地“连”在一起，使用简单的组合来构建应用程序栈。另外一

些工具提供了在更大规模多个Docker宿主机上进行协作的能力，以及复杂的调度和

执行能力。

刚才提到的这些领域，每个领域都值得写一本书。不过本书只介绍这些领域里几个

有用的工具，这些工具可以让读者了解应该如何实际对容器进行编配。希望这些工

具可以帮读者构建自己的Docker环境。

本章将关注以下3个领域。

简单的容器编配。这部分内容会介绍Docker Compose。Docker Compose（之

前的Fig）是由Orchard团队开发的开源Docker编配工具，后来2014年被

Docker公司收购。这个工具用Python编写，遵守Apache 2.0许可。

分布式服务发现。这部分内容会介绍Consul。Consul使用Go语言开发，以MPL

2.0许可授权开源。这个工具提供了分布式且高可用的服务发现功能。本书会展

示如何使用Consul和Docker来管理应用，发现相关服务。

Docker的编配和集群。在这里我们将会介绍Swarm。Swarm是一个开源的、基于

Apache 2.0许可证发布的软件。它用Go语言编写，由Docker公司团队开发。

提示 本章的后面我还会谈论可用的许多其他的编配工具。

本书由「ePUBw.COM」整理，ePUBw.COM 提供最新最全的优质

电子书下载！！！





7.1　Docker Compose

现在先来熟悉一下Docker Compose。使用Docker Compose，可以用一个YAML文件

定义一组要启动的容器，以及容器运行时的属性。Docker Compose称这些容器

为“服务”，像这样定义：

容器通过某些方法并指定一些运行时的属性来和其他容器产生交互。

下面会介绍如何安装Docker Compose，以及如何使用Docker Compose构建一个简

单的多容器应用程序栈。

7.1.1　安装Docker Compose

现在开始安装Docker Compose。Docker Compose目前可以在Linux、Windows和

OS X上使用。可以通过直接安装可执行包来安装，或者通过Docker Toolbox安

装，也可以通过Python Pip包来安装。

为了在Linux上安装Docker Compose，可以从GitHub下载Docker Compose的可执

行包，并让其可执行。和Docker一样，Docker Compose目前只能安装在64位

Linux上。可以使用curl命令来完成安装，如代码清单7-1所示。

代码清单7-1　在Linux上安装Docker Compose

$ sudo bash -c "curl -L https://github.com/docker/compose/

releases/download/1.5.0/docker-compose-ùname -s`-ùname -m` >

/usr/local/bin/docker-compose"

$ sudo chmod +x /usr/local/bin/docker-compose

这个命令会从GitHub下载docker-compose可执行程序并安装到/usr/local/bin目

录中。之后使用chmod命令确保可执行程序docker-compose可以执行。

如果是在OS X上，Docker Toolbox 已经包含了Docker Compose，或者可以像代

码清单7-2所示这样进行安装。

代码清单7-2　在OS X上安装Docker Compose

$ sudo bash -c "curl -L https://github.com/docker/compose/

releases/download/1.5.0/docker-compose-Darwin-x86_64 > /usr/

local/bin/docker-compose"

$ sudo chmod +x /usr/local/bin/docker-compose

如果是在Windows平台上，也可以用Docker Toolbox，里面包含了Docker

Compose。

如果是在其他平台上或者偏好使用包来安装，Compose也可以作为Python包来安

装。这需要预先安装Python-Pip工具，保证存在pip命令。这个命令在大部分Red

Hat、Debian或者Ubuntu发行版里，都可以通过python-pip包安装，如代码清单

7-3所示。

代码清单7-3　通过Pip安装Docker Compose

$ sudo pip install -U docker-compose

安装好docker-compose可执行程序后，就可以通过使用--version标志调

用docker-compose命令来测试其可以正常工作，如代码清单7-4所示。

代码清单7-4　测试Docker Compose是否工作





$ docker-compose --version

docker-compose 1.5.0

注意 如果从1.3.0之前的版本升级而来，那么需要将容器格式也升级到1.3.0版本，这可以通过docker-

compose migrate-to-labels命令来实现。

7.1.2　获取示例应用

为了演示Compose是如何工作的，这里使用一个Python Flask应用作为例子，这个

例子使用了以下两个容器。

应用容器，运行Python示例程序。

Redis容器，运行Redis数据库。

现在开始构建示例应用。首先，创建一个目录并创建Dockerfile，如代码清单7-5

所示。

代码清单7-5　创建composeapp目录

$ mkdir composeapp

$ cd composeapp

$ touch Dockerfile

这里创建了一个叫作composeapp的目录来保存示例应用。之后进入这个目录，创建

了一个空Dockerfile，用于保存构建Docker镜像的指令。

之后，需要添加应用程序的源代码。创建一个名叫app.py的文件，并写入代码清单

7-6所示的Python代码。

代码清单7-6　app.py文件

from flask import Flask

from redis import Redis

import os

app = Flask(__name__)

redis = Redis(host="redis_1", port=6379)

@app.route('/')

def hello():

　　redis.incr('hits')

　　return 'Hello Docker Book reader! I have been seen {0} times'

　　　.format(redis.get('hits'))

if __name__ == "__main__":

　　app.run(host="0.0.0.0", debug=True)

提示 读者可以在GitHub[1]或者本书官网[2]找到源代码。

这个简单的Flask应用程序追踪保存在Redis里的计数器。每次访问根路径/时，计

数器会自增。

现在还需要创建requirements.txt文件来保存应用程序的依赖关系。创建这个文

件，并加入代码清单7-7列出的依赖。

代码清单7-7　requirements.txt文件

flask

redis

现在来看看Dockerfile，如代码清单7-8所示。

代码清单7-8　composeapp的Dockerfile

# Compose示例应用的镜像

FROM python:2.7

MAINTAINER James Turnbull <james@example.com> ADD . /composeapp

WORKDIR /composeapp

RUN pip install -r requirements.txt

这个Dockerfile很简单。它基于python:2.7镜像构建。首先添加文件app.py和

requirements.txt到镜像中的/composeapp目录。之后Dockerfile将工作目录设

置为/composeapp，并执行pip命令来安装应用的依赖：flask和redis。

使用docker build来构建镜像，如代码清单7-9所示。

代码清单7-9　构建composeapp应用

$ sudo docker build -tjamtur01/composeapp .

Sending build context to Docker daemon　16.9 kB

Sending build context to Docker daemon

Step 0 : FROM python:2.7

---> 1c8df2f0c10b

Step 1 : MAINTAINER James Turnbull <james@example.com>

---> Using cache

---> aa564fe8be5a

Step 2 : ADD . /composeapp

---> c33aa147e19f

Removing intermediate container 0097bc79d37b

Step 3 : WORKDIR /composeapp

---> Running in 76e5ee8544b3

---> d9da3105746d

Removing intermediate container 76e5ee8544b3

Step 4 : RUN pip install -r requirements.txt

---> Running in e71d4bb33fd2

Downloading/unpacking flask (from -r requirements.txt (line 1))





. . .

Successfully installed flask redis Werkzeug Jinja2 itsdangerous markupsafe Cleaning up...

---> bf0fe6a69835

Removing intermediate container e71d4bb33fd2

Successfully built bf0fe6a69835

这样就创建了一个名叫jamtur01/composeapp的容器，这个容器包含了示例应用和

应用需要的依赖。现在可以使用Docker Compose来部署应用了。

注意 之后会从Docker Hub上的默认Redis镜像直接创建Redis容器，这样就不需要重新构建或者定制Redis

容器了。

7.1.3　docker-compose.yml文件

现在应用镜像已经构建好了，可以配置Compose来创建需要的服务了。在Compose

中，我们定义了一组要启动的服务（以Docker容器的形式表现），我们还定义了我

们希望这些服务要启动的运行时属性，这些属性和docker run命令需要的参数类

似。将所有与服务有关的属性都定义在一个YAML文件里。之后执行docker-

compose up命令， Compose会启动这些容器，使用指定的参数来执行，并将所有

的日志输出合并到一起。

先来为这个应用创建docker-compose.yml文件，如代码清单7-10所示。

代码清单7-10　创建docker-compose.yml文件

$ touch docker-compose.yml

现在来看看docker-compose.yml文件的内容。docker-compose.yml是YAML格式

的文件，包括了一个或者多个运行Docker容器的指令。现在来看看示例应用使用的

指令，如代码清单7-11所示。

代码清单7-11　docker-compose.yml文件

web:

　image: jamtur01/composeapp

　command: python app.py

　ports:

　 - "5000:5000"

　volumes:

　 - .:/composeapp

　links:

　 - redis

redis:

　image: redis

每个要启动的服务都使用一个YAML的散列键定义：web和redis。

对于web服务，指定了一些运行时参数。首先，使用image指定了要使用的镜

像：jamtur01/composeapp镜像。Compose也可以构建Docker镜像。可以使

用build指令，并提供一个到Dockerfile的路径，让Compose构建一个镜像，并使

用这个镜像创建服务，如代码清单7-12所示。

代码清单7-12　build指令的示例

web:

　build: /home/james/composeapp

. . .

这个build指令会使用/home/james/composeapp目录下的Dockerfile来构建





Docker镜像。

我们还使用command指定服务启动时要执行的命令。接下来使用ports和volumes指

定了我们的服务要映射到的端口和卷，我们让服务里的5000端口映射到主机的5000

端口，并创建了卷/composeapp。最后使用links指定了要连接到服务的其他服务：

将redis服务连接到web服务。

如果想用同样的配置，在代码行中使用docker run执行服务，需要像代码清单7-13

所示这么做。

代码清单7-13　同样效果的docker run命令

$ sudo docker run -d -p 5000:5000 -v .:/composeapp --link redis:redis \

--name jamtur01/composeapp python app.py

之后指定了另一个名叫redis的服务。这个服务没有指定任何运行时的参数，一切使

用默认的配置。之前也用过这个redis镜像，这个镜像默认会在标准端口上启动一个

Redis数据库。这里没必要修改这个默认配置。

提示 可以在Docker Compose官网[3]查看docker-compose.yml所有可用的指令列表。

7.1.4　运行Compose

一旦在docker-compose.yml中指定了需要的服务，就可以使用docker-compose

up命令来执行这些服务，如代码清单7-14所示。

代码清单7-14　使用docker-compose up启动示例应用服务

$ cd composeapp

$ sudo docker-compose up

Creating composeapp_redis_1...

Creating composeapp_web_1...

Attaching to composeapp_redis_1, composeapp_web_1

redis_1 |　|`-._`-._　　`-.__.-'　　_.-'_.-'|

redis_1 |　|　　`-._`-._　　　　_.-'_.-'　　　|

redis_1 |　 `-._　　`-._`-.__.-'_.-'　　_.-'

redis_1 |　　　 `-._　　`-.__.-'　　_.-'

redis_1 |　　　　　 `-._　　　　_.-'

redis_1 |　　　　　　　 `-.__.-'

redis_1 |

redis_1 | [1] 13 Aug 01:48:32.218 # Server started, Redis version 2.8.13

redis_1 | [1] 13 Aug 01:48:32.218 # WARNING overcommit_memory is set to 0! Background save may fail under low memory condition. To fix this issue add 'vm.overcommit_memory = 1' to /etc/sysctl.conf and then reboot or run the command 'sysctl vm.overcommit_memory=1' for this to take effect.

redis_1 | [1] 13 Aug 01:48:32.218 * The server is now ready to accept connections on port 6379

web_1　 |　* Running on http://0.0.0.0:5000/

web_1　 |　* Restarting with reloader

提示 必须在docker-compose.yml文件所在的目录执行大多数Compose命令。

可以看到Compose创建了composeapp_redis_1和composeapp_web_1这两个新的服

务。那么，这两个名字是从哪儿来的呢？为了保证服务是唯一的，Compose

将docker- compose.yml文件中指定的服务名字加上了目录名作为前缀，并分别使

用数字作为后缀。

Compose之后接管了每个服务输出的日志，输出的日志每一行都使用缩短的服务名

字作为前缀，并交替输出在一起，如代码清单7-15所示。

代码清单7-15　Compose服务输出的日志





redis_1 | [1] 13 Aug 01:48:32.218 # Server started, Redis version 2.8.13

服务（和Compose）交替运行。这意味着，如果使用Ctrl+C来停止Compose运行，

也会停止运行的服务。也可以在运行Compose时指定-d标志，以守护进程的模式来

运行服务（类似于docker run -d标志），如代码清单7-16所示。

代码清单7-16　以守护进程方式运行Compose

$ sudo docker-compose up -d

来看看现在宿主机上运行的示例应用。这个应用绑定在宿主机所有网络接口的5000

端口上，所以可以使用宿主机的IP或者通过localhost来浏览该网站。

图7-1　Compose示例应用

可以看到这个页面上显示了当前计数器的值。刷新网站，会看到这个值在增加。每

次刷新都会增加保存在Redis里的值。Redis更新是通过由Compose控制的Docker容

器之间的链接实现的。

提示 在默认情况下，Compose会试图连接到本地的Docker守护进程，不过也会受到DOCKER_`` ``HOST环境

变量的影响，去连接到一个远程的Docker宿主机。

7.1.5　使用Compose

现在来看看Compose的其他选项。首先，使用Ctrl+C关闭正在运行的服务，然后以

守护进程的方式启动这些服务。

在composeapp目录下按Ctrl+C，之后使用-d标志重新运行docker-compose up命

令，如代码清单7-17所示。

代码清单7-17　使用守护进程模式重启Docker Compose

$ sudo docker-compose up -d

Recreating composeapp_redis_1...

Recreating composeapp_web_1...

$ . . .

可以看到Docker Compose重新创建了这些服务，启动它们，最后返回到命令行。

现在，在宿主机上以守护进程的方式运行了受Docker Compose管理的服务。使

用docker-compose ps命令（docker ps命令的近亲）可以查看这些服务的运行状

态。

提示 执行docker-compose help加上想要了解的命令，可以看到相关的Compose帮助，比如docker-

compose help ps命令可以看到与ps相关的帮助。

docker-compose ps命令列出了本地docker-compose.yml文件里定义的正在运行

的所有服务，如代码清单7-18所示。

代码清单7-18　运行docker-compose ps命令

$ cd composeapp

$ sudo docker-compose ps

　　Name　　　　　 Command　　　　State　　　　Ports

-----------------------------------------------------

composeapp_redis_1　 redis-server　　 Up　　　6379/tcp

composeapp_web_1　　　python app.py　 Up　　　5000->5000/tcp

这个命令展示了正在运行的Compose服务的一些基本信息：每个服务的名字、启动

服务的命令以及每个服务映射到的端口。

还可以使用docker-compose logs命令来进一步查看服务的日志事件，如代码清单

7-19所示。

代码清单7-19　显示Docker Compose服务的日志

$ sudo docker-compose logs

docker-compose logs

Attaching to composeapp_redis_1, composeapp_web_1

redis_1 |　(　　'　　　,　　　　 .-`　　| `,　　) Running in stand alone mode redis_1 |　|`-._`-...-` __...-.``-._|'` _.-'| Port: 6379

redis_1 |　|　　`-._　 `._　　/　　 _.-'　　| PID: 1

. . .

这个命令会追踪服务的日志文件，很类似tail -f命令。与tail -f命令一样，想要

退出可以使用Ctrl+C。

使用docker-compose stop命令可以停止正在运行的服务，如代码清单7-20所示。

代码清单7-20　停止正在运行的服务

$ sudo docker-compose stop

Stopping composeapp_web_1...

Stopping composeapp_redis_1...

这个命令会同时停止两个服务。如果该服务没有停止，可以使用docker-compose

kill命令强制杀死该服务。

现在可以用docker-compose ps命令来验证服务确实停止了，如代码清单7-21所

示。

代码清单7-21　验证Compose服务已经停止了

$ sudo docker-compose ps

　　Name　　　　　　Command　　　　State　　 Ports

---------------------------------------------

composeapp_redis_1　 redis-server　　Exit 0

composeapp_web_1　　 python app.py　 Exit 0

如果使用docker-compose stop或者docker-compose kill命令停止服务，还可

以使用docker-compose start命令重新启动这些服务。这与使用docker start命

令重启服务很类似。

最后，可以使用docker-compose rm命令来删除这些服务，如代码清单7-22所示。

代码清单7-22　删除Docker Compose服务

$ sudo docker-compose rm

Going to remove composeapp_redis_1, composeapp_web_1

Are you sure? [yN] y

Removing composeapp_redis_1...

Removing composeapp_web_1...

首先会提示你确认需要删除服务，确认之后两个服务都会被删除。docker-compose

ps命令现在会显示没有运行中或者已经停止的服务，如代码清单7-23所示。





代码清单7-23　显示没有Compose服务

$ sudo docker-compose ps

Name　 Command　 State　 Ports

------------------------------

7.1.6　Compose小结

现在，使用一个文件就可以构建好一个简单的Python-Redis栈！可以看出使用这种

方法能够非常简单地构建一个需要多个Docker容器的应用程序。而这个例子，只展

现了Compose最表层的能力。在Compose官网上有很多例子，比如使用Rails[4]、

Django[5]和Wordpress[6]，来展现更高级的概念。还可以将Compose与提供图形化

用户界面的Shipyard[7]一起使用。

提示 在Compose官网[8]可以找到完整的命令行参考手册。

本书由「ePUBw.COM」整理，ePUBw.COM 提供最新最全的优质

电子书下载！！！





7.2　Consul、服务发现和Docker

服务发现是分布式应用程序之间管理相互关系的一种机制。一个分布式程序一般由

多个组件组成。这些组件可以都放在一台机器上，也可以分布在多个数据中心，甚

至分布在不同的地理区域。这些组件通常可以为其他组件提供服务，或者为其他组

件消费服务。

服务发现允许某个组件在想要与其他组件交互时，自动找到对方。由于这些应用本

身是分布式的，服务发现机制也需要是分布式的。而且，服务发现作为分布式应用

不同组件之间的“胶水”，其本身还需要足够动态、可靠，适应性强，而且可以快

速且一致地共享关于这些服务的数据。

另外，Docker主要关注分布式应用以及面向服务架构与微服务架构。这些关注点很

适合与某个服务发现工具集成。每个Docker容器可以将其中运行的服务注册到服务

发现工具里。注册的信息可以是IP地址或者端口，或者两者都有，以便服务之间进

行交互。

这本书使用Consul[9]作为服务发现工具的例子。Consul是一个使用一致性算法的

特殊数据存储器。Consul使用Raft[10]一致性算法来提供确定的写入机制。Consul 暴露了键值存储系统和服务分类系统，并提供高可用性、高容错能力，并保证强一

致性。服务可以将自己注册到Consul，并以高可用且分布式的方式共享这些信息。

Consul还提供了一些有趣的功能。

提供了根据API进行服务分类，代替了大部分传统服务发现工具的键值对存储。

提供两种接口来查询信息：基于内置的DNS服务的DNS查询接口和基于HTTP的





REST API查询接口。选择合适的接口，尤其是基于DNS的接口，可以很方便地

将Consul与现有环境集成。

提供了服务监控，也称作健康监控。Consul内置了强大的服务监控系统。

为了更好地理解Consul是如何工作的，本章先介绍如何在Docker容器里分布式运行

Consul。之后会从Docker容器将服务注册到Consul，并从其他Docker容器访问注

册的数据。为了更有挑战，会让这些容器运行在不同的Docker宿主机上。

为了做到这些，需要做到以下几点。

创建Consul服务的Docker镜像。

构建3台运行Docker的宿主机，并在每台上运行一个Consul。这3台宿主机会提

供一个分布式环境，来展现Consul如何处理弹性和失效工作的。

构建服务，并将其注册到Consul，然后从其他服务查询该数据。

注意 可以在http://www.consul.io/intro/index.html找到对Consul更通用的介绍。

7.2.1　构建Consul镜像

首先创建一个Dockerfile来构建Consul镜像。先来创建用来保存Dockerfile的目

录，如代码清单7-24所示。

代码清单7-24　创建目录来保存Consul的Dockerfile

$ mkdir consul

$ cd consul

$ touch Dockerfile

现在来看看用于Consul镜像的Dockerfile的内容，如代码清单7-25所示。

代码清单7-25　Consul Dockerfile

FROM ubuntu:14.04

MAINTAINER James Turnbull <james@example.com>

ENV REFRESHED_AT 2014-08-01

RUN apt-get -qqy update

RUN apt-get -qqy install curl unzip

ADD https://dl.bintray.com/mitchellh/consul/0.3.1_linux_amd64.zip

/tmp/consul.zip

RUN cd /usr/sbin && unzip /tmp/consul.zip && chmod +x /usr/sbin/

consul && rm /tmp/consul.zip

ADD https://dl.bintray.com/mitchellh/consul/0.3.1_web_ui.zip

/tmp/webui.zip

RUN cd /tmp/ && unzip webui.zip && mv dist/ /webui/

ADD consul.json /config/

EXPOSE 53/udp 8300 8301 8301/udp 8302 8302/udp 8400 8500

VOLUME ["/data"]

ENTRYPOINT [ "/usr/sbin/consul", "agent", "-config-dir=/config" ]

CMD []

这个Dockerfile很简单。它是基于Ubuntu 14.04镜像，它安装了curl和unzip。

然后我们下载了包含consul可执行程序的zip文件。将这个可执行文件移动

到/usr/sbin并修改属性使其可以执行。我们还下载了Consul网页界面，将其放在

名为/webui的目录里。一会儿就会看到这个界面。

之后将Consul配置文件consul.json添加到/config目录。现在来看看配置文件的

内容，如代码清单7-26所示。

代码清单7-26　consul.json配置文件

{

　"data_dir": "/data",

　"ui_dir": "/webui",

　"client_addr": "0.0.0.0",

　"ports": {

　　"dns": 53

　},

　"recursor": "8.8.8.8"

}

consul.json配置文件是做过JSON格式化后的配置，提供了Consul运行时需要的信

息。我们首先指定了数据目录/data来保存Consul的数据，之后指定了网页界面文

件的位置：/webui。设置client_addr变量，将Consul绑定到容器内的所有网页界

面。

之后使用ports配置Consul服务运行时需要的端口。这里指定Consul的DNS服务运

行在53端口。之后，使用recursor选项指定了DNS服务器，这个服务器会用来解析

Consul无法解析的DNS请求。这里指定的8.8.8.8是Google的公共DNS服务[11]的一

个IP地址。

提示 可以在http://www.consul.io/docs/agent/options.html找到所有可用的Consul配置选项。

回到之前的Dockerfile，我们用EXPOSE指令打开了一系列端口，这些端口是

Consul运行时需要操作的端口。表7-1列出了每个端口的用途。

表7-1Consul的默认端口

端　　口

用　　途

53/udp

DNS服务器

8300

服务器 RPC

8301+udp

Serf的LAN端口

8302+udp

Serf的WAN端口

8400

RPC接入点

8500

HTTP API

就本章的目的来说，不需要关心表7-1里的大部分内容。比较重要的是53/udp端

口，Consul会使用这个端口运行DNS。之后会使用DNS来获取服务信息。另一个要关

注的是8500端口，它用于提供HTTP API和网页界面。其余的端口用于处理后台通

信，将多个Consul节点组成集群。之后我们会使用这些端口配置Docker容器，但并

不深究其用途。

注意 可以在http://www.consul.io/docs/agent/options.html找到每个端口更详细的信息。

之后，使用VOLUME指令将/data目录设置为卷。如果看过第6章，就知道这样可以更

方便地管理和处理数据。

最后，使用ENTRYPOINT指令指定从镜像启动容器时，启动Consul服务的consul可

执行文件。

现在来看看使用的命令行选项。首先我们已经指定了consul执行文件所在的目录

为/usr/sbin/。参数agent告诉Consul以代理节点的模式运行，-config-dir标志





指定了配置文件consul.json所在的目录是/config。

现在来构建镜像，如代码清单7-27所示。

代码清单7-27　构建Consul镜像

$ sudo docker build -t="jamtur01/consul" .

注意 可以从官网[12]或者GitHub[13]获得Consul的Dockerfile以及相关的配置文件。

7.2.2　在本地测试Consul容器

在多个宿主机上运行Consul之前，先来看看在本地单独运行一个Consul的情况。从

jamtur01/consul镜像启动一个容器，如代码清单7-28所示。

代码清单7-28　执行一个本地Consul节点

$ sudo docker run -p 8500:8500 -p 53:53/udp \

-h node1 jamtur01/consul -server -bootstrap

==> Starting Consul agent...

==> Starting Consul agent RPC...

==> Consul agent running!

　　　　 Node name: 'node1'

　　　　Datacenter: 'dc1'

. . .

2014/08/25 21:47:49 [WARN] raft: Heartbeat timeout reached, starting election 2014/08/25 21:47:49 [INFO] raft: Node at 172.17.0.26:8300 [Candidate]

entering Candidate state

2014/08/25 21:47:49 [INFO] raft: Election won. Tally: 1

2014/08/25 21:47:49 [INFO] raft: Node at 172.17.0.26:8300 [Leader]

entering Leader state

2014/08/25 21:47:49 [INFO] consul: cluster leadership acquired



2014/08/25 21:47:49 [INFO] consul: New leader elected: node1

2014/08/25 21:47:49 [INFO] consul: member 'node1' joined, marking health alive 使用docker run创建了一个新容器。这个容器映射了两个端口，容器中的8500端口

映射到了主机的8500端口，容器中的53端口映射到了主机的53端口。我们还使用-h

标志指定了容器的主机名node。这个名字也是Consul节点的名字。之后我们指定了

要启动的Consul镜像jamtur01/consul。

最后，给consul可执行文件传递了两个标志：-server和-bootstrap。标志-

server告诉Consul代理以服务器的模式运行，标志-bootstrap告诉Consul本节点

可以自选举为集群领导者。这个参数会让本节点以服务器模式运行，并可以执行

Raft领导者选举。

警告 有一点很重要：每个数据中心最多只有一台Consul服务器可以用自启动（bootstrap）模式运行。否

则，如果有多个可以进行自选举的节点，整个集群无法保证一致性。后面将其他节点加入集群时会介绍更多的

相关信息。

可以看到，Consul启动了node1节点， 并在本地进行了领导者选举。因为没有别的

Consul节点运行，刚启动的节点也没有其余的连接动作。

还可以通过Consul网页界面来查看节点情况，在浏览器里打开本地IP的8500端口。

图7-2　Consul网页界面





7.2.3　使用Docker运行Consul集群

由于Consul是分布式的，通常可以简单地在不同的数据中心、云服务商或者不同地

区创建3个（或者更多）服务器。甚至给每个应用服务器添加一个Consul代理，以

保证分布服务具有足够的可用性。本章会在3个运行Docker守护进程的宿主机上运

行Consul，来模拟这种分布环境。首先创建3个Ubutnu 14.04宿主

机：larry、curly和moe。每个主机上都已经安装了Docker守护进程，之后拉取

jamtur01/consul镜像，如代码清单7-29所示。

提示 要安装Docker可以使用第2章中介绍的安装指令。

代码清单7-29　拉取Consul镜像

$ sudo docker pull jamtur01/consul

在每台宿主机上都使用jamtur01/consul镜像运行一个Docker容器。要做到这一

点，首先需要选择运行Consul的网络。大部分情况下，这个网络应该是个私有网

络，不过既然是模拟Consul集群，这里使用每台宿主机上的公共接口，让Consul运

行在一个公共网络上。这需要每台宿主机都有一个公共IP地址。这个地址也是

Consul代理要绑定到的地址。

首先来获取larry的公共IP地址，并将这个地址赋值给环境变量$PUBLIC_IP，如代

码清单7-30所示。

代码清单7-30　给larry主机设置公共IP地址

larry$ PUBLIC_IP="$(ifconfig eth0 | awk -F ' *|:' '/inet addr/{

print $4}')"

larry$ echo $PUBLIC_IP

104.131.38.54

之后在curly和moe上创建同样的$PUBLIC_IP变量，如代码清单7-31所示。

代码清单7-31　在curly和moe上设置公共IP地址

curly$ PUBLIC_IP="$(ifconfig eth0 | awk -F ' *|:' '/inet addr/{print $4}')"

curly$ echo $PUBLIC_IP

104.131.38.55

moe$ PUBLIC_IP="$(ifconfig eth0 | awk -F ' *|:' '/inet addr/{print $4}')"

moe$ echo $PUBLIC_IP

104.131.38.56

现在3台宿主机有3个IP地址（如表7-2所示），每个地址都赋值给了$PUBLIC_IP环

境变量。

表7-2 Consul宿主机IP地址

宿 　 主 　 机

IP地址

larry

104.131.38.54

curly

104.131.38.55

moe

104.131.38.56

现在还需要指定一台宿主机为自启动的主机，来启动整个集群。这里指定larry主

机。这意味着，需要将larry的IP地址告诉curly和moe，以便让后两个宿主机知道

要连接到Consul节点的哪个集群。现在将larry的IP地址104.131.38.54添加到宿

主机curly和moe的环境变量$JOIN_IP，如代码清单7-32所示。

代码清单7-32　添加集群IP地址

curly$ JOIN_IP=104.131.38.54

moe$ JOIN_IP=104.131.38.54

最后，修改每台宿主机上的Docker守护进程的网络配置，以便更容易使用Consul。

将Docker守护进程的DNS查找设置为：

本地Docker的IP地址，以便使用Consul来解析DNS；

Google的DNS服务地址，来解析其他请求；

为Consul查询指定搜索域。

要做到这一点，首先需要知道Docker接口docker0的IP地址，如代码清单7-33所

示。

代码清单7-33　获取docker0的IP地址

larry$ ip addr show docker0

3: docker0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP group default

　　link/ether 56:84:7a:fe:97:99 brd ff:ff:ff:ff:ff:ff

　　inet 172.17.42.1/16 scope global docker0

　　　 valid_lft forever preferred_lft forever

　　inet6 fe80::5484:7aff:fefe:9799/64 scope link

　　　 valid_lft forever preferred_lft forever

可以看到这个接口的IP地址是172.17.42.1。

使用这个地址， 将/etc/default/docker文件中的Docker启动选项从代码清单7-





34所示的默认值改为代码清单7-35所示的新配置。

代码清单7-34　Docker的默认值

#DOCKER_OPTS="--dns 8.8.8.8 --dns 8.8.4.4"

代码清单7-35　larry上新的Docker默认值

DOCKER_OPTS='--dns 172.17.42.1 --dns 8.8.8.8 --dns-search service .consul'

之后在curly和moe上进行同样的设置：找到docker0的IP地址，并更新到/etc/

default/docker文件中的DOCKER_OPTS标志里。

提示 其他的分布式环境需要使用合适的机制更新Docker守护进程的默认值。更多信息参见第2章。

之后在每台宿主机上重启Docker守护进程，如代码清单7-36所示。

代码清单7-36　在larry上重启Docker守护进程

larry$ sudo service docker restart

7.2.4　启动具有自启动功能的Consul节点

现在在larry启动用来初始化整个集群的自启动节点。由于要映射很多端口，使用的

docker run命令会有些复杂。实际上，这个命令要映射表7-1里列出的所有端口。

而且，由于Consul既要运行在容器里，又要和其他宿主机上的容器通信，所以需要

将每个端口都映射到本地宿主机对应的端口上。这样可以既在本内部又可以在外部

访问Consul了。

来看看要用到的docker run命令，如代码清单7-37所示。

代码清单7-37　启动具有自启动功能的Consul节点

larry$ sudo docker run -d -h $HOSTNAME \

-p 8300:8300 -p 8301:8301 \

-p 8301:8301/udp -p 8302:8302 \

-p 8302:8302/udp -p 8400:8400 \

-p 8500:8500 -p 53:53/udp \

--name larry_agent jamtur01/consul \

-server -advertise $PUBLIC_IP -bootstrap-expect 3

这里以守护进程的方式从jamtur01/consul镜像启动了一个容器，用来运行Consul

代理。命令使用-h标志将容器的主机名设置为$HOSTNAME环境变量。这会让Consul

代理使用本地主机名larry。该命令还将8个端口映射到本地宿主机对应的端口。

该命令还指定了一些Consul代理的命令行参数，如代码清单7-38所示。

代码清单7-38　Consul代理的命令行参数

-server -advertise $PUBLIC_IP -bootstrap-expect 3

-server标志告诉代理运行在服务器模式。-advertise标志告诉代理通过环境变量

$PUBLIC_IP指定的IP广播自己。最后，-bootstrap-expect标志告诉Consul集群

中有多少代理。在这个例子里，指定了3个代理。这个标志还指定了本节点具有自启

动的功能。

现在使用docker logs命令来看看初始Consul容器的日志，如代码清单7-39所示。

代码清单7-39　启动具有自启动功能的Consul节点

larry$ sudo docker logs larry_agent

==> Starting Consul agent...

==> Starting Consul agent RPC...

==> Consul agent running!

　　　　 Node name: 'larry'

　　　　Datacenter: 'dc1'

　　　　　　 Server: true (bootstrap: false)

　　　 Client Addr: 0.0.0.0 (HTTP: 8500, DNS: 53, RPC: 8400)

　　　Cluster Addr: 104.131.38.54 (LAN: 8301, WAN: 8302)

　　Gossip encrypt: false, RPC-TLS: false, TLS-Incoming: false

. . .

2014/08/31 18:10:07 [WARN] memberlist: Binding to public address without encryption!

2014/08/31 18:10:07 [INFO] serf: EventMemberJoin: larry

104.131.38.54

2014/08/31 18:10:07 [WARN] memberlist: Binding to public address without encryption!

2014/08/31 18:10:07 [INFO] serf: EventMemberJoin: larry.dc1

104.131.38.54

2014/08/31 18:10:07 [INFO] raft: Node at 104.131.38.54:8300

[Follower] entering Follower state

2014/08/31 18:10:07 [INFO] consul: adding server larry (Addr:

104.131.38.54:8300) (DC: dc1)

2014/08/31 18:10:07 [INFO] consul: adding server larry.dc1 (Addr: 104.131.38.54:8300) (DC: dc1)

2014/08/31 18:10:07 [ERR] agent: failed to sync remote state: No cluster leader

2014/08/31 18:10:08 [WARN] raft: EnableSingleNode disabled, and no known peers. Aborting election.

可以看到larry上的代理已经启动了，但是因为现在还没有其他节点加入集群，所以

并没有触发选举操作。从仅有的一条错误信息（如代码清单7-40所示）可以看到这

一点。





代码清单7-40　有关集群领导者的错误信息

[ERR] agent: failed to sync remote state: No cluster leader

7.2.5　启动其余节点

现在集群已经启动好了，需要将剩下的curly和moe节点加入进来。先来启动

curly。使用docker run命令来启动第二个代理，如代码清单7-41所示。

代码清单7-41　在curly上启动代理

curly$ sudo docker run -d -h $HOSTNAME \

-p 8300:8300 -p 8301:8301 \

-p 8301:8301/udp -p 8302:8302 \

-p 8302:8302/udp -p 8400:8400 \

-p 8500:8500 -p 53:53/udp \

--name curly_agent jamtur01/consul \

-server -advertise $PUBLIC_IP -join $JOIN_IP

这个命令与larry上的自启动命令很相似，只是传给Consul代理的参数有变化，如

代码清单7-42所示。

代码清单7-42　在curly上启动Consul代理

-server -advertise $PUBLIC_IP -join $JOIN_IP

首先，还是使用-server启动了Consul代理的服务器模式，并将代理绑定到用-

advertise标志指定的公共IP地址。最后，-join告诉Consul要连接由环境变量

$JOIN_IP指定的larry主机的IP所在的Consul集群。

现在来看看启动容器后发生了什么，如代码清单7-43所示。

代码清单7-43　查看Curly代理的日志

curly$ sudo docker logs curly_agent

==> Starting Consul agent...

==> Starting Consul agent RPC...

==> Joining cluster...

　　 Join completed. Synced with 1 initial agents

==> Consul agent running!

　　　　 Node name: 'curly'

　　　　Datacenter: 'dc1'

　　　　　　 Server: true (bootstrap: false)

　　　 Client Addr: 0.0.0.0 (HTTP: 8500, DNS: 53, RPC: 8400)

　　　Cluster Addr: 104.131.38.55 (LAN: 8301, WAN: 8302)

　　Gossip encrypt: false, RPC-TLS: false, TLS-Incoming: false

. . .

2014/08/31 21:45:49 [INFO] agent: (LAN) joining: [104.131.38.54]

2014/08/31 21:45:49 [INFO] serf: EventMemberJoin: larry 104.131.38.54

2014/08/31 21:45:49 [INFO] agent: (LAN) joined: 1 Err: <nil> 2014/08/31 21:45:49 [ERR] agent: failed to sync remote state: No cluster leader

2014/08/31 21:45:49 [INFO] consul: adding server larry (Addr:

104.131.38.54:8300) (DC: dc1)

2014/08/31 21:45:51 [WARN] raft: EnableSingleNode disabled, and no known peers. Aborting election.

可以看到curly已经连接了larry，而且在larry上应该可以看到代码清单7-44所示

的日志。

代码清单7-44　curly加入larry

2014/08/31 21:45:49 [INFO] serf: EventMemberJoin: curly

104.131.38.55

2014/08/31 21:45:49 [INFO] consul: adding server curly (Addr: 104.131.38.55:8300) (DC: dc1)

这还没有达到集群的要求数量，还记得之前-bootstrap-expect参数指定了3个节

点吧。所以现在在moe上启动最后一个代理，如代码清单7-45所示。

代码清单7-45　在moe上启动代理

moe$ sudo docker run -d -h $HOSTNAME \

-p 8300:8300 -p 8301:8301 \

-p 8301:8301/udp -p 8302:8302 \

-p 8302:8302/udp -p 8400:8400 \

-p 8500:8500 -p 53:53/udp \

--name moe_agent jamtur01/consul \

-server -advertise $PUBLIC_IP -join $JOIN_IP

这个docker run命令基本上和在curly上执行的命令一样。只是这次整个集群有了3

个代理。现在，如果查看容器的日志，应该能看到整个集群的状态，如代码清单7-

46所示。

代码清单7-46　moe上的Consul日志

moe$ sudo docker logs moe_agent

==> Starting Consul agent...

==> Starting Consul agent RPC...

==> Joining cluster...

　　 Join completed. Synced with 1 initial agents

==> Consul agent running!

　　　　 Node name: 'moe'

　　　　Datacenter: 'dc1'

　　　　　　 Server: true (bootstrap: false)

　　　 Client Addr: 0.0.0.0 (HTTP: 8500, DNS: 53, RPC: 8400) Cluster Addr: 104.131.38.56 (LAN: 8301, WAN: 8302)

　　Gossip encrypt: false, RPC-TLS: false, TLS-Incoming: false

. . .

2014/08/31 21:54:03 [ERR] agent: failed to sync remote state: No cluster leader

2014/08/31 21:54:03 [INFO] consul: adding server curly (Addr:

104.131.38.55:8300) (DC: dc1)

2014/08/31 21:54:03 [INFO] consul: adding server larry (Addr:

104.131.38.54:8300) (DC: dc1)

2014/08/31 21:54:03 [INFO] consul: New leader elected: larry

从这个日志中可以看出，moe已经连接了集群。这样Consul集群就达到了预设的节

点数量，并且触发了领导者选举。这里larry被选举为集群领导者。

在larry上也可以看到最后一个代理节点加入Consul的日志，如代码清单7-47所

示。

代码清单7-47　在larry上的Consul领导者选举日志

2014/08/31 21:54:03 [INFO] consul: Attempting bootstrap with nodes:

[104.131.38.55:8300 104.131.38.56:8300 104.131.38.54:8300]

2014/08/31 21:54:03 [WARN] raft: Heartbeat timeout reached,

starting election

2014/08/31 21:54:03 [INFO] raft: Node at 104.131.38.54:8300

[Candidate] entering Candidate state

2014/08/31 21:54:03 [WARN] raft: Remote peer 104.131.38.56:8300

does not have local node 104.131.38.54:8300 as a peer

2014/08/31 21:54:03 [INFO] raft: Election won. Tally: 2

2014/08/31 21:54:03 [INFO] raft: Node at 104.131.38.54:8300

[Leader] entering Leader state



2014/08/31 21:54:03 [INFO] consul: cluster leadership acquired

2014/08/31 21:54:03 [INFO] consul: New leader elected: larry

. . .

2014/08/31 21:54:03 [INFO] consul: member 'larry' joined, marking health alive 2014/08/31 21:54:03 [INFO] consul: member 'curly' joined, marking health alive 2014/08/31 21:54:03 [INFO] consul: member 'moe' joined, marking health alive 通过浏览Consul的网页界面，选择Consul服务也可以看到当前的状态，如图7-3所

示。

图7-3　网页界面中的Consul服务

最后，可以通过dig命令测试DNS服务正在工作，如代码清单7-48所示。

代码清单7-48　测试Consul的DNS服务

larry$ dig @172.17.42.1 consul.service.consul

; <<>> DiG 9.9.5-3-Ubuntu <<>> @172.17.42.1 consul.service.consul

; (1 server found)

;; global options: +cmd

;; Got answer:

;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 13502

;; flags: qr aa rd ra; QUERY: 1, ANSWER: 3, AUTHORITY: 0, ADDITIONAL: 0

;; QUESTION SECTION:

;consul.service.consul.　　 IN　A

;; ANSWER SECTION:

consul.service.consul.　0　 IN　A　 104.131.38.55

consul.service.consul.　0　 IN　A　 104.131.38.54

consul.service.consul.　0　 IN　A　 104.131.38.56

;; Query time: 2 msec

;; SERVER: 172.17.42.1#53_(172.17.42.1)_

;; WHEN: Sun Aug 31 21:30:27 EDT 2014

;; MSG SIZE rcvd: 150

这里查询了本地Docker接口的IP地址，并将其作为DNS服务器地址，请求它返回关

于consul.service.consul的相关信息。这个域名的格式是使用Consul 的DNS快

速查询相关服务时的格式：consul是主机名，而service.consul是域名。这里

consul.service.consul代表Consul服务的DNS入口。

例如，代码清单7-49所示的代码会返回所有关于webservice服务的DNS A记录。

代码清单7-49　通过DNS查询其他Consul服务

larry$ dig @172.17.42.1 webservice.service.consul

提示 可以在http://www.consul.io/docs/agent/dns.html找到更多关于Consul的DNS接口的信息。

现在，这3台不同的宿主机依靠其中运行的Docker容器组成了一个Consul集群。这

看上去很酷，但还没有什么实际用处。下面来看看如何在Consul中注册一个服务，

并获得相关数据。





7.2.6　配合Consul，在Docker里运行一个分布式服务

为了演示如何注册服务，先基于uWSGI框架[14]创建一个演示用的分布式应用程序。

这个应用程序由以下两部分组成。

一个Web应用：distributed_app。它在启动时会启动相关的Web工作进程

（worker），并将这些程序作为服务注册到Consul。

●`` 一个应用客户端：distributed_client。它从Consul读取

与distributed_app相关的信息，并报告当前应用程序的状态和配置。

distributed_app会在两个Consul节点（即larry和curly）上运行，

而distributed_client客户端会在moe节点上运行。

1．构建分布式应用

现在来创建用于构建distributed_app的Dockerfile。先来创建用于保存镜像的目

录，如代码清单7-50所示。

代码清单7-50　创建用于保存distributed_app的Dockerfile的目录

$ mkdir distributed_app

$ cd distributed_app

$ touch Dockerfile

现在来看看用于构建distributed_app的Dockerfile的内容，如代码清单7-51所

示。

代码清单7-51　distributed_app使用的Dockerfile

FROM ubuntu:14.04

MAINTAINER James Turnbull "james@example.com"

ENV REFRESHED_AT 2014-06-01

RUN apt-get -qqy update

RUN apt-get -qqy install ruby-dev git libcurl4-openssl-dev curl build-essential python

RUN gem install --no-ri --no-rdoc uwsgi sinatra

RUN uwsgi --build-plugin https://github.com/unbit/uwsgi-consul

RUN mkdir -p /opt/distributed_app

WORKDIR /opt/distributed_app

ADD uwsgi-consul.ini /opt/distributed_app/

ADD config.ru /opt/distributed_app/

ENTRYPOINT [ "uwsgi", "--ini", "uwsgi-consul.ini", "--ini",

"uwsgi-consul.ini:server1", "--ini", "uwsgi-consul.ini:server2" ]

CMD []

这个Dockerfile安装了一些需要的包，包括uWSGI框架和Sinatra框架，以及一个

可以让uWSGI写入Consul的插件[15]。之后创建了目录/opt/distributed_app/，

并将其作为工作目录。之后将两个文件uwsgi-consul.ini和config.ru加到这个目

录中。

文件uwsgi-consul.ini用于配置uWSGI，来看看这个文件的内容，如代码清单7-52

所示。

代码清单7-52　uWSGI的配置

[uwsgi]

plugins = consul

socket = 127.0.0.1:9999

master = true

enable-threads = true

[server1]

consul-register = url=http://%h.node.consul:8500,name=

distributed_app, id=server1,port=2001

mule = config.ru

[server2]

consul-register = url=http://%h.node.consul:8500,name=

distributed_app, id=server2,port=2002

mule = config.ru

文件uwsgi-consul.ini使用uWSGI的Mule结构来运行两个不同的应用程序，这两个

应用都是在Sinatra框架中写成的输出“Hello World”的。现在来看

看config.ru文件，如代码清单7-53所示。

代码清单7-53　distributed_app使用的config.ru文件

require 'rubygems'

require 'sinatra'

get '/' do

"Hello World!"

end

run Sinatra::Application

文件uwsgi-consul.ini每个块里定义了一个应用程序，分别标记为server1和

server2。每个块里还包含一个对uWSGI Consul插件的调用。这个调用连到Consul

实例，并将服务以distributed_app的名字，与服务名server1或者server2，一

同注册到Consul。每个服务使用不同的端口，分别是2001和2002。

当该框架开始运行时，会创建两个Web应用的工作进程，并将其分别注册到

Consul。应用程序会使用本地的Consul节点来创建服务。参数%h是主机名的简写，

执行时会使用正确的主机名替换，如代码清单7-54所示。

代码清单7-54　Consul插件的URL

url=http://%h.node.consul:8500...

最后，Dockerfile会使用ENTRYPOINT指令来自动运行应用的工作进程。

现在来构建镜像，如代码清单7-55所示。

代码清单7-55　构建distributed_app镜像

$ sudo docker build -t="jamtur01/distributed_app" .

注意 可以从官网[16]或者GitHub[17]获取distributed_app的Dockerfile、相关配置和应用程序文件。

2．构建分布式客户端

现在来创建用于构建distributed_client镜像的Dockerfile文件。先来创建用来

保存镜像的目录，如代码清单7-56所示。

代码清单7-56　创建保存distributed_client的Dockerfile的目录

$ mkdir distributed_client

$ cd distributed_client

$ touch Dockerfile

现在来看看distributed_client应用程序的Dockerfile的内容，如代码清单7-57

所示。

代码清单7-57　distributed_client使用的Dockerfile

FROM ubuntu:14.04

MAINTAINER James Turnbull "james@example.com"

ENV REFRESHED_AT 2014-06-01

RUN apt-get -qqy update

RUN apt-get -qqy install ruby ruby-dev build-essential

RUN gem install --no-ri --no-rdoc json

RUN mkdir -p /opt/distributed_client

ADD client.rb /opt/distributed_client/

WORKDIR /opt/distributed_client

ENTRYPOINT [ "ruby", "/opt/distributed_client/client.rb" ]

CMD []

这个Dockerfile先安装了Ruby以及一些需要的包和gem，然后创建了/opt

/distributed_client目录，并将其作为工作目录。之后将包含了客户端应用程序

代码的client.rb文件复制到镜像的/opt/distributed_client目录。

现在来看看这个应用程序的代码，如代码清单7-58所示。

代码清单7-58　distributed_client应用程序

require "rubygems"

require "json"

require "net/http"

require "uri"

require "resolv"

uri = URI.parse("http://consul.service.consul:8500/v1/catalog/service/

distributed_app")

http = Net::HTTP.new(uri.host, uri.port)

request = Net::HTTP::Get.new(uri.request_uri)

response = http.request(request)

while true

　if response.body == "{}"

　　puts "There are no distributed applications registered in Consul"

　　sleep(1)

　elsif

　　result = JSON.parse(response.body)

　　result.each do |service|

　　　puts "Application #{service['ServiceName']} with element #{service

["ServiceID"]} on port #{service["ServicePort"]} found on node #{

service["Node"]} (#{service["Address"]})."

　　　dns = Resolv::DNS.new.getresources("distributed_app.service.consul", Resolv::DNS::Resource::IN::A)

　　　puts "We can also resolve DNS - #{service['ServiceName']}resolves to #{dns.collect { |d| d.address }.join(" and ")}."

　　　sleep(1)

　　end

　end

end

这个客户端首先检查Consul HTTP API和Consul DNS，判断是否存在名叫

distributed_app的服务。客户端查询宿主机consul.service.consul，返回的结

果和之前看到的包含所有Connsul集群节点的A记录的DNS CNAME记录类似。这可以

让我们的查询变简单。

如果没有找到服务，客户端会在控制台（consloe）上显示一条消息。如果检查

到distributed_app服务，就会：

解析从API返回的JSON输出，并将一些有用的信息输出到控制台；

对这个服务执行DNS查找，并将返回的所有A记录输出到控制台。

这样，就可以查看启动Consul集群中distributed_app容器的结果。

最后，Dockerfile用ENTRYPOINT命令指定了容器启动时，运行client.rb命令来

启动应用。

现在来构建镜像，如代码清单7-59所示。

代码清单7-59　构建distributed_client镜像

$ sudo docker build -t="jamtur01/distributed_client" .

注意 可以从官网[18]或者GitHub[19]下载distributed_client的Dockerfile和应用程序文件。

3．启动分布式应用

现在已经构建好了所需的镜像，可以在larry和curly上启动distributed_app应用

程序容器了。假设这两台机器已经按照之前的配置正常运行了Consul。先从

在larry上运行一个应用程序实例开始，如代码清单7-60所示。

代码清单7-60　在larry启动distributed_app

larry$ sudo docker run -h $HOSTNAME -d --name larry_distributed \

jamtur01/distributed_app

这里启动了jamtur01/distributed_app镜像，并且使用-h标志指定了主机名。主

机名很重要，因为uWSGI使用主机名来获知Consul服务注册到了哪个节点。之后将

这个容器命名为larry_distributed，并以守护进方式模式运行该容器。

如果检查容器的输出日志，可以看到uWSGI启动了Web应用工作进程，并将其作为服

务注册到Consul，如代码清单7-61所示。

代码清单7-61　distributed_app日志输出



larry$ sudo docker logs larry_distributed

[uWSGI] getting INI configuration from uwsgi-consul.ini

*** Starting uWSGI 2.0.6 (64bit) on [Tue Sep 2 03:53:46 2014] ***

. . .

[consul] built service JSON: {"Name":"distributed_app","ID":"server1",

"Check":{"TTL":"30s"},"Port":2001}

[consul] built service JSON: {"Name":"distributed_app","ID":"server2",

"Check":{"TTL":"30s"},"Port":2002}

[consul] thread for register_url=http://larry.node.consul:8500/v1/

agent/service/register check_url=http://larry.node.consul:8500/v1/

agent/check/pass/service:server1 name=distributed_app

id=server1 started

. . .

Tue Sep 2 03:53:47 2014 - [consul] workers ready, let's register the service to the agent

[consul] service distributed_app registered successfully

这里展示了部分日志。从这个日志里可以看到uWSGI已经启动了。Consul插件为每

个distributed_app工作进程构造了一个服务项[20]，并将服务项注册到Consul 里。如果现在检查Consul网页界面，应该可以看到新注册的服务，如图7-4所示。

图7-4　Consul网页界面中的distributed_app服务

现在在curly上再启动一个Web应用工作进程，如代码清单7-62所示。

代码清单7-62　在curly上启动distributed_app

curly$ sudo docker run -h $HOSTNAME -d --name curly_distributed \

jamtur01/distributed_app

如果查看日志或者Consul网页界面，应该可以看到更多已经注册的服务，如图7-5

所示。

4．启动分布式应用客户端

现在已经在larry和curly启动了Web应用工作进程，继续在moe上启动应用客户端，

看看能不能从Consul查询到数据，如代码清单7-63所示。

代码清单7-63　在moe上启动distributed_client

moe$ sudo docker run -h $HOSTNAME -d --name moe_distributed \

jamtur01/distributed_client

这次，在moe上运行了jamtur01/distributed_client镜像，并将容器命名

为moe_distributed。现在来看看容器输出的日志，看一下分布式客户端是不是找

到了Web应用工作进程的相关信息，如代码清单7-64所示。



图7-5　Consul网页界面上的更多distributed_app服务

代码清单7-64　moe上的distributed_client日志

moe$ sudo docker logs moe_distributed

Application distributed_app with element server2 on port 2002 found on node larry (104.131.38.54).

We can also resolve DNS - distributed_app resolves to 104.131.38.55 and 104.131.38.54.

Application distributed_app with element server1 on port 2001 found on node larry (104.131.38.54).

We can also resolve DNS - distributed_app resolves to 104.131.38.54 and 104.131.38.55.

Application distributed_app with element server2 on port 2002 found on node curly (104.131.38.55).

We can also resolve DNS - distributed_app resolves to 104.131.38.55 and 104.131.38.54.

Application distributed_app with element server1 on port 2001 found on node curly (104.131.38.55).

从这个日志可以看到，应用distributed_client查询了HTTP API，找到了关于

distributed_app及其server1和server2工作进程的服务项，这两个服务项分别

运行在larry和curly上。客户端还通过DNS查找到运行该服务的节点的IP地

址104.131.38.54和104.131.38.55。

在真实的分布式应用程序里，客户端和工作进程可以通过这些信息在分布式应用的

节点间进行配置、连接、分派消息。这提供了一种简单、方便且有弹性的方法来构

建分离的Docker容器和宿主机里运行的分布式应用程序。

本书由「ePUBw.COM」整理，ePUBw.COM 提供最新最全的优质

电子书下载！！！





7.3　Docker Swarm

Docker Swarm是一个原生的Docker集群管理工具。Swarm将一组Docker主机作为

一个虚拟的Docker主机来管理。Swarm有一个非常简单的架构，它将多台Docker主

机作为一个集群，并在集群级别上以标准Docker API的形式提供服务。这非常强

大，它将Docker容器抽象到集群级别，而又不需要重新学习一套新的API。这也使

得Swarm非常容易和那些已经集成了Docker的工具再次集成，包括标准的Docker客

户端。对Docker客户端来说，Swarm集群不过是另一台普通的Docker主机而已。

Swarm也像其他Docker工具一样，遵循了类似笔记本电池一样的设计原则，虽然自

带了电池，但是也可以选择不使用它。这意味着，Swarm提供了面向简单应用场景的

工具以及后端集成，同时提供了API（目前还处于成长期）用于与更复杂的工具及应

用场景进行集成。Swarm基于Apache 2.0许可证发布，可以在GitHub[21]上找到它

的源代码。

注意 Swarm仍是一个新项目，它的基本雏形已现，但是亦可以期待随着项目的进化，它可以开发和演化更多

的API。可以在GitHub上找到它的发展蓝图[22]。

7.3.1　安装Swarm

安装Swarm最简单的方法就是使用Docker自己。我知道这听起来可能有一点儿超

前，但是Docker公司为Swarm提供了一个实时更新的Docker镜像，可以轻易下载并

运行这个镜像。我们这里也将采用这种安装方式。

因此，Swarm没有像第2章那样需要很多前提条件。这里我们假设读者已经按照第2

章的指导安装好了Docker。





要想支持Swarm，Docker有一个最低的版本。用户的所有Docker主机都必须在

1.4.0或者更高版本之上。此外，运行Swarm的所有Docker节点也都必须运行着同一

个版本的Docker。不能混合搭配不同的版本，比如应该让Docker上的每歌节点都运

行在1.6.0之上，而不能混用1.5.0版本和1.6.0版本的节点。

我们将在两台主机上安装Swarm，这两台主机分别为smoker和joker。smoker的主

机IP是10.0.0.125，joker的主机IP是10.0.0.135。两台主机都安装并运行着最

新版本的Docker。

让我们先从smoker主机开始，在其上拉取swarm镜像，如代码清单7-65所示。

代码清单7-65　在smoker上拉取Docker Swarm镜像

smoker$ sudo docker pull swarm

之后再到joker上做同样的操作，如代码清单7-66所示。

代码清单7-66　在joker上拉取Docker Swarm镜像

joker$ sudo docker pull swarm

我们可以确认一下Swarm镜像是否下载成功，如代码清单7-67所示。

代码清单7-67　查看Swarm镜像

$ docker images swarm

REPOSITORY　　TAG　　 IMAGE ID　　　CREATED　　　VIRTUAL SIZE

swarm　　　　　latest bf8b6923851d 6 weeks ago 7.19 MB

7.3.2　创建Swarm集群

我们已经在两台主机上下载了swarm镜像，之后就可以创建Swarm集群了。集群中的

每台主机上都运行着一个Swarm节点代理。每个代理都将该主机上的相关Docker守

护进程注册到集群中。和节点代理相对的是Swarm管理者，用于对集群进行管理。

集群注册可以通过多种可能的集群发现后端（discovery backend）来实现。默认

的集群发现后端是基于Docker Hub。它允许用户在Docker Hub中注册一个集群，

然后返回一个集群ID，我们之后可以使用这个集群ID向集群添加额外的节点。

提示 其他的集群发现后端包括etcd、Consul和Zookeeper，甚至是一个IP地址的静态列表。我们能使用之前

创建的Consule集群为Docker Swarm集群提供发现方式。可以在https://docs.

ocker.com/swarm/discovery/获得更多关于集群发现的说明。

这里我们使用默认的Docker Hub作为集群发现服务创建我们的第一个Swarm集群。

我们还是在smoker主机上创建Swarm集群，如代码清单7-68所示。

代码清单7-68　创建Docker Swarm

smoker$ sudo docker run --rm swarm create

b811b0bc438cb9a06fb68a25f1c9d8ab

我们看到该命令返回了一个字符串b811b0bc438cb9a06fb68a25f1c9d8ab。这是我

们的集群ID。这是一个唯一的ID，我们能利用这个ID向Swarm集群中添加节点。用

户应该保管好这个ID，并且只有当用户想向集群中添加节点时才拿出来使用。

接着我们在每个节点上运行Swarm代理。让我们从smoker主机开始，如代码清单7-

69所示。

代码清单7-69　在smoker上运行swarm代理

smoker$ sudo docker run -d swarm join --addr=10.0.0.125:2375

token://b811b0bc438cb9a06fb68a25f1c9d8ab

b5fb4ecab5cc0dadc0eeb8c157b537125d37e541d0d96e11956c2903ca69eff0

接着在joker上运行Swarm代理，如代码清单7-70所示。

代码清单7-70　在joker上运行swarm代理

joker$ sudo docker run -d swarm join --addr=10.0.0.135:2375 token

://b811b0bc438cb9a06fb68a25f1c9d8ab

537bc90446f12bfa3ba41578753b63f34fd5fd36179bffa2dc152246f4b449d7

这将创建两个Swarm代理，这些代理运行在运行了swarm镜像的Docker化容器中。我

们通过传递给容器的join标志，通过—addr``选项传递的本机IP地址，以及代表集

群ID的token，启动一个代理。每个代理都会绑定到它们所在主机的IP地址上。每

个代理都会加入Swarm集群中去。

提示 像Docker一样，用户也可以让自己的Swarm通过TLS和Docker节点进行连接。我们将会在第8章介绍如何

配置Docker来使用TLS。

我们可以通过查看代理容器的日志来了解代理内部是如何工作的，如代码清单7-71

所示。

代码清单7-71　查看smoker代理的日志

smoker$ docker logs b5fb4ecab5cc

time="2015-04-12T17:54:35Z" level=info msg="Registering on the discovery service every 25 seconds..." addr="10.0.0.125:2375"

discovery="token://b811b0bc438cb9a06fb68a25f1c9d8ab"

time="2015-04-12T17:55:00Z" level=info msg="Registering on the discovery service every 25 seconds..." addr="10.0.0.125:2375"

discovery="token://b811b0bc438cb9a06fb68a25f1c9d8ab"

. . .

我们可以看到，代理每隔25秒就会向发现服务进行注册。这将告诉发现后端Docker

Hub该代理可用，该Docker服务器也可以被使用。

下面我们就来看看集群是如何工作的。我们可以在任何运行着Docker的主机上执行

这一操作，而不一定必须要在Swarm集群的节点中。我们甚至可以在自己的笔记本电

脑上安装好Docker并下载了swarm镜像后，本地运行Swarm集群，如代码清单7-72

所示。

代码清单7-72　列出我们的Swarm节点

$ docker run --rm swarm list token://

b811b0bc438cb9a06fb68a25f1c9d8ab

10.0.0.125:2375

10.0.0.135:2375

这里我们运行了swarm镜像，并指定了list标志以及集群的token。该命令返回了集

群中所有节点的列表。下面让我们来启动Swarm集群管理者。我们可以通过Swarm集

群管理者来对集群进行管理。同样，我们也可以在任何安装了Docker的主机上执行

以下命令，如代码清单7-73所示。

代码清单7-73　启动Swarm集群管理者

$ docker run -d -p 2380:2375 swarm manage token://

b811b0bc438cb9a06fb68a25f1c9d8ab

这将创建一个新容器来运行 Swarm集群管理者。同时我们还将2380端口映射到了

2375端口。我们都知道2375是Docker的标准端口。我们将使用这个端口来和标准





Docker客户端或者API进行交互。我们运行了swarm镜像，并通过指定managè`r选

项来启动管理者，还指定了集群ID。现在我们就可以通过这个管理者来向集群发送

命令了。让我们从在Swarm 集群中运行docker info开始。这里我们通过-H选项来

指定Swarm 集群管理节点的API端点，如代码清单7-74所示。

代码清单7-74　在Swarm集群中运行docker info命令

$ sudo docker -H tcp://localhost:2380 info

Containers: 4

Nodes: 2

joker: 10.0.0.135:2375└

Containers: 2└

Reserved CPUs: 0 / 1└

Reserved Memory: 0 B / 994 MiB

smoker: 10.0.0.125:2375└

Containers: 2└

Reserved CPUs: 0 / 1└

Reserved Memory: 0 B / 994 MiB

我们看到，除了标准的docker info输出之外，Swarm还向我们输出了所有节点信

息。我们可以看到每个节点、节点的IP地址、每台节点上有多少容器在运行，以及

CPU和内存这样的容量信息。

7.3.3　创建容器

现在让我们通过一个小的shell循环操作来创建6个Nginx容器，如代码清单7-75所

示。

代码清单7-75　通过循环创建6个Nginx容器

$ for i in `seq 1 6`;do sudo docker -H tcp://localhost:2380 run -

d --name www-$i -p 80 nginx;done

37d5c191d0d59f00228fbae86f54280ddd116677a7cfcb8be7ff48977206d1e2

b194a69468c03cee9eb16369a3f9b157413576af3dcb78e1a9d61725c26c2ec7

47923801a6c6045427ca49054fb988ffe58e3e9f7ff3b1011537acf048984fe7

90f8bf04d80421888915c9ae8a3f9c35cf6bd351da52970b0987593ed703888f 5bf0ab7ddcd72b11dee9064e504ea6231f9aaa846a23ea65a59422a2161f6ed4

b15bce5e49fcee7443a93601b4dde1aa8aa048393e56a6b9c961438e419455c5

这里我们运行了包装在一个shell循环里的docker run命令。我们通过-H选项为

Docker客户端指定了 tcp://localhost:2380地址，也就是Swarm管理者的地址。

我们告诉Docker以守护方式启动容器，并将容器命名为www-加上一个循环变量$i。

这些容器都是基于nginx镜像创建的，并都打开了80端口。

我们看到上面的命令返回了6个容器的ID，这也是Swarm在集群中启动的6个容器。

让我们来看看这些正在运行中的容器，如代码清单7-76所示。

代码清单7-76　 Swarm在集群中执行docker ps的输出

$ sudo docker -H tcp://localhost:2380 ps

CONTAINER ID IMAGE ... PORTS　　　　　　　　　　　　　　　　　NAMES

b15bce5e49fc nginx　　　443/tcp,10.0.0.135:49161->80/tcp joker/www-6

47923801a6c6 nginx　　　443/tcp,10.0.0.125:49158->80/tcp smoker/www-3

5bf0ab7ddcd7 nginx　　　443/tcp,10.0.0.135:49160->80/tcp joker/www-5

90f8bf04d804 nginx　　　443/tcp,10.0.0.125:49159->80/tcp smoker/www-4

b194a69468c0 nginx　　　443/tcp,10.0.0.135:49157->80/tcp joker/www-2

37d5c191d0d5 nginx　　　443/tcp,10.0.0.125:49156->80/tcp smoker/www-1

注意 这里我们省略了输出中部分列的信息，以节省篇幅，包括容器启动时运行的命令、容器当前状态以及容

器创建的时间。





我们可以看到我们已经运行了docker ps命令，但它不是在本地Docker守护进程

中，而是跨Swarm集群运行的。我们看到结果中有6个容器在运行，平均分配在集群

的两个节点上。那么，Swarm是如何决定容器应该在哪个节点上运行呢？

Swarm根据过滤器（filter）和策略（strategy）的结合来决定在哪个节点上运行

容器。

7.3.4　过滤器

过滤器是告知Swarm该优先在哪个节点上运行容器的明确指令。

目前Swarm具有如下5种过滤器：

约束过滤器（constraint filter）；

亲和过滤器（affinity filter）；

依赖过滤器（dependency filter）；

端口过滤器（port filter）；

健康过滤器（health filter）。

下面我们就来逐个了解一下这些过滤器。

1．约束过滤器

约束过滤器依赖于用户给各个节点赋予的标签。举例来说，用户想为使用特殊存储

类型或者指定操作系统的节点来分组。约束过滤器需要在启动Docker守护进程时，

设置键值对标签，通过—label标注来设置，如代码清单7-77所示。

代码清单7-77　运行Docker守护进程时设置约束标签

$ sudo docker daemon --label datacenter=us-east1

Docker还提供了一些Docker守护进程启动时标准的默认约束，包括内核版本、操作

系统、执行驱动（execution driver）和存储驱动（storage driver）。如果我

们将这个Docker实例加入Swarm集群，就可以通过代码清单7-78所示的方式在容器

启动时选择这个Docker实例。

代码清单7-78　启动容器时指定约束过滤器

$ sudo docker -H tcp://localhost:2380 run -e constraint:

datacenter==us-east1 -d --name www-use1 -p 80 nginx

这里我们启动了一个名为 www-use1的容器，并通过-e选项指定约束条件，这里用

来匹配datacenter==us-east1。这样将会在设置了这个标签的Docker守护进程中

启动该容器。这个约束过滤器支持相等匹配==和不等匹配!=，也支持使用正则表达

式，如代码清单7-79所示。

代码清单7-79　启动容器时在约束过滤器中使用正则表达式

$ sudo docker -H tcp://localhost:2380 run -e constraint:

datacenter==us-east* -d --name www-use1 -p 80 nginx

这会在任何设置了datacenter标签并且标签值匹配us-east*的Swarm节点上启动容

器。

2．亲和过滤器

亲和过滤器让容器运行更互相接近，比如让容器web1挨着haproxy1容器或者挨着指

定ID的容器运行，如代码清单7-80所示。

代码清单7-80　启动容器时指定亲和过滤器

$ sudo docker run -d --name www-use2 -e affinity:container==www-use1

nginx

这里我们通过亲和过滤器启动了一个容器，并告诉这个容器运行在www-use1容器所

在的Swarm节点上。我们也可以使用不等于条件，如代码清单7-81所示。

代码清单7-81　启动容器时在亲和过滤器中使用不等于条件

$ sudo docker run -d --name db1 -e affinity:container!=www-use1

mysql

读者会看到这里我们在亲和过滤器中使用了!=比较操作符。这将告诉Docker在任何

没有运行www-use1容器的Swarm节点上运行这个容器。

我们也能匹配已经拉取了指定镜像的节点，如affinity :image==nginx将会让容

器在任何已经拉取了nginx镜像的节点上运行。或者，像约束过滤器一样，我们也可

以通过按名字或者正则表达式来搜索容器来匹配特定的节点，如代码清单7-82所

示。

代码清单7-82　启动容器时在亲和过滤器中使用正则表达式

$ sudo docker run -d --name db1 -e affinity:container!=www-use*

mysql

3．依赖过滤器

在具备指定卷或容器链接的节点上启动容器。

4．端口过滤器





通过网络端口进行调度，在具有指定端口可用的节点上启动容器，如代码清单7-83

所示。

代码清单7-83　使用端口过滤器

$ sudo docker -H tcp://localhost:2380 run -d --name haproxy -p

80:80 haproxy

5．健康过滤器

利用健康过滤器，Swarm就不会将任何容器调度到被认为不健康的节点上。通常来

说，不健康是指Swarm管理者或者发现服务报告某集群节点有问题。

可以在http://docs.docker.com/swarm/scheduler/filter/查看到Swarm过滤

器的完整列表，以及它们的具体配置。

提示 可以通过为swarm manage命令传递—filter标志来控制哪些过滤器能用。

7.3.5　策略

策略允许用户用集群节点更隐式的特性来对容器进行调度，比如该节点可用资源的

数量等，只在拥有足够内存或者CPU的节点上启动容器。Docker Swarm现在有3种策

略：平铺（Spread）策略、紧凑（BinPacking）策略和随机（Random）策略。但

只有平铺策略和紧凑策略才真正称得上是策略。默认的策略是平铺策略。

可以在执行swarm manage命令时，通过—strategy标志设置用户想选用的策略。

1．平铺策略

平铺策略会选择已运行容器数量最少的节点。使用平铺策略会让所有容器比较平均





地分配到集群中的每个节点上。

2．紧凑策略

紧凑策略会根据每个节点上可用的CPU和内存资源为节点打分，它会先返回使用最紧

凑的节点。这将会保证节点最大程度地被使用，避免碎片化，并确保在需要启动更

大的容器时有最大数量的空间可用。

3．随机策略

随机策略会随机选择一个节点来运行容器。这主要用于调试中，生产环境下请不要

使用这种策略。

7.3.6　小结

读者可能希望看到Swarm还是很有潜力的，也有了足够的基础知识来尝试一下

Swam。这里我再次提醒一下，Swarm还处于beta阶段，还不推荐在生产环境中使

用。

本书由「ePUBw.COM」整理，ePUBw.COM 提供最新最全的优质

电子书下载！！！





7.4　其他编配工具和组件

正如前面提到的，Compose和Consul不是Docker编配工具这个家族里唯一的选择。

编配工具是一个快速发展的生态环境，没有办法列出这个领域中的所有可用的工

具。这些工具的功能不尽相同，不过大部分都属于以下两个类型：

调度和集群管理；

服务发现。

注意 本节中列出的服务都在各自的许可下开源了。

7.4.1　Fleet和etcd

Fleet和etcd由CoreOS[23]团队发布。Fleet[24]是一个集群管理工具，而etcd[25]

是一个高可用性的键值数据库，用于共享配置和服务发现。Fleet与systemd和etcd

一起，为容器提供了集群管理和调度能力。可以把Fleet看作是systemd的扩展，只

是不是工作在主机层面上，而是工作在集群这个层面上。

7.4.2　Kubernetes

Kubernetes[26]是由Google开源的容器集群管理工具。这个工具可以使用Docker在

多个宿主机上分发并扩展应用程序。Kubernetes主要关注需要使用多个容器的应用

程序，如弹性分布式微服务。

7.4.3　Apache Mesos

Apache Mesos[27]项目是一个高可用的集群管理工具。Mesos从Mesos 0.20开始，





已经内置了Docker集成，允许利用Mesos使用容器。Mesos在一些创业公司里很流

行，如著名的Twitter和AirBnB。

7.4.4　Helios

Helios[28]项目由Spotify的团队发布，是一个为了在全流程中发布和管理容器而设

计的Docker编配平台。这个工具可以创建一个抽象的“作业”（job），之后可以

将这个作业发布到一个或者多个运行Docker的Helios宿主机。

7.4.5　Centurion

Centurion[29]是一个基于Docker的部署工具，由New Relic团队打造并开源。

Centurion从Docker Registry里找到容器，并在一组宿主机上使用正确的环境变

量、主机卷映射和端口映射来运行这个容器。这个工具的目的是帮助开发者利用

Docker做持续部署。

本书由「ePUBw.COM」整理，ePUBw.COM 提供最新最全的优质

电子书下载！！！





7.5　小结

本章介绍了如何使用Compose进行编配工作，展示了如何添加一个Compose配置文件

来创建一个简单的应用程序栈，还展示了如何运行Compose并构建整个栈，以及如

何用Compose完成一些基本的管理工作。

本章还展示了服务发现工具Consul，介绍了如何将Consul安装到Docker以及如何

创建Consul节点集群，还演示了在Docker上简单的分布式应用如何工作。

我们还介绍了Docker自己的集群和调度工具Docker Swarm。

我们学习了如何安装Swarm，如何对Swarm进行管理，以及如何在Swarm集群间进行

任务调度。

本章最后展示了可以用在Docker生态环境中的其他编配工具。

下一章会介绍Docker API，如何使用这些API，以及如何通过TLS与Docker守护进

程建立安全的链接。

[1] 　 https://github.com/jamtur01/dockerbook-code/tree/master/code/7/composeapp

[2] 　 http://www.dockerbook.com/code/

[3] 　 https://docs.docker.com/compose/yml/

[4] 　 https://docs.docker.com/compose/rails/

[5] 　 https://docs.docker.com/compose/django/

[6] 　 https://docs.docker.com/compose/wordpress/

[7] 　 https://github.com/shipyard/shipyard

[8] 　 https://docs.docker.com/compose/cli/

[9] 　 http://www.consul.io

[10] 　 http://en.wikipedia.org/wiki/Raft_(computer_science)

[11] 　 https://developers.google.com/speed/public-dns/

[12] 　 http://dockerbook.com/code/7/consul/

[13] 　 https://github.com/jamtur01/dockerbook-code/tree/master/code/7/consul/

[14] 　 http://uwsgi-docs.readthedocs.org/en/latest/

[15] 　 https://github.com/unbit/uwsgi-consul

[16] 　 http://dockerbook.com/code/7/consul/

[17] 　 https://github.com/jamtur01/dockerbook-code/tree/master/code/7/consul/

[18] 　 http://dockerbook.com/code/7/consul/

[19] 　 https://github.com/jamtur01/dockerbook-

code/tree/master/code/7/consul/

[20] 　 http://www.consul.io/docs/agent/services.html

[21] 　 https://github.com/docker/swarm

[22] 　 https://github.com/docker/swarm/blob/master/ROADMAP.md

[23] 　 https://coreos.com/

[24] 　 https://github.com/coreos/fleet

[25] 　 https://github.com/coreos/etcd

[26] 　 https://github.com/GoogleCloudPlatform/kubernetes

[27] 　 http://mesos.apache.org/

[28] 　 https://github.com/spotify/helios

[29] 　 https://github.com/newrelic/centurion 本书由「ePUBw.COM」整理，ePUBw.COM 提供最新最全的优质

电子书下载！！！





第8章　使用Docker API

在第6章中，我们已经学习了很多优秀的例子，关于如何在Docker中运行服务和构

建应用程序，以及以Docker为中心的工作流。TProv应用就是其中一例，它主要以

在命令行中使用docker程序，并且获取标准输出的内容。从与Docker进行集成的角

度来看，这并不是一个很理想的方案，尤其是Docker提供了强大的API，用户完全

可以直接将这些API用于集成。

在本章中，我们将会介绍Docker API，并看看如何使用它。我们已经了解了如何将

Docker守护进程绑定到网络端口，从现在开始我们将会从一个更高的层次对Docker

API进行审视，并抓住它的核心内容。我们还会再回顾一下TProv这个应用，这个应

用我们在第6章里已经见过了，在本章我们会将其中直接使用了docker命令行程序

的部分用Docker API进行重写。最后，我们还会再看一下如何使用TLS来实现API中

的认证功能。

本书由「ePUBw.COM」整理，ePUBw.COM 提供最新最全的优质

电子书下载！！！





8.1　Docker API

在Docker生态系统中一共有3种API[1]。

Registry API：提供了与来存储Docker镜像的Docker Registry集成的功

能。

Docker Hub API：提供了与Docker Hub[2]集成的功能。

Docker Remote API：提供与Docker守护进程进行集成的功能。

所有这3种API都是RESTful[3]风格的。在本章中，我们将会着重对Remote API进行

介绍，因为它是通过程序与Docker进行集成和交互的核心内容。

本书由「ePUBw.COM」整理，ePUBw.COM 提供最新最全的优质

电子书下载！！！





8.2　初识Remote API

让我们浏览一下Docker Remote API，并看看它都提供了哪些功能。首先需要牢记

的是，Remote API是由Docker守护进程提供的。在默认情况下，Docker守护进程

会绑定到一个所在宿主机的套接字，即unix:///var/run/docker.sock。Docker

守护进程需要以root权限来运行，以便它有足够的权限去管理所需要的资源。也正

如在第2章所阐述的那样，如果系统中存在一个名为docker用户组，Docker会将上

面所说的套接字的所有者设为该用户组。因此任何属于docker用户组的用户都可以

运行Docker而无需root权限。

警告 谨记，虽然docker用户组让我们的工作变得更轻松，但它依旧是一个值得注意的安全隐患。可以认

为docker用户组和root具有相当的权限，应该确保只有那些需要此权限的用户和应用程序才能使用该用户

组。

如果我们只查询在同一台宿主机上运行Docker的Remote API，那么上面的机制看起

来没什么问题，但是如果我们想远程访问Remote API，我们就需要将Docker守护进

程绑定到一个网络接口上去。我们只需要给Docker守护进程传递一个-H标志即可做

到这一点。

如果用户可以在本地使用Docker API，那么就可以使用nc命令来进行查询，如代码

清单8-1所示。

代码清单8-1　 在本地查询Docker API

$ echo -e "GET /info HTTP/1.0\r\n" | sudo nc -U /var/run/docker.

sock

在大多数操作系统上，可以通过编辑守护进程的启动配置文件将Docker守护进程绑

定到指定网络接口。对于Ubuntu或者Debian，我们需要编

辑/etc/default/docker文件；对于使用了Upstart的系统，则需要编

辑/etc/init/docker.conf文件；对于Red Hat、Redora及相关发布版本，则需要

编辑/etc/sysconfig/docker文件；对于那些使用了``Systemd的发布版本，则需

要编辑/usr/lib/systemd/system/docker.service文件。

让我们来看看如何在一个运行systemd的Red Hat衍生版上将Docker守护进程绑定

到一个网络接口上。我们将编辑/usr/lib/systemd/system/docker.service文

件，将代码清单8-2所示的内容修改为代码清单8-3所示的内容。

代码清单8-2　默认的Systemd守护进程启动选项

ExecStart=/usr/bin/docker -d --selinux-enabled

代码清单8-3　绑定到网络接口的Systemd守护进程启动选项

ExecStart=/usr/bin/docker -d --selinux-enabled -H tcp://0.0.0.0:2375

这将把Docker守护进程绑定到该宿主机的所有网络接口的2375端口上。之后需要使

用systemctl命令来重新加载并启动该守护进程，如代码清单8-4所示。

代码清单8-4　重新加载和启动Docker守护进程

$ sudo systemctl --system daemon-reload

提示 用户还需要确保任何Docker宿主机上的防火墙或者自己和Docker主机之间的防火墙能允许用户在2375

端口上与该IP地址进行TCP通信。

现在我们可以通过docker客户端命令的-H标志来测试一下刚才的配置是否生效。让

我们从一台远程主机来访问Docker守护进程，如代码清单8-5所示。

代码清单8-5　连接到远程Docker守护进程

$ sudo docker -H docker.example.com:2375 info

Containers: 0

Images: 0

Driver: devicemapper

　Pool Name: docker-252:0-133394-pool

　Data file: /var/lib/docker/devicemapper/devicemapper/data

　Metadata file: /var/lib/docker/devicemapper/devicemapper/metadata

. . .

这里假定Docker所在主机名为docker.example.com，并通过-H标志来指定了该主

机名。Docker提供了更优雅的DOCKER_HOST环境变量（见代码清单8-6），这样就

省掉了每次都需要设置-H标志的麻烦。

代码清单8-6　检查DOCKER_HOST环境变量

$ export DOCKER_HOST="tcp://docker.example.com:2375"

警告 请记住，与Docker守护进程之间的网络连接是没有经过认证的，是对外开放的。在本章的后面，我们将

会看到如何为网络连接加入认证功能。

本书由「ePUBw.COM」整理，ePUBw.COM 提供最新最全的优质

电子书下载！！！





8.3　测试Docker Remote API

现在已经通过docker程序建立并确认了与Docker守护进程之间的网络连通性，接着

我们来试试直接连接到API。为了达到此目的，会用到curl命令。接下来连接

到info API接入点，如代码清单8-7所示，这会返回与docker info命令大致相同

的信息。

代码清单8-7　使用info API接入点

$ curl http://docker.example.com:2375/info

{

　"Containers": 0,

　"Debug": 0,

　"Driver": "devicemapper",

　. . .

　"IPv4Forwarding": 1,

　"Images": 0,

　"IndexServerAddress": "https://index.docker.io/v1/",

　"InitPath": "/usr/libexec/docker/dockerinit",

　"InitSha1": "dafd83a92eb0fc7c657e8eae06bf493262371a7a",

　"KernelVersion": "3.9.8-300.fc19.x86_64",

　"LXCVersion": "0.9.0",

　"MemoryLimit": 1,

　"NEventsListener": 0,

　"NFd": 10,

　"NGoroutines": 14,

　"SwapLimit": 0

}





这里通过curl命令连接到了提供了Docker API的网

址http://docker.example.com:2375，并指定了到 Docker API的路径：主

机docker.example.com上的2375端口，info接入点。

可以看出，API返回的都是JSON散列数据，上面的例子的输出里包括了关于Docker

守护进程的系统信息。这展示出Docker API可以正常工作并返回了一些数据。

8.3.1　通过API来管理Docker镜像

让我们从一些基础的API开始：操作Docker镜像的API。我们将从获取Docker守护

进程中所有镜像的列表开始，如代码清单8-8所示。

代码清单8-8　通过API获取镜像列表

$ curl http://docker.example.com:2375/images/json | python -mjson.tool

[

　{

　　"Created": 1404088258,

　　"Id": "2

　　e9e5fdd46221b6d83207aa62b3960a0472b40a89877ba71913998ad9743e065",

　　"ParentId":"7

cd0eb092704d1be04173138be5caee3a3e4bea5838dcde9ce0504cdc1f24cbb",

　　　　"RepoTags": [

　　　　　　"docker:master"

　　　　],

　　　　"Size": 186470239,

　　　　"VirtualSize": 1592910576

　},

. . .

　{

　　"Created": 1403739688,

　　"Id": "15

d0178048e904fee25354db77091b935423a829f171f3e3cf27f04ffcf7cf56",

　　"ParentId": "74830

af969b02bb2cec5fe04bb2e168a4f8d3db3ba504e89cacba99a262baf48",

　　　　"RepoTags": [

　　　　　　"jamtur01/jekyll:latest"

　　　　],

　　　　"Size": 0,

　　　　"VirtualSize": 607622922

　}

. . .

]

注意 我们已经使用Python的JSON工具对API的返回结果进行了格式化处理。

这里使用了/images/json这个接入点，它将返回Docker守护进程中的所有镜像的

列表。它的返回结果提供了与docker images命令非常类似的信息。我们也可以通

过镜像ID来查询某一镜像的信息，如代码清单8-9所示，这非常类似于使用docker

inspect命令来查看某镜像ID。

代码清单8-9　获取指定镜像

curl <a>http://docker.example.com:2375/images/</a>

15

d0178048e904fee25354db77091b935423a829f171f3e3cf27f04ffcf7cf56/

json | python -mjson.tool

{

　　"Architecture": "amd64",

　　"Author": "James Turnbull <james@example.com>",

　　"Comment": "",

　　"Config": {

　　　　"AttachStderr": false,

　　　　"AttachStdin": false,

　　　　"AttachStdout": false,

　　　　"Cmd": [

　　　　　　"--config=/etc/jekyll.conf"

　　　　],

. . .

}

上面是我们查看jamtur01/jekyll镜像时输出的一部分内容。最后，也像命令行一

样，我们也可以在Docker Hub上查找镜像，如代码清单8-10所示。

代码清单8-10　通过API搜索镜像

$ curl "http://docker.example.com:2375/images/search?term=

jamtur01" | python -mjson.tool

[

　　{

　　　　"description": "",

　　　　"is_official": false,

　　　　"is_trusted": true,

　　　　"name": "jamtur01/docker-presentation",

　　　　"star_count": 2

　　},

　　{

　　　　"description": "",

　　　　"is_official": false,

　　　　"is_trusted": false,

　　　　"name": "jamtur01/dockerjenkins",

　　　　"star_count": 1

　　},





. . .

]

在上面的例子里我们搜索了名字中带jamtur01的所有镜像，并显示了该搜索返回结

果的一部分内容。这只是使用Docker API能完成的工作的一个例子而已，实际上还

能用API进行镜像构建、更新和删除。

8.3.2　通过API管理Docker容器

Docker Remote API也提供了所有在命令行中能使用的对容器的所有操作。我们可

以使用/containers接入点列出所有正在运行的容器，如代码清单8-11所示，就像

使用docker ps命令一样。

代码清单8-11　列出正在运行的容器

$ curl -s "http://docker.example.com:2375/containers/json" |

python -mjson.tool

[

　　{

　　　　"Command": "/bin/bash",

　　　　"Created": 1404319520,

　　　　"Id":

　　　　"cf925ad4f3b9fea231aee386ef122f8f99375a90d47fc7cbe43fac1d962dc51b",

　　　　"Image": "ubuntu:14.04",

　　　　"Names": [

　　　　　　"/desperate_euclid"

　　　　],

　　　　"Ports": [],

　　　　"Status": "Up 3 seconds"

　　}

]

这个查询将会显示出在Docker宿主机上正在运行的所有容器，在这个例子里只有一

个容器在运行。如果想同时列出正在运行的和已经停止的容器，我们可以在接入点

中增加all标志，并将它的值设置为1，如代码清单8-12所示。

代码清单8-12　通过API列出所有容器

http://docker.example.com:2375/containers/json?all=1

我们也可以通过使用POST请求来调用/containers/create 接入点来创建容器，如

代码清单8-13所示。这是用来创建容器的API调用的一个最简单的例子。

代码清单8-13　通过API创建容器

$ curl -X POST -H "Content-Type: application/json" \

http://docker.example.com:2375/containers/create \

-d '{

　　 "Image":"jamtur01/jekyll"

}'

{"Id":"591

ba02d8d149e5ae5ec2ea30ffe85ed47558b9a40b7405e3b71553d9e59bed3",

"Warnings":null}

我们调用了/containers/create接入点，并POST了一个JSON散列数据，这个结构

中包括要启动的镜像名。这个API返回了刚创建的容器的ID，以及可能的警告信息。

这条命令将会创建一个容器。

我们可以在创建新容器的时候提供更多的配置，这可以通过在JSON散列数据中加入

键值对来实现，如代码清单8-14所示。

代码清单8-14　通过API配置容器启动选项

$ curl -X POST -H "Content-Type: application/json" \

"http://docker.example.com:2375/containers/create?name=jekyll" \

-d '{

　　 "Image":"jamtur01/jekyll",

　　 "Hostname":"jekyll"

}'

{"Id":"591

ba02d8d149e5ae5ec2ea30ffe85ed47558b9a40b7405e3b71553d9e59bed3",

"Warnings":null}

上面的例子中我们指定了Hostname键，它的值为jekyll，用来为所要创建的容器设

置主机名。

要启动一个容器，需要使用/containers/start接入点，如代码清单8-15所示。

代码清单8-15　通过API启动容器

$ curl -X POST -H "Content-Type: application/json" \

http://docker.example.com:2375/containers/591

　ba02d8d149e5ae5ec2ea30ffe85ed47558b9a40b7405e3b71553d9e59bed3/start \

-d '{

　　　　 "PublishAllPorts":true

}'

将这两个API组合在一起，就提供了与docker run相同的功能，如代码清单8-16所

示。

代码清单8-16　API等同于docker run命令

$ sudo docker run jamtur01/jekyll

我们也可以通过/containers/接入点来得到刚创建的容器的详细信息，如代码清单

8-17所示。

代码清单8-17 通过API列出所有容器

$ curl <a>http://docker.example.com:2375/containers/</a> 591

　ba02d8d149e5ae5ec2ea30ffe85ed47558b9a40b7405e3b71553d9e59bed3/

　json | python -mjson.tool

{

　　"Args": [

　　　　"build",

　　　　"--destination=/var/www/html"

　　],

. . .

　　　　"Hostname": "591ba02d8d14",

　　　　"Image": "jamtur01/jekyll",

. . .

　　"Id": "591

ba02d8d149e5ae5ec2ea30ffe85ed47558b9a40b7405e3b71553d9e59bed3",

　　"Image": "29

d4355e575cff59d7b7ad837055f231970296846ab58a037dd84be520d1cc31",

. . .

　　"Name": "/hopeful_davinci",

. . .

}

在这里可以看到，我们使用了容器ID查询了我们的容器，并展示了提供给我们的数

据的示例。

本书由「ePUBw.COM」整理，ePUBw.COM 提供最新最全的优质

电子书下载！！！





8.4　改进TProv应用

现在让来看看第6章的TProv应用里所使用的方法。我们来看看用来创建和删除

Docker容器的具体方法，如代码清单8-18所示。

代码清单8-18　旧版本TProv容器启动方法

def get_war(name, url)

　cid = `docker run --name #{name} jamtur01/fetcher #{url} 2>&1`.chop puts cid

　[$?.exitstatus == 0, cid]

end

def create_instance(name)

　cid = `docker run -P --volumes-from #{name} -d -t jamtur01/

tomcat7 2>&1`.chop

　[$?.exitstatus == 0, cid]

end

def delete_instance(cid)

　kill = `docker kill #{cid} 2>&1`

　[$?.exitstatus == 0, kill]

end

注意 可以在本书网站[4]或者GitHub[5]看到之前版本的TProv代码。

很粗糙，不是吗？我们直接使用了docker程序，然后再捕获它的输出结果。从很多

方面来说这都是有问题的，其中最重要的是用户的TProv应用将只能运行在安装了

Docker客户端的机器上。

我们可以使用Docker的客户端库利用Docker API来改善这种问题。在本例中，我们

将使用Ruby Docker-API客户端库[6]。

提示 可以在http://docs.docker.com/reference/api/remote_api_client_libraries/找到可用的

Docker客户端库的完整列表。目前Docker已经拥有了Ruby、Python、Node.JS、Go、Erlang、Java以及其

他语言的库。

让我们先来看看如何建立到Docker API的连接，如代码清单8-19所示。

代码清单8-19　Docker Ruby客户端库

require 'docker'

. . .

module TProv

　class Application < Sinatra::Base

. . .

　　Docker.url = ENV['DOCKER_URL'] || 'http://localhost:2375'

　　Docker.options = {

　　　:ssl_verify_peer => false

　　}

我们通过require指令引入了docker-api这个gem。为了能让程序正确运行，需要

事先安装这个gem，或者把它加到TProv应用的gem specification中去。

之后我们可以用Docker.url方法指定我们想要连接的Docker宿主机的地址。在上面

的代码里，我们用了DOCKER_URL这个环境变量来指定这个地址，或者使用默认

值http://localhost:2375。

我们还通过Docker.options指定了我们想传递给Docker守护进程连接的选项。

我们还可以通过IRB shell在本地来验证我们的设想。现在就来试一试。用户需要

在自己想测试的机器上先安装Ruby，如代码清单8-20所示。这里假设我们使用的事

Fedora宿主机。

代码清单8-20　安装Docker Ruby客户端API

$ sudo yum -y install ruby ruby-irb

. . .

$ sudo gem install docker-api json

. . .

现在我们就可以用irb命令来测试Docker API连接了，如代码清单8-21所示。

代码清单8-21　用irb测试Docker API连接

$ irb

irb(main):001:0> require 'docker'; require 'pp'

=> true

irb(main):002:0> Docker.url = 'http://docker.example.com:2375'

=> "http://docker.example.com:2375"

irb(main):003:0> Docker.options = { :ssl_verify_peer => false }

=> {:ssl_verify_peer=>false}

irb(main):004:0> pp Docker.info

{"Containers"=>9,

"Debug"=>0,

"Driver"=>"aufs",

"DriverStatus"=>[["Root Dir", "/var/lib/docker/aufs"], ["Dirs", "882"]],

"ExecutionDriver"=>"native-0.2",

. . .

irb(main):005:0> pp Docker.version

{"ApiVersion"=>"1.12",

"Arch"=>"amd64",

"GitCommit"=>"990021a",

"GoVersion"=>"go1.2.1",

"KernelVersion"=>"3.8.0-29-generic",

"Os"=>"linux",

"Version"=>"1.0.1"}

. . .

在上面我们启动了irb并且加载了docker（通过require指令）和pp这两个gem，pp

用来对输出进行格式化以方便查看。之后我们调用了Docker.url和

Docker.options两个方法，来设置目的Docker主机地址和我们需要的一些选项

（这里将禁用SSL对等验证，这样就可以在不通过客户端认证的情况下使用TLS）。

之后我们又执行了两个全局方法Docker.info和Docker.version，这两个Ruby客

户端API提供了与docker info及docker version两个命令相同的功能。

现在我们就可以在TProv应用中通过docker-api这个客户端库，来使用API进行容

器管理。让我们来看一下相关代码，如代码清单8-22所示。

代码清单8-22　修改后的TProv的容器管理方法

def get_war(name, url)

　container = Docker::Container.create('Cmd' => url, 'Image' =>

'jamtur01/fetcher', 'name' => name)

　container.start

　container.id

end

def create_instance(name)

　container = Docker::Container.create('Image' => 'jamtur01/tomcat7') container.start('PublishAllPorts' => true, 'VolumesFrom' => name) container.id

end

def delete_instance(cid)

　container = Docker::Container.get(cid)

　container.kill

end

可以看到，我们用Docker API替换了之前使用的docker程序之后，代码变得更清晰

了。我们的get_war方法使用Docker::Container.create和

Docker::Container.start方法来创建和启动我们的jamtur01/fetcher容

器。delete_instance也能完成同样的工作，不过创建的是jamtur01/tomcat7容

器。最后，我们对delete_instance方法进行了修改，首先会通过

Docker::Container.get方法根据参数的容器ID来取得一个容器实例，然后再通过

Docker::Container.kill方法销毁该容器。

注意 读者可以在本书网站[7]或者GitHub[8]上看到改进后的TProv代码。

本书由「ePUBw.COM」整理，ePUBw.COM 提供最新最全的优质

电子书下载！！！





8.5　对Docker Remote API进行认证

我们已经看到了如何连接到Docker Remote API，不过这也意味着任何其他人都能

连接到同样的API。从安全的角度上看，这存在一点儿安全问题。不过值得感谢的

是，自Docker的0.9版本开始Docker Remote API开始提供了认证机制。这种认证

机制采用了TLS/SSL证书来确保用户与API之间连接的安全性。

提示 该认证不仅仅适用于API。通过这个认证，还需要配置Docker客户来支持TLS认证。在本节中我们也将看

到如何对客户端进行配置。

有几种方法可以对我们的连接进行认证，包括使用一个完整的PKI基础设施，我们可

以选择创建自己的证书授权中心（Certificate Authority，CA），或者使用已有

的CA。在这里我们将建立自己的证书授权中心，因为这是一个简单、快速的开始。

警告 这依赖于运行在Docker宿主机上的本地CA。它也不像使用一个完整的证书授权中心那样安全。

8.5.1　建立证书授权中心

我们将快速了解一下创建所需CA证书和密钥（key）的方法，在大多数平台上这都是

一个非常标准的过程。在开始之前，我们需要先确保系统已经安装好了openssl，

如代码清单8-23所示。

代码清单8-23　检查是否已安装openssl

$ which openssl

/usr/bin/openssl

让我们在Docker宿主机上创建一个目录来保存我们的CA和相关资料，如代码清单8-

24所示。

代码清单8-24　创建CA目录

$ sudo mkdir /etc/docker

现在就来创建一个CA。

我们需要先生成一个私钥（private key），如代码清单8-25所示。

代码清单8-25　生成私钥

$ cd /etc/docker

$ echo 01 | sudo tee ca.srl

$ sudo openssl genrsa -des3 -out ca-key.pem

Generating RSA private key, 512 bit long modulus

....++++++++++++

.................++++++++++++

e is 65537 (0x10001)

Enter pass phrase for ca-key.pem:

Verifying - Enter pass phrase for ca-key.pem:

在创建私钥的过程中，我们需要为CA密钥设置一个密码，我们需要牢记这个密码，

并确保它的安全性。在新CA中，我们需要用这个密码来创建并对证书签名。

上面的操作也将创建一个名为ca-key.pem的新文件。这个文件是我们的CA的密钥。

我们一定不能将这个文件透露给别人，也不能弄丢这个文件，因为此文件关系到我

们整个解决方案的安全性。

现在就让我们来创建一个CA证书，如代码清单8-26所示。

代码清单8-26　创建CA证书





$ sudo openssl req -new -x509 -days 365 -key ca-key.pem -out ca.pem Enter pass phrase for ca-key.pem:

You are about to be asked to enter information that will be incorporated into your certificate request.

What you are about to enter is what is called a Distinguished Name or a DN.

There are quite a few fields but you can leave some blank

For some fields there will be a default value,

If you enter '.', the field will be left blank.

-----

Country Name (2 letter code) [AU]:

State or Province Name (full name) [Some-State]:

Locality Name (eg, city) []:

Organization Name (eg, company) [Internet Widgits Pty Ltd]:

Organizational Unit Name (eg, section) []:

Common Name (e.g. server FQDN or YOUR name) []:docker.example.com Email Address []:

这将创建一个名为ca.pem的文件，这也是我们的CA证书。我们之后也会用这个文件

来验证连接的安全性。

现在我们有了自己的CA，让我们用它为我们的Docker服务器创建证书和密钥。

8.5.2　创建服务器的证书签名请求和密钥

我们可以用新CA来为Docker服务器进行证书签名请求（certificate signing

request，CSR）和密钥的签名和验证。让我们从为Docker服务器创建一个密钥开

始，如代码清单8-27所示。

代码清单8-27　创建服务器密钥

$ sudo openssl genrsa -des3 -out server-key.pem

Generating RSA private key, 512 bit long modulus

...................++++++++++++

...............++++++++++++

e is 65537 (0x10001)

Enter pass phrase for server-key.pem:

Verifying - Enter pass phrase for server-key.pem:

这将为我们的服务器创建一个密钥server-key.pem。像前面一样，我们要确保此密

钥的安全性，这是保证我们的Docker服务器安全的基础。

注意 请在这一步设置一个密码。我们将会在正式使用之前清除这个密码。用户只需要在后面的几步中使用该

密码。

接着，让我们创建服务器的证书签名请求（CSR），如代码清单8-28所示。

代码清单8-28　创建服务器CSR

$ sudo openssl req -new -key server-key.pem -out server.csr

Enter pass phrase for server-key.pem:

You are about to be asked to enter information that will be incorporated into your certificate request.

What you are about to enter is what is called a Distinguished Name or a DN.

There are quite a few fields but you can leave some blank

For some fields there will be a default value,

If you enter '.', the field will be left blank.

-----

Country Name (2 letter code) [AU]:

State or Province Name (full name) [Some-State]:

Locality Name (eg, city) []:

Organization Name (eg, company) [Internet Widgits Pty Ltd]:

Organizational Unit Name (eg, section) []:

Common Name (e.g. server FQDN or YOUR name) []:*

Email Address []:

Please enter the following 'extra' attributes

to be sent with your certificate request

A challenge password []:

An optional company name []:

这将创建一个名为server.csr的文件。这也是一个请求，这个请求将为创建我们的

服务器证书进行签名。在这些选项中最重要的是Common Name或CN。该项的值要么

为Docker服务器（即从DNS中解析后得到的结果，比如docker.example.com）的

FQDN（fully qualified domain name，完全限定的域名）形式，要么为*，这将

允许在任何服务器上使用该服务器证书。

现在让我们来对CSR进行签名并生成服务器证书，如代码清单8-29所示。

代码清单8-29　对CSR进行签名

$ sudo openssl x509 -req -days 365 -in server.csr -CA ca.pem \

-CAkey ca-key.pem -out server-cert.pem

Signature ok

subject=/C=AU/ST=Some-State/O=Internet Widgits Pty Ltd/CN=*

Getting CA Private Key

Enter pass phrase for ca-key.pem:

在这里，需要输入CA密钥文件的密码，该命令会生成一个名为server-cert.pem的

文件，这个文件就是我们的服务器证书。

现在就让我们来清除服务器密钥的密码，如代码清单8-30所示。我们不想在Docker

守护进程启动的时候再输入一次密码，因此需要清除它。

代码清单8-30　移除服务器端密钥的密码





$ sudo openssl rsa -in server-key.pem -out server-key.pem Enter pass phrase for server-key.pem:

writing RSA key

现在，让我们为这些文件添加一些更为严格的权限来更好地保护它们，如代码清单

8-31所示。

代码清单8-31　设置Docker服务器端密钥和证书的安全属性

$ sudo chmod 0600 /etc/docker/server-key.pem /etc/docker/server-cert.pem \

/etc/docker/ca-key.pem /etc/docker/ca.pem

8.5.3　配置Docker守护进程

现在我们已经得到了我们的证书和密钥，让我们配置Docker守护进程来使用它们。

因为我们会在Docker守护进程中对外提供网络套接字服务，因此需要先编辑它的启

动配置文件。和之前一样，对于Ubuntu或者Debian系统，我们需要编

辑/etc/default/docker文件；对于使用了Upstart的系统，则需要编

辑/etc/init/docker.conf文件；对于Red Hat、Fedora及相关发布版本，则需要

编辑/etc/sysconfig/docker文件；对于那些使用了Systemd的发布版本，则需要

编辑/usr/lib/systemd/docker.service文件。

这里我们仍然使用运行Systemd的Red Hat衍生版本为例继续说明。编辑/usr/``

``lib/systemd/system/docker.service文件内容，如代码清单8-32所示。

代码清单8-32　在Systemd中启用Docker TLS

ExecStart=/usr/bin/docker -d -H tcp://0.0.0.0:2376 --tlsverify

　--tlscacert=/etc/docker/ca.pem --tlscert=/etc/docker/server-cert.pem





　--tlskey=/etc/docker/server-key.pem

注意 可以看到，这里我们使用了2376端口，这是Docker中TLS/SSL的默认端口号。对于非认证的连接，只能

使用2375这个端口。

这段代码通过使用--tlsverify``标志来启用``TLS``。我们还使用``--

tlscacert``、``--tlscert``和``—tlskey``这``3``个参数指定了CA证书、证

书和密钥的位置。关于TLS还有很多其他选项可以使用，请参

考http://docs.docker.com/articles/https/。

提示 可以使用--tls标志来只启用TLS，而不启用客户端认证功能。

然后我们需要重新加载并启动Docker守护进程，这可以使用systemctl命令来完

成，如代码清单8-33所示。

代码清单8-33　重新加载并启动Docker守护进程

$ sudo systemctl --system daemon-reload

8.5.4　创建客户端证书和密钥

我们的服务器现在已经启用了TLS；接下来，我们需要创建和签名证书和密钥，以保

证我们Docker客户端的安全性。让我们先从创建客户端密钥开始，如代码清单8-34

所示。

代码清单8-34　创建客户端密钥

$ sudo openssl genrsa -des3 -out client-key.pem

Generating RSA private key, 512 bit long modulus

..........++++++++++++

.......................................++++++++++++

e is 65537 (0x10001)

Enter pass phrase for client-key.pem:

Verifying - Enter pass phrase for client-key.pem:

这将创建一个名为client-key.pem的密钥文件。我们同样需要在创建阶段设置一个

临时性的密码。

现在让我们来创建客户端CSR，如代码清单8-35所示。

代码清单8-35　创建客户端CSR

$ sudo openssl req -new -key client-key.pem -out client.csr

Enter pass phrase for client-key.pem:

You are about to be asked to enter information that will be incorporated into your certificate request.

What you are about to enter is what is called a Distinguished Name or a DN.

There are quite a few fields but you can leave some blank

For some fields there will be a default value,

If you enter '.', the field will be left blank.

-----

Country Name (2 letter code) [AU]:

State or Province Name (full name) [Some-State]:

Locality Name (eg, city) []:

Organization Name (eg, company) [Internet Widgits Pty Ltd]:

Organizational Unit Name (eg, section) []:

Common Name (e.g. server FQDN or YOUR name) []:

Email Address []:

Please enter the following 'extra' attributes

to be sent with your certificate request

A challenge password []:

An optional company name []:





接下来，我们需要通过添加一些扩展的SSL属性，来开启我们的密钥的客户端身份认

证，如代码清单8-36所示。

代码清单8-36　添加客户端认证属性

$ echo extendedKeyUsage = clientAuth > extfile.cnf

现在让我们在自己的CA中对客户端CSR进行签名，如代码清单8-37所示。

代码清单8-37　对客户端CSR进行签名

$ sudo openssl x509 -req -days 365 -in client.csr -CA ca.pem \

-CAkey ca-key.pem -out client-cert.pem -extfile extfile.cnf

Signature ok

subject=/C=AU/ST=Some-State/O=Internet Widgits Pty Ltd

Getting CA Private Key

Enter pass phrase for ca-key.pem:

我们再使用CA密钥的密码创建另一个证书：client-cert.pem。

最后，我们需要清除client-cert.pem文件中的密码，以便在Docker客户端中使用

该文件，如代码清单8-38所示。

代码清单8-38　移除客户端密钥的密码

$ sudo openssl rsa -in client-key.pem -out client-key.pem

Enter pass phrase for client-key.pem:

writing RSA key

8.5.5　配置Docker客户端开启认证功能

接下来，配置我们的Docker客户端来使用我们新的TLS配置。之所以需要这么做，

是因为Docker守护进程现在已经准备接收来自客户端和API的经过认证的连接。

我们需要将ca.pem、client-cert.pem和client-key.pem这3个文件复制到想运行

Docker客户端的宿主机上。

提示 请牢记，有了这些密钥就能以root身份访问Docker守护进程，应该妥善保管这些密钥文件。

让我们把它们复制到.docker目录下，这也是Docker查找证书和密钥的默认位置。

Docker默认会查找key.pem、cert.pem和我们的CA证书ca.pem，如代码清单8-39

所示。

代码清单8-39　复制Docker客户端的密钥和证书

$ mkdir -p ~/.docker/

$ cp ca.pem ~/.docker/ca.pem

$ cp client-key.pem ~/.docker/key.pem

$ cp client-cert.pem ~/.docker/cert.pem

$ chmod 0600 ~/.docker/key.pem ~/.docker/cert.pem

现在来测试从客户端到Docker守护进程的连接。要完成此工作，我们将使用docker

info命令，如代码清单8-40所示。

代码清单8-40　测试TLS认证过的连接

$ sudo docker -H=docker.example.com:2376 --tlsverify info

Containers: 33

Images: 104

Storage Driver: aufs

Root Dir: /var/lib/docker/aufs

Dirs: 170

Execution Driver: native-0.1

Kernel Version: 3.8.0-29-generic

Username: jamtur01

Registry: [https://index.docker.io/v1/]

WARNING: No swap limit support

可以看到��我们已经指定了-H标志来告诉客户端要连接到哪台主机。如果不想在每

次启动Docker客户端时都指定-H标志，那么可以使用DOCKER_HOST环境变量。另

外，我们也指定了--tlsverify标注，它使我们通过TLS方式连接到Docker守护进

程。我们不需要指定任何证书或者密钥文件，因为Docker会自己在我们的

~/.docker/目录下查找这些文件。如果确实需要指定这些文件，则可以使用--

tlscacert、--tlscert和--tlskey标志来指定这些文件的位置。

如果不指定TLS连接将会怎样呢？让我们去掉--tlsverify标志后再试一下，入代码

清单8-41所示。

代码清单8-41　测试TLS连接过的认证

$ sudo docker -H=docker.example.com:2376 info

2014/04/13 17:50:03 malformed HTTP response "\x15\x03\x01\x00\x02\x02"

哦，出错了。如果看到这样的错误，用户就应该知道自己可能是没有在连接上启用

TLS，可能是没有指定正确的TLS配置，也可能是用户的证书或密钥不正确。

如果一切都能正常工作，现在就有了一个经过认证的Docker连接了。

本书由「ePUBw.COM」整理，ePUBw.COM 提供最新最全的优质

电子书下载！！！





8.6　小结

在这一章中我们介绍了Docker Remote API。我们还了解了如何通过SSL/TLS证书

来保护Docker Remote API；研究了Docker API，以及如何使用它来管理镜像和

容器；看到了如何使用Docker API客户端库之一来改写我们的TProv应用，让该程

序直接使用Docker API。

在下一章也就是最后一章中，我们将讨论如何对Docker做出贡献。

[1] 　 http://docs.docker.com/reference/api/

[2] 　 http://hub.docker.com

[3] 　 http://en.wikipedia.org/wiki/Representational_state_transfer

[4] 　 http://dockerbook.com/code/6/tomcat/tprov/

[5] 　 https://github.com/jamtur01/dockerbook-code/tree/master/code/6/tomcat/tprov

[6] 　 https://github.com/swipely/docker-api

[7] 　 http://dockerbook.com/code/8/tprov_api/

[8] 　 https://github.com/jamtur01/dockerbook-code/tree/master/code/8/tprov_api

本书由「ePUBw.COM」整理，ePUBw.COM 提供最新最全的优质

电子书下载！！！





第9章　获得帮助和对Docker进行改

进

Docker目前还处在婴儿期，还会经常出错。本章将会讨论如下内容。

如何以及从哪里获得帮助。

向Docker贡献补丁和新特性。

读者会发现在哪里可以找到Docker的用户，以及寻求帮助的最佳途径。读者还会学

到如何参与到Docker的开发者社区：在Docker开源社区有数百提交者，他们贡献了

大量的开发工作。如果对Docker感到兴奋，为Docker项目做出自己的贡献是很容易

的。本章还会介绍关于如何贡献Docker项目，如何构建一个Docker开发环境，以及

如何建立一个良好的pull request的基础知识。

注意 本章假设读者都具备Git、GitHub和Go语言的基本知识，但不要求读者一定是特别精通这些知识的开发

者。

本书由「ePUBw.COM」整理，ePUBw.COM 提供最新最全的优质

电子书下载！！！





9.1　获得帮助

Docker的社区庞大且友好。大多数Docker用户都集中使用下面3节中介绍的3种方

式。

注意 Docker公司也提供了对企业的付费Docker支持。可以在支持页面看到相关信息。

9.1.1　Docker用户、开发邮件列表及论坛

Docker用户和开发邮件列表具体如下。

Docker用户邮件列表[1]。

Docker开发者邮件列表[2]。

Docker用户列表一般都是关于Docker的使用方法和求助的问题。Docker开发者列

表则更关注与开发相关的疑问和问题。

还有Docker论坛[3]可用。

9.1.2　IRC上的Docker

Docker社区还有两个很强大的IRC频道：#docker和#docker-dev。这两个频道都

在Freenode IRC网络[4]上。

#docker频道一般也都是讨论用户求助和基本的Docker问题的，而#docker-dev都

是Docker贡献者用来讨论Docker源代码的。

可以在https://botbot.me/freenode/docker/查看#docker频道的历史信息，





在https:// botbot. me/freenode/docker-dev/查看#docker-dev频道的历史

信息。

9.1.3　GitHub上的Docker

Docker（和它的大部分组件以及生态系统）都托管在

GitHub（http://www.github.com）上。Docker本身的核心仓库

在https://github.com/docker/docker/。

其他一些要关注的仓库如下。

distribution[5]：能独立运行的Docker Registry分发工具。

runc[6]：Docker容器格式和CLI工具。

Docker Swarm[7]：Docker的编配框架。

Docker Compose[8]：Docker Compose工具。

本书由「ePUBw.COM」整理，ePUBw.COM 提供最新最全的优质

电子书下载！！！





9.2　报告Docker的问题

让我们从基本的提交问题和补丁以及与Docker社区进行互动开始。在提交Docker

问　题[9]的时候，要牢记我们要做一个良好的开源社区公民，为了帮助社区解决你

的问题，一定要提供有用的信息。当你描述一个问题的时候，记住要包含如下背景

信息：

docker info和docker version命令的输出；

uname -a命令的输出。

然后还需要提供关于你遇到的问题的具体说明，以及别人能够重现该问题的详细步

骤。

如果你描述的是一个功能需求，那么需要仔细解释你想要的是什么以及你希望它将

是如何工作的。请仔细考虑更通用的用例：你的新功能只能帮助你自己，还是能帮

助每一个人？

在提交新问题之前，请花点儿时间确认问题库里没有和你的bug报告或者功能需求一

样的问题。如果已经有类似问题了，那么你就可以简单地添加一个“+1”或者“我

也有类似问题”的说明，如果你觉得你的输入能加速建议的实现或者bug修正，你可

以添加额外的有实际意义的更新。

本书由「ePUBw.COM」整理，ePUBw.COM 提供最新最全的优质

电子书下载！！！





9.3　搭建构建环境

为了使为Docker做出贡献更容易，我们接下来会介绍如何构建一个Docker开发环

境。这个开发环境提供了所有为了让Docker工作而必需的依赖和构建工具。

9.3.1　安装Docker

为了建立开发环境，用户必须先安装Docker，因为构建环境本身就在一个Docker容

器里面。我们将使用Docker来构建和开发Docker。请参照第2章的内容安装

Docker，应该安装当前最新版的Docker。

9.3.2　安装源代码和构建工具

接着，需要安装Make和Git，这样就可以检出Docker的源代码并且运行构建过程。

Docker的源代码都保存在GitHub上，而构建过程则围绕着Makefile来进行。

在Ubuntu上，使用代码清单9-1所示的命令安装git包。

代码清单9-1　在Ubuntu上安装git

$ sudo apt-get -y install git make

在Red Hat及其衍生版本上使用代码清单9-2所示的命令。

代码清单9-2　在Red Hat及其相关衍生版本上安装git

$ sudo yum install git make

9.3.3　检出源代码





现在让我们检出（check out）Docker的源代码（如果是在Docker其他模块上工

作，请选择对应的源代码仓库），并换到源代码所在目录，如代码清单9-3所示。

代码清单9-3　Check out Docker源代码

$ git clone <a>https://github.com/docker/docker.git</a> $ cd docker

现在就可以在Docker源代码上进行工作和修正bug、更新文档或者编写非常棒的新

功能了。

9.3.4　贡献文档

让人兴奋的是，任何人，即使他不是开发者或者不精通Go语言，都可以通过更新、

增强或编写新文档的方式为向Docker做出贡献。Docker文档[10]都在Docker官方网

站上。文档的源代码、主题以及用来生成官方文档网站的工具都保存在 在GitHub

上的Docker仓库[11]中。

可以在https://github.com/docker/docker/blob/master/docs/README.md找

到关于Docker文档的具体指导方针和基本风格指南。

可以在本地使用Docker本身来构建整个文档。

在对文档源代码进行了一些修改之后，可以使用make命令来构建文档，如代码清单

9-4所示。

代码清单9-4　构建Docker文档

$ cd docker





$ make docs

...

docker run --rm -it　-e AWS_S3_BUCKET -p 8000:8000 "docker-docs:master"

　mkdocs serve

Running at: http://0.0.0.0:8000/

Live reload enabled.

Hold ctrl+c to quit.

之后就可以在浏览器中打开8080端口来查看本地版本的Docker文档了。

9.3.5　构建开发环境

如果不只是满足于为Docker的文档做出贡献，可以使用make和Docker来构建一个开

发环境，如代码清单9-5所示。在Docker的源代码中附带了一个Dockerfile文件，

我们使用这个文件来安装所有必需的编译和运行时依赖，来构建和测试Docker。

代码清单9-5　构建Docker环境

$ sudo make build

提示 如果是第一次执行这个命令，要完成这个过程将会花费较长的时间。

上面的命令会创建一个完整的运行着的Docker开发环境。它会将当前的源代码目录

作为构建上下文（build context）上传到一个Docker镜像，这个镜像包含了Go和

其他所有必需的依赖，之后会基于这个镜像启动一个容器。

使用这个开发镜像，也可以创建一个Docker可执行程序来测试任何bug修正或新功

能，如代码清单9-6所示。这里我们又用到了make工具。

代码清单9-6　构建Docker可执行程序

$ sudo make binary

这条命令将会创建Docker可执行文件，该文件保存

在./bundles/&lt;version&gt;-dev/binary/卷中。比如，在这个例子里我们得

到的结果如代码清单9-7所示。

代码清单9-7　dev版本的Docker dev可执行程序

$ ls -l ~/docker/bundles/1.0.1-dev/binary/docker

lrwxrwxrwx 1 root root 16 Jun 29 19:53 ~/docker/bundles/1.7.1-dev/binary/

docker -> docker-1.7.1-dev

之后就可以使用这个可执行程序进行测试了，方法是运行它而不是运行本地Docker

守护进程。为此，我们需要先停止之前的Docker然后再运行这个新的Docker可执行

程序，如代码清单9-8所示。

代码清单9-8　使用开发版的Docker守护进程

$ sudo service docker stop

$ ~/docker/bundles/1.7.1-dev/binary/docker -d

这会以交互的方式运行开发版本Docker守护进程。但是，如果你愿意的话也可以将

守护进程放到后台。

接着我们就可以使用新的Docker可执行程序来和刚刚启动的Docker守护进程进行交

互操作了，如代码清单9-9所示。

代码清单9-9　使用开发版的docker可执行文件

$ ~/docker/bundles/1.7.1-dev/binary/docker version

Client version: 1.7.1-dev





Client API version: 1.19

Go version (client): go1.2.1

Git commit (client): d37c9a4

Server version: 1.7.1-dev

Server API version: 1.19

Go version (server): go1.2.1

Git commit (server): d37c9a

可以看到，我们正在运行版本为1.0.1-dev的客户端，这个客户端正好和我们刚启

动的1.0.1-dev版本的守护进程相对应。可以通过这种组合来测试和确保对Docker

所做的所有修改都能正常工作。

9.3.6　运行测试

在提交贡献代码之前，确保所有的Docker测试都能通过也是非常重要的。为了运行

所有Docker的测试，需要执行代码清单9-10所示的命令。

代码清单9-10　运行Docker测试

$ sudo make test

这条命令也会将当前代码作为构建上下文上传到镜像并创建一个新的开发镜像。之

后会基于此镜像启动一个容器，并在该容器中运行测试代码。同样，如果是第一次

做这个操作，那么也将会花费一些时间。

如果所有的测试都通过的话，那么该命令输出的最后部分看起来会如代码清单9-11

所示。

代码清单9-11　Dcoker测试输出结果





...

[PASSED]: save - save a repo using stdout

[PASSED]: load - load a repo using stdout

[PASSED]: save - save a repo using -o

[PASSED]: load - load a repo using -i

[PASSED]: tag - busybox -> testfoobarbaz

[PASSED]: tag - busybox's image ID -> testfoobarbaz

[PASSED]: tag - busybox fooo/bar

[PASSED]: tag - busybox fooaa/test

[PASSED]: top - sleep process should be listed in non privileged mode

[PASSED]: top - sleep process should be listed in privileged mode

[PASSED]: version - verify that it works and that the output is properly formatted

PASS

PASS　　github.com/docker/docker/integration-cli　　178.685s

提示 可以在测试运行时通过$TESTFLAGS环境变量来传递参数。

9.3.7　在开发环境中使用Docker

也可以在新构建的开发容器中启动一个交互式会话，如代码清单9-12所示。

代码清单9-12　启动交互式会话

$ sudo make shell

要想从容器中退出，可以输入exit或者Ctrl+D。

9.3.8　发起pull request

如果对自己所做的文档更新、bug修正或者新功能开发非常满意，你就可以在

GitHub上为你的修改提交一个pull request。为了提交pull request，需要已经

fork了Docker仓库，并在你自己的功能分支上进行修改。

如果是一个bug修正分支，那么分支名为XXXX-something，这里的XXXX为该问

题的编号。

如果是一个新功能开发分支，那么需要先创建一个新功能问题宣布你都要干什

么，并将分支命名为XXXX-something，这里的XXXX也是该问题的编号。

你必须同时提交针对你所做修改的单元测试代码。可以参考一下既有的测试代码来

寻找一些灵感。在提交pull request之前，你还需要在自己的分支上运行完整的测

试集。

任何包含新功能的pull request都必须同时包括更新过的文档。在提交pull

request之前，应该使用上面提到的流程来测试你对文档所做的修改。当然你也需

要遵循一些其他的使用指南（如上面提到的）。

我们有以下一些简单的规则，遵守这些规则有助于你的pull request会尽快被评审

（review）和合并。

在提交代码之前必须总是对每个被修改的文件运行gofmt -s -w file.go。这

将保证代码的一致性和整洁性。

pull request的描述信息应该尽可能清晰，并且包括到该修改解决的所有问题

的引用。

pull request不能包括来自其他人或者分支的代码。

提交注释（commit message）必须包括一个以大写字母开头且长度在50字符之

内的简明扼要的说明，简要说明后面可以跟一段更详细的说明，详细说明和简

要说明之间需要用空行隔开。

通过git rebase -i和git push -f尽量将你的提交集中到一个逻辑可工作单

元。同时对文档的修改也应该放到同一个提交中，这样在撤销（revert）提交

时，可以将所有与新功能或者bug修正相关的信息全部删除。

最后需要注意的是，Docker项目采用了开发者原产证明书（Developer

Certificate of Origin，DCO）机制，以确认你所提交的代码都是你自己写的或

者你有权将其以开源的方式发布。你可以阅读一篇文章[12]来了解一下我们为什么要

这么做。应用这个证书非常简单，你需要做的只是在每个Git提交消息中添加如代码

清单9-13所示的一行而已。

代码清单9-13　Docker DCO

Docker-DCO-1.1-Signed-off-by: Joe Smith <joe.smith@email.com> (github: github_handle)

注意 用户必须使用自己的真实姓名。出于法律考虑，我们不允许假名或匿名的贡献。

关于签名（signing）的需求，这里也有几个小例外，具体如下。

你的补丁修改的是拼写或者语法错误。

你的补丁只修改了docs目录下的文档的一行。

你的补丁修改了docs目录下的文档中的Markdown格式或者语法错误。

还有一种对Git提交进行签名的更简单的方式是使用git commit -s命令。

注意 老的Docker-DCO-1.1-Signed-off-by方式现在还能继续使用，不过在以后的贡献中，还是请使用这种

方法。





9.3.9　批准合并和维护者

在提交了pull request之后，首先要经过评审，你也可能会收到一些反馈。

Docker采用了与Linux内核维护者类似的机制。Docker的每个组件都有一个或者若

干个维护者，维护者负责该组件的质量、稳定性以及未来的发展方向。维护者的背

后则是仁慈的独裁者兼首席维护者 Solomon Hykes[13]，他是唯一一个权利凌驾于

其他维护者之上的人，他也全权负责任命新的维护者。

Docker的维护者通过在代码评审中使用LGTM（Looks Good To Me）注解来表示接

受此pull request。变更要想获得通过，需要受影响的每个组件的绝对多数维护者

（或者对于文档，至少两位维护者）都认为LGTM才行。比如，如果一个变更影响到

了docs/和registry/两个模块，那么这个变更就需要获得docs/的两个拥护者和

registry/的绝对多数维护者的同意。

提示 可以查看维护者工作流程手册[14]来了解更多关于维护者的详细信息。

本书由「ePUBw.COM」整理，ePUBw.COM 提供最新最全的优质

电子书下载！！！





9.4　小结

在本章中，我们学习了如何获得Docker帮助，以及有用的Docker社区成员和开发者

聚集的地方。我们也学习了记录Docker问题的最佳方法，包括各种要提供的必要信

息，以帮你得到最好的反馈。

我们也看到了如何配置一个开发环境来修改Docker源代码或者文档，以及如何在开

发环境中进行构建和测试，以保证自己所做的修改或者新功能能正常工作。最后，

我们学习了如何为你的修改创建一个结构良好且品质优秀的pull request。

[1]　 https://groups.google.com/forum/#!forum/docker-user

[2]　 https://groups.google.com/forum/#!forum/docker-dev

[3]　 https://forums.docker.com/

[4]　 http://freenode.net/

[5]　 https://github.com/docker/docker-registry

[6]　 https://github.com/docker/libcontainer

[7]　 https://github.com/docker/libswarm

[8]　 https://github.com/docker/compose

[9]　 https://github.com/docker/docker/issues

[10]　 http://docs.docker.com

[11]　 https://github.com/docker/docker/tree/master/docs

[12]　 http://blog.docker.com/2014/01/docker-code-contributions-require-developer-certificate-of-origin/

[13]　 https://github.com/shykes

[14]

https://github.com/docker/docker/blob/master/hack/MAINTAINERS.md 本书由「ePUBw.COM」整理，ePUBw.COM 提供最新最全的优质

电子书下载！！！





看完了

如果您对本书内容有疑问，可发邮件至contact@epubit.com.cn，会有编辑或作译

者协助答疑。也可访问异步社区，参与本书讨论。

如果是有关电子书的建议或问题，请联系专用客服邮箱：ebook@epubit.com.cn。

在这里可以找到我们：

微博：@人邮异步社区

QQ群：368449889

091507240605ToBeReplacedWithUserId

本书由「ePUBw.COM」整理，ePUBw.COM 提供最新最全的优质

电子书下载！！！





Document Outline


版权信息

作者简介

译者简介

本书特色

版权声明

内容提要

对本书的赞誉

序

我们走在容器化的大道上

前言 本书面向的读者 致谢





技术审稿人团队 Scott Collier

John Ferlito

Paul Nasrat





技术插图作家

校对者

排版约定

代码及示例

说明

勘误

版本





第1章 简介 1.1 Docker简介 1.1.1 提供一个简单、轻量的建模方式

1.1.2 职责的逻辑分离

1.1.3 快速、高效的开发生命周期

1.1.4 鼓励使用面向服务的架构





1.2 Docker组件 1.2.1 Docker客户端和服务器

1.2.2 Docker镜像

1.2.3 Registry

1.2.4 容器





1.3 能用Docker做什么

1.4 Docker与配置管理

1.5 Docker的技术组件

1.6 本书的内容

1.7 Docker资源





第2章 安装Docker 2.1 安装Docker的先决条件

2.2 在Ubuntu和Debian中安装Docker 2.2.1 检查前提条件

2.2.2 安装Docker

2.2.3 Docker与UFW





2.3 在Red Hat和Red Hat系发行版中安装Docker 2.3.1 检查前提条件

2.3.2 安装Docker

2.3.3 在Red Hat系发行版中启动Docker守护进程





2.4 在OS X中安装Docker Toolbox 2.4.1 在OS X中安装Docker Toolbox

2.4.2 在OS X中启动Docker Toolbox

2.4.3 测试Docker Toolbox





2.5 在Windows中安装Docker Toolbox 2.5.1 在Windows中安装Docker Toolbox

2.5.2 在Windows中启动Docker Toolbox

2.5.3 测试Docker Toolbox





2.6 使用本书的Docker Toolbox示例

2.7 Docker安装脚本

2.8 二进制安装

2.9 Docker守护进程 2.9.1 配置Docker守护进程

2.9.2 检查Docker守护进程是否正在运行





2.10 升级Docker

2.11 Docker用户界面

2.12 小结





第3章 Docker入门 3.1 确保Docker已经就绪

3.2 运行我们的第一个容器

3.3 使用第一个容器

3.4 容器命名

3.5 重新启动已经停止的容器

3.6 附着到容器上

3.7 创建守护式容器

3.8 容器内部都在干些什么

3.9 Docker日志驱动

3.10 查看容器内的进程

3.11 Docker统计信息

3.12 在容器内部运行进程

3.13 停止守护式容器

3.14 自动重启容器

3.15 深入容器

3.16 删除容器

3.17 小结





第4章 使用Docker镜像和仓库 4.1 什么是Docker镜像

4.2 列出镜像

4.3 拉取镜像

4.4 查找镜像

4.5 构建镜像 4.5.1 创建Docker Hub账号

4.5.2 用Docker的commit命令创建镜像

4.5.3 用Dockerfile构建镜像

4.5.4 基于Dockerfile构建新镜像

4.5.5 指令失败时会怎样

4.5.6 Dockerfile和构建缓存

4.5.7 基于构建缓存的Dockerfile模板

4.5.8 查看新镜像

4.5.9 从新镜像启动容器

4.5.10 Dockerfile指令





4.6 将镜像推送到Docker Hub 自动构建





4.7 删除镜像

4.8 运行自己的Docker Registry 4.8.1 从容器运行Registry

4.8.2 测试新Registry





4.9 其他可选Registry服务 Quay





4.10 小结





第5章 在测试中使用Docker 5.1 使用Docker测试静态网站 5.1.1 Sample网站的初始Dockerfile

5.1.2 构建Sample网站和Nginx镜像

5.1.3 从Sample网站和Nginx镜像构建容器

5.1.4 修改网站





5.2 使用Docker构建并测试Web应用程序 5.2.1 构建Sinatra应用程序

5.2.2 创建Sinatra容器

5.2.3 扩展Sinatra应用程序来使用Redis

5.2.4 将Sinatra应用程序连接到Redis容器

5.2.5 Docker内部连网

5.2.6 Docker Networking

5.2.7 使用容器连接来通信

5.2.8 连接容器小结





5.3 Docker用于持续集成 5.3.1 构建Jenkins和Docker服务器

5.3.2 创建新的Jenkins作业

5.3.3 运行Jenkins作业

5.3.4 与Jenkins作业有关的下一步

5.3.5 Jenkins设置小结





5.4 多配置的Jenkins 5.4.1 创建多配置作业

5.4.2 测试多配置作业

5.4.3 Jenkins多配置作业小结





5.5 其他选择 5.5.1 Drone

5.5.2 Shippable





5.6 小结





第6章 使用Docker构建服务 6.1 构建第一个应用 6.1.1 Jekyll基础镜像

6.1.2 构建Jekyll基础镜像

6.1.3 Apache镜像

6.1.4 构建Jekyll Apache镜像

6.1.5 启动Jekyll网站

6.1.6 更新Jekyll网站

6.1.7 备份Jekyll卷

6.1.8 扩展Jekyll示例网站





6.2 使用Docker构建一个Java应用服务 6.2.1 WAR文件的获取程序

6.2.2 获取WAR文件

6.2.3 Tomecat7应用服务器

6.2.4 运行WAR文件

6.2.5 基于Tomcat应用服务器的构建服务





6.3 多容器的应用栈 6.3.1 Node.js镜像

6.3.2 Redis基础镜像

6.3.3 Redis主镜像

6.3.4 Redis副本镜像

6.3.5 创建Redis后端集群

6.3.6 创建Node容器

6.3.7 捕获应用日志

6.3.8 Node程序栈的小结





6.4 不使用SSH管理Docker容器

6.5 小结



