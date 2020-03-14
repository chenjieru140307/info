- [GitLab](https://library.prof.wang/handbook_html/h02-linux/GitLab/index.html#toc-0)[简介](https://library.prof.wang/handbook_html/h02-linux/GitLab/index.html#toc-1)[安装过程](https://library.prof.wang/handbook_html/h02-linux/GitLab/index.html#toc-2)[一、在线安装](https://library.prof.wang/handbook_html/h02-linux/GitLab/index.html#toc-3)[二、离线包安装](https://library.prof.wang/handbook_html/h02-linux/GitLab/index.html#toc-6)[SSL 配置](https://library.prof.wang/handbook_html/h02-linux/GitLab/index.html#toc-9)[常用命令](https://library.prof.wang/handbook_html/h02-linux/GitLab/index.html#toc-10)[服务管理](https://library.prof.wang/handbook_html/h02-linux/GitLab/index.html#toc-11)[运维管理](https://library.prof.wang/handbook_html/h02-linux/GitLab/index.html#toc-12)[日志排查](https://library.prof.wang/handbook_html/h02-linux/GitLab/index.html#toc-13)[常见报错](https://library.prof.wang/handbook_html/h02-linux/GitLab/index.html#toc-14)[备份与恢复](https://library.prof.wang/handbook_html/h02-linux/GitLab/index.html#toc-15)[扩展：自动备份到阿里云 OSS](https://library.prof.wang/handbook_html/h02-linux/GitLab/index.html#toc-16)

# GitLab

## 简介

GitLab 是一个利用 Ruby on Rails 开发的开源版本控制系统，实现一个自托管的 Git 项目仓库，可通过 Web 界面进行访问公开的或者私人项目。

它拥有与 GitHub 类似的功能，能够浏览源代码，管理缺陷和注释。可以管理团队对仓库的访问，它非常易于浏览提交过的版本并提供一个文件历史库。团队成员可以利用内置的简单聊天程序（Wall）进行交流。它还提供一个代码片段收集功能可以轻松实现代码复用，便于日后有需要的时候进行查找。

开源中国代码托管平台就是基于 GitLab 项目搭建。

[http://git.oschina.net/](http://git.oschina.net/)

GitLab 分为 GitLab Community Edition(CE) 社区版 和 GitLab Enterprise Edition(EE) 专业版。社区版免费，专业版收费，两个版本在功能上的差异对比，可以参考官方对比说明：

[https://about.gitlab.com/features/#compare](https://about.gitlab.com/features/#compare)。

## 安装过程

### 一、在线安装

本文以 CentOS 7.4 && Ubuntu16.04为例，更多安装方法参考：[https://about.gitlab.com/installation/](https://about.gitlab.com/installation/)

- 准备工作

推荐配置至少2C4G.(官方推荐配置，低配机器可自行测试)

#### 1. YUM（RedHat 体系）

以 CentOS7.4为例：

- 安装依赖

sudo yum install -y curl policycoreutils-python openssh-server

```
sudo systemctl enable sshd
sudo systemctl start sshd
sudo firewall-cmd --permanent --add-service=http
sudo systemctl reload firewalld

sudo yum install postfix
sudo systemctl enable postfix
sudo systemctl start postfix

```

- 设置 gitlab 安装源

官方源安装方法：

```
curl https://packages.gitlab.com/install/repositories/gitlab/gitlab-ee/script.rpm.sh | sudo bash

```

由于网络问题，国内推荐使用清华大学镜像源进行安装：

```
cat>/etc/yum.repos.d/gitlab-ce.repo<<'EOF'
[gitlab-ce]
name=Gitlab CE Repository
baseurl=https://mirrors.tuna.tsinghua.edu.cn/gitlab-ce/yum/el$releasever/
gpgcheck=0
enabled=1
EOF

```

- 安装 GitLab

首先可以设置域名，域名需要保证能够正常解析，可以根据使用场景选择公网域名或者是本地 hosts 测试。

```
sudo EXTERNAL_URL="http://gitlab.ikiwi.me" yum install -y gitlab-ce

```

也可以不设置域名，域名绑定可以随时在配置文件`/etc/gitlab/gitlab.rb`里进行修改：

```
external_url 'http://gitlab.xxx.com'

```

直接安装：

```
sudo yum install -y gitlab-ce

```

安装完成以后，运行下面的命令进行配置：

```
sudo gitlab-ctl reconfigure

```

#### 2. APT（Debian 体系）

以 Ubuntu 16.04为例：

- 安装依赖

```
sudo apt-get update
sudo apt-get install -y curl openssh-server ca-certificates
sudo apt-get install -y postfix

```

- 设置 gitlab 安装源

官方源安装方法：

```
curl https://packages.gitlab.com/install/repositories/gitlab/gitlab-ee/script.deb.sh | sudo bash

```

- 安装 GitLab

首先可以设置域名，域名需要保证能够正常解析，可以根据使用场景选择公网域名或者是本地 hosts 测试。

```
sudo EXTERNAL_URL="http://gitlab.ikiwi.me" apt-get install gitlab-ce

```

也可以不设置域名，域名绑定可以随时在配置文件`/etc/gitlab/gitlab.rb`里进行修改：

```
external_url 'http://gitlab.xxx.com'

```

直接安装：

```
sudo apt-get install gitlab-ce

```

### 二、离线包安装

#### 1. RPM 包（RedHat 体系）

以 CentOS7.4为例：

```
wget --content-disposition <https://packages.gitlab.com/gitlab/gitlab-ce/packages/el/7/gitlab-ce-11.0.3-ce.0.el7.x86_64.rpm/download.rpm>

rpm -ivh [gitlab-ce-11.0.3-ce.0.el7.x86_64.rpm](https://packages.gitlab.com/gitlab/gitlab-ce/packages/el/7/gitlab-ce-11.0.3-ce.0.el7.x86_64.rpm/download.rpm)

```

#### 2. DEB 包（Debian 体系）

以 Ubuntu 16.04为例：

```
wget --content-disposition <https://packages.gitlab.com/gitlab/gitlab-ce/packages/ubuntu/xenial/gitlab-ce_11.3.8-ce.0_amd64.deb/download.deb>

sudo dpkg -i [gitlab-ce_11.3.8-ce.0_amd64.deb](https://packages.gitlab.com/gitlab/gitlab-ce/packages/ubuntu/xenial/gitlab-ce_11.3.8-ce.0_amd64.deb/download.deb)

```

安装完成以后，运行下面的命令进行配置：

```
sudo gitlab-ctl reconfigure

```

> 整个 Gitlab 项目构成：

- nginx: 静态 web 服务器
- gitlab-shell: 用于处理 Git 命令和修改 authorized
- keys 列表
- gitlab-workhorse: 轻量级的反向代理服务器
- logrotate：日志文件管理工具
- postgresql：数据库
- redis：缓存数据库
- sidekiq：用于在后台执行队列任务（异步执行）
- unicorn：An HTTP server for Rack applications，GitLab Rails 应用是托管在这个服务器上面的。

> 安装目录信息：

- 主配置文件: /etc/gitlab/gitlab.rb
- GitLab 文档根目录: /opt/gitlab
- 默认存储库位置: /var/opt/gitlab/git-data/repositories
- GitLab Nginx 配置文件路径: /var/opt/gitlab/nginx/conf/gitlab-http.conf
- Postgresql 数据目录: /var/opt/gitlab/postgresql/data

安装工作完成后，登录 web 界面以及修改账户密码：

访问之前设置的域名

[http://gitlab.example.com/](http://gitlab.example.com/)

初次登录强制要求修改 root 用户的密码，修改后并登录。

![img](https://library.prof.wang/handbook_html/h02-linux/GitLab/1.png)

登录成功后的主界面:

![img](https://library.prof.wang/handbook_html/h02-linux/GitLab/2.png)

## SSL 配置

GitLab 默认是使用 HTTP 的，可以手动配置为 HTTPS.

- 上传 SSL 证书

创建 ssl 目录，用于存放 SSL 证书

```
# mkdir -p /etc/gitlab/ssl
# chmod 0700 /etc/gitlab/ssl

```

上传证书并修改证书权限

```
# chmod 600 /etc/gitlab/ssl/*

```

- 修改 GitLab 的配置文件

修改配置文件`/etc/gitlab/gitlab.rb`

```
external_url "https://gitlab.xxx.com"
nginx['redirect_http_to_https'] = true
nginx['ssl_certificate'] = "/etc/gitlab/ssl/gitlab.xxx.com.crt"
nginx['ssl_certificate_key'] = "/etc/gitlab/ssl/gitlab.xxx.com.key"

```

- 重启服务，使其生效

```
# gitlab-ctl restart

```

以上操作后，GitLab 自带的 Nginx 服务的配置文件 `/var/opt/gitlab/nginx/conf/gitlab-http.conf` 会被重新修改：

```
server {
 listen *:80;
 server_name gitlab.xxx.com;
 server_tokens off; ## Don't show the nginx version number, a security best practice
 return 301 https://gitlab.xxx.com:443$request_uri;
 access_log /var/log/gitlab/nginx/gitlab_access.log gitlab_access;
 error_log   /var/log/gitlab/nginx/gitlab_error.log;
}

```

不用额外再配置，HTTP 会自动跳转到 HTTPS 。记得要对外放行443端口。

![img](https://library.prof.wang/handbook_html/h02-linux/GitLab/3.png)

## 常用命令

### 服务管理

```
# 启动所有 gitlab 组件：
gitlab-ctl start
# 停止所有 gitlab 组件：
gitlab-ctl stop
# 停止所有 gitlab postgresql 组件：
gitlab-ctl stop postgresql
# 停止相关数据连接服务
gitlab-ctl stop unicorn
gitlab-ctl stop sidekiq
# 重启所有 gitlab 组件：
gitlab-ctl restart
# 重启所有 gitlab gitlab-workhorse 组件：
gitlab-ctl restart gitlab-workhorse
# 查看服务状态
gitlab-ctl status
# 生成配置并启动服务
gitlab-ctl reconfigure

```

### 运维管理

```
# 查看版本
cat /opt/gitlab/embedded/service/gitlab-rails/VERSION
# 检查 gitlab
gitlab-rake gitlab:check SANITIZE=true --trace
# 实时查看日志
gitlab-ctl tail
# 数据库关系升级
gitlab-rake db:migrate
# 清理 redis 缓存
gitlab-rake cache:clear
# 升级 GitLab-ce 版本
yum update gitlab-ce
# 升级 PostgreSQL 最新版本
gitlab-ctl pg-upgrade

```

### 日志排查

```
# 实时查看所有日志
gitlab-ctl tail
# 实时检查 redis 的日志
gitlab-ctl tail redis
# 实时检查 postgresql 的日志
gitlab-ctl tail postgresql
# 检查 gitlab-workhorse 的日志
gitlab-ctl tail gitlab-workhorse
# 检查 logrotate 的日志
gitlab-ctl tail logrotate
# 检查 nginx 的日志
gitlab-ctl tail nginx
# 检查 sidekiq 的日志
gitlab-ctl tail sidekiq
# 检查 unicorn 的日志
gitlab-ctl tail unicorn

```

## 常见报错

**1.postfix 启动出现 fatal: parameter inet ··· 错误**

```
[root@iZbp1h5pmg6spg0eodt4laZ ~]# sudo systemctl start postfix
Job for postfix.service failed because the control process exited with error code. See "systemctl status postfix.service" and "journalctl -xe" for details.
[root@iZbp1h5pmg6spg0eodt4laZ ~]# systemctl status postfix.service
● postfix.service - Postfix Mail Transport Agent
   Loaded: loaded (/usr/lib/systemd/system/postfix.service; enabled; vendor preset: disabled)
   Active: failed (Result: exit-code) since 六 2018-09-15 14:48:32 CST; 23s ago
  Process: 1856 ExecStart=/usr/sbin/postfix start (code=exited, status=1/FAILURE)
  Process: 1854 ExecStartPre=/usr/libexec/postfix/chroot-update (code=exited, status=0/SUCCESS)
  Process: 1851 ExecStartPre=/usr/libexec/postfix/aliasesdb (code=exited, status=75)

```

解决方案：

```
1. 查看 centos 的 postfix 日志
[root@iZbp1h5pmg6spg0eodt4laZ sbin]# more  /var/log/maillog
Sep 15 14:44:24 localhost postfix/sendmail[961]: fatal: parameter inet_interface
s: no local interface found for ::1
Sep 15 14:44:24 localhost postfix[967]: fatal: parameter inet_interfaces: no loc
al interface found for ::1
Sep 15 14:48:31 localhost postfix/sendmail[1853]: fatal: parameter inet_interfac
es: no local interface found for ::1
Sep 15 14:48:31 localhost postfix[1856]: fatal: parameter inet_interfaces: no lo
cal interface found for ::1
Sep 15 14:49:38 localhost postfix/sendmail[1883]: fatal: parameter inet_interfac
es: no local interface found for ::1
Sep 15 14:49:38 localhost postfix[1887]: fatal: parameter inet_interfaces: no lo
cal interface found for ::1

2. vim /etc/postfix/main.cf
修改
inet_interfaces = localhost

inet_protocols = all

为

inet_interfaces = all

inet_protocols = all

即可。

```

**2. /var/spool/postfix 目录无权限问题**

```
[root@linux115 spool]# service postfix start
启动 postfix： [失败]

[root@linux115 log]# postfix start
postsuper: fatal: scan_dir_push: open directory defer: Permission denied
postfix/postfix-script: fatal: Postfix integrity check failed!
[root@linux115 log]# postfix check
postsuper: fatal: scan_dir_push: open directory defer: Permission denied

1.查看日志文件

more /var/log/maillog

有如下两行：
May 26 09:01:51 linux115 postfix/postsuper[6199]: fatal: scan_dir_push: open directory defer: Permission denied
May 26 09:01:52 linux115 postfix/postfix-script[6200]: fatal: Postfix integrity check failed!

问题原因：
那是/var/spool/postfix 这个目录拥有都权限的问题，原来默认的拥有都是 root，需要将拥有者改为 postfix，如（1）

解决命令：
（1）
[root@linux115 spool]# chown -R postfix:postfix /var/spool/postfix/

[root@linux115 spool]# service postfix start
启动 postfix： [确定]

```

## 备份与恢复

GitLab 作为公司项目代码的版本管理系统，数据非常重要，所以应做好备份工作。

- 修改备份目录

GitLab 备份的默认目录是`/var/opt/gitlab/backups` ，如果想改备份目录，可修改`/etc/gitlab/gitlab.rb`：

```
gitlab_rails['backup_path'] = '/data/backups'

```

修改配置后，记得：

```
gitlab-ctl reconfigure

```

- 备份命令

```
gitlab-rake gitlab:backup:create

```

该命令会在备份目录（默认：/var/opt/gitlab/backups/）下创建一个 tar 压缩包 xxxxxxxx_gitlab_backup.tar，其中开头的 xxxxxx 是备份创建的时间戳，这个压缩包包括 GitLab 整个的完整部分。

- 自动备份

通过任务计划 crontab 实现自动备份

```
# 每天2点备份 gitlab 数据
0 2 * * * /usr/bin/gitlab-rake gitlab:backup:create

```

- 备份保留7天

可设置只保留最近7天的备份，编辑配置文件`/etc/gitlab/gitlab.rb`

```
# 数值单位：秒
gitlab_rails['backup_keep_time'] = 604800

```

重新加载 gitlab 配置文件

```
gitlab-ctl reconfigure

```

- 恢复

备份文件：

```
/var/opt/gitlab/backups/1499244722_2017_07_05_9.2.6_gitlab_backup.tar

```

停止 unicorn 和 sidekiq ，保证数据库没有新的连接，不会有写数据情况。

```
# 停止相关数据连接服务
gitlab-ctl stop unicorn
gitlab-ctl stop sidekiq

# 指定恢复文件，会自动去备份目录找。确保备份目录中有这个文件。
# 指定文件名的格式类似：1499242399_2017_07_05_9.2.6，程序会自动在文件名后补上：“_gitlab_backup.tar”
# 一定按这样的格式指定，否则会出现 The backup file does not exist! 的错误
gitlab-rake gitlab:backup:restore BACKUP=1499242399_2017_07_05_9.2.6

# 启动 Gitlab
gitlab-ctl start

```

### 扩展：自动备份到阿里云 OSS

Gitlab 还支持自动备份文件到阿里云 OSS，具体配置如下：

- 修改配置文件 /etc/gitlab/gitlab.rb

```
gitlab_rails['backup_upload_connection']={
'provider'=>'aliyun',
'aliyun_accesskey_id'=>'xxxxx', # 阿里云 ak
'aliyun_accesskey_secret'=>'xxxxxxxxxxxxxxxx', # 阿里云 aks
'aliyun_oss_endpoint'=>'oss-cn-shenzhen-internal.aliyuncs.com', # 阿里云 oss 连接地址
'aliyun_oss_bucket'=>'xy-gitlab', # oss 的 bucket 地址
'aliyun_oss_location'=>'cn-shenzhen', # oss 地域
}
gitlab_rails['backup_upload_remote_directory']='gitlab' # 远端备份路径，即为 bucket 的子目录

```

- 重启 gitlab，使配置生效

```
gitlab-ctl reconfigure
gitlab-ctl restart
```