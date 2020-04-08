
## visdo

sudo visdo

```
git ALL=(ALL) NOPASSWD: /home/iterategitbook.git/hooks/post-update
git ALL=(ALL) NOPASSWD: /usr/bin/gitbook
```

添加这两句，按 Ctrl+s 保存，Ctrl+x 退出。

## post-update


post-update 内容如下：

```sh
unset GIT_DIR
DIR_ONE=/home/bloggitbook/
cd $DIR_ONE

# git init
# git remote add origin /home/iteratesite.git
# git clean -df
# git reset --hard
git pull origin master

echo "root密码写在这里" | sudo -S gitbook build

# hugo -t even -F
# hugo server --bind=0.0.0.0 --minify --theme book
# sudo hugo -t book
exec git update-server-info
```


post-update 改为 root 权限：

```
sudo chown -R root:root post-update
```


## nginx

定位 nginx.conf

```
locate nginx.conf
```

修改文件：

```
sudo vim /etc/nginx/nginx.conf
```

http 中的 server 设置如下：

```conf
server {
        listen 80;
        server_name www.iterate.site;
        location /{
                root /home/bloggitbook/_book;
        }
}
```


重启 nginx

```
sudo nginx -s reload
```


## gitbook 


调试：

```
gitbook build ./ --log=debug --debug
```



- [如何在脚本中运行’sudo’命令？](https://ubuntuqa.com/article/1440.html)