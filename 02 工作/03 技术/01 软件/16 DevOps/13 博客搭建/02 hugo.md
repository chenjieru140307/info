
# hugo 搭建博客



下载：

https://github.com/gohugoio/hugo/releases

我用的是 hugo_extended_0.68.3_Linux-64bit.deb 版本的。

注意

- 有 extended
- 阿里云上直接下可能比较慢，可以下在本地然后传上去。


```
wget https://github.com/spf13/hugo/releases/download/v0.14/hugo_0.14_amd64.deb
sudo dpkg -i hugo*.deb
hugo version
```

环境：

```
sudo apt-get update
sudo apt-get install git
```

创建新的：

```
hugo new site mydocs; cd mydocs
git init
git submodule add https://github.com/alex-shpak/hugo-book themes/book
cp -R themes/book/exampleSite/content .
```

mydocs 是我们的 site 的名字。


开启：


```
hugo server --minify --theme book
```


win10 测试端口：


开启 telnet：

- 按 windows 键 
- 设置 
- 应用(卸载,默认应用,可选功能) 
- (右上角)相关设置 [程序和功能] 
- 左侧[启动或关闭 windows 功能] 
- 弹出的对话框中 勾选 Telnet客户端,确定OK

 
使用 telnet 测试：

- telnet ip地址 端口号

服务器上端口开放：

- [安全组应用案例](https://cloud.tencent.com/document/product/213/34601)

注意：

- 服务器端口在安全组开放后。需要该端口在监听状态，并且无防火墙及安全则的限制，端口才可通过外网访问。
- 因此，可以：登录云服务器，开启 server：`hugo server --bind=0.0.0.0 --minify --theme book`，在另一个 ssh 中执行 `netstat -tunlp` 命令查看端口是否监听。如果此时对应端口已在监听，则在浏览器中 http://xxx.xxx.xxx.xxx:1313/ 进行访问。





netstat -lnp 查看当前哪些端口在监听。



nohup sudo hugo server -t book &
sudo hugo server -t book --disableFastRender
nohup sudo hugo server -t book --disableFastRender &

可能删除文件时，public 没有对应删除。