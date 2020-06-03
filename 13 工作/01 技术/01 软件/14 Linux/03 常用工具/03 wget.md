# wget



## 抓取整个网站

指令

- 抓取整站：`wget -r  -p -np -k -E http://www.xxx.com`
- 抓取第一级：`wget -l 1 -p -np -k http://www.xxx.com`
- 将全站下载以本地的当前工作目录，生成可访问、完整的镜像：`wget -m -e robots=off -k -E "http://www.xxx.net/"`

 
说明：

- `-r` 递归抓取
- `-k` 抓取之后修正链接，适合本地浏览
- `-m` 镜像，就是整站抓取
- `-e robots=off` 忽略robots协议，强制、流氓抓取
- `-k` 将绝对 `URL` 链接转换为本地相对URL
- `-E` 将所有 `text/html` 文档以 `.html` 扩展名保存
