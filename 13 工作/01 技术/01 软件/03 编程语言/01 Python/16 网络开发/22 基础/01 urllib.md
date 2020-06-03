# urllib

作用：

- urllib提供了一系列用于操作URL的功能。

### Get

urllib的`request`模块可以非常方便地抓取URL内容，也就是发送一个GET请求到指定的页面，然后返回HTTP的响应：

例如，对豆瓣的一个URL`https://api.douban.com/v2/book/2129650`进行抓取，并返回响应：

```
from urllib import request

with request.urlopen('https://api.douban.com/v2/book/2129650') as f:
    data = f.read()
    print('Status:', f.status, f.reason)
    for k, v in f.getheaders():
        print('%s: %s' % (k, v))
    print('Data:', data.decode('utf-8'))
```

可以看到HTTP响应的头和JSON数据：

```
Status: 200 OK
Server: nginx
Date: Tue, 26 May 2015 10:02:27 GMT
Content-Type: application/json; charset=utf-8
Content-Length: 2049
Connection: close
Expires: Sun, 1 Jan 2006 01:00:00 GMT
Pragma: no-cache
Cache-Control: must-revalidate, no-cache, private
X-DAE-Node: pidl1
Data: {"rating":{"max":10,"numRaters":16,"average":"7.4","min":0},"subtitle":"","author":["廖雪峰编著"],"pubdate":"2007-6",...}
```

如果我们要想模拟浏览器发送GET请求，就需要使用`Request`对象，通过往`Request`对象添加HTTP头，我们就可以把请求伪装成浏览器。例如，模拟iPhone 6去请求豆瓣首页：

```
from urllib import request

req = request.Request('http://www.douban.com/')
req.add_header('User-Agent', 'Mozilla/6.0 (iPhone; CPU iPhone OS 8_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/8.0 Mobile/10A5376e Safari/8536.25')
with request.urlopen(req) as f:
    print('Status:', f.status, f.reason)
    for k, v in f.getheaders():
        print('%s: %s' % (k, v))
    print('Data:', f.read().decode('utf-8'))
```

这样豆瓣会返回适合iPhone的移动版网页：

```
...
    <meta name="viewport" content="width=device-width, user-scalable=no, initial-scale=1.0, minimum-scale=1.0, maximum-scale=1.0">
    <meta name="format-detection" content="telephone=no">
    <link rel="apple-touch-icon" sizes="57x57" href="http://img4.douban.com/pics/cardkit/launcher/57.png" />
...
```

### Post

如果要以POST发送一个请求，只需要把参数`data`以bytes形式传入。

我们模拟一个微博登录，先读取登录的邮箱和口令，然后按照weibo.cn的登录页的格式以`username=xxx&password=xxx`的编码传入：

```
from urllib import request, parse

print('Login to weibo.cn...')
email = input('Email: ')
passwd = input('Password: ')
login_data = parse.urlencode([
    ('username', email),
    ('password', passwd),
    ('entry', 'mweibo'),
    ('client_id', ''),
    ('savestate', '1'),
    ('ec', ''),
    ('pagerefer', 'https://passport.weibo.cn/signin/welcome?entry=mweibo&r=http%3A%2F%2Fm.weibo.cn%2F')
])

req = request.Request('https://passport.weibo.cn/sso/login')
req.add_header('Origin', 'https://passport.weibo.cn')
req.add_header('User-Agent', 'Mozilla/6.0 (iPhone; CPU iPhone OS 8_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/8.0 Mobile/10A5376e Safari/8536.25')
req.add_header('Referer', 'https://passport.weibo.cn/signin/login?entry=mweibo&res=wel&wm=3349&r=http%3A%2F%2Fm.weibo.cn%2F')

with request.urlopen(req, data=login_data.encode('utf-8')) as f:
    print('Status:', f.status, f.reason)
    for k, v in f.getheaders():
        print('%s: %s' % (k, v))
    print('Data:', f.read().decode('utf-8'))
```

如果登录成功，我们获得的响应如下：

```
Status: 200 OK
Server: nginx/1.2.0
...
Set-Cookie: SSOLoginState=1432620126; path=/; domain=weibo.cn
...
Data: {"retcode":20000000,"msg":"","data":{...,"uid":"1658384301"}}
```

如果登录失败，我们获得的响应如下：

```
...
Data: {"retcode":50011015,"msg":"\u7528\u6237\u540d\u6216\u5bc6\u7801\u9519\u8bef","data":{"username":"example@python.org","errline":536}}
```

### Handler

如果还需要更复杂的控制，比如通过一个Proxy去访问网站，我们需要利用`ProxyHandler`来处理，示例代码如下：

```
proxy_handler = urllib.request.ProxyHandler({'http': 'http://www.example.com:3128/'})
proxy_auth_handler = urllib.request.ProxyBasicAuthHandler()
proxy_auth_handler.add_password('realm', 'host', 'username', 'password')
opener = urllib.request.build_opener(proxy_handler, proxy_auth_handler)
with opener.open('http://www.example.com/login.html') as f:
    pass
```

### 小结

urllib提供的功能就是利用程序去执行各种HTTP请求。如果要模拟浏览器完成特定功能，需要把请求伪装成浏览器。伪装的方法是先监控浏览器发出的请求，再根据浏览器的请求头来伪装，`User-Agent`头就是用来标识浏览器的。


## 抓取整个网站

```py
# 网页url采集爬虫，给定网址，以及存储文件，将该网页内全部网址采集下，可指定文件存储方式
import requests, time
from lxml import etree
from urllib import parse

"""
    url:给定的url
    save_file_name:为url存储文件
"""

OLD_URL = 'http://www.research.pku.edu.cn'
level = 2;  # 递归层级，广度优先


def Redirect(url):
    try:
        res = requests.get(url, timeout=10)
        url = res.url
    except Exception as e:
        print("4", e)
        time.sleep(1)
    return url


def requests_for_url(url, save_file_name, file_model, features):
    global OLD_URL
    headers = {
        'pragma': "no-cache",
        'accept-encoding': "gzip, deflate, br",
        'accept-language': "zh-CN,zh;q=0.8",
        'upgrade-insecure-requests': "1",
        'user-agent': "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36",
        'accept': "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        'cache-control': "no-cache",
        'connection': "keep-alive",
    }
    try:
        response = requests.request("GET", url, headers=headers)
        selector = etree.HTML(response.text, parser=etree.HTMLParser(encoding='utf-8'))
    except Exception as e:
        print("页面加载失败", e)

    return_set = set()
    with open(save_file_name, file_model,encoding='utf-8') as f:
        try:
            context = selector.xpath('//a/@href')
            for i in context:
                try:
                    print("-------------" + i)
                    print(type(i))

                    if i[:10] == "javascript":  # javascript跳转
                        continue
                    elif i[0] == "/":  # 相对路径跳转
                        # print i
                        # i = OLD_URL + i
                        i = parse.urljoin(url, i)  # 通过parse组装相对路径为绝对路径
                    elif i[0:4] == "http":  # 如果是完整的URL
                        i = i
                    elif i[0] == ".":  # 如果是有层级的相对路径
                        i = parse.urljoin(url, i)  # 通过parse组装相对路径为绝对路径
                    elif i[0] != ".":  # 如果是没有层级的相对路径
                        if url[-1] != "/":  # 如果不是“/”结尾
                            if url[8:].rfind("/") != -1:  # 如果连接中包含“/”
                                i = url[:url.rfind("/") + 1] + i  # 截取掉最后一个“/"之后字符串
                            else:
                                i = url + "/" + i
                        else:
                            i = url + i
                    if features in i:  # 不符合特征值，舍弃
                        f.write(i)
                        f.write("\n")
                        return_set.add(i)
                        # print(len(return_set))
                        print(len(return_set), i)
                except Exception as e:
                    print("1", e)
        except Exception as e:
            print("2", e)
    return return_set


def Recursion_GETurl(return_set, features):
    global level
    level -= 1
    print("目前递归层级：" + str(level))
    if level < 0:
        return
    else:
        return_all = set()
        for value in return_set:
            # print(value)
            http = "http://"
            if http in value:
                return_set2 = requests_for_url(value, save_file_name, "a", features)
                return_all = return_all | return_set2  # 合并SET()集
        Recursion_GETurl(return_all, features)


if __name__ == '__main__':
    # 网页url采集爬虫，给定网址，以及存储文件，将该网页内全部网址采集下，可指定文件存储方式
    url = "http://www.huaxiaozhuan.com/"
    features = "com"  # 特征值，URL不包含该字段的，舍弃。二级域名的情况下，应该定义为一级域名
    save_file_name = "url.txt"
    return_set = requests_for_url(url, save_file_name, "a", features)  # “a”:追加
    Recursion_GETurl(return_set, features)
    # 对url.txt进行数据去重
    print("终于爬完了，辛苦，伙计！")
```
