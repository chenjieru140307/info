# Pip


## Pip 安装和升级

```py
sudo easy_install pip # 安装 pip
pip install -U pip # 升级pip
```

## 常用命令

```py
# 在线安装 通过使用== >= <= > <来指定版本，不写则安装最新版
pip install <包名>
# 安装本地安装包
pip install <目录>/<文件名>
# 搜索并安装本地安装包
pip install --no-index -f=<目录>/ <包名>
# 下载包而不安装
pip install <包名> -d <目录>

# 查询可升级的包
pip list -o
# 升级包
pip install -U <包名>

# 卸载包
pip uninstall <包名>

# 显示包所在的目录
pip show -f <包名>
# 搜索包
pip search <搜索关键字>

# 列出已安装的包
pip freeze 或者 pip list

# 打包
pip wheel <包名>
```

不清楚：

- <span style="color:red;">这个 pip wheel 会打包成什么？</span>


## 命令对应的可选参数

```
-h,--help # 显示帮助
-v,--verbose # 更多的输出，最多可使用3次。
-V,--version # 显示版本信息后退出
-q,--quiet # 最少的输出
--log-file <path> # 覆盖的方式记录 verbose 错误日志，默认文件：/root/.pip/pip.log
--log <path> # 不覆盖记录 verbose 输出的日志
--proxy <proxy> # 指定如下形式的 proxy [user:passwd@]proxy.server:port
--timeout <sec> # 连接超时时间（默认 15 秒）
--exists-action <action> # 指定 当路径存在时候的对应：(s)witch (i)gnore (w)ipe (b)ackup
--cert <path> # 证书
```


## 指定源进行安装

<span style="color:red">**不知是否过期**</span>

国内 pypi 镜像：

- 阿里：<https://mirrors.aliyun.com/pypi/simple>
- 中国科学技术大学：<http://pypi.mirrors.ustc.edu.cn/simple/>

指定单次安装源：

```py
pip install <包名> -i https://mirrors.aliyun.com/pypi/simple
```

指定全局安装源：

在unix和macos，配置文件为：`$HOME/.pip/pip.conf`
在windows上，配置文件为：`%HOME%\pip\pip.ini`

```conf
[global]
timeout = 6000
  index-url = https://mirrors.aliyun.com/pypi/simple
```

```txt
pip install -i https://pypi.douban.com/simple module # 使用豆瓣源
pip install -i http://mirrors.aliyun.com/pypi/simple/ module # 阿里云
pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ module # 中国科技大学
pip install -i http://pypi.douban.com/simple/ module # 豆瓣(douban)
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ module # 清华大学
pip install -i http://pypi.mirrors.ustc.edu.cn/simple/ module # 中国科学技术大学
```



## requirements.txt 文件使用


可以导出当前包到 requirements.txt 文本中。这样别人如果使用你的程序，安装环境的时候子需要把 requirements.txt 安装进来即可。


```py
# 导出
pip freeze > requirements.txt

# 导入
(sudo) pip install –r requirements.txt

# 下载包而不安装
pip install -d <目录> -r requirements.txt

# 卸载
pip uninstall -r requirements.txt
```



## 问题

### 如何使用本地源进行安装

<span style="color:red;">这个问题没有得到解决。
</span>

- 首先你需要一台可以连接其他 pip 源的电脑，通常也就是你自己的开发环境，并且安装了 pip.
- `pip install pip2pi`
- 用 pip freeze 在你的开发环境上 制作一个 requirements 文件：`pip freeze > requirements.txt`
- 手动更新下 requirements.txt 文件，只留一行：pyecharts==0.4.1
- 建立一个 pacakges 文件夹，作为存放本地源的路径
- 假设你的 packages 和 requirements.txt 都在 `c:\` 下
- 执行：`pip2pi package --no-binary :all: -r requirements.txt`，取得所有需要的包
- 执行：`pip2tgz packages -r requirements.txt`，取得所有需要的 wheel
- 用 u 盘把 packages 和 requirements.txt 拷贝到内网
- 内网执行：`pip install --no-index --find-links=packages -r requirements.txt`

上面这个在执行到 `pip2pi package --no-binary :all: -r requirements.txt` 的时候有问题：`module 'pip' has no attribute 'main'`




### 问题：Could not install packages due to an EnvironmentError: raw write() returned invalid length 2

在 Pycharm 的 console 里面进行更新的收遇到了一个问题:

```
(venv) E:\02.try\092001>pip install pytesseract
Collecting pytesseract
Collecting Pillow (from pytesseract)
  Downloading https://files.Pythonhosted.org/packages/2e/5f/2829276d720513a434f5bcbf61316d98369a5707f6128b34c03f2213feb1/Pillow-5.2.0-cp35-cp35m-win_amd64.whl (1.6MB)
Could not install packages due to an EnvironmentError: raw write() returned invalid length 2 (should have been between 0 and 1)
```


这个好像是 windows 的问题：

[Python throws IOError in some scripts in Windows Integrated Terminal](https://github.com/Microsoft/vscode/issues/36630)

windows 升级到 1803 应该是可以的。
