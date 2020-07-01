# Anaconda 

## 下载

- [Anaconda 官方](https://www.anaconda.com/)
- [Anaconda 镜像 清华](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)



# Anaconda 环境安装

镜像文件的下载：

地址：

https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/

在 ubuntu 中可以这么下载：

```
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.2.0-Linux-x86_64.sh
```


## 环境操作

指令：

```
conda create -n py36 Python=3.6
conda remove -n py36 --all
source activate py36
source deactivate
```

说明：

- `conda create -n py36 Python=3.6 ` 创建一个 Python=3.6版本的环境，取名叫 py36
- `conda remove -n py36 --all` 删除环境。注意，不要乱删。
- `source activate py36` 激活 环境名为 py36 的环境
- `source deactivate` 退出当前所在的环境。


## 库的安装


在线：

```
conda install xxx
```

离线：

1. 下载：可以在 Anaconda 的网站上下载安装包，搜索地址：<https://anaconda.org/>。将包下载至本地。
2.  ```
    conda install --use-local ffmpeg-2.7.0-0.tar.bz2
    conda install --offline -f ***.tar.bz2
    conda install /path/***.tar.bz2
    ```

## 注意

### 与 Pip 混合使用时

包可以使用 conda 下载和更新，也可以使用 pip 下载和更新：

```
conda install package_name # conda 下载
pip install package_name # pip 下载

conda update package_name # conda 更新
pip install --upgrade package_name # pip更新
```

注意：

- 下载。优先使用 conda 下载。当包并不在 conda 服务器上时，使用 pip 下载。应该不会有冲突。
- 更新。**不要使用 pip 来更新用 conda 下载的包，这会导致库之间的依赖出现问题。** 


## 问题

### 问题：windows下安装 Anaconda3 之后再 cmd 下出现'activate' 不是内部或外部命令，也不是可运行的程序 或批处理文件

windows 下安装 Anaconda3 之后再 cmd 下出现 'activate' 不是内部或外部命令，也不是可运行的程序 或批处理文件。

输入 conda 时也会出现：'conda' 不是内部或外部命令，也不是可运行的程序 或批处理文件。

经过查找，问题在于应该将 `D:\software\Anaconda3\Scripts` 加入到环境变量之中。因为这里存在 conda 和 activate。

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20190624/57VqcqXEClwf.png?imageslim">
</p>
<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20190624/2fIsjvF2BAVK.png?imageslim">
</p>

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20190624/gyFjoHbr20YB.png?imageslim">
</p>


原文及相关：

- [果冻先生的专栏](https://blog.csdn.net/Homewm/article/details/84886033)





### 问题：Intel MKL FATAL ERROR: Cannot load mkl_intel_thread.dll.

这个问题，当我把 pycharm 升级到最新的时候，再安装 matplotlib 的时候总是遇到。

而且，是这样的，在 console 里执行这个的时候是没有问题的：

```py
from matplotlib import pyplot as plt
fig = plt.figure()
```

但是，在 pycharm 里用 py 脚本执行这个的时候就有问题：

```
Intel MKL FATAL ERROR: Cannot load mkl_intel_thread.dll.
```

之前查了下，以为是 numpy+mkl 的问题，因此，把 mkl 卸载了，然后重新用 pip 安装了 numpy，但是后面 用 conda 安装的时候又把 mkl 安装上了，所以这个问题还是没解决。

后来看了下：

- [Intel MKL FATAL ERROR: Cannot load mkl_intel_thread.dll. ](https://intellij-support.jetbrains.com/hc/en-us/community/posts/360000365510-Intel-MKL-FATAL-ERROR-Cannot-load-mkl-intel-thread-dll-)
- [Anaconda Intel MKL FATAL ERROR when running scripts in 2017.3 and no py.test suites found in project](https://youtrack.jetbrains.com/issue/PY-27466)

这个问题之后，发现，好像新版本的 pycharm 就是有这个问题，应该是 pycharm 的 科学计算模式是要对接 matplotlib 来画图。

所以是有这个问题的。

换了低版本的 pycharm 应该是没有这个问题的。


原文及相关：

- [Intel MKL FATAL ERROR: Cannot load mkl_intel_thread.dll. ](https://intellij-support.jetbrains.com/hc/en-us/community/posts/360000365510-Intel-MKL-FATAL-ERROR-Cannot-load-mkl-intel-thread-dll-)
- [Anaconda Intel MKL FATAL ERROR when running scripts in 2017.3 and no py.test suites found in project](https://youtrack.jetbrains.com/issue/PY-27466)
 project](https://youtrack.jetbrains.com/issue/PY-27466)






## Anaconda 换源

这个还是必须要知道的。因为使用默认的国外的源速度有的时候非常的慢，因此还是会经常要配置国内的源的：

不错的源有两个：中科大的和清华的：

添加中科大的源：

```
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```

添加清华的源：（清华的源有的时候不是很稳定，有的快，有的慢）

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```

当用户第一次运行 `conda config` 命令时，将会在用户的家目录创建一个 `.condarc` 配置文件，一般会在 windows：`C:\users\username\`，linux：`/home/username/` 下生成。

在这个配置文件中可以修改删除或者添加想要的源：

比如我现在的 `.condarc` 文件内容是这样的：

```
channels:
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - defaults
show_channel_urls: true
```

defaults 是官方默认的源，当 conda 在寻找安装包的时候，应该是从上到下查找源的。<span style="color:red;">这个不确定</span>

