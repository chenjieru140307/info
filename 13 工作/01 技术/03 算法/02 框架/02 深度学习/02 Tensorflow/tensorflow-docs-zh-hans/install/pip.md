### 可用的包

- `tensorflow` —— 当前的 CPU-only 发布版本 *（推荐初学者使用）*
- `tensorflow-gpu` —— [支持 GPU ](./gpu)的当前发布版本 *（适用于 Ubuntu 和 Windows 系统）*
- `tf-nightly` —— Nightly build for CPU-only *（不稳定）*
- `tf-nightly-gpu` —— [支持 GPU](./gpu)的每日构建版本 *（不稳定，适用于 Ubuntu 和 Windows 系统）*

### 系统需求

- Ubuntu 16.04 或更高（64 位）
- macOS 10.12.6 (Sierra) 或更高（64 位）*（不支持 GPU）*
- Windows 7 或更高（64 位）*（仅支持 Python 3）*
- Raspbian 9.0 或更高

### 硬件需求

- 自 TensorFlow 1.6 起，二进制文件会使用 [AVX 指令集](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#CPUs_with_AVX)，这在旧的 CPU 上可能无法运行。
- 阅读 [GPU 支持](./gpu) 来在 Ubuntu 或 Windows 上配置支持 CUDA® 加速的显卡。

## 1. 在系统中安装 Python 开发环境

 Python 3 Python 2.7

检查 Python 环境是否正确配置：

{% dynamic if request.query_string.lang == "python2" %}

```bsh
python --version
pip --version
virtualenv --version
```

{% dynamic else %}

Requires Python 3.4, 3.5, or 3.6

```bsh
python3 --version
pip3 --version
virtualenv --version
```

{% dynamic endif %}

如果你已经安装了这些包，可以跳到下一步。
否则，安装 [Python](https://www.python.org/)， [pip 包管理器](https://pip.pypa.io/en/stable/installing/)， 和 [Virtualenv](https://virtualenv.pypa.io/en/stable/)：

{% dynamic if request.query_string.lang == "python2" %}

### Ubuntu

```bsh
sudo apt update
sudo apt install python-dev python-pip
sudo pip install -U virtualenv  # 系统层面安装
```

### mac OS

通过 [Homebrew](https://brew.sh/) 包管理器安装：

```bsh
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
brew update
brew install python@2  # Python 2
sudo pip install -U virtualenv  # 系统层面安装
```

### Raspberry Pi

需要 [Raspbian](https://www.raspberrypi.org/downloads/raspbian/) 操作系统：

```bsh
sudo apt update
sudo apt install python-dev python-pip
sudo apt install libatlas-base-dev     # numpy 需要的库
sudo pip install -U virtualenv         # 系统层面安装
```

### 其它系统

```bsh
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
sudo pip install -U virtualenv  # 系统层面安装
```

{% dynamic else %}

### Ubuntu

```bsh
sudo apt update
sudo apt install python3-dev python3-pip
sudo pip3 install -U virtualenv  # 系统层面安装
```

### mac OS

通过 [Homebrew](https://brew.sh/) 包管理器安装：

```bsh
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
brew update
brew install python  # Python 3
sudo pip3 install -U virtualenv  # 系统层面安装
```

### Windows

安装 [Python 3 的 Windows 发布版本](https://www.python.org/downloads/windows/)（勾选 `pip` 安装选项）。

```
pip3 install -U pip virtualenv
```

### Raspberry Pi

需要 [Raspbian](https://www.raspberrypi.org/downloads/raspbian/) 操作系统：

```bsh
sudo apt update
sudo apt install python3-dev python3-pip
sudo apt install libatlas-base-dev        # numpy 需要的库
sudo pip3 install -U virtualenv           # 系统层面安装
```

### 其它系统

```bsh
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
sudo pip3 install -U virtualenv  # 系统层面安装
```

{% dynamic endif %}

## 2. 创建虚拟环境（推荐）

Python 虚拟环境用于让包安装独立于系统环境。

### Ubuntu / mac OS

新建一个 `./venv` 目录，选择一个 Python 解释器，在其中创建一个新的虚拟环境：

{% dynamic if request.query_string.lang == "python2" %}

```
virtualenv --system-site-packages -p python2.7 ./venv
```

{% dynamic else %}

```
virtualenv --system-site-packages -p python3 ./venv
```

{% dynamic endif %}

用一个 shell 类的命令激活虚拟环境：

```bsh
source ./venv/bin/activate  # 使用 sh、bash、ksh 或者 zsh
```

当 virtualenv 激活时，你的 shell 提示符会带上 `(venv)` 前缀。

在虚拟环境内安装包将不会影响主机的系统环境。 从更新 `pip` 开始：

```bsh
pip install --upgrade pip

pip list  # 展示虚拟环境中已安装的包列表
```

需要退出 virtualenv 时：

```bsh
deactivate  # 运行完 TensorFlow 后再退出
```

### Windows

新建一个 `./venv` 目录，选择一个 Python 解释器，在其中创建一个新的虚拟环境：

{% dynamic if request.query_string.lang == "python2" %}

在 Windows 上，TensorFlow 不支持 Python 2.7

{% dynamic else %}

```
virtualenv --system-site-packages -p python3 ./venv
```

{% dynamic endif %}

激活虚拟环境：

```
.\venv\Scripts\activate
```

在虚拟环境内安装包将不会影响主机的系统环境。 从更新 `pip` 开始：

```bsh
pip install --upgrade pip

pip list  # 展示虚拟环境中已安装的包列表
```

需要退出 virtualenv 时：

```bsh
deactivate  # 运行完 TensorFlow 后再退出
```

### Conda

虽然我们推荐使用 TensorFlow 官方出品的 *pip* 包，但 *社区支持的* [Anaconda 包](https://anaconda.org/conda-forge/tensorflow) 也是可以使用的。

新建一个 `./venv` 目录，选择一个 Python 解释器，在其中创建一个新的虚拟环境：

{% dynamic if request.query_string.lang == "python2" %}

```
conda create -n venv pip python=2.7
```

{% dynamic else %}

```
conda create -n venv pip python=3.6  # 选择 python 版本
```

{% dynamic endif %}

激活虚拟环境：

```
source activate venv
```

在虚拟环境中，通过[完整的 URL](#package-location) 安装 TensorFlow pip 包：

```bsh
pip install --ignore-installed --upgrade packageURL
```

需要退出 virtualenv 时：

```bsh
source deactivate
```

## 3. 安装 TensorFlow pip 包

从 [PyPI](https://pypi.org/project/tensorflow/) 选择以下一种 TensorFlow 包来安装：

- `tensorflow` —— 当前的 CPU-only 发布版本*（推荐初学者使用）*
- `tensorflow-gpu` —— [支持 GPU ](./gpu)的当前发布版本*（适用于 Ubuntu 和 Windows 系统）*
- `tf-nightly` —— Nightly build for CPU-only *（不稳定）*
- `tf-nightly-gpu` —— [支持 GPU](./gpu)的每日构建版本*（不稳定，适用于 Ubuntu 和 Windows 系统）*

依赖的包将被自动安装。`REQUIRED_PACKAGES` 下的 [`setup.py`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/setup.py) 文件中有完整的列表。

### Virtualenv 安装

```bsh
pip install --upgrade tensorflow
```

验证安装：

```bsh
python -c "import tensorflow as tf; print(tf.__version__)"
```

### 系统安装

{% dynamic if request.query_string.lang == "python2" %}

```bsh
pip install --user --upgrade tensorflow  # install in $HOME
```

验证安装：

```bsh
python -c "import tensorflow as tf; print(tf.__version__)"
```

{% dynamic else %}

```bsh
pip3 install --user --upgrade tensorflow  # install in $HOME
```

验证安装：

```bsh
python3 -c "import tensorflow as tf; print(tf.__version__)"
```

{% dynamic endif %}

**成功：** TensorFlow 已被成功安装。阅读[教程](../tutorials)来开始使用。

## 包地址

一些安装机制需要 TensorFlow Python 包的 URL。 具体的地址取决于你的 Python 版本。

| 版本                   | URL                                                          |
| ---------------------- | ------------------------------------------------------------ |
| Linux                  |                                                              |
| Python 2.7 CPU only    | https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.10.1-cp27-none-linux_x86_64.whl |
| Python 2.7 GPU support | https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.10.1-cp27-none-linux_x86_64.whl |
| Python 3.4 CPU only    | https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.10.1-cp34-cp34m-linux_x86_64.whl |
| Python 3.4 GPU support | https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.10.1-cp34-cp34m-linux_x86_64.whl |
| Python 3.5 CPU only    | https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.10.1-cp35-cp35m-linux_x86_64.whl |
| Python 3.5 GPU support | https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.10.1-cp35-cp35m-linux_x86_64.whl |
| Python 3.6 CPU only    | https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.10.1-cp36-cp36m-linux_x86_64.whl |
| Python 3.6 GPU support | https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.10.1-cp36-cp36m-linux_x86_64.whl |
| macOS                  |                                                              |
| Python 2.7             | https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.10.1-py2-none-any.whl |
| Python 3.4, 3.5, 3.6   | https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.10.1-py3-none-any.whl |