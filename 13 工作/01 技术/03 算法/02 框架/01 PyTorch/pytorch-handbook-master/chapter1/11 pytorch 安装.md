

[anaconda 清华镜像](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)，之前由于合规问题中科大、清华镜像都已经关闭。目前只有清华镜像恢复，所以目前可以继续使用

## 安装 Pytorch

```bash
#默认 使用 cuda10.1
pip3 install torch===1.3.0 torchvision===0.4.1 -f https://download.pytorch.org/whl/torch_stable.html

#cuda 9.2
pip3 install torch==1.3.0+cu92 torchvision==0.4.1+cu92 -f https://download.pytorch.org/whl/torch_stable.html

#cpu版本
pip3 install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```


验证：

```python
import torch
torch.__version__
```

