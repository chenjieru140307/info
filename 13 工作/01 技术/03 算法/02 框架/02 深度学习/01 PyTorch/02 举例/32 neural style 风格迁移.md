# neural style 风格迁移


## 数据准备


- [COCO](http://cocodataset.org/#download)
- [百度网盘](https://pan.baidu.com/s/1IWZqX5RXVUzLJA_71zI0tw) 提取码：eu0q 
- 也可以使用其它数据，比如ImageNet。请尽量保证数据的多样性，**不建议**使用单一种类的数据集，比如LSUN或者人脸识别数据集之类的。

请确保所有的图片保存于`data/coco/`文件夹下,形如：

```Bash
data
 └─ coco
    ├── COCO_train2014_000000000009.jpg
    ├── COCO_train2014_000000000025.jpg
    ├── COCO_train2014_000000000030.jpg
```


## 代码

utils.py

```py
# coding:utf8
from itertools import chain
import visdom
import torch as t
import time
import torchvision as tv
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def gram_matrix(y):
    """
    Input shape: b,c,h,w
    Output shape: b,c,c
    """
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


class Visualizer():
    """
    wrapper on visdom, but you may still call native visdom by `self.vis.function`
    """

    def __init__(self, env='default', **kwargs):
        import visdom
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)

        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        
        """
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)
        return self

    def plot_many(self, d):
        """
        plot multi values in a time
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append'
                      )
        self.index[name] = x + 1

    def img(self, name, img_):
        """
        self.img('input_img',t.Tensor(64,64))
        """

        if len(img_.size()) < 3:
            img_ = img_.cpu().unsqueeze(0)
        self.vis.image(img_.cpu(),
                       win=name,
                       opts=dict(title=name)
                       )

    def img_grid_many(self, d):
        for k, v in d.items():
            self.img_grid(k, v)

    def img_grid(self, name, input_3d):
        """
        convert batch images to grid of images
        i.e. input（36，64，64） ->  6*6 grid，each grid is an image of size 64*64
        """
        self.img(name, tv.utils.make_grid(
            input_3d.cpu()[0].unsqueeze(1).clamp(max=1, min=0)))

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win=win)

    def __getattr__(self, name):
        return getattr(self.vis, name)


def get_style_data(path):
    """
    load style image，
    Return： tensor shape 1*c*h*w, normalized
    """
    style_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    style_image = tv.datasets.folder.default_loader(path)
    style_tensor = style_transform(style_image)
    return style_tensor.unsqueeze(0)


def normalize_batch(batch):
    """
    Input: b,ch,h,w  0~255
    Output: b,ch,h,w  -2~2
    """
    mean = batch.data.new(IMAGENET_MEAN).view(1, -1, 1, 1)
    std = batch.data.new(IMAGENET_STD).view(1, -1, 1, 1)
    mean = (mean.expand_as(batch.data))
    std = (std.expand_as(batch.data))
    return (batch / 255.0 - mean) / std
```

PackedVGG.py

```py
# coding:utf8
import torch
import torch.nn as nn
from torchvision.models import vgg16
from collections import namedtuple


class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(vgg16(pretrained=True).features)[:23]
        # the 3rd, 8th, 15th and 22nd layer of \ 
        # self.features are: relu1_2,relu2_2,relu3_3,relu4_3
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {3, 8, 15, 22}:
                results.append(x)

        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        return vgg_outputs(*results)
```

transformer_net.py

```py
# coding:utf8
"""
code refer to https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/transformer_net.py
"""
import torch as t
from torch import nn
import numpy as np


class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()

        # Down sample layers
        self.initial_layers = nn.Sequential(
            ConvLayer(3, 32, kernel_size=9, stride=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(True),
            ConvLayer(32, 64, kernel_size=3, stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(True),
            ConvLayer(64, 128, kernel_size=3, stride=2),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(True),
        )

        # Residual layers
        self.res_layers = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )

        # Upsampling Layers
        self.upsample_layers = nn.Sequential(
            UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(True),
            UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(True),
            ConvLayer(32, 3, kernel_size=9, stride=1)
        )

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.res_layers(x)
        x = self.upsample_layers(x)
        return x


class ConvLayer(nn.Module):
    """
    add ReflectionPad for Conv
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(nn.Module):
    """UpsampleConvLayer
    instead of ConvTranspose2d, we do UpSample + Conv2d
    see ref for why.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = t.nn.functional.interpolate(x_in, scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class ResidualBlock(nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out

```


main.py

```py
# coding:utf8

import torch as t
import torchvision as tv
import torchnet as tnt

from torch.utils import data
from transformer_net import TransformerNet
import utils
from PackedVGG import Vgg16
from torch.nn import functional as F
import tqdm
import os
import ipdb

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class Config(object):
    # General Args
    use_gpu = True
    model_path = None  # pretrain model path (for resume training or test)

    # Train Args
    image_size = 256  # image crop_size for training
    batch_size = 8
    data_root = 'D://BaiduNetdiskDownload/coco/'  # dataset root：$data_root/coco/a.jpg
    num_workers = 4  # dataloader num of workers

    lr = 1e-3
    epoches = 2  # total epoch to train
    content_weight = 1e5  # weight of content_loss
    style_weight = 1e10  # weight of style_loss

    style_path = 'style.jpg'  # style image path
    env = 'neural-style'  # visdom env
    plot_every = 10  # visualize in visdom for every 10 batch

    debug_file = '/tmp/debugnn'  # touch $debug_fie to interrupt and enter ipdb

    # Test Args
    content_path = 'input.png'  # input file to do style transfer [for test]
    result_path = 'output.png'  # style transfer result [for test]


def train(**kwargs):
    opt = Config()
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = t.device('cuda') if opt.use_gpu else t.device('cpu')
    vis = utils.Visualizer(opt.env)

    # Data loading
    transfroms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x * 255)
    ])
    dataset = tv.datasets.ImageFolder(opt.data_root, transfroms)
    dataloader = data.DataLoader(dataset, opt.batch_size)

    # style transformer network
    transformer = TransformerNet()
    if opt.model_path:
        transformer.load_state_dict(t.load(opt.model_path, map_location=lambda _s, _: _s))
    transformer.to(device)

    # Vgg16 for Perceptual Loss
    vgg = Vgg16().eval()
    vgg.to(device)
    for param in vgg.parameters():
        param.requires_grad = False

    # Optimizer
    optimizer = t.optim.Adam(transformer.parameters(), opt.lr)

    # Get style image
    style = utils.get_style_data(opt.style_path)
    vis.img('style', (style.data[0] * 0.225 + 0.45).clamp(min=0, max=1))
    style = style.to(device)

    # gram matrix for style image
    with t.no_grad():
        features_style = vgg(style)
        gram_style = [utils.gram_matrix(y) for y in features_style]

    # Loss meter
    style_meter = tnt.meter.AverageValueMeter()
    content_meter = tnt.meter.AverageValueMeter()

    for epoch in range(opt.epoches):
        content_meter.reset()
        style_meter.reset()

        for ii, (x, _) in tqdm.tqdm(enumerate(dataloader)):

            # Train
            optimizer.zero_grad()
            x = x.to(device)
            y = transformer(x)
            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)
            features_y = vgg(y)
            features_x = vgg(x)

            # content loss
            content_loss = opt.content_weight * F.mse_loss(features_y.relu2_2, features_x.relu2_2)

            # style loss
            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gram_y = utils.gram_matrix(ft_y)
                style_loss += F.mse_loss(gram_y, gm_s.expand_as(gram_y))
            style_loss *= opt.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            # Loss smooth for visualization
            content_meter.add(content_loss.item())
            style_meter.add(style_loss.item())

            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # visualization
                vis.plot('content_loss', content_meter.value()[0])
                vis.plot('style_loss', style_meter.value()[0])
                # denorm input/output, since we have applied (utils.normalize_batch)
                vis.img('output', (y.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))
                vis.img('input', (x.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))

        # save checkpoint
        vis.save([opt.env])
        t.save(transformer.state_dict(), 'checkpoints/%s_style.pth' % epoch)


@t.no_grad()
def stylize(**kwargs):
    """
    perform style transfer
    """
    opt = Config()

    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    device = t.device('cuda') if opt.use_gpu else t.device('cpu')

    # input image preprocess
    content_image = tv.datasets.folder.default_loader(opt.content_path)
    content_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device).detach()

    # model setup
    style_model = TransformerNet().eval()
    style_model.load_state_dict(t.load(opt.model_path, map_location=lambda _s, _: _s))
    style_model.to(device)

    # style transfer and save output
    output = style_model(content_image)
    output_data = output.cpu().data[0]
    tv.utils.save_image(((output_data / 255)).clamp(min=0, max=1), opt.result_path)


if __name__ == '__main__':
    import fire
    fire.Fire()
```



## 用法

- 启动visdom服务：
  - `python -m visdom.server`
- 训练：
  - `python main.py train --use-gpu --data-root=data --batch-size=2`
  - 如果需要训练其它风格的图片，只需要修改 `--style-path` 对应的风格图片。
- 测试
  - 建议下载一张高清的图片，这样风格迁移的效果会比较好。
  - `python main.py stylize  --model-path='transformer.pth' --content-path='amber.jpg' --result-path='output2.png' --use-gpu=False`

完整的选项及默认值：

```python
image_size = 256 # 图片大小
batch_size = 8  
data_root = 'data/' # 数据集存放路径：data/coco/a.jpg
num_workers = 4 # 多线程加载数据
use_gpu = True # 使用GPU

style_path= 'style.jpg' # 风格图片存放路径
lr = 1e-3 # 学习率

env = 'neural-style' # visdom env
plot_every=10 # 每10个batch可视化一次

epoches = 2 # 训练epoch

content_weight = 1e5 # content_loss 的权重 
style_weight = 1e10 # style_loss的权重

model_path = None # 预训练模型的路径
debug_file = '/tmp/debugnn' # touch $debug_fie 进入调试模式 

content_path = 'input.png' # 需要进行风格迁移的图片
result_path = 'output.png' # 风格迁移结果的保存路径
```


训练：

- `python main.py train --use-gpu`

其中，style.jpg：

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200527/Ok45s1rajSM6.jpg?imageslim">
</p>



输出：


```txt
Setting up a new session...
Without the incoming socket you cannot receive events from the server or register event handlers to your Visdom
 client.
10348it [1:06:10,  2.61it/s]
10348it [1:11:15,  2.42it/s]
```


style_loss：

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200527/48FKG05mq7dz.png?imageslim">
</p>


content_loss：

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200527/GBwDRorLGE0w.png?imageslim">
</p>

测试：

- `python main.py stylize  --model-path='./checkpoints/1_style.pth' --content-pa
th='cat.jpg' --result-path='output2.png' --use-gpu=False`

cat.jpg：

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200527/uFdngTI18BUI.jpg?imageslim">
</p>

output2.png：

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200527/vO3ySfJY1JsV.png?imageslim">
</p>


