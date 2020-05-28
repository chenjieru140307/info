# PyTorch常用工具模块

在训练神经网络过程中，需要用到很多工具，其中最重要的三部分是：数据、可视化和GPU加速。本章主要介绍Pytorch在这几方面的工具模块，合理使用这些工具能够极大地提高编码效率。


## 数据处理


### 数据加载

在PyTorch中，数据加载可通过自定义的数据集对象。数据集对象被抽象为`Dataset`类，实现自定义的数据集需要继承Dataset，并实现两个Python魔法方法：

- `__getitem__`：返回一条数据，或一个样本。`obj[index]`等价于`obj.__getitem__(index)`
- `__len__`：返回样本的数量。`len(obj)`等价于`obj.__len__()`


```py
import torch as t
from torch.utils import data

import os
from PIL import Image
import numpy as np


class DogCat(data.Dataset):
    def __init__(self, root):
        imgs = os.listdir(root)
        # 所有图片的绝对路径 这里不实际加载图片，只是指定路径，当调用__getitem__时才会真正读图片
        self.imgs = [os.path.join(root, img) for img in imgs]

    def __getitem__(self, index):
        img_path = self.imgs[index]
        # dog->1， cat->0
        label = 1 if 'dog' in img_path.split('/')[-1] else 0
        pil_img = Image.open(img_path)
        array = np.asarray(pil_img)
        data = t.from_numpy(array)
        return data, label

    def __len__(self):
        return len(self.imgs)


dataset = DogCat('./data/dogcat/')
img, label = dataset[0]  # 相当于调用dataset.__getitem__(0)
for img, label in dataset:
    print(img.size(), img.float().mean(), label)
```

输出：

```txt
torch.Size([500, 497, 3]) tensor(106.4915) 0
torch.Size([499, 379, 3]) tensor(171.8085) 0
torch.Size([236, 289, 3]) tensor(130.3004) 0
torch.Size([374, 499, 3]) tensor(115.5177) 0
torch.Size([375, 499, 3]) tensor(116.8139) 1
torch.Size([375, 499, 3]) tensor(150.5080) 1
torch.Size([377, 499, 3]) tensor(151.7174) 1
torch.Size([400, 300, 3]) tensor(128.1550) 1
```

注，data 下 catdog 文件夹如下：

<p align="center">
    <img width="40%" height="70%" src="http://images.iterate.site/blog/image/20200526/SKbLDXsmEqRH.png?imageslim">
</p>



这里返回的数据不适合实际使用，因其具有如下两方面问题：

- 返回样本的形状不一，因每张图片的大小不一样，这对于需要取batch训练的神经网络来说很不友好
- 返回样本的数值较大，未归一化至[-1, 1]

针对上述问题，PyTorch提供了torchvision。它是一个视觉工具包，提供了很多视觉图像处理的工具，其中 `transforms` 模块提供了对PIL `Image` 对象和 `Tensor` 对象的常用操作。


对PIL Image的操作包括：


- `Scale`：调整图片尺寸，长宽比保持不变
- `CenterCrop`、`RandomCrop`、`RandomResizedCrop`： 裁剪图片
- `Pad`：填充
- `ToTensor`：将PIL Image对象转成Tensor，会自动将[0, 255]归一化至[0, 1]

对Tensor的操作包括：

- Normalize：标准化，即减均值，除以标准差
- ToPILImage：将Tensor转为PIL Image对象

如果要对图片进行多个操作，可通过`Compose` 函数将这些操作拼接起来，类似于`nn.Sequential`。注意，这些操作定义后是以函数的形式存在，真正使用时需调用它的`__call__`方法，这点类似于 `nn.Module`。例如要将图片调整为 $224\times 224$，首先应构建这个操作 `trans = Resize((224, 224))`，然后调用 `trans(img)`。




下面我们就用transforms的这些操作来优化上面实现的dataset。


```py
import os
from PIL import Image
from torchvision import transforms as T
from torch.utils import data

transform = T.Compose([
    T.Resize(224),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
    T.CenterCrop(224),  # 从图片中间切出224 * 224 的图片
    T.ToTensor(),  # 将图片(Image) 转成Tensor，归一化至[0, 1]
    T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至[-1, 1]，规定均值和标准差
])


class DogCat(data.Dataset):
    def __init__(self, root, transforms=None):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 0 if 'dog' in img_path.split('/')[-1] else 1
        data = Image.open(img_path)
        if self.transforms:
            data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


dataset = DogCat('./data/dogcat/', transforms=transform)
img, label = dataset[0]
for img, label in dataset:
    print(img.size(), label)

```

输出：

```txt
torch.Size([3, 224, 224]) 1
torch.Size([3, 224, 224]) 1
torch.Size([3, 224, 224]) 1
torch.Size([3, 224, 224]) 1
torch.Size([3, 224, 224]) 0
torch.Size([3, 224, 224]) 0
torch.Size([3, 224, 224]) 0
torch.Size([3, 224, 224]) 0
```


除了上述操作之外，transforms还可通过`Lambda`封装自定义的转换策略。例如想对PIL Image进行随机旋转，则可写成这样`trans=T.Lambda(lambda img: img.rotate(random()*360))`。

torchvision已经预先实现了常用的Dataset，包括前面使用过的CIFAR-10，以及ImageNet、COCO、MNIST、LSUN等数据集，可通过诸如`torchvision.datasets.CIFAR10`来调用，具体使用方法请参看[官方文档](http://pytorch.org/docs/master/torchvision/datasets.html)。

在这里介绍一个会经常使用到的Dataset——`ImageFolder`，它的实现和上述的`DogCat`很相似。`ImageFolder`假设所有的文件按文件夹保存，每个文件夹下存储同一个类别的图片，文件夹名为类名，其构造函数如下：

- `ImageFolder(root, transform=None, target_transform=None, loader=default_loader)`
  - `root`：在root指定的路径下寻找图片
  - `transform`：对PIL Image进行的转换操作，transform的输入是使用loader读取图片的返回对象
  - `target_transform`：对label的转换
  - `loader`：给定路径后如何读取图片，默认读取为RGB格式的PIL Image对象

label 是按照文件夹名顺序排序后存成字典，即{类名:类序号(从0开始)}，一般来说最好直接将文件夹命名为从0开始的数字，这样会和ImageFolder实际的label一致，如果不是这种命名规范，建议看看 `self.class_to_idx` 属性以了解label和文件夹名的映射关系。


举例：

```py
import os
from PIL import Image
from torchvision import transforms as T
from torch.utils import data
from torchvision.datasets import ImageFolder

dataset = ImageFolder('data/dogcat_2/')
print(dataset.class_to_idx) # cat文件夹的图片对应label 0，dog对应1
print(dataset.imgs) # 所有图片的路径和对应的label
# 没有任何的transform，所以返回的还是PIL Image对象
print(dataset[0][1])  # 第一维是第几张图，第二维为1返回label
print(dataset[0][0])  # 为0返回图片数据
print()

# 加上transform
normalize = T.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    normalize,
])
dataset = ImageFolder('data/dogcat_2/', transform=transform)
# 深度学习中图片数据一般保存成CxHxW，即通道数x图片高x图片宽
print(dataset[0][0].size())
to_img = T.ToPILImage()
# 0.2和0.4是标准差和均值的近似
img = to_img(dataset[0][0] * 0.2 + 0.4)
```

输出：

```txt
{'cat': 0, 'dog': 1}
[('data/dogcat_2/cat\\cat.12484.jpg', 0), ('data/dogcat_2/cat\\cat.12485.jpg', 0), ('data/dogcat_2/cat\\cat.12486.jpg', 0), ('data/dogcat_2/cat\\cat.12487.jpg', 0), ('data/dogcat_2/dog\\dog.12496.jpg', 1), ('data/dogcat_2/dog\\dog.12497.jpg', 1), ('data/dogcat_2/dog\\dog.12498.jpg', 1), ('data/dogcat_2/dog\\dog.12499.jpg', 1)]
0
<PIL.Image.Image image mode=RGB size=497x500 at 0x1F3C685CB08>

torch.Size([3, 224, 224])
```




- DataLoader
  - `Dataset`只负责数据的抽象，一次调用`__getitem__`只返回一个样本。前面提到过，在训练神经网络时，最好是对一个batch的数据进行操作，同时还需要对数据进行shuffle和并行加速等。对此，PyTorch提供了`DataLoader`帮助我们实现这些功能。
  - DataLoader的函数定义如下：`DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False)`
    - dataset：加载的数据集(Dataset对象)
    - batch_size：batch size
    - shuffle:：是否将数据打乱
    - sampler： 样本抽样，后续会详细介绍
    - num_workers：使用多进程加载的进程数，0代表不使用多进程
    - collate_fn： 如何将多个样本数据拼接成一个batch，一般使用默认的拼接方式即可
    - pin_memory：是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些
    - drop_last：dataset中的数据个数可能不是batch_size的整数倍，drop_last为True会将多出来不足一个batch的数据丢弃
- dataloader 是一个可迭代的对象，意味着我们可以像使用迭代器一样使用它
- 在数据处理中，有时会出现某个样本无法读取等问题，比如某张图片损坏。这时在 `__getitem__` 函数中将出现异常，此时最好的解决方案即是将出错的样本剔除。如果实在是遇到这种情况无法处理，则可以返回None对象，然后在 `Dataloader` 中实现自定义的 `collate_fn`，将空对象过滤掉。但要注意，在这种情况下dataloader 返回的 batch 数目会少于batch_size。


举例：

```py
import torch as t
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
from PIL import Image
from torchvision import transforms as T
from torch.utils import data





# 加上transform
normalize = T.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    normalize,
])
dataset = ImageFolder('data/dogcat_2/', transform=transform)


# 使用 DataLoader
dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0, drop_last=False)
dataiter = iter(dataloader)
imgs, labels = next(dataiter)
print(imgs.size())  # batch_size, channel, height, weight
for batch_datas, batch_labels in dataloader:
    # train()
    pass



transform = T.Compose([
    T.Resize(224),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
    T.CenterCrop(224),  # 从图片中间切出224 * 224 的图片
    T.ToTensor(),  # 将图片(Image) 转成Tensor，归一化至[0, 1]
    T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至[-1, 1]，规定均值和标准差
])


class DogCat(data.Dataset):
    def __init__(self, root, transforms=None):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 0 if 'dog' in img_path.split('/')[-1] else 1
        data = Image.open(img_path)
        if self.transforms:
            data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


dataset = DogCat('./data/dogcat/', transforms=transform)
img, label = dataset[0]
for img, label in dataset:
    print(img.size(), label)


class NewDogCat(DogCat):  # 继承前面实现的DogCat数据集
    def __getitem__(self, index):
        try:
            # 调用父类的获取函数，即 DogCat.__getitem__(self, index)
            return super(NewDogCat, self).__getitem__(index)
        except:
            return None, None


from torch.utils.data.dataloader import default_collate  # 导入默认的拼接方式
def my_collate_fn(batch):
    '''
    batch中每个元素形如(data, label)
    '''
    # 过滤为None的数据
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        return t.Tensor()
    return default_collate(batch)  # 用默认方式拼接过滤后的batch数据

dataset = NewDogCat('data/dogcat_wrong/', transforms=transform)
print(dataset[5])

dataloader = DataLoader(dataset, batch_size=2, collate_fn=my_collate_fn, shuffle=True, num_workers=0)
for batch_datas, batch_labels in dataloader:
    print(batch_datas.size(), batch_labels.size())
```

输出：

```txt
torch.Size([3, 3, 224, 224])
torch.Size([3, 224, 224]) 1
torch.Size([3, 224, 224]) 1
torch.Size([3, 224, 224]) 1
torch.Size([3, 224, 224]) 1
torch.Size([3, 224, 224]) 0
torch.Size([3, 224, 224]) 0
torch.Size([3, 224, 224]) 0
torch.Size([3, 224, 224]) 0
(tensor([[[-0.1137, -0.0667, -0.0667,  ...,  0.4902,  0.4118,  0.4039],
         [-0.2235, -0.1843, -0.1608,  ...,  0.4824,  0.4039,  0.3961],
         [-0.3020, -0.2471, -0.2078,  ...,  0.4980,  0.4039,  0.3961],
         ...,
         [-0.7255, -0.7176, -0.7333,  ..., -0.5608, -0.5451, -0.4039],
         [-0.7647, -0.7176, -0.7098,  ..., -0.5608, -0.5451, -0.4039],
         [-0.7804, -0.7569, -0.7176,  ..., -0.5608, -0.5451, -0.4039]],

        [[ 0.1451,  0.1765,  0.1686,  ...,  0.4510,  0.3490,  0.3255],
         [ 0.0275,  0.0510,  0.0745,  ...,  0.4431,  0.3490,  0.3098],
         [-0.0667, -0.0196,  0.0118,  ...,  0.4510,  0.3490,  0.3176],
         ...,
         [-0.6235, -0.6392, -0.6471,  ..., -0.4196, -0.4353, -0.3098],
         [-0.6627, -0.6392, -0.6235,  ..., -0.4196, -0.4353, -0.3098],
         [-0.6784, -0.6784, -0.6314,  ..., -0.4196, -0.4353, -0.3098]],

        [[ 0.6784,  0.7490,  0.7490,  ...,  0.0275, -0.0667, -0.0745],
         [ 0.5294,  0.5843,  0.6235,  ...,  0.0196, -0.0745, -0.0902],
         [ 0.3882,  0.4588,  0.5059,  ...,  0.0275, -0.0745, -0.0902],
         ...,
         [-0.5529, -0.5216, -0.5373,  ..., -0.5765, -0.5373, -0.3804],
         [-0.5922, -0.5216, -0.5137,  ..., -0.5686, -0.5373, -0.3804],
         [-0.6078, -0.5608, -0.5137,  ..., -0.5686, -0.5373, -0.3804]]]), 0)
torch.Size([2, 3, 224, 224]) torch.Size([2])
torch.Size([1, 3, 224, 224]) torch.Size([1])
torch.Size([2, 3, 224, 224]) torch.Size([2])
torch.Size([2, 3, 224, 224]) torch.Size([2])
torch.Size([1, 3, 224, 224]) torch.Size([1])
```

说明：

- 来看一下上述 batch_size 的大小。其中第2个的 batch_size为1，这是因为有一张图片损坏，导致其无法正常返回。而最后1个的batch_size也为1，这是因为共有9张（包括损坏的文件）图片，无法整除2（batch_size），因此最后一个batch的数据会少于batch_szie，可通过指定`drop_last=True`来丢弃最后一个不足batch_size的batch。


对于诸如样本损坏或数据集加载异常等情况，还可以通过其它方式解决。例如但凡遇到异常情况，就随机取一张图片代替：



```py
class NewDogCat(DogCat):
    def __getitem__(self, index):
        try:
            return super(NewDogCat, self).__getitem__(index)
        except:
            new_index = random.randint(0, len(self)-1)
            return self[new_index]
```


相比较丢弃异常图片而言，这种做法会更好一些，因为它能保证每个batch的数目仍是batch_size。但在大多数情况下，最好的方式还是对数据进行彻底清洗。

DataLoader里面并没有太多的魔法方法，它封装了 Python 的标准库`multiprocessing`，使其能够实现多进程加速。在此提几点关于Dataset和DataLoader使用方面的建议：

- 高负载的操作放在 `__getitem__` 中，如加载图片等。
   - 因为多进程会并行的调用`__getitem__` 函数，将负载高的放在`__getitem__` 函数中能够实现并行加速。
- dataset 中应尽量只包含只读对象，避免修改任何可变对象，利用多线程进行操作。
  - 因为dataloader使用多进程加载，如果在`Dataset`实现中使用了可变对象，可能会有意想不到的冲突。在多线程/多进程中，修改一个可变对象，需要加锁，但是dataloader 的设计使得其很难加锁（在实际使用中也应尽量避免锁的存在），因此最好避免在dataset中修改可变对象。例如下面就是一个不好的例子，在多进程处理中`self.num` 可能与预期不符，这种问题不会报错，因此难以发现。如果一定要修改可变对象，建议使用Python标准库 `Queue` 中的相关数据结构。


```python
class BadDataset(Dataset):
    def __init__(self):
        self.datas = range(100)
        self.num = 0 取数据的次数
    def __getitem__(self, index):
        self.num += 1
        return self.datas[index]
```



使用 Python `multiprocessing` 库的另一个问题是，在使用多进程时，如果主程序异常终止（比如用Ctrl+C强行退出），相应的数据加载进程可能无法正常退出。这时你可能会发现程序已经退出了，但GPU显存和内存依旧被占用着，或通过 `top`、`ps aux` 依旧能够看到已经退出的程序，这时就需要手动强行杀掉进程。

建议使用如下命令：

- `ps x | grep <cmdline> | ps x`
  - 先用这句打印确认一下是否会误杀其它进程
- `ps x | grep <cmdline> | awk '{print $1}' | xargs kill`
  - `ps x`：获取当前用户的所有进程
  - `grep <cmdline>`：找到已经停止的PyTorch程序的进程，例如你是通过python train.py启动的，那你就需要写`grep 'python train.py'`
  - `awk '{print $1}'`：获取进程的pid
  - `xargs kill`：杀掉进程，根据需要可能要写成`xargs kill -9`强制杀掉进程




PyTorch中还单独提供了一个 `sampler` 模块，用来对数据进行采样。

常用的有随机采样器：`RandomSampler`，当dataloader的`shuffle`参数为True时，系统会自动调用这个采样器，实现打乱数据。默认的是采用`SequentialSampler`，它会按顺序一个一个进行采样。

这里介绍另外一个很有用的采样方法：`WeightedRandomSampler`，它会根据每个样本的权重选取数据，在样本比例不均衡的问题中，可用它来进行重采样。

构建 `WeightedRandomSampler` 时需提供两个参数：每个样本的权重`weights`、共选取的样本总数`num_samples`，以及一个可选参数`replacement`。权重越大的样本被选中的概率越大，待选取的样本数目一般小于全部的样本数目。`replacement`用于指定是否可以重复选取某一个样本，默认为True，即允许在一个epoch中重复采样某一个数据。如果设为False，则当某一类的样本被全部选取完，但其样本数目仍未达到num_samples时，sampler将不会再从该类中选择数据，此时可能导致`weights`参数失效。

下面举例说明。

```py
import torch as t
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
from PIL import Image
from torchvision import transforms as T
from torch.utils import data

# 加上transform
normalize = T.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    normalize,
])
dataset = ImageFolder('data/dogcat_2/', transform=transform)

# 使用 DataLoader
dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0, drop_last=False)
dataiter = iter(dataloader)
imgs, labels = next(dataiter)
print(imgs.size())  # batch_size, channel, height, weight
for batch_datas, batch_labels in dataloader:
    # train()
    pass

transform = T.Compose([
    T.Resize(224),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
    T.CenterCrop(224),  # 从图片中间切出224 * 224 的图片
    T.ToTensor(),  # 将图片(Image) 转成Tensor，归一化至[0, 1]
    T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至[-1, 1]，规定均值和标准差
])


class DogCat(data.Dataset):
    def __init__(self, root, transforms=None):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 0 if 'dog' in img_path.split('/')[-1] else 1
        data = Image.open(img_path)
        if self.transforms:
            data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


dataset = DogCat('data/dogcat/', transforms=transform)
# 狗的图片被取出的概率是猫的概率的两倍
# 两类图片被取出的概率与weights的绝对大小无关，只和比值有关
weights = [2 if label == 1 else 1 for data, label in dataset]
print(weights)

from torch.utils.data.sampler import WeightedRandomSampler

sampler = WeightedRandomSampler(weights, num_samples=9, replacement=True)
dataloader = DataLoader(dataset,
                        batch_size=3,
                        sampler=sampler)
for datas, labels in dataloader:
    print(labels.tolist())
print()

sampler = WeightedRandomSampler(weights, 8, replacement=False)
dataloader = DataLoader(dataset, batch_size=4, sampler=sampler)
for datas, labels in dataloader:
    print(labels.tolist())
```

输出：

```txt
torch.Size([3, 3, 224, 224])
[2, 2, 2, 2, 1, 1, 1, 1]
[0, 1, 1]
[1, 1, 1]
[1, 1, 0]

[0, 0, 1, 0]
[1, 1, 1, 0]
```

说明：

- 可见猫狗样本比例约为1:2，另外一共只有8个样本，但是却返回了9个，说明肯定有被重复返回的，这就是 replacement 参数的作用，下面将 replacement 设为 False 试试。
-  replacement 设为 False 后，num_samples 等于dataset的样本总数，为了不重复选取，sampler 会将每个样本都返回，这样就失去 weight 参数的意义了。
- 从上面的例子可见sampler在样本采样中的作用：如果指定了sampler，shuffle将不再生效，并且sampler.num_samples会覆盖dataset的实际大小，即一个epoch返回的图片总数取决于`sampler.num_samples`。



## 计算机视觉工具包：torchvision


安装：

- `pip instal torchvision`


torchvision 主要包含三部分：


- models：提供深度学习中各种经典网络的网络结构以及预训练好的模型，包括`AlexNet`、VGG系列、ResNet系列、Inception系列等。
- datasets： 提供常用的数据集加载，设计上都是继承`torhc.utils.data.Dataset`，主要包括`MNIST`、`CIFAR10/100`、`ImageNet`、`COCO`等。
- transforms：提供常用的数据预处理操作，主要包括对Tensor以及PIL Image对象的操作。


说明：

- Transforms中涵盖了大部分对Tensor和PIL Image的常用处理，这些已在上文提到，这里就不再详细介绍。
  - 需要注意的是转换分为两步，
    - 第一步：构建转换操作，例如`transf = transforms.Normalize(mean=x, std=y)`，
    - 第二步：执行转换操作，例如`output = transf(input)`。另外还可将多个处理操作用Compose拼接起来，形成一个处理转换流程。
- torchvision还提供了两个常用的函数。一个是`make_grid`，它能将多张图片拼接成一个网格中；另一个是`save_img`，它能将Tensor保存成图片。

举例：

```py
from torchvision import models
from torch import nn
from torchvision import transforms
import torch as t
from torchvision.utils import make_grid, save_image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
from PIL import Image
from torchvision import transforms as T
from torch.utils import data
from torchvision import datasets

# 加载预训练好的模型，如果不存在会进行下载
# 预训练好的模型保存在 ~/.torch/models/下面
resnet34 = models.squeezenet1_1(pretrained=True, num_classes=1000)
# 修改最后的全连接层为10分类问题（默认是ImageNet上的1000分类）
resnet34.fc = nn.Linear(512, 10)

transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
])
# 指定数据集路径为data，如果数据集不存在则进行下载
# 通过train=False获取测试集
dataset = datasets.MNIST('data/', download=True, train=False, transform=transform)
print(len(dataset))

dataloader = DataLoader(dataset, shuffle=True, batch_size=16)
dataiter = iter(dataloader)
img_tensor = make_grid(next(dataiter)[0], 4)  # 拼成4*4网格图片，且会转成３通道
print(img_tensor.size())

to_pil = transforms.ToPILImage()
# img_pil=to_pil(t.randn(3, 64, 64))
img_pil = to_pil(img_tensor)

save_image(img_tensor, 'a.png')
Image.open('a.png')
```

输出：

```txt
10000
torch.Size([3, 906, 906])
```

图像：

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200527/wqQXo1eeQa6B.png?imageslim">
</p>




## 使用GPU加速：cuda

这部分内容在前面介绍Tensor、Module时大都提到过，这里将做一个总结，并深入介绍相关应用。

在PyTorch中以下数据结构分为CPU和GPU两个版本：

- Tensor
- nn.Module（包括常用的layer、loss function，以及容器Sequential等）

它们都带有一个`.cuda`方法，调用此方法即可将其转为对应的GPU对象。注意，`tensor.cuda`会返回一个新对象，这个新对象的数据已转移至GPU，而之前的tensor还在原来的设备上（CPU）。而 `module.cuda`则会将所有的数据都迁移至GPU，并返回自己。所以 `module = module.cuda()`和 `module.cuda()` 所起的作用一致。


nn.Module 在 GPU 与 CPU 之间的转换，本质上还是利用了Tensor在GPU和CPU之间的转换。`nn.Module` 的cuda方法是将nn.Module 下的所有 parameter（包括子module的parameter）都转移至GPU，而Parameter 本质上也是 tensor(Tensor的子类)。

下面将举例说明，这部分代码需要你具有两块GPU设备。

P.S. 为什么将数据转移至GPU的方法叫做`.cuda`而不是`.gpu`，就像将数据转移至CPU调用的方法是`.cpu`？这是因为GPU的编程接口采用CUDA，而目前并不是所有的GPU都支持CUDA，只有部分 Nvidia 的 GPU 才支持。PyTorch 未来可能会支持 AMD 的 GPU，而AMD GPU的编程接口采用OpenCL，因此 PyTorch 还预留着 `.cl` 方法，用于以后支持 AMD 等的 GPU。

举例：

```py
import torch as t
import torch.nn as nn

tensor = t.Tensor(3, 4)
# 返回一个新的tensor，保存在第1块GPU上，但原来的tensor并没有改变
print(tensor.cuda(0))
print(tensor.is_cuda) # False


# 不指定所使用的GPU设备，将默认使用第1块GPU
tensor = tensor.cuda()
print(tensor.is_cuda) # True


module = nn.Linear(3, 4)
module.cuda(device = 1)
print(module.weight.is_cuda) # True


class VeryBigModule(nn.Module):
    def __init__(self):
        super(VeryBigModule, self).__init__()
        self.GiantParameter1 = t.nn.Parameter(t.randn(100000, 20000)).cuda(0)
        self.GiantParameter2 = t.nn.Parameter(t.randn(20000, 100000)).cuda(1)

    def forward(self, x):
        x = self.GiantParameter1.mm(x.cuda(0))
        x = self.GiantParameter2.mm(x.cuda(1))
        return x
```

输出：

- 没有实际运行，因为 VeryBigModule 需要两个GPU。


说明：

- 上面最后一部分中，两个Parameter所占用的内存空间都非常大，大概是8个G，如果将这两个都同时放在一块GPU上几乎会将显存占满，无法再进行任何其它运算。此时可通过这种方式将不同的计算分布到不同的GPU中。



关于使用GPU的一些建议：

- GPU运算很快，但对于很小的运算量来说，并不能体现出它的优势，因此对于一些简单的操作可直接利用CPU完成
- 数据在CPU和GPU之间，以及GPU与GPU之间的传递会比较耗时，应当尽量避免
- 在进行低精度的计算时，可以考虑`HalfTensor`，它相比于`FloatTensor`能节省一半的显存，但需千万注意数值溢出的情况。


另外这里需要专门提一下，大部分的损失函数也都属于`nn.Moudle`，但在使用GPU时，很多时候我们都忘记使用它的`.cuda`方法，这在大多数情况下不会报错，因为损失函数本身没有可学习的参数（learnable parameters）。但在某些情况下会出现问题，为了保险起见同时也为了代码更规范，应记得调用`criterion.cuda`。

举例：

```py
import torch as t

# 交叉熵损失函数，带权重
criterion = t.nn.CrossEntropyLoss(weight=t.Tensor([1, 3]))
input = t.randn(4, 2).cuda()
target = t.Tensor([1, 0, 0, 1]).long().cuda()

# 下面这行会报错，因weight未被转移至GPU
# loss = criterion(input, target)

# 这行则不会报错
criterion.cuda()
loss = criterion(input, target)

print(criterion._buffers)
```

输出：

```txt
OrderedDict([('weight', tensor([1., 3.], device='cuda:0'))])
```


而除了调用对象的 `.cuda` 方法之外，还可以使用 `torch.cuda.device`，来指定默认使用哪一块GPU，或使用`torch.set_default_tensor_type`使程序默认使用GPU，不需要手动调用cuda。

举例：

```py
import torch as t

# 如果未指定使用哪块GPU，默认使用GPU 0
x = t.cuda.FloatTensor(2, 3)
x.get_device() == 0
y = t.FloatTensor(2, 3).cuda()
y.get_device() == 0

# 指定默认使用GPU 1
with t.cuda.device(1):
    # 在GPU 1上构建tensor
    a = t.cuda.FloatTensor(2, 3)

    # 将tensor转移至GPU 1
    b = t.FloatTensor(2, 3).cuda()
    print(a.get_device() == b.get_device() == 1)

    c = a + b
    print(c.get_device() == 1)

    z = x + y
    print(z.get_device() == 0)

    # 手动指定使用GPU 0
    d = t.randn(2, 3).cuda(0)
    print(d.get_device() == 2)

t.set_default_tensor_type('torch.cuda.FloatTensor')  # 指定默认tensor的类型为GPU上的FloatTensor
a = t.ones(2, 3)
print(a.is_cuda)
```

输出：

- 未实际运行


说明：

- 如果服务器具有多个GPU，`tensor.cuda()`方法会将tensor保存到第一块GPU上，等价于`tensor.cuda(0)`。此时如果想使用第二块GPU，需手动指定`tensor.cuda(1)`，而这需要修改大量代码，很是繁琐。这里有两种替代方法：
  - 一种是先调用`t.cuda.set_device(1)`指定使用第二块GPU，后续的`.cuda()`都无需更改，切换GPU只需修改这一行代码。
  - 更推荐的方法是设置环境变量`CUDA_VISIBLE_DEVICES`，例如当`export CUDA_VISIBLE_DEVICE=1`（下标是从0开始，1代表第二块GPU），只使用第二块物理GPU，但在程序中这块GPU会被看成是第一块逻辑GPU，因此此时调用`tensor.cuda()`会将Tensor转移至第二块物理GPU。`CUDA_VISIBLE_DEVICES`还可以指定多个GPU，如`export CUDA_VISIBLE_DEVICES=0,2,3`，那么第一、三、四块物理GPU会被映射成第一、二、三块逻辑GPU，`tensor.cuda(1)`会将Tensor转移到第三块物理GPU上。
    - 设置`CUDA_VISIBLE_DEVICES`有两种方法，一种是在命令行中`CUDA_VISIBLE_DEVICES=0,1 python main.py`，一种是在程序中`import os;os.environ["CUDA_VISIBLE_DEVICES"] = "2"`。如果使用IPython或者Jupyter notebook，还可以使用`%env CUDA_VISIBLE_DEVICES=1,2`来设置环境变量。


从 0.4 版本开始，pytorch新增了`tensor.to(device)`方法，能够实现设备透明，便于实现CPU/GPU兼容。这部份内容已经在第三章讲解过了。


从PyTorch 0.2版本中，PyTorch新增分布式GPU支持。分布式是指有多个GPU在多台服务器上，而并行一般指的是一台服务器上的多个GPU。分布式涉及到了服务器之间的通信，因此比较复杂，PyTorch封装了相应的接口，可以用几句简单的代码实现分布式训练。分布式对普通用户来说比较遥远，因为搭建一个分布式集群的代价十分大，使用也比较复杂。相比之下一机多卡更加现实。对于分布式训练，这里不做太多的介绍，感兴趣的读者可参考[文档](http://pytorch.org/docs/distributed.html)。

### 单机多卡并行

要实现模型单机多卡十分容易，直接使用 `new_module = nn.DataParallel(module, device_ids)`, 默认会把模型分布到所有的卡上。

多卡并行的机制如下：

- 将模型（module）复制到每一张卡上
- 将形状为（N,C,H,W）的输入均等分为 n份（假设有n张卡），每一份形状是（N/n, C,H,W）,然后在每张卡前向传播，反向传播，梯度求平均。要求batch-size 大于等于卡的个数(N>=n)

在绝大多数情况下，new_module的用法和module一致，除了极其特殊的情况下（RNN中的PackedSequence）。另外想要获取原始的单卡模型，需要通过`new_module.module`访问。


### 多机分布式

## 持久化

在PyTorch中，以下对象可以持久化到硬盘，并能通过相应的方法加载到内存中：

- Tensor
- Variable
- nn.Module
- Optimizer

本质上上述这些信息最终都是保存成Tensor。Tensor的保存和加载十分的简单，使用 t.save 和 t.load 即可完成相应的功能。在 save/load 时可指定使用的 pickle 模块，在 load 时还可将 GPU tensor 映射到 CPU 或其它 GPU 上。


我们可以通过`t.save(obj, file_name)`等方法保存任意可序列化的对象，然后通过`obj = t.load(file_name)`方法加载保存的数据。对于Module和Optimizer对象，这里建议保存对应的`state_dict`，而不是直接保存整个Module/Optimizer对象。

Optimizer对象保存的主要是参数，以及动量信息，通过加载之前的动量信息，能够有效地减少模型震荡，下面举例说明。



举例：

```py
import torch as t

a = t.Tensor(3, 4)
if t.cuda.is_available():
    a = a.cuda(1)
    # 把a转为GPU1上的tensor,
    t.save(a, 'a.pth')

    # 加载为b, 存储于GPU1上(因为保存时tensor就在GPU1上)
    b = t.load('a.pth')

    # 加载为c, 存储于CPU
    c = t.load('a.pth', map_location=lambda storage, loc: storage)

    # 加载为d, 存储于GPU0上
    d = t.load('a.pth', map_location={'cuda:1': 'cuda:0'})

t.set_default_tensor_type('torch.FloatTensor')
from torchvision.models import SqueezeNet

model = SqueezeNet()
# module的state_dict是一个字典
print(model.state_dict().keys())

# Module对象的保存与加载
t.save(model.state_dict(), 'squeezenet.pth')
model.load_state_dict(t.load('squeezenet.pth'))

optimizer = t.optim.Adam(model.parameters(), lr=0.1)

t.save(optimizer.state_dict(), 'optimizer.pth')
optimizer.load_state_dict(t.load('optimizer.pth'))

all_data = dict(
    optimizer=optimizer.state_dict(),
    model=model.state_dict(),
    info=u'模型和优化器的所有参数'
)
t.save(all_data, 'all.pth')

all_data = t.load('all.pth')
print(all_data.keys())
```

输出：

```txt
odict_keys(['features.0.weight', 'features.0.bias', 'features.3.squeeze.weight', ...略... 'features.12.expand1x1.bias', 'features.12.expand3x3.weight', 'features.12.expand3x3.bias', 'classifier.1.weight', 'classifier.1.bias'])
dict_keys(['optimizer', 'model', 'info'])
```


