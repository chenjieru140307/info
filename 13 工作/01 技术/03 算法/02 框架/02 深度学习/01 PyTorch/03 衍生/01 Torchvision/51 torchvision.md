# torchvision

介绍：

- 它包含了处理一些基本图像数据集的方法。这些数据集包括 Imagenet, CIFAR10, MNIST 等。
- 除了数据加载以外，`torchvision` 还包含了图像转换器，`torchvision.datasets` 和 `torch.utils.data.DataLoader`。

用于图像处理较为方便。



# 数据集

Torchvision 包括了目前流行的数据集、模型结构和常用的图片转换工具。<span style="color:red;">有常用的图片转换工具吗？总结下。</span>

## torchvision.datasets

torchvision.datasets 中包含了以下数据集：

- MNIST
- COCO（用于图像标注和目标检测）（Captioning and Detection）
- LSUN Classification
- ImageFolder
- Imagenet-12
- CIFAR10 and CIFAR100
- STL10
- SVHN
- PhotoTour


## torchvision.models

torchvision.models 模块的子模块中包含以下预训练的模型结构。

- AlexNet
- VGG
- ResNet
- SqueezeNet
- DenseNet



可以通过调用构造函数来构造具有随机权重的模型：

```py
import torchvision.models as models

resnet18 = models.resnet18()
alexnet = models.alexnet()
squeezenet = models.squeezenet1_0()
densenet = models.densenet_161()
```

PyTorch 提供了大量的预训练的模型，利用 PyTorch 的 torch.utils.model_zoo 来加载预训练模型。这些可以通过构建 pretrained=True，如我们加载预训练的 resnet18 模型。

```py
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
alexnet=torchvision.models.alexnet(pretrained=False)
```




# 大纲

torchvision 包由 computer vision 方向的以下几部分组成：

- popular datasets
- model architecture
- common image transformations

具体如下：

- torchvision.datasets
  - [MNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#mnist)
  - [Fashion-MNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#fashion-mnist)
  - [KMNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#kmnist)
  - [EMNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#emnist)
  - [FakeData](https://pytorch.org/docs/stable/torchvision/datasets.html#fakedata)
  - [COCO](https://pytorch.org/docs/stable/torchvision/datasets.html#coco)
  - [LSUN](https://pytorch.org/docs/stable/torchvision/datasets.html#lsun)
  - [ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder)
  - [DatasetFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#datasetfolder)
  - [ImageNet](https://pytorch.org/docs/stable/torchvision/datasets.html#imagenet)
  - [CIFAR](https://pytorch.org/docs/stable/torchvision/datasets.html#cifar)
  - [STL10](https://pytorch.org/docs/stable/torchvision/datasets.html#stl10)
  - [SVHN](https://pytorch.org/docs/stable/torchvision/datasets.html#svhn)
  - [PhotoTour](https://pytorch.org/docs/stable/torchvision/datasets.html#phototour)
  - [SBU](https://pytorch.org/docs/stable/torchvision/datasets.html#sbu)
  - [Flickr](https://pytorch.org/docs/stable/torchvision/datasets.html#flickr)
  - [VOC](https://pytorch.org/docs/stable/torchvision/datasets.html#voc)
  - [Cityscapes](https://pytorch.org/docs/stable/torchvision/datasets.html#cityscapes)
  - [SBD](https://pytorch.org/docs/stable/torchvision/datasets.html#sbd)
- torchvision.models
  - [Classification](https://pytorch.org/docs/stable/torchvision/models.html#classification)
  - [Semantic Segmentation](https://pytorch.org/docs/stable/torchvision/models.html#semantic-segmentation)
  - [Object Detection, Instance Segmentation and Person Keypoint Detection](https://pytorch.org/docs/stable/torchvision/models.html#object-detection-instance-segmentation-and-person-keypoint-detection)
- torchvision.transforms
  - [Transforms on PIL Image](https://pytorch.org/docs/stable/torchvision/transforms.html#transforms-on-pil-image)
  - [Transforms on torch.*Tensor](https://pytorch.org/docs/stable/torchvision/transforms.html#transforms-on-torch-tensor)
  - [Conversion Transforms](https://pytorch.org/docs/stable/torchvision/transforms.html#conversion-transforms)
  - [Generic Transforms](https://pytorch.org/docs/stable/torchvision/transforms.html#generic-transforms)
  - [Functional Transforms](https://pytorch.org/docs/stable/torchvision/transforms.html#functional-transforms)
- [torchvision.utils](https://pytorch.org/docs/stable/torchvision/utils.html)



- `torchvision.``get_image_backend`()[[SOURCE\]](https://pytorch.org/docs/stable/_modules/torchvision.html#get_image_backend)

  Gets the name of the package used to load images

- `torchvision.``set_image_backend`(*backend*)[[SOURCE\]](https://pytorch.org/docs/stable/_modules/torchvision.html#set_image_backend)

  Specifies the package used to load images.Parameters**backend** (*string*) – Name of the image backend. one of {‘PIL’, ‘accimage’}. The `accimage`package uses the Intel IPP library. It is generally faster than PIL, but does not support as many operations.
