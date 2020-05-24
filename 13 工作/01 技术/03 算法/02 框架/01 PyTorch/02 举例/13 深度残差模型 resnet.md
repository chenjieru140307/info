
# ResNet

```py
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable

# 导入模块包
transform = transforms.Compose([
    transforms.Scale(40),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])

train_dataset = datasets.CIFAR10(root='./data/',
                                 train=True,
                                 transform=transform,
                                 download=True)
test_dataset = datasets.CIFAR10(root='./data/',
                                train=False,
                                transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100,
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False)


# 加载数据

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(8) # 这个没有很明白。
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(conv3x3(self.in_channels, out_channels, stride=stride),
                                       nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



def train_model():
    resnet = ResNet(ResidualBlock, [3, 3, 3])
    resnet.cuda()

    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)
    for epoch in range(80):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = resnet(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' % (epoch + 1, 80, i + 1, 500, loss.data[0]))
        if (epoch + 1) % 20 == 0:
            lr /= 3
            optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)
            model_name="resnet.pkl"
            # 保存模型
            torch.save(resnet.state_dict(), model_name.replace('.pkl','_{0}.pkl'.format(epoch)))


def predict_with_path_model(path_model):
    # 加载模型
    resnet = ResNet(ResidualBlock, [3, 3, 3])
    resnet.cuda()
    resnet.load_state_dict(torch.load(path_model))

    # 测试结果
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.cuda())
        outputs = resnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()  # Tensor.cpu() to copy the tensor to host memory
    print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))


if __name__=="__main__":
    is_train=False
    if is_train:
        train_model()
    else:
        path_model = "./resnet.pkl"
        predict_with_path_model(path_model)

```

<span style="color:red;">哇塞，这个真的很不错，还是要再理解下的，感觉很多点都可以更明确下的。</span>

输出如下：

`is_train==True` 时：


```
Files already downloaded and verified
Epoch [1/80], Iter [100/500] Loss: 1.5502
Epoch [1/80], Iter [200/500] Loss: 1.3850
Epoch [1/80], Iter [300/500] Loss: 1.2312
Epoch [1/80], Iter [400/500] Loss: 1.1334
略..
Epoch [80/80], Iter [300/500] Loss: 0.0961
Epoch [80/80], Iter [400/500] Loss: 0.1103
Epoch [80/80], Iter [500/500] Loss: 0.1856
```

`is_train==False` 时：

```
Files already downloaded and verified
Accuracy of the model on the test images: 85 %
```


说明：

- 本节使用的是比较经典的数据集叫作 CIFAR-10，包含 60000 张 32×32 的彩色图像，因为是彩色图像，所以这个数据集是三通道的，分别是 R、G、B 三个通道。CIFAR-10，一共有 10 类图片，每一类图片有 6000 张，有飞机、鸟、猫、狗等，而且其中没有任何重叠的情况。现在还有一个版本， CIFAR-100，里面有 100 类。这里还要提到一个数据增广的问题，对于数据集比较小，数据量远远不够的情况，我们可以对图片进行翻转、随机剪切等操作来增加数据，制造出更多的样本，提高对图片的利用率。

对输入的数据进行二维卷积。卷积的本质就是用卷积核的参数来提取原始数据的特征，通过矩阵点乘的运算，提取出和卷积核特征一致的值，如果卷积层有多个卷积核，则神经网络会自动学习卷积核的参数值，使得每个卷积核代表一个特征。

参数说明如下。

- input：输入的 Tensor 数据，格式为（batch,channels,W），三维数组，第一维度是样本数量，第二维度是通道数或者记录数，第三维度是宽度。
- weight：过滤器，也叫卷积核权重。是一个三维数组，（out_channels, in_channels/groups,kW）。out_channels是卷积核输出层的神经元个数，也就是这层有多少个卷积核；in_channels是输入通道数；kW是卷积核的宽度。
- bias：位移参数。
- stride：滑动窗口，默认为 1，指每次卷积对原数据滑动 1 个单元格。
- padding：是否对输入数据填充 0。padding可以将输入数据的区域改造成卷积核大小的整数倍，这样对不满足卷积核大小的部分数据就不会忽略了。通过 padding 参数指定填充区域的高度和宽度，默认为 0。
- dilation：卷积核之间的空格，默认为 1。
- groups：将输入数据分成组，in_channels应该被组数整除，默认为 1。
- Conv2d 是二维卷积，它和 conv1d 的区别在于对宽度进行卷积，对高度进行卷积，而一维卷积对高度不进行卷积。



残差结构，深度网络容易造成梯度在 back propagation 的过程中消失，导致训练效果很差，而深度残差网络在神经网络的结构层面解决了这一问题，使得就算网络很深，梯度也不会消失。使用预激活残差单元构筑的残差网络，相较于使用原始单元更易收敛，且有一定正则化的效果，测试集上性能也普遍好于原始残差单元。<span style="color:red;">嗯，挺好的。</span>

每个 Convx_x 中都含有 3 个残差模块，每个模块的卷积核都是 3×3 大小的，pad 为 1,stride 为 1。Con4_x 的输出通过 global_average_pooling 映射到 64 个 1×1 大小的特征图上，最后再通过含有 10 个神经元的全连接层输出分类结果。



一个残差模块是由两层卷积再加一个恒等映射组成的。特征图的大小是一样的，残差模块的输入输出的维度大小也是一样的，可以直接进行相加。

网络的层数越深，可覆盖的解空间越广，理论上应该精度越高。

但简单地累加层数，并不能直接带来更好的收敛性和精度。

损失和优化函数，学习速率为 0.001，如果选择较大的学习速率会导致训练速度过快，导致效果不好。如果学习速率过小，导致训练速度太慢，所以选择合适的学习速率可以提高效果和缩短时间。

进行训练时，总共训练 80 批次，同时把图片数据转换成 PyTorch 可识别的变量，利用 CUDA 进行加速运算。


