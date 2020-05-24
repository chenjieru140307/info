
# 可以补充进来的

- 挺好的，但是想要更深的理解下。而且，想更多的补充一些例子，或者拆分出去好好总结下各种 RNN 。这个感觉是不够的。
- 双向的还没跑过。要跑一下。



# 循环神经网络实现


前面我们已经学习了循环神经网络的基本原理，下面我们用具体的案例来实现。

## 单向 RNN 案例

例子：

```py
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


# Hyper Parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# MNIST Dataset
train_dataset = dsets.MNIST(root='./data/',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data/',
                           train=False,
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# TODO 对于这个地方还是有些不清楚
# RNN Model (Many-to-One)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) # 这个 LSTM 的参数要看下，batch_first 是什么？
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out

rnn = RNN(input_size, hidden_size, num_layers, num_classes)
rnn.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, sequence_length, input_size)).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = rnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, sequence_length, input_size)).cuda()
    outputs = rnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Model
torch.save(rnn.state_dict(), 'rnn.pkl')
```

输出：


```
Epoch [1/2], Step [100/600], Loss: 0.6595
Epoch [1/2], Step [200/600], Loss: 0.6234
Epoch [1/2], Step [300/600], Loss: 0.1465
Epoch [1/2], Step [400/600], Loss: 0.1174
Epoch [1/2], Step [500/600], Loss: 0.1437
Epoch [1/2], Step [600/600], Loss: 0.1984
Epoch [2/2], Step [100/600], Loss: 0.0365
Epoch [2/2], Step [200/600], Loss: 0.1402
Epoch [2/2], Step [300/600], Loss: 0.0338
Epoch [2/2], Step [400/600], Loss: 0.0222
Epoch [2/2], Step [500/600], Loss: 0.0426
Epoch [2/2], Step [600/600], Loss: 0.0995
Test Accuracy of the model on the 10000 test images: 96 %
```

<span style="color:red;">这也太快了？这个高的准确率吗？为啥？</span>


<span style="color:red;">上面这个程序有几个地方想知道，想更多了解下 `nn.LSTM` 的参数啥的，想知道更关于 forward 是怎么写的，比如说，现在我们要搭建任何一个 网络，那么怎么写对应的 forward？</span>




下载数据 MNIST 数据集来自美国国家标准与技术研究所，National Institute of Standards and Technology（NIST）。它包含了四个部分。

- Training set images:train-images-idx3-ubyte.gz（9.9 MB，解压后 47 MB，包含 60000 个样本）
- Training set labels:train-labels-idx1-ubyte.gz（29 KB，解压后 60 KB，包含 60000 个标签）
- Test set images:t10k-images-idx3-ubyte.gz（1.6 MB，解压后 7.8 MB，包含 10000 个样本）
- Test set labels:t10k-labels-idx1-ubyte.gz（5 KB，解压后 10 KB，包含 10000 个标签）



`torch.nn.LSTM（input_size, hidden_size, num_layers, batch_first=True）`将一个多层的（LSTM）应用到输入序列。

参数说明如下。

- input_size：输入的特征维度。
- hidden_size：隐状态的特征维度。
- num_layers：层数。
- bias：如果为 False，那么 LSTM 将不会使用 $b_{ih},b_{hh}$，默认为 True。
- batch_first：如果为 True，那么输入和输出 Tensor 的形状为（batch,seq, feature）。<span style="color:red;">如果为 False 呢？到底有什么区别？真实使用中用什么？</span>
- dropout：如果非零的话，将会在 RNN 的输出上加个 dropout，最后一层除外。<span style="color:red;">RNN 的 dropout 是这样加的吗？可以像普通的网络一样直接写 dropout 来加吗？还是说必须这样？RNN 的 dropout 有效果吗？</span>
- Bidirectional：如果为 True，将会变成一个双向 RNN，默认为 False。<span style="color:red;">嗯，双向的话要怎么用？forward 要怎么写？</span>

循环神经网络工作的关键点就是使用历史信息来帮助当前的决策。LSTM 靠一些“门”的结构让信息有选择性地影响每个时刻循环神经网络中的状态。

从输出结果来看，精度为 92%，说明循环神经网络在 MNIST 中，效果良好。<span style="color:red;">是呀？没想通，为啥连 RNN 在 MNIST 中效果都这么好？不科学呀？还是说随便一个网络在 MNIST 中效果都很好？也不科学呀？这个总共就训练了两个 epoch ，到底学到了啥？</span>




## 双向 RNN 案例

前面我们已经简单介绍了 Bidirectional RNN原理，下面我们用代码来实现。


```py
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


# Hyper Parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.003

# MNIST Dataset
train_dataset = dsets.MNIST(root='./data/',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data/',
                           train=False,
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# BiRNN Model (Many-to-One)
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection

    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)).cuda() # 2 for bidirection
        c0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)).cuda()

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out

rnn = BiRNN(input_size, hidden_size, num_layers, num_classes)
rnn.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, sequence_length, input_size)).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = rnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, sequence_length, input_size)).cuda()
    outputs = rnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Model
torch.save(rnn.state_dict(), 'rnn.pkl')
```

输出：

```

```

由于标准的循环神经网络（RNN）在时序上处理序列，往往忽略了未来的上下文信息。一种很显而易见的解决办法是在输入和目标之间添加延迟，进而可以给网络一些时步来加入未来的上下文信息，双向循环神经网络（BRNN）的基本思想是每一个训练序列向前和向后分别是两个循环神经网络（RNN），而且这两个都连接着一个输出层。这个结构提供给输出层输入序列中每一个点的完整的过去和未来的上下文信息。输入在向前和向后隐含层，隐含层到隐含层，向前和向后隐含层到输出层之中。每一个时刻都在重复利用。值得注意的是：向前和向后隐含层之间没有信息流，这保证了展开图是非循环的。<span style="color:red;">没有很明白，对于双向循环神经网络还有些不够理解。</span>

损失和优化函数，损失函数采用交叉熵损失函数，交叉熵可在神经网络中作为损失函数，$p$ 表示真实标记的分布，$q$ 则为训练后模型的预测标记分布，交叉熵损失函数可以衡量 $p$ 与 $q$ 的相似性。交叉熵作为损失函数还有一个好处是使用 Sigmoid 函数在梯度下降时能避免均方误差损失函数学习速率降低的问题，因为学习速率可以被输出的误差所控制。<span style="color:red;">啥？学习速率可以被输出的误差所控制？Sigmod 可以避免均方误差损失函数学习速率下降的问题吗？</span>

从输出的结果来看，双向循环网络模型经过训练之后，对图片进行训练的精度为 98%，说明双向循环神经网络效果还是很不错的。<span style="color:red;">这也太牛逼了吧？</span>





# 相关

- 《深度学习框架 Pytorch 快速开发与实战》
