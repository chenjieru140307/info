
# 优化算法

torch.optim 是实现各种优化算法的包。

在使用 torch.optim 包构建 Optimizer 对象中，可以指定 Optimizer 参数，包括学习速率，权重衰减等。

例如：

```py
import torch
import torch.optim as optim

optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
optimizer=optim.Adam([var1,var2],lr=0.0001)
```

同时也可以为每个参数单独设置选项，利用 dict 定义一组参数，进行传入数据。

下面我们来看一个例子。

当我们想指定每一层的学习速率时，可以使用下面的方法：

```py
import torch
import torch.optim as optim

optim.SGD([{'params': model.base.parameters()},
           {'params': model.classifier.parameters(), 'lr': 1e-3}],
          lr=1e-2,
          momentum=0.9)
```

<span style="color:red;">这样真的可以吗？一般什么时候回用到指定每一层的学习率？这种指定真的会有效果吗？一般对什么层这样指定？</span>

`model.base` 参数将使用默认的学习速率 `1e-2` ，`model.classifier` 参数将使用学习速率`1e-3`，并且 `0.9` 的 `momentum` 将会被用于所有的参数。

所有的 Optimizer 都会实现 `step()` 更新参数的方法，使用方法如下：

```py
optimizer.step()
```


一旦梯度被如 `backward()` 之类的函数计算好后，我们就可以调用该函数。

例子如下。

```py
import torch
import torch.optim as optim

for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    optimizer.step(closure)
```

<span style="color:red;">上面这个地方为什么有了 `optimizer.step()` 之后 又有 `optimizer.step(closure)` ？</span>

一些优化算法例如 Conjugate Gradient 和 LBFGS 需要重复多次计算函数，因此你需要传入一个闭包来允许它们重新计算你的模型。这个闭包会清空梯度，计算损失，然后返回。<span style="color:red;">这个地方没有需要补充下，没有很透彻，Conjugate Gradient 和 LBFGS 需要补充下。</span>

例子如下。

```py
import torch
import torch.optim as optim

for input, target in dataset:
    def closure():
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        return loss
    optimizer.step(closure)
```

<span style="color:red;">这他妈啥例子？感觉没有抓住痛点呀？删了重写！</span>


## 常见的随机梯度优化算法

下面介绍常见的随机梯度优化算法：

```
torch.optim.SGD(params,lr=,momentum=0,dampening=0,weight_decay=0, nesterov=False)
```

以上可以实现随机梯度下降算法（momentum可选）。

SGD 全名 Stochastic Gradient Descent，即随机梯度下降。其特点是训练速度快，对于很大的数据集，也能够以较快的速度收敛。

由于是抽取，因此得到的梯度肯定有误差。因此学习速率需要逐渐减小，否则模型无法收敛。因为误差，所以每一次迭代的梯度受抽样的影响比较大，也就是说梯度含有比较大的噪声，不能很好地反映真实梯度。<span style="color:red;">嗯。</span>

选择合适的学习速率比较困难，所以对所有的参数更新使用同样的学习速率。对于稀疏数据或者特征，有时我们可能，所以对于不经常出现的特征，对于常出现的特征更新慢一些，这时候 SGD 就不太能满足要求了。<span style="color:red;">什么意思？对于不经常出现的特征，对于常出现的特征更新慢一些，这个是什么优化算法？</span>

参数说明如下。

- params（iterable）：用于优化，可以迭代参数或定义参数组。
- lr（float）：学习速率。
- momentum（float，可选）：动量因子（默认：0）。
- weight_decay（float，可选）：权重衰减（L2范数）（默认：0）。<span style="color:red;">哦，知道了看到 L2范数知道了。</span>
- dampening（float，可选）：动量的抑制因子（默认：0）。<span style="color:red;">什么是动量的抑制因子？</span>
- nesterov（bool，可选）：使用 Nesterov 动量（默认：False）。<span style="color:red;">动量有哪些类型？在什么情况下分别使用哪种？为什么会有这么多种？</span>


例子如下。

```
import torch
import torch.optim as optim

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer.zero_grad()
loss_fn(model(input), target).backward()
optimizer.step()
```


`torch.optim.lr_scheduler` 提供了几种方法来根据 `epoches` 的数量调整学习速率。`torch.optim.lr_scheduler.ReduceLROnPlateau` 允许基于一些验证测量来降低动态学习速率。

<span style="color:red;">这个 lr_scheduler 好像没有用过哎，到底效果怎么样？</span>

具体函数如下：

```py
torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda,last_epoch=-1)
```

将每个参数组的学习速率设置为初始的 `lr` 乘以一个给定的函数。当 `last_epoch=-1` 时，将初始 `lr` 设置为 `lr`。

参数说明如下。

- optimizer（Optimizer）：包装的优化器。
- lr_lambda（function or list）：一个函数来计算一个乘法因子，给定一个整数参数的 epoch ，或列表等功能，为每个组 optimizer.param_groups。<span style="color:red;">什么？说的是什么？没明白？</span>
- last_epoch（int）：最后一个时期的索引。默认：-1。


例子：


```py
import torch
import torch.optim as optim

lambda1 = lambda epoch: epoch // 30
lambda2 = lambda epoch: 0.95 ** epoch

scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
for epoch in range(100):
    scheduler.step()
    train(...)
    validate(...)
```

<span style="color:red;">没明白，这个到底是咋用的？一般什么时候使用这个？</span>

