
## backward

```py
import torch

x = torch.rand(5, 5, requires_grad=True)
y = torch.rand(5, 5, requires_grad=True)
z = torch.sum(x + y)
z.backward()
print(x.grad)
print(y.grad)


x = torch.rand(5, 5, requires_grad=True)
y = torch.rand(5, 5, requires_grad=True)
z = x ** 2 + y ** 3
z.backward(torch.ones_like(x))
print(x.grad)
```

- 如果Tensor类表示的是一个标量（即它包含一个元素的张量），则不需要为backward()指定任何参数，但是如果它有更多的元素，则需要指定一个gradient参数，它是形状匹配的张量。 以上的 `z.backward()`相当于是`z.backward(torch.tensor(1.))`的简写。 这种参数常出现在图像分类中的单标签分类，输出一个标量代表图像的标签。(还没有很清楚，怎么指定 gradient 参数。)
- 如果我们的返回值不是一个标量，那么需要输入一个大小相同的张量作为参数，这里我们用ones_like函数根据x生成一个张量
- 使用 with torch.no_grad() 进行嵌套后，代码不会跟踪历史记录，也就是说保存的这部分记录会减少内存的使用量并且会加快少许的运算速度。







## Autograd 过程解析




为了说明Pytorch的自动求导原理，我们来尝试分析一下PyTorch的源代码，虽然Pytorch的 Tensor和 TensorBase都是使用CPP来实现的，但是可以使用一些Python的一些方法查看这些对象在Python的属性和状态。 Python的 `dir()` 返回参数的属性、方法列表。`z`是一个Tensor变量，看看里面有哪些成员变量。



dir(z)
print("x.is_leaf="+str(x.is_leaf))
print("z.is_leaf="+str(z.is_leaf))



- `.is_leaf`：记录是否是叶子节点。通过这个属性来确定这个变量的类型 在官方文档中所说的“graph leaves”，“leaf variables”，都是指像`x`，`y`这样的手动创建的、而非运算得到的变量，这些变量成为创建变量。 像`z`这样的，是通过计算后得到的结果称为结果变量。一个变量是创建变量还是结果变量是通过`.is_leaf`来获取的。

- 为什么我们执行`z.backward()`方法会更新`x.grad`和`y.grad`呢？ `.grad_fn`属性记录的就是这部分的操作，虽然`.backward()`方法也是CPP实现的，但是可以通过Python来进行简单的探索。

`grad_fn`：记录并且编码了完整的计算历史，`grad_fn`是一个`AddBackward0`类型的变量 `AddBackward0`这个类也是用Cpp来写的，但是我们从名字里就能够大概知道，他是加法(ADD)的反反向传播（Backward），看看里面有些什么东西







[12]



dir(z.grad_fn)
z.grad_fn.next_functions

xg = z.grad_fn.next_functions[0][0]
dir(xg)

x_leaf=xg.next_functions[0][0]

print(type(x_leaf))
print(x_leaf.variable)
print("x_leaf.variable的id:"+str(id(x_leaf.variable)))
print("x的id:"+str(id(x)))
assert(id(x_leaf.variable)==id(x))


- `next_functions`就是`grad_fn`的精华
- `next_functions`是一个tuple of tuple of PowBackward0 and int。为什么是2个tuple ？ 因为我们的操作是`z= x**2+y**3` 刚才的`AddBackward0`是相加，而前面的操作是乘方 `PowBackward0`。tuple第一个元素就是x相关的操作记录
- 在PyTorch的反向图计算中，`AccumulateGrad`类型代表的就是叶子节点类型，也就是计算图终止节点。`AccumulateGrad`类中有一个`.variable`属性指向叶子节点。
- 这个`.variable`的属性就是我们的生成的变量`x`


这样整个规程就很清晰了：

1. 当我们执行z.backward()的时候。这个操作将调用z里面的grad_fn这个属性，执行求导的操作。
2. 这个操作将遍历grad_fn的next_functions，然后分别取出里面的Function（AccumulateGrad），执行求导操作。这部分是一个递归的过程直到最后类型为叶子节点。
3. 计算出结果以后，将结果保存到他们对应的variable 这个变量所引用的对象（x和y）的 grad这个属性里面。
4. 求导结束。所有的叶节点的grad变量都得到了相应的更新

最终当我们执行完c.backward()之后，a和b里面的grad值就得到了更新。









## 扩展Autograd

一个自定义的Function需要一下三个方法：

- __init__ (optional)：如果这个操作需要额外的参数则需要定义这个Function的构造函数，不需要的话可以忽略。
- forward()：执行前向传播的计算代码
- backward()：反向传播时梯度计算的代码。 参数的个数和forward返回值的个数一样，每个参数代表传回到此操作的梯度。


```py

import torch

from torch.autograd.function import Function


# 定义一个乘以常数的操作(输入参数是张量)
# 方法必须是静态方法，所以要加上@staticmethod
class MulConstant(Function):
    @staticmethod
    def forward(ctx, tensor, constant):
        # ctx 用来保存信息这里类似self，并且ctx的属性可以在backward中调用
        ctx.constant = constant
        return tensor * constant

    @staticmethod
    def backward(ctx, grad_output):
        # 返回的参数要与输入的参数一样.
        # 第一个输入为3x3的张量，第二个为一个常数
        # 常数的梯度必须是 None.
        return grad_output, None


a = torch.rand(3, 3, requires_grad=True)
b = MulConstant.apply(a, 5)
print("a:" + str(a))
print("b:" + str(b))  # b为a的元素乘以5
b.backward(torch.ones_like(a))
print(a.grad)
```

输出：

```txt
a:tensor([[0.6279, 0.1056, 0.1455],
        [0.3992, 0.6376, 0.0991],
        [0.9012, 0.2152, 0.3778]], requires_grad=True)
b:tensor([[3.1396, 0.5281, 0.7275],
        [1.9961, 3.1882, 0.4954],
        [4.5062, 1.0758, 1.8889]], grad_fn=<MulConstantBackward>)
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]])
```




- 反向传播，返回值不是标量，所以`backward`方法需要参数
