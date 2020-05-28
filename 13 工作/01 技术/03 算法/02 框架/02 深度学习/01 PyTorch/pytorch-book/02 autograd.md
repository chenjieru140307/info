# autograd

torch.autograd：

- 为方便用户使用，而专门开发的一套自动求导引擎
- 它能够根据输入和前向传播过程自动构建计算图，并执行反向传播。

计算图(Computation Graph)：

- 是现代深度学习框架如PyTorch和TensorFlow等的核心，其为高效自动求导算法——反向传播(Back Propogation)提供了理论支持，了解计算图在实际写程序过程中会有极大的帮助。

## requires_grad

autograd记录对tensor的操作记录用来构建计算图。


Variable提供了大部分tensor支持的函数，但其不支持部分`inplace`函数，因这些函数会修改tensor自身，而在反向传播中，variable需要缓存原来的tensor来计算反向传播梯度。如果想要计算各个Variable的梯度，只需调用根节点variable的`backward`方法，autograd会自动沿着计算图反向传播，计算每一个叶子节点的梯度。


`variable.backward(gradient=None, retain_graph=None, create_graph=None)`主要有如下参数：

- grad_variables：形状与variable一致，对于`y.backward()`，grad_variables相当于链式法则 ${dz \over dx}={dz \over dy} \times {dy \over dx}$ 中的$\textbf {dz} \over \textbf {dy}$。grad_variables也可以是tensor或序列。
- retain_graph：反向传播需要缓存一些中间结果，反向传播之后，这些缓存就被清空，可通过指定这个参数不清空缓存，用来多次反向传播。
- create_graph：对反向传播过程再次构建计算图，可通过`backward of backward`实现求高阶导数。

**举例1：**

```py
from __future__ import print_function
import torch as t

# 在创建tensor的时候指定requires_grad
a = t.randn(3, 4, requires_grad=True)
print(a)
a = t.randn(3, 4).requires_grad_()
print(a)
a = t.randn(3, 4)
a.requires_grad = True
print(a)

b = t.zeros(3, 4).requires_grad_()
print(b)
c = a.add(b)
print(c)
d = c.sum()
print(d)
d.backward()
print(a.grad)

print(a.requires_grad, b.requires_grad, c.requires_grad)
print(a.is_leaf, b.is_leaf, c.is_leaf)
print(c.grad is None)
```

输出：

```txt
tensor([[ 1.0502,  0.7429,  0.1080,  0.9992],
        [ 0.4496,  0.5917, -1.3125, -0.4538],
        [ 0.7961,  0.1142,  1.6440, -0.7867]], requires_grad=True)
tensor([[ 0.9580, -1.1934, -1.0399, -0.4744],
        [-1.6322,  1.4437, -0.1134,  1.0515],
        [-0.6909,  0.9586,  1.0919,  0.3692]], requires_grad=True)
tensor([[ 1.3863, -0.0729,  0.3923,  0.8025],
        [-0.5874, -0.5295, -0.4078,  1.6387],
        [ 0.6210, -0.1543,  1.4724,  0.5341]], requires_grad=True)
tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]], requires_grad=True)
tensor([[ 1.3863, -0.0729,  0.3923,  0.8025],
        [-0.5874, -0.5295, -0.4078,  1.6387],
        [ 0.6210, -0.1543,  1.4724,  0.5341]], grad_fn=<AddBackward0>)
tensor(5.0955, grad_fn=<SumBackward0>)
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])
True True True
True True False
True
```

说明：

- 此处虽然没有指定c需要求导，但c依赖于a，而a需要求导，因此 c 的requires_grad 属性会自动设为True
- 由用户创建的variable属于叶子节点，对应的grad_fn是None
- c.grad是None, 因c不是叶子节点，它的梯度是用来计算a的梯度，所以虽然 c.requires_grad = True,但其梯度计算完之后即被释放即： c.grad is None



**举例2：**

- 对比 autograd 的计算结果与手动求导计算结果。
$$
y = x^2\cdot e^x
$$
$$
{dy \over dx} = 2x\cdot e^x + x^2 \cdot e^x
$$

```py
import torch as t


def f(x):
    y = x ** 2 * t.exp(x)
    return y


x = t.randn(3, 4, requires_grad=True)
y = f(x)

# 自动计算
y.backward(t.ones(y.size()))  # gradient形状与y一致
print(x.grad)


# 手动计算
def gradf(x):
    dx = 2 * x * t.exp(x) + x ** 2 * t.exp(x)
    return dx
print(gradf(x))
```

输出：

```txt
tensor([[ 7.9349e+00,  4.3697e+01, -1.3455e-01, -4.5642e-01],
        [ 3.3105e+00,  4.2967e+01, -4.2251e-01, -4.8609e-03],
        [-1.2595e-02,  8.2021e+01, -4.5658e-01,  3.1067e+01]])
tensor([[ 7.9349e+00,  4.3697e+01, -1.3455e-01, -4.5642e-01],
        [ 3.3105e+00,  4.2967e+01, -4.2251e-01, -4.8609e-03],
        [-1.2595e-02,  8.2021e+01, -4.5658e-01,  3.1067e+01]],
       grad_fn=<AddBackward0>)
```


说明：

- autograd的计算结果与利用公式手动计算的结果一致


## 计算图


计算图：

- 计算图是一种特殊的有向无环图（DAG）
  - PyTorch中 `autograd` 的底层采用了计算图，用于记录算子与变量之间的关系。一般用矩形表示算子，椭圆形表示变量。

举例：

- 表达式 $\textbf {z = wx + b}$ 可分解为$\textbf{y = wx}$ 和 $\textbf{z = y + b}$
- 其计算图如图所示，图中`MUL`，`ADD`都是算子，$\textbf{w}$，$\textbf{x}$，$\textbf{b}$即变量。


<p align="center">
    <img width="40%" height="70%" src="http://images.iterate.site/blog/image/20200525/QdY92VpoFizq.svg">
</p>

- 图中，$\textbf{X}$ 和 $\textbf{b}$ 是叶子节点（leaf node），这些节点通常由用户自己创建，不依赖于其他变量。$\textbf{z}$ 称为根节点，是计算图的最终目标。利用链式法则很容易求得各个叶子节点的梯度。
- 
    $${\partial z \over \partial b} = 1,\space {\partial z \over \partial y} = 1\\
    {\partial y \over \partial w }= x,{\partial y \over \partial x}= w\\
    {\partial z \over \partial x}= {\partial z \over \partial y} {\partial y \over \partial x}=1 * w\\
    {\partial z \over \partial w}= {\partial z \over \partial y} {\partial y \over \partial w}=1 * x\\
    $$


而有了计算图，上述链式求导即可利用计算图的反向传播自动完成，其过程如下图所示。

计算图的反向传播：

<p align="center">
    <img width="50%" height="70%" src="http://images.iterate.site/blog/image/20200525/OgfzXix4MeAi.svg">
</p>


在PyTorch实现中：

- autograd会随着用户的操作，记录生成当前variable的所有操作，并由此建立一个有向无环图。
- 用户每进行一个操作，相应的计算图就会发生改变。更底层的实现中，图中记录了操作`Function`，每一个变量在图中的位置可通过其`grad_fn`属性在图中的位置推测得到。
- 在反向传播过程中，autograd沿着这个图从当前变量（根节点$\textbf{z}$）溯源，可以利用链式求导法则计算所有叶子节点的梯度。每一个前向传播操作的函数都有与之对应的反向传播函数用来计算输入的各个variable的梯度，这些函数的函数名通常以`Backward`结尾。

举例：


```py
import torch as t

x = t.ones(1)
b = t.rand(1, requires_grad=True)
w = t.rand(1, requires_grad=True)
y = w * x  # 等价于y=w.mul(x)
z = y + b  # 等价于z=y.add(b)

print(x.requires_grad, b.requires_grad, w.requires_grad)
print(y.requires_grad)
print(x.is_leaf, w.is_leaf, b.is_leaf)
print(y.is_leaf, z.is_leaf)

print(z.grad_fn)
print(z.grad_fn.next_functions)

print(z.grad_fn.next_functions[0][0] == y.grad_fn)
print(y.grad_fn.next_functions)
print(w.grad_fn, x.grad_fn)

# 使用retain_graph来保存buffer
z.backward(retain_graph=True)
print(w.grad)
z.backward()
print(w.grad)


def abs(x):
    if x.data[0] > 0:
        return x
    else:
        return -x


print()

x = t.ones(1, requires_grad=True)
y = abs(x)
y.backward()
print(x.grad)
print(y)

x = -1 * t.ones(1)
x = x.requires_grad_()
y = abs(x)
y.backward()
print(x.grad)
print(y)
print(x.requires_grad)
cc = x * 3
print(cc.requires_grad)


def f(x):
    result = 1
    for ii in x:
        if ii.item() > 0:
            result = ii * result
    return result


x = t.arange(-2, 4, dtype=t.float32).requires_grad_()
y = f(x)  # y = x[3]*x[4]*x[5]
y.backward()
print(x.grad)
print(y)

print()

x = t.ones(1, requires_grad=True)
w = t.rand(1, requires_grad=True)
y = x * w
# y依赖于w，而w.requires_grad = True
print(x.requires_grad, w.requires_grad, y.requires_grad)
with t.no_grad():
    x = t.ones(1)
    w = t.rand(1, requires_grad=True)
    y = x * w
# y依赖于w和x，虽然w.requires_grad = True，但是y的requires_grad依旧为False
print(x.requires_grad, w.requires_grad, y.requires_grad)

t.set_grad_enabled(False)
x = t.ones(1)
w = t.rand(1, requires_grad=True)
y = x * w
# y依赖于w和x，虽然w.requires_grad = True，但是y的requires_grad依旧为False
print(x.requires_grad, w.requires_grad, y.requires_grad)
t.set_grad_enabled(True) # 设置回默认


print()
a = t.ones(3, 4, requires_grad=True)
b = t.ones(3, 4, requires_grad=True)
c = a * b
print(a.data)
print(a.data.requires_grad)
d = a.data.sigmoid_()  # sigmoid_ 是个inplace操作，会修改a自身的值
print(d.requires_grad)
print(a.requires_grad)

tensor = a.detach()
print(tensor.requires_grad)
mean = tensor.mean()
std = tensor.std()
maximum = tensor.max()
tensor[0] = 1
# c.sum().backward()

print()
x = t.ones(3, requires_grad=True)
w = t.rand(3, requires_grad=True)
y = x * w
# y依赖于w，而w.requires_grad = True
z = y.sum()
print(x.requires_grad, w.requires_grad, y.requires_grad)
# 非叶子节点grad计算完之后自动清空，y.grad是None
z.backward()
print(x.grad, w.grad, y.grad)

# 第一种方法：使用grad获取中间变量的梯度
x = t.ones(3, requires_grad=True)
w = t.rand(3, requires_grad=True)
y = x * w
z = y.sum()
# z对y的梯度，隐式调用backward()
t.autograd.grad(z, y) # 这个地方是不是使用的有问题？
z.backward()
print(x.grad, w.grad, y.grad)



# 第二种方法：使用hook ，hook是一个函数，输入是梯度，不应该有返回值
def variable_hook(grad):
    print('y的梯度：', grad)
x = t.ones(3, requires_grad=True)
w = t.rand(3, requires_grad=True)
y = x * w
# 注册hook
hook_handle = y.register_hook(variable_hook)
z = y.sum()
z.backward()
print(x.grad, w.grad, y.grad)
# 除非你每次都要用hook，否则用完之后记得移除hook
hook_handle.remove()

print()

x = t.arange(0,3, requires_grad=True,dtype=t.float)
y = x**2 + x*2
z = y.sum()
z.backward() # 从z开始反向传播
print(x.grad)

x = t.arange(0,3, requires_grad=True,dtype=t.float)
y = x**2 + x*2
z = y.sum()
y_gradient = t.Tensor([1,1,1]) #dz/dy
y.backward(y_gradient) #从y开始反向传播
print(x.grad)
```

输出：

```txt
False True True
True
True True True
False False
<AddBackward0 object at 0x000001F0094A66C8>
((<MulBackward0 object at 0x000001F0094A67C8>, 0), (<AccumulateGrad object at 0x000001F0094A6708>, 0))
True
((<AccumulateGrad object at 0x000001F0094A6708>, 0), (None, 0))
None None
tensor([1.])
tensor([2.])

tensor([1.])
tensor([1.], requires_grad=True)
tensor([-1.])
tensor([1.], grad_fn=<NegBackward>)
True
True
tensor([0., 0., 0., 6., 3., 2.])
tensor(6., grad_fn=<MulBackward0>)

True True True
False True False
False True False

tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])
False
False
True
False

True True True
tensor([0.4555, 0.9967, 0.1555]) tensor([1., 1., 1.]) None
tensor([0.8630, 0.5319, 0.3128]) tensor([1., 1., 1.]) None
y的梯度： tensor([1., 1., 1.])
tensor([0.6451, 0.9878, 0.0616]) tensor([1., 1., 1.]) None

tensor([2., 4., 6.])
tensor([2., 4., 6.])
```


说明：

- 虽然未指定y.requires_grad为True，但由于y依赖于需要求导的w，故而y.requires_grad为True

- grad_fn可以查看这个variable的反向传播函数，z是add函数的输出，所以它的反向传播函数是AddBackward
- next_functions 保存grad_fn的输入，是一个tuple，tuple 的元素也是 Function 
  - 第一个是y，它是乘法(mul)的输出，所以对应的反向传播函数 y.grad_fn 是MulBackward
  - 第二个是b，它是叶子节点，由用户创建，grad_fn为None，但是有 z.grad_fn.next_functions 
- variable的grad_fn对应着和图中的function相对应
  - 第一个是w，叶子节点，需要求导，梯度是累加的
  - 第二个是x，叶子节点，不需要求导，所以为None
- 叶子节点的grad_fn是None

- 计算w的梯度的时候，需要用到x的数值(${\partial y\over \partial w} = x$)，这些数值在前向过程中会保存成 buffer，在计算完梯度之后会自动清空。为了能够多次反向传播需要指定 `retain_graph` 来保留这些buffer。
- 多次反向传播，梯度累加，这也就是w中AccumulateGrad标识的含义

- PyTorch使用的是动态图，它的计算图在每次前向传播时都是从头开始构建，所以它能够使用Python控制语句（如for、if等）根据需求创建计算图。这点在自然语言处理领域中很有用，它意味着你不需要事先构建所有可能用到的图的路径，图在运行时才构建。
- 变量的`requires_grad`属性默认为False，如果某一个节点requires_grad被设置为True，那么所有依赖它的节点`requires_grad`都是True。这其实很好理解，对于 $\textbf{x}\to \textbf{y} \to \textbf{z}$，`x.requires_grad = True`，当需要计算 $\partial z \over \partial x$ 时，根据链式法则，$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \frac{\partial y}{\partial x}$，自然也需要求 $\frac{\partial z}{\partial y}$，所以 y.requires_grad 会被自动标为True. 
- 有些时候我们可能不希望autograd对tensor求导。认为求导需要缓存许多中间结构，增加额外的内存/显存开销，那么我们可以关闭自动求导。对于不需要反向传播的情景（如inference，即测试推理时），关闭自动求导可实现一定程度的速度提升，并节省约一半显存，因其不需要分配空间计算梯度。
- `a.detach()` 与 `a.data`
  - 如果我们想要修改 tensor 的数值，但是又不希望被 autograd 记录，那么我么可以对tensor.data 进行操作：`a.data` 还是一个tensor，`a.data.requires_grad` 但是已经是独立于计算图之外
  - 如果我们希望对tensor，但是又不希望被记录, 可以使用tensor.data 或者tensor.detach()
  - 注意：`tensor = a.detach()` 近似于 `tensor=a.data`, 但是如果 `tensor` 被修改，backward 可能会报错
  - `c.sum().backward()` 会报错，因为 c=a*b, b的梯度取决于a，现在修改了tensor，其实也就是修改了a，梯度不再准确：RuntimeError: one of the variables needed for gradient  computation has been modified by an inplace operation


- 非叶子节点的梯度查看
  - 在反向传播过程中非叶子节点的导数计算完之后即被清空。若想查看这些变量的梯度，有两种方法：
    - 使用autograd.grad函数
    - 使用hook
    - 注意：`autograd.grad`和`hook`方法都是很强大的工具，更详细的用法参考官方api文档，这里举例说明基础的使用。推荐使用`hook`方法，但是在实际使用中应尽量避免修改grad的值。
- 最后再来看看variable中grad属性和backward函数`grad_variables`参数的含义，这里直接下结论：
  - variable $\textbf{x}$ 的梯度是目标函数 ${f(x)}$ 对 $\textbf{x}$ 的梯度，$\frac{df(x)}{dx} = (\frac {df(x)}{dx_0},\frac {df(x)}{dx_1},...,\frac {df(x)}{dx_N})$，形状和 $\textbf{x}$一致。
  - 对于y.backward(grad_variables)中的grad_variables相当于链式求导法则中的$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \frac{\partial y}{\partial x}$ 中的 $\frac{\partial z}{\partial y}$。$z$ 是目标函数，一般是一个标量，故而 $\frac{\partial z}{\partial y}$ 的形状与variable $\textbf{y}$的形状一致。`z.backward()`在一定程度上等价于y.backward(grad_y)。`z.backward()`省略了grad_variables参数，是因为 $z$ 是一个标量，而$\frac{\partial z}{\partial z} = 1$
- 另外值得注意的是，只有对variable的操作才能使用autograd，如果对variable的data直接进行操作，将无法使用反向传播。除了对参数初始化，一般我们不会修改variable.data的值。



在PyTorch中计算图的特点可总结如下：

- autograd 根据用户对 variable 的操作构建其计算图。对变量的操作抽象为`Function`。
- 对于那些不是任何函数(Function)的输出，由用户创建的节点称为叶子节点，叶子节点的 `grad_fn` 为None。叶子节点中需要求导的variable，具有 `AccumulateGrad`标识，因其梯度是累加的。
- variable 默认是不需要求导的，即`requires_grad` 属性默认为False，如果某一个节点requires_grad被设置为True，那么所有依赖它的节点 `requires_grad`都为True。
- variable的`volatile`属性默认为False，如果某一个variable的`volatile`属性被设为True，那么所有依赖它的节点`volatile`属性都为True。volatile属性为True的节点不会求导，volatile的优先级比`requires_grad`高。
- 多次反向传播时，梯度是累加的。反向传播的中间缓存会被清空，为进行多次反向传播需指定`retain_graph`=True来保存这些缓存。
- 非叶子节点的梯度计算完之后即被清空，可以使用`autograd.grad`或`hook`技术获取非叶子节点的值。
- variable的grad与data形状一致，应避免直接修改variable.data，因为对data的直接操作无法利用autograd进行反向传播
- 反向传播函数`backward`的参数`grad_variables`可以看成链式求导的中间结果，如果是标量，可以省略，默认为1
- PyTorch采用动态图设计，可以很方便地查看中间层的输出，动态的设计计算图结构。

这些知识不懂大多数情况下也不会影响对pytorch的使用，但是掌握这些知识有助于更好的理解pytorch，并有效的避开很多陷阱







## 自定义 Function


目前绝大多数函数都可以使用 `autograd` 实现反向求导，但如果需要自己写一个复杂的函数，不支持自动反向求导怎么办? 

写一个`Function`，实现它的前向传播和反向传播代码，`Function` 对应于计算图中的矩形， 它接收参数，计算并返回结果。下面给出一个例子。

```py
import torch as t
from torch.autograd import Function

class Mul(Function):
    @staticmethod
    def forward(ctx, w, x, b, x_requires_grad=True):
        ctx.x_requires_grad = x_requires_grad
        ctx.save_for_backward(w, x)
        output = w * x + b
        return output

    @staticmethod
    def backward(ctx, grad_output):
        w, x = ctx.saved_tensors
        grad_w = grad_output * x
        if ctx.x_requires_grad:
            grad_x = grad_output * w
        else:
            grad_x = None
        grad_b = grad_output * 1
        return grad_w, grad_x, grad_b, None


from torch.autograd import Function


class MultiplyAdd(Function):

    @staticmethod
    def forward(ctx, w, x, b):
        ctx.save_for_backward(w, x)
        output = w * x + b
        return output

    @staticmethod
    def backward(ctx, grad_output):
        w, x = ctx.saved_tensors
        grad_w = grad_output * x
        grad_x = grad_output * w
        grad_b = grad_output * 1
        return grad_w, grad_x, grad_b

x = t.ones(1)
w = t.rand(1, requires_grad=True)
b = t.rand(1, requires_grad=True)
# 开始前向传播
z = MultiplyAdd.apply(w, x, b)
# 开始反向传播
z.backward()
# x不需要求导，中间过程还是会计算它的导数，但随后被清空
print(x.grad, w.grad, b.grad)


x = t.ones(1)
w = t.rand(1, requires_grad=True)
b = t.rand(1, requires_grad=True)
# print('开始前向传播')
z = MultiplyAdd.apply(w, x, b)
# print('开始反向传播')
# 调用MultiplyAdd.backward
# 输出grad_w, grad_x, grad_b
z.grad_fn.apply(t.ones(1))
```

输出：

```txt
None tensor([1.]) tensor([1.])
```


说明：

- 自定义的 Function
  - 自定义的Function需要继承autograd.Function，没有构造函数`__init__`，forward和backward函数都是静态方法
  - backward函数的输出和forward函数的输入一一对应，backward函数的输入和forward函数的输出一一对应
  - backward函数的grad_output参数即t.autograd.backward中的`grad_variables`
  - 如果某一个输入不需要求导，直接返回None，如forward中的输入参数x_requires_grad显然无法对它求导，直接返回None即可
  - 反向传播可能需要利用前向传播的某些中间结果，需要进行保存，否则前向传播结束后这些对象即被释放
- Function的使用利用Function.apply(variable)

疑问：

- 没有明白 z.grad_fn.apply 这个。



之所以forward函数的输入是tensor，而backward函数的输入是variable，是为了实现高阶求导。backward函数的输入输出虽然是variable，但在实际使用时autograd.Function会将输入variable提取为tensor，并将计算结果的tensor封装成variable返回。在backward函数中，之所以也要对variable进行操作，是为了能够计算梯度的梯度（backward of backward）。下面举例说明，有关torch.autograd.grad的更详细使用请参照文档。

```py
import torch as t
from torch.autograd import Function



x = t.tensor([5], requires_grad=True,dtype=t.float)
y = x ** 2
grad_x = t.autograd.grad(y, x, create_graph=True)
print(grad_x) #dy/dx = 2 * x
grad_grad_x = t.autograd.grad(grad_x[0],x)
print(grad_grad_x) # 二阶导数 d(2x)/dx = 2
```

输出：

```txt
(tensor([10.], grad_fn=<MulBackward0>),)
(tensor([2.]),)
```


这种设计虽然能让 `autograd` 具有高阶求导功能，但其也限制了 Tensor 的使用，因autograd 中反向传播的函数只能利用当前已经有的 Variable 操作。这个设计是在`0.2`版本新加入的，为了更好的灵活性，也为了兼容旧版本的代码，PyTorch 还提供了另外一种扩展 autograd 的方法。PyTorch 提供了一个装饰器 `@once_differentiable`，能够在 backward 函数中自动将输入的variable 提取成 tensor，把计算结果的tensor 自动封装成 variable。有了这个特性我们就能够很方便的使用 numpy/scipy 中的函数，操作不再局限于 variable 所支持的操作。但是这种做法正如名字中所暗示的那样只能求导一次，它打断了反向传播图，不再支持高阶求导。



此外在实现了自己的Function之后，还可以使用`gradcheck`函数来检测实现是否正确。`gradcheck`通过数值逼近来计算梯度，可能具有一定的误差，通过控制 `eps` 的大小可以控制容忍的误差。




下面举例说明如何利用Function实现sigmoid Function。

举例：

```py
import torch as t
from torch.autograd import Function
from timeit import timeit


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, x):
        output = 1 / (1 + t.exp(-x))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        grad_x = output * (1 - output) * grad_output
        return grad_x


# 采用数值逼近方式检验计算梯度的公式对不对
test_input = t.randn(3, 4, requires_grad=True).double()
print(t.autograd.gradcheck(Sigmoid.apply, (test_input,), eps=1e-3))


def f_sigmoid(x):
    y = Sigmoid.apply(x)
    y.backward(t.ones(x.size()))


def f_naive(x):
    y = 1 / (1 + t.exp(-x))
    y.backward(t.ones(x.size()))


def f_th(x):
    y = t.sigmoid(x)
    y.backward(t.ones(x.size()))


x = t.randn(100, 100, requires_grad=True)

setup = 'from __main__ import f_sigmoid,f_naive,f_th,x'
num = 1000
t1 = timeit('f_sigmoid(x)', setup=setup, number=num)
t2 = timeit('f_naive(x)', setup=setup, number=num)
t3 = timeit('f_th(x)', setup=setup, number=num)
print(t1, t2, t3)
```

说明：

```txt
True
0.25091730000000007 0.24855169999999993 0.19176300000000002
```

说明：

- 显然 `f_sigmoid` 要比单纯利用`autograd`加减和乘方操作实现的函数快不少，因为 f_sigmoid 的 backward 优化了反向传播的过程。（并没有，为什么）
- 另外可以看出系统实现的 built-in 接口(t.sigmoid)更快。



## 举例：用Variable实现线性回归


在上一节中讲解了利用 tensor 实现线性回归，在这一小节中，将讲解如何利用 autograd/Variable 实现线性回归，以此感受 autograd 的便捷之处。

举例：

```py
import torch as t
from matplotlib import pyplot as plt
import numpy as np

# 设置随机数种子，为了在不同人电脑上运行时下面的输出一致
t.manual_seed(1000)

def get_fake_data(batch_size=8):
    ''' 产生随机数据：y = x*2 + 3，加上了一些噪声'''
    x = t.rand(batch_size, 1) * 5
    y = x * 2 + 3 + t.randn(batch_size, 1)
    return x, y


# 来看看产生x - y分布是什么样的
x, y = get_fake_data()
plt.scatter(x.squeeze().numpy(), y.squeeze().numpy())

# 随机初始化参数
w = t.rand(1, 1, requires_grad=True)
b = t.zeros(1, 1, requires_grad=True)
losses = np.zeros(500)

lr = 0.005  # 学习率

for ii in range(500):
    x, y = get_fake_data(batch_size=32)

    # forward：计算loss
    y_pred = x.mm(w) + b.expand_as(y)
    loss = 0.5 * (y_pred - y) ** 2
    loss = loss.sum()
    losses[ii] = loss.item()

    # backward：手动计算梯度
    loss.backward()

    # 更新参数
    w.data.sub_(lr * w.grad.data)
    b.data.sub_(lr * b.grad.data)

    # 梯度清零
    w.grad.data.zero_()
    b.grad.data.zero_()

    if ii % 50 == 0:
        # 画图
        x = t.arange(0, 6).view(-1, 1).float()
        y = x.mm(w.data) + b.data.expand_as(x)
        plt.plot(x.numpy(), y.numpy())  # predicted

        x2, y2 = get_fake_data(batch_size=20)
        plt.scatter(x2.numpy(), y2.numpy())  # true data

        plt.xlim(0, 5)
        plt.ylim(0, 13)
        plt.show()
        plt.pause(0.5)

print(w.item(), b.item())

plt.plot(losses)
plt.ylim(5, 50)
plt.show()
```

输出：

```txt
2.026895761489868 2.9732823371887207
```

图像：

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200526/PGTqoWECfdIo.png?imageslim">
</p>


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200526/1CBKg315NjRC.png?imageslim">
</p>

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200526/mQRbKw0j3aoM.png?imageslim">
</p>

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200526/KhAQ3bJL9NyT.png?imageslim">
</p>





用autograd实现的线性回归最大的不同点就在于autograd不需要计算反向传播，可以自动计算微分。这点不单是在深度学习，在许多机器学习的问题中都很有用。另外需要注意的是在每次反向传播之前要记得先把梯度清零。

本章主要介绍了PyTorch中两个基础底层的数据结构：Tensor和autograd中的Variable。Tensor是一个类似Numpy数组的高效多维数值运算数据结构，有着和Numpy相类似的接口，并提供简单易用的GPU加速。Variable是autograd封装了Tensor并提供自动求导技术的，具有和Tensor几乎一样的接口。`autograd`是PyTorch的自动微分引擎，采用动态计算图技术，能够快速高效的计算导数。
