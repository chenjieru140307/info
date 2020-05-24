# autograd


`autograd`包为张量上的所有操作提供了自动求导。 它是一个在运行时定义的框架，这意味着反向传播是根据你的代码来确定如何运行，并且每次迭代可以是不同的。

示例

## 张量（Tensor）

- `torch.Tensor`是这个包的核心类。如果设置 `.requires_grad` 为 `True`，那么将会追踪所有对于该张量的操作。 当完成计算后通过调用 `.backward()`，自动计算所有的梯度， 这个张量的所有梯度将会自动积累到 `.grad` 属性。要阻止张量跟踪历史记录，可以调用`.detach()`方法将其与计算历史记录分离，并禁止跟踪它将来的计算记录。

为了防止跟踪历史记录（和使用内存），可以将代码块包装在`with torch.no_grad()：`中。 在评估模型时特别有用，因为模型可能具有`requires_grad = True`的可训练参数，但是我们不需要梯度计算。

在自动梯度计算中还有另外一个重要的类`Function`.

`Tensor` 和 `Function`互相连接并生成一个非循环图，它表示和存储了完整的计算历史。 每个张量都有一个`.grad_fn`属性，这个属性引用了一个创建了`Tensor`的`Function`（除非这个张量是用户手动创建的，即，这个张量的 `grad_fn` 是 `None`）。

如果需要计算导数，你可以在`Tensor`上调用`.backward()`。 如果`Tensor`是一个标量（即它包含一个元素数据）则不需要为`backward()`指定任何参数， 但是如果它有更多的元素，你需要指定一个`gradient`参数来匹配张量的形状。





```py
import torch


x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
  y = y * 2
print(y)
gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(gradients)
print(x.grad)

print(x.requires_grad)
print((x ** 2).requires_grad)
with torch.no_grad():
  print((x ** 2).requires_grad)
```

输出：

```txt
tensor([1.0240e+02, 1.0240e+03, 1.0240e-01])
True
True
False
```



- 如果`.requires_grad=True`但是你又不希望进行autograd的计算， 那么可以将变量包裹在 `with torch.no_grad()`中:

