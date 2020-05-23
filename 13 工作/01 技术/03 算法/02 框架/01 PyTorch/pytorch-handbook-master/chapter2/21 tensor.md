### 基本类型

Tensor的基本数据类型有五种：

- 32位浮点型：torch.FloatTensor。 (默认)
- 64位整型：torch.LongTensor。
- 32位整型：torch.IntTensor。
- 16位整型：torch.ShortTensor。
- 64位浮点型：torch.DoubleTensor。

除以上数字类型外，还有 byte和chart型

转换，举例：

```py
import torch

tensor = torch.tensor([3.1433223])
print(tensor)
print(tensor.size())


print(tensor.long())
print(tensor.half())
print(tensor.int())
print(tensor.float())
print(tensor.short())
print(tensor.char())
print(tensor.byte())
```

输出：

```txt
tensor([3.1433])
torch.Size([1])
tensor([3])
tensor([3.1426], dtype=torch.float16)
tensor([3], dtype=torch.int32)
tensor([3.1433])
tensor([3], dtype=torch.int16)
tensor([3], dtype=torch.int8)
tensor([3], dtype=torch.uint8)
```




### 设备间转换




```py
import torch

# cpu 与 gpu 移动
cpu_a = torch.rand(4, 3)
print(cpu_a.type())
gpu_a = cpu_a.cuda()
print(gpu_a.type())
cpu_b = gpu_a.cpu()
print(cpu_b.type())
print()

# 使用torch.cuda.is_available()来确定是否有cuda设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
gpu_b = cpu_b.to(device)
print(gpu_b.type())
```

输出：

```txt
torch.FloatTensor
torch.cuda.FloatTensor
torch.FloatTensor

cuda
torch.cuda.FloatTensor
```


- 一般情况下可以使用.cuda方法将tensor移动到gpu，这步操作需要cuda设备支持
- 使用.cpu方法将tensor移动到cpu
- 如果我们有多GPU的情况，可以使用to方法来确定使用那个设备，这里只做个简单的实例：