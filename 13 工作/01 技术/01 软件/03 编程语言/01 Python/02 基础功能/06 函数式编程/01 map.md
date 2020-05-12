# map

map 函数：

- 接收两个参数，一个是函数，一个是序列，将传入的函数依次作用到序列的每个元素，并把结果作为新的 list 返回。

举例：

```py
a = [1, 2, 3]
l = list(map(lambda i: i + 1, a))
print(l)

l = list(map(lambda x, y: x + y, [3], [5]))
print(l)
l = list(map(lambda x, y: x + y, [3,4], [5,6]))
print(l)
```

输出：

```
[2, 3, 4]
[8]
[8, 10]
```

注意：

- 这个地方要注意，**map中的 lambda 表达式后面的参数，一定要是列表格式的。因为是要从每个列表里分别提取 item 进行运算。**

（这个例子还没有很透彻）
