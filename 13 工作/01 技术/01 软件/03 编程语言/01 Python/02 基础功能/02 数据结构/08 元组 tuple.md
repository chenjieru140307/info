

# 可以补充进来的


# 元组 tuple

元组：

- 与列表类似
- 元组的元素不能修改。

使用：

- 因为 tuple 不可变，所以代码更安全。如果一组数据是固定不变的，还是最好创建为 tuple 。

举例

## 创建

```py
tup1 = ()
tup2 = ('aaa')
tup3 = ('aaa',)
tup4 = ('physics', 'chemistry', 1997, 2000)
tup5 = (1, 2, 3, 4, 5 )
tup6 = "a", "b", "c", "d"

print(tup1)
print(tup2)
print(tup3)
print(tup4)
print(tup5)
print(tup6)
```

输出：

```
()
aaa
('aaa',)
('physics', 'chemistry', 1997, 2000)
(1, 2, 3, 4, 5)
('a', 'b', 'c', 'd')
```

说明：

- 注意到，`('aaa')` 实际输出的是一个字符串。因此，当元组中只包含一个元素时，需要在元素后面添加逗号，传参的时候经常使用。


## 访问


举例：

```py
tup1 = ('physics', 'chemistry', 1997, 2000);
tup2 = (1, 2, 3, 4, 5, 6, 7 );

print "tup1[0]: ", tup1[0]
print "tup2[1:5]: ", tup2[1:5]
```

输出：

```
tup1[0]:  physics
tup2[1:5]:  (2, 3, 4, 5)
```


## 转换

举例：

```py
print(tuple([4, 0, 2]))
print(tuple('string'))
```

输出：

```
(4, 0, 2)
('s', 't', 'r', 'i', 'n', 'g')
```



## 连接

- 元组中的元素值是不允许修改的，但我们可以对元组进行连接组合，创建一个新的元组。

举例：


```py
tup1 = (12, 34.56);
tup2 = ('abc', 'xyz');

# 以下修改元组元素操作是非法的。
# tup1[0] = 100;

# 创建一个新的元组
tup3 = tup1 + tup2;
print(tup3)
```

输出：

```
(12, 34.56, 'abc', 'xyz')
```

## 删除

- 元组中的元素值是不允许删除的，但我们可以使用 del 语句来删除整个元组，如下实例:

```py
tup = ('physics', 'chemistry', 1997, 2000)

print tup
del tup
print "After deleting tup : "
print tup
```

输出：

```py
('physics', 'chemistry', 1997, 2000)
After deleting tup :
Traceback (most recent call last):
File "test.py", line 9, in <module>
  print tup;
NameError: name 'tup' is not defined
```

说明：

- 以上实例元组被删除后，输出变量会有异常信息。



## 元组运算符



| 表达式               | 结果                           | 描述 |
| ---------------------------- | ---------------------------- | ------------ |
| `len((1, 2, 3))`               | 3                            | 计算元素个数 |
| `(1, 2, 3) + (4, 5, 6)`        | (1, 2, 3, 4, 5, 6)           | 连接         |
| `['Hi!'] * 4`                  | ['Hi!', 'Hi!', 'Hi!', 'Hi!'] | 复制         |
| `3 in (1, 2, 3)`               | True                         | 元素是否存在 |
| `for x in (1, 2, 3): print x` | 1 2 3                        | 迭代         |




## 元组内置函数

Python 元组包含了以下内置函数

| 序号 | 方法               | 描述                   |
| ---- | ------------------ | ---------------------- |
|   1   | `cmp(tuple1,tuple2)` | 比较两个元组元素       |
|   2   | `len(tuple)`         | 计算元组元素个数。     |
|   3   | `max(tuple)`        | 返回元组中元素最大值。 |
|   4   | `min(tuple)`         | 返回元组中元素最小值。 |
|   5   | `tuple(seq)`         | 将列表转换为元组。     |




## unpacking 解包

举例：

```
a = 1
b = 10
b, a = a, b
print(a, b)

seq = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
for a, b, c in seq:
    print('a=%s, b=%s, c=%s'%(a, b, c))

values = 1, 2, 3, 4, 5
a, b, *rest = values
print(a,b)
print(rest)
```

输出：

```
10 1
a=1, b=2, c=3
a=4, b=5, c=6
a=7, b=8, c=9
1 2
[3, 4, 5]
```

说明：

- `a, b, *rest` 只取出 tuple 中开头几个元素，剩下的元素直接赋给`*rest`。如果 rest 部分是你想要丢弃的，名字本身无所谓，通常用下划线来代替：`a, b, *_` 



## 元组中包含 list 时候

举例：

```py
t = ('a', 'b', ['A', 'B'])
t[2][0]='X'
t[2][1]='Y'
print(t)
```

输出：

```
('a', 'b', ['X', 'Y'])
```

说明：

- tuple 一开始指向的 list 并没有改成别的 list，所以，tuple所谓的“不变”是说，tuple的每个元素，指向永远不变。即指向`'a'`，就不能改成指向`'b'`，指向一个 list，就不能改成指向其他对象。
- 但指向的这个 list 本身是可变的！
- 所以，要创建一个内容也不变的 tuple ，就必须保证 tuple 的每一个元素本身也不能变。



