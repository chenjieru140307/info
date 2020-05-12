

# 匿名函数 lambda

**举例1：**

```py
sum = lambda x, y: x + y
print(sum(10, 20))

g = lambda x: x * 2
print(g(3))

print((lambda x: x * 2)(4))
```

输出：

```
30
6
8
```

**举例2：**

- 用于 sort 的 key。

```py
l = ['foo', 'card', 'bar', 'aaaa', 'abab']
l.sort(key=lambda x: len(set(list(x))))
print(l)
```

输出：

```
['aaaa', 'foo', 'abab', 'bar', 'card']
```

