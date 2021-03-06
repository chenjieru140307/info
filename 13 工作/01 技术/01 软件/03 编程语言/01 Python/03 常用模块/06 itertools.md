# itertools

作用：

- 用于操作迭代对象。
- `itertools`模块提供的全部是处理迭代功能的函数，它们的返回值不是list，而是`Iterator`，只有用`for`循环迭代的时候才真正计算。

文档：

- [文档](https://docs.Python.org/3.6/library/itertools.html)

举例：

```py
import itertools
natuals = itertools.count(1)
for n in natuals:
    print(n)

cs = itertools.cycle('ABC') # 注意字符串也是序列的一种
for c in cs:
    print(c)

ns = itertools.repeat('A', 3)
for n in ns:
    print(n)

natuals = itertools.count(1)
ns = itertools.takewhile(lambda x: x <= 10, natuals)
print(list(ns))


for c in itertools.chain('ABC', 'XYZ'):
    print(c)

for key, group in itertools.groupby('AAABBBCCAAA'):
    print(key, list(group))
for key, group in itertools.groupby('AaaBBbcCAAa', lambda c: c.upper()):
    print(key, list(group))
```

输出：

```
...
1
2
3
...

...
'A'
'B'
'C'
'A'
'B'
'C'
...

A
A
A

[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 迭代效果：'A' 'B' 'C' 'X' 'Y' 'Z'

A ['A', 'A', 'A']
B ['B', 'B', 'B']
C ['C', 'C']
A ['A', 'A', 'A']

A ['A', 'a', 'a']
B ['B', 'B', 'b']
C ['c', 'C']
A ['A', 'A', 'a']
```

说明：

- 无限的迭代器
  - `count()`会创建一个无限的迭代器，所以上述代码会打印出自然数序列，根本停不下来，只能按`Ctrl+C`退出。
  - `cycle()`会把传入的一个序列无限重复下去，同样停不下来。
  - `repeat()`负责把一个元素无限重复下去，不过如果提供第二个参数就可以限定重复次数：
  - 无限序列虽然可以无限迭代下去，但是通常我们会通过`takewhile()`等函数根据条件判断来截取出一个有限的序列。
- `chain()`可以把一组迭代对象串联起来，形成一个更大的迭代器。
- `groupby()`把迭代器中相邻的重复元素挑出来放在一起
  - 挑选规则是通过函数完成的，只要作用于函数的两个元素返回的值相等，这两个元素就被认为是在一组的，而函数返回值作为组的key。如果我们要忽略大小写分组，就可以让元素`'A'`和`'a'`都返回相同的key：


### 练习

计算圆周率可以根据公式：

利用Python提供的itertools模块，我们来计算这个序列的前N项和：

`# -*- coding: utf-8 -*- import itertools ``# 测试: print(pi(10)) print(pi(100)) print(pi(1000)) print(pi(10000)) assert 3.04 < pi(10) < 3.05 assert 3.13 < pi(100) < 3.14 assert 3.140 < pi(1000) < 3.141 assert 3.1414 < pi(10000) < 3.1415 print('ok') ` Run

### 小结
