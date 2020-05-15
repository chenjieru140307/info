
# collections

作用：

- 提供了一些集合结构

主要：

- namedtuple
- deque
- defaultdict Ordereddict
- Counter


## namedtuple

作用：

- `namedtuple` 是一个函数，它用来创建一个自定义的`tuple`对象，并且规定了`tuple`元素的个数，并可以用属性而不是索引来引用 `tuple` 的某个元素。
- 这样一来，我们用 `namedtuple` 可以很方便地定义一种数据类型，它具备tuple的不变性，又可以根据属性来引用，使用十分方便。

```Python
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
print(Point)
p = Point(1, 2)
print(p.x, p.y)
Circle = namedtuple('Circle', ['x', 'y', 'r'])
print(Circle)
```

输出：


```Python
<class '__main__.Point'>
1 2
<class '__main__.Circle'>
```

疑问：

- 这个 namedtuple 一般用在什么地方？嗯，这个一般是作为没有方法的类使用的。


## deque

作用：

- 使用 `list` 存储数据时，按索引访问元素很快，但是插入和删除元素就很慢了，因为`list`是线性存储，数据量大的时候，插入和删除效率很低。
- `deque` 是为了高效实现插入和删除操作的双向列表，适合用于队列和栈。
- `deque`除了实现list的`append()`和`pop()`外，还支持`appendleft()`和`popleft()`，这样就可以非常高效地往头部添加或删除元素。

```py
from collections import deque

q = deque(['a', 'b', 'c'])
q.append('x')
q.appendleft('y')
print(q)
```

输出：

```
deque(['y', 'a', 'b', 'c', 'x'])
```



## defaultdict

作用：

- 使用`dict`时，如果引用的Key不存在，就会抛出`KeyError`。如果希望key不存在时，返回一个默认值，就可以用`defaultdict`。

举例：

```py
from collections import defaultdict
import traceback

d = dict({'key1': 1})
print(d['key1'])
try:
    print(d['key2'])# 普通的 dict 如果没有这个 key 会报 KeyError
except:
    traceback.print_exc()

dd = defaultdict(lambda: 'N/A')  # 为什么这个地方是一个 lambda 表达式？
dd['key1'] = 'abc'
print(dd['key1'])  # key1存在
print(dd['key2'])  # key2不存在，返回默认值
```


输出：

```
1
Traceback (most recent call last):
  File "E:/01.Learn/01.Python/01.PythonBasic/c7.py", line 6, in <module>
    print(d['key2'])
KeyError: 'key2'
abc
N/A
```

说明：

- 注意：默认值是调用函数返回的，而函数在创建`defaultdict`对象时传入。
- 除了在Key不存在时返回默认值，`defaultdict`的其他行为跟`dict`是完全一样的。


## Ordereddict

作用：

- 使用`dict`时，Key是无序的。在对`dict`做迭代时，我们无法确定Key的顺序。如果要保持Key的顺序，可以用`OrderedDict`。

```py
from collections import OrderedDict

print(dict([('a', 1), ('b', 2), ('c', 3)]))# dict的Key是无序的
print(OrderedDict([('a', 1), ('b', 2), ('c', 3)]))# OrderedDict的Key是有序的

od = OrderedDict()
od['z'] = 1
od['y'] = 2
od['x'] = 3
print(list(od.keys()))  # 按照插入的Key的顺序返回


class LastUpdatedOrderedDict(OrderedDict):

    def __init__(self, capacity):
        super(LastUpdatedOrderedDict, self).__init__()
        self._capacity = capacity

    def __setitem__(self, key, value):
        containsKey = 1 if key in self else 0
        if len(self) - containsKey >= self._capacity:
            last = self.popitem(last=False)
            print('remove:', last)
        if containsKey:
            del self[key]
            print('set:', (key, value))
        else:
            print('add:', (key, value))
        OrderedDict.__setitem__(self, key, value)

```

输出：

```txt
{'a': 1, 'b': 2, 'c': 3}
OrderedDict([('a', 1), ('b', 2), ('c', 3)])
['z', 'y', 'x']
```

说明：

- 注意，`OrderedDict`的Key会按照插入的顺序排列，不是Key本身排序：
- `OrderedDict` 可以实现一个FIFO（先进先出）的 dict `LastUpdatedOrderedDict`，当容量超出限制时，先删除最早添加的 Key。


## ChainMap

作用：

- `ChainMap` 可以把一组`dict`串起来并组成一个逻辑上的`dict`。`ChainMap`本身也是一个dict，但是查找的时候，会按照顺序在内部的dict依次查找。
- 什么时候使用`ChainMap`最合适？举个例子：应用程序往往都需要传入参数，参数可以通过命令行传入，可以通过环境变量传入，还可以有默认参数。我们可以用`ChainMap`实现参数的优先级查找，即先查命令行参数，如果没有传入，再查环境变量，如果没有，就使用默认参数。

举例：

- 下面的代码演示了如何查找`user`和`color`这两个参数：

```py
from collections import ChainMap
import os, argparse

# 构造缺省参数:
defaults = {
    'color': 'red',
    'user': 'guest'
}

# 构造命令行参数:
parser = argparse.ArgumentParser()
parser.add_argument('-u', '--user')
parser.add_argument('-c', '--color')
namespace = parser.parse_args()
command_line_args = { k: v for k, v in vars(namespace).items() if v }

# 组合成ChainMap:
combined = ChainMap(command_line_args, os.environ, defaults)

# 打印参数:
print('color=%s' % combined['color'])
print('user=%s' % combined['user'])
```

没有任何参数时，打印出默认参数：

```
$ python3 use_chainmap.py 
color=red
user=guest
```

当传入命令行参数时，优先使用命令行参数：

```
$ python3 use_chainmap.py -u bob
color=red
user=bob
```

同时传入命令行参数和环境变量，命令行参数的优先级较高：

```
$ user=admin color=green python3 use_chainmap.py -u bob
color=green
user=bob
```

## Counter

作用：

- `Counter`是一个简单的计数器，例如，统计字符出现的个数：

```py
from collections import Counter

c = Counter()
for ch in 'programming':
    c[ch] = c[ch] + 1
print(c)
c.update('hello')  # 也可以一次性update
print(c)
```

输出：


```
Counter({'r': 2, 'g': 2, 'm': 2, 'p': 1, 'o': 1, 'a': 1, 'i': 1, 'n': 1})
Counter({'r': 2, 'o': 2, 'g': 2, 'm': 2, 'l': 2, 'p': 1, 'a': 1, 'i': 1, 'n': 1, 'h': 1, 'e': 1})
```


说明：

- Counter 类似一个 dict 一样，不过用起来稍微方便些。counter的确是 dict 的一个子类

