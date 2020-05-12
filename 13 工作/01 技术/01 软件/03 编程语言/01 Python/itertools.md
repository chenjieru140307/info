

## itertools module

<span style="color:red;">补充整理下。</span>

itertools 模块有很多常用算法的生成器。

比如 groupby 能取任意序列当做函数：


```py
import itertools

first_letter = lambda x: x[0]
names = ['Alan', 'Adam', 'Wes', 'Will', 'Albert', 'Steven']
for letter, group in itertools.groupby(names, first_letter):
    print(letter, list(group))
```

输出：

```
A ['Alan', 'Adam']
W ['Wes', 'Will']
A ['Albert']
S ['Steven']
```

说明：

- groupby 会根据 key 进行划分 group，然后对这一小 group 的东西返回一个迭代器，这样每个小组都有每个小组的迭代器，上面的函数中的 group 就是一个迭代器。

注意：

- **如果想按key 完全划分，那么需要先将列表排列好。**


一些迭代工具函数：<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180615/g24jae5ALg.png?imageslim">
</p>


