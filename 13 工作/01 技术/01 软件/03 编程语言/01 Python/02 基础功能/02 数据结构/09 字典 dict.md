
# 字典 dict


字典：

- 类似其他语言的 map。
- 使用键-值存储，具有极快的查找速度。
- 可存储任意类型对象。
- **键必须是不可变的，如字符串，数字或元组。**
  - 纯粹的 tuple 是可以作为 dict 的 key 的，但是含有 list 的 tuple 是不能作为 key 的。
  - 可以用 hash 函数查看一个 object 是否是 hashable，只要是 hashable 的，就可以当做 dict 中的 key 。如：`hash('string')` 或`hash((1, 2, (2, 3)))`。


## 访问

举例：

```py
dict = {'Name': 'Zara', 'Age': 7, 'Class': 'First'};

print(dict['Name'])
print(dict['Age'])
print(dict['aaa'])

print('aaa' in dict)
print(dict.get('aaa'))
print(dict.get('aaa', -1))
```

输出：

```
Zara
7
Traceback (most recent call last):
  File "D:/21.Practice/demo/f.py", line 7, in <module>
    print(dict['aaa'])
KeyError: 'aaa'
False
None
-1
```

说明：

- `dict['aaa']` 如果访问字典里没有的键，会输出 KeyError 错误。
- 可以用 `'aaa' in dict` 预先判断是否有这个键
- 或者可以用 `dict.get('aaa')` 来取值，没有这个键，将返回 `None`。
- 推荐使用 `dict.get()` 的方式来获得 value。


## 修改字典

举例：

```py
dict = {'Name': 'Zara', 'Age': 7, 'Class': 'First'};

dict['Age'] = 8
dict['School'] = "DPS School"

print(dict['Age'])
print(dict['School'])
```

输出：

```
8
DPS School
```



## 删除字典元素

- 删除单一元素，可以用 del，也可以用 pop
- 清空用 clear。
- 删除一个完整字典用 del

举例：


```py
a = {'Name': 'Zara', 'Age': 7, 'Class': 'First', 'Score': 88};

del a['Name']
item = a.pop('Age')
print(item)
item = a.popitem()
print(item)
print(a)

a.clear()  # 清空词典所有条目
print(a)

del a  # 删除词典
print(a)
```


输出：

```
7
('Score', 88)
{'Class': 'First'}
{}
Traceback (most recent call last):
  File "D:/21.Practice/demo/f.py", line 14, in <module>
    print(a)
NameError: name 'a' is not defined
```

说明：

- `del dict` 后字典不再存在。



## 遍历字典

举例：

```py
d={'a':1,'b':2,'c':3,1:'one',2:'two','m':[1,3,3,3]}
for key in d:
    print(key,d[key])
for key,value in d.items():
    print(key,value)

print(list(d.keys())[0])
print(d.keys()[0])
```

输出：

```
a 1
b 2
c 3
1 one
2 two
m [1, 3, 3, 3]
a 1
b 2
c 3
1 one
2 two
m [1, 3, 3, 3]
a
Traceback (most recent call last):
File "D:/21.Practice/demo/f.py", line 8, in <module>
  print(d.keys()[0])
TypeError: 'dict_keys' object is not subscriptable
```

说明：

- `dict.keys()` 返回的是 dict_keys 对象，支持 iterable 但不支持 indexable，我们可以将其明确的转化成 list。



## 字典内置函数

方法：

- `cmp(dict1, dict2)` 比较两个字典元素。
- `len(dict)` 计算字典元素个数，即键的总数。
- `str(dict)` 输出字典可打印的字符串表示。
- `type(variable)` 返回输入的变量类型，如果变量是字典就返回字典类型。


内置函数：


- `dict.clear()` 删除字典内所有元素
- `dict.copy()` 返回一个字典的浅复制
- `dict.fromkeys(seq[, val\])` 创建一个新字典，以序列 seq 中元素做字典的键，val 为字典所有键对应的初始值
- `dict.get(key, default=None)` 返回指定键的值，如果值不在字典中返回 default 值 
- `dict.has_key(key)` 如果键在字典 dict 里返回 true，否则返回 false
- `dict.items()` 以列表返回可遍历的(键, 值) 元组数组
- `dict.keys()` 以列表返回一个字典所有的键
- `dict.setdefault(key, default=None)` 和 get()类似, 但如果键不存在于字典中，将会添加键并将值设为 default
- `dict.update(dict2)` 把字典 dict2 的键/值对更新到 dict 里
- `dict.values()` 以列表返回字典中的所有值
- `pop(key[,default\])` 删除字典给定键 key 所对应的值，返回值为被删除的值。key值必须给出。 否则，返回 default 值。
- `popitem()` 随机返回并删除字典中的一对键和值。

举例：

**update**

- 可以用 update 来合并两个 dict。


```py
d1 = {'b': 'foo', 'c': 12}
d1.update({'b': 'aa', 'a': 1})
print(d1)
```

输出：

```
{'b': 'aa', 'c': 12, 'a': 1}
```

注意：

- 这个 update 是更改了原有的 dict，不会返回新的 dict
- 重复的 key 会被新的 value 覆盖。


## 设定默认值

- 我们想要把一些单词按首字母归类。

举例：


```py
words = ['apple', 'bat', 'bar', 'atom', 'book']
by_letter = {}
for word in words:
    letter = word[0]
    if letter not in by_letter:
        by_letter[letter] = [word]
    else:
        by_letter[letter].append(word)
print(by_letter)
```

输出：

```
{'a': ['apple', 'atom'], 'b': ['bat', 'bar', 'book']}
```

**`setdefault` 方法**

- 而 `setdefault` 方法是专门为这个用途存在的。

```py
words = ['apple', 'bat', 'bar', 'atom', 'book']
by_letter={}
for word in words:
    letter = word[0]
    by_letter.setdefault(letter, []).append(word)
print(by_letter)
```

输出：

```
{'a': ['apple', 'atom'], 'b': ['bat', 'bar', 'book']}
```

说明：

- 如果 key 不在 dictionary 中那么就添加它并把它对应的值初始为空列表 `[]` ，然后把元素 append 到空列表中。

**defaultdict 类**

- 内建的 collections 模块有一个有用的类 defaultdict，这个能让上述过程更简单。

举例：

```py
from collections import defaultdict

words = ['apple', 'bat', 'bar', 'atom', 'book']
by_letter = defaultdict(list)
for word in words:
    by_letter[word[0]].append(word)
print(by_letter)
```

输出：

```
defaultdict(<class 'list'>, {'a': ['apple', 'atom'], 'b': ['bat', 'bar', 'book']})
```

说明：

- 创建 defaultdict 的方法是传递一个 type 或是函数。
