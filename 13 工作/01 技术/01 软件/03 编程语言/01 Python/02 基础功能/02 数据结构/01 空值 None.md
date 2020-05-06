
# 空值 None

None：

- None 本质是一个值。表示一个空值。
- `None`不能理解为`0`，因为`0`是有意义的，而`None`是一个特殊的空值。

## 比较

举例：

```py
a=list()
print(type(a))
print(a)
print(a is None)
print(a == None)

import cv2

a=cv2.imread('./b.png')
print(type(a))
print(a)
print(a is None)
print(a == None)
```

输出：

```
<class 'list'>
[]
False
False
// 当不存在 b.png 图片时，输出：
<class 'NoneType'>
None
True
True
//当存在 b.png 图片时，输出：
<class 'numpy.ndarray'>
[[[0 0 0]
  [0 0 0]
  [0 0 0]
  ...
  [0 0 0]
  [0 0 0]
  [0 0 0]]
 ...
 ...
 [[0 0 0]
  [0 0 0]
  [0 0 0]
  ...
  [0 0 0]
  [0 0 0]
  [0 0 0]]]
False
[[[False False False]
  [False False False]
  [False False False]
  ...
  [False False False]
  [False False False]
  [False False False]]
 ...
 ...
 [[False False False]
  [False False False]
  [False False False]
  ...
  [False False False]
  [False False False]
  [False False False]]]
```

说明：

- 空的 list 仍然是一个 list，并不是 None。
- cv2.imread 当加载的图片存在时，返回 `<class 'numpy.ndarray'>` 类型，不存在时，返回 `<class 'NoneType'>` 类型

注意：

- **判断一个对象是否是 None，最好用 is None 来判断。**


