# 枚举 enum


举例：


```py
from enum import Enum

Month = Enum('Month', ('Jan', 'Feb', 'Mar', 'Apr'))
print(Month)

for name, member in Month.__members__.items():
    print(name, '=>', member, ',', member.value)

jan = Month.Jan
print(jan)
```

输出：

```
<enum 'Month'>
Jan => Month.Jan , 1
Feb => Month.Feb , 2
Mar => Month.Mar , 3
Apr => Month.Apr , 4
Month.Jan
```
