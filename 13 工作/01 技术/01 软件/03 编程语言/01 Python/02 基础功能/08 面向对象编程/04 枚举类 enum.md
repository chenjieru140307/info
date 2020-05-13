# 枚举类

**举例1：**

- 直接创建 Enum 对象。

```python
from enum import Enum

Month = Enum('Month', ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))
for name, member in Month.__members__.items():
    print(name, '=>', member, ',', member.value)
```

输出：

```
Jan => Month.Jan , 1
Feb => Month.Feb , 2
Mar => Month.Mar , 3
Apr => Month.Apr , 4
May => Month.May , 5
Jun => Month.Jun , 6
Jul => Month.Jul , 7
Aug => Month.Aug , 8
Sep => Month.Sep , 9
Oct => Month.Oct , 10
Nov => Month.Nov , 11
Dec => Month.Dec , 12
```

说明：

- `value` 属性则是自动赋给成员的 `int` 常量，默认从 `1` 开始计数。


**举例2：**

- 从 `Enum` 派生出自定义类。
- 可以更精确地控制枚举类型。

```py
from enum import Enum, unique

@unique
class Weekday(Enum):
    Sun = 0  # Sun的 value 被设定为 0
    Mon = 1
    Tue = 2
    Wed = 3
    Thu = 4
    Fri = 5
    Sat = 6

day1 = Weekday.Mon
print(day1)
print(Weekday.Tue)
print(Weekday['Tue'])
print(Weekday.Tue.value)
print(day1 == Weekday.Mon)
print(day1 == Weekday.Tue)
print(Weekday(1))
print(day1 == Weekday(1))
for name, member in Weekday.__members__.items():
    print(name, '=>', member)
# print(Weekday(7))
```

输出：

```txt
Weekday.Mon
Weekday.Tue
Weekday.Tue
2
True
False
Weekday.Mon
True
Sun => Weekday.Sun
Mon => Weekday.Mon
Tue => Weekday.Tue
Wed => Weekday.Wed
Thu => Weekday.Thu
Fri => Weekday.Fri
Sat => Weekday.Sat
```


说明：

- `@unique`装饰器可以帮助我们检查保证没有重复值。
- 既可以用成员名称引用枚举常量，又可以直接根据 value的值获得枚举常量。
- 这个 `Weekday(1)` 还是第一次见到可以这样写，`Weekday['Tue']` 这样的写法也是之前没有使用过的。

