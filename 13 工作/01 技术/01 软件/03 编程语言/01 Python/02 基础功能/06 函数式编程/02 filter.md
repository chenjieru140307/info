

# filter


说明：

- `filter()` 接收一个函数和一个序列。把传入的函数依次作用于每个元素，若 `True` 则保存， `False` 则丢弃。
- `filter()` 函数返回的是一个 `Iterator`，也就是一个惰性序列，所以要强迫 `filter()` 完成计算结果，可以用 `list()` 函数获得所有结果并返回 list。

**举例1：**

```py
l = [1, 2, 3, 4]
new_list = list(filter(lambda x: x < 3, l))
print(new_list)
```

输出：

```
[1, 2]
```

**举例2：**

（这个例子还没有很理解）

- 求素数

```py
def _odd_iter():
    n = 1
    while True:
        n = n + 2
        yield n

def _not_divisible(n):
    return lambda x: x % n > 0


def primes():
    yield 2
    it = _odd_iter() # 初始序列
    while True:
        n = next(it) # 返回序列的第一个数
        yield n
        it = filter(_not_divisible(n), it) # 构造新序列

for n in primes():
    if n < 1000:
        print(n)
    else:
        break
```

输出：1000 以内的素数

```txt
2
3
5
...
983
991
997
```


说明：

- `_odd_iter()`，是一个从`3`开始的奇数序列生成器。并且是一个无限序列。
- `_not_divisible(n)` 筛选函数。
- `primes()` 生成器，不断返回下一个素数。
  - 这个生成器先返回第一个素数`2`，然后，利用`filter()`不断产生筛选后的新的序列。
