# 集合 set

集合：

- 和字典类似，也是一组 key 的集合，但不存储 value。
- 由于 key 不能重复，所以，在 set 中，没有重复的 key。


## 初始化

举例：

```py
a = set([1, 2, 2, 3, 4, 5, 6])
a = {1, 2, 2, 3, 4, 5, 6}
b = {4, 5, 6, 7}
print(a)
print(b)
a = {(1, 2, 3), (1, 2, 3), 3}
print(a)
a = {(1, [1, 2], 3), (1, 2, 3), 3}
print(a)
```

输出：


```
{1, 2, 3, 4, 5, 6}
{4, 5, 6, 7}
{3, (1, 2, 3)}
Traceback (most recent call last):
  File "D:/21.Practice/demo/f.py", line 7, in <module>
    a = {(1, [1, 2], 3), (1, 2, 3), 3}
TypeError: unhashable type: 'list'
```

说明：

- 初始化 set 的时候 重复的元素自动只保留一个。

注意：

- 纯粹的 tuple 是可以作为 set 的 key 的，但是里面含有 list 的 tuple 是无法作为 key 被 hash 的。

## 集合的运算

举例：

```py
s_a = set([1, 2, 2, 3, 4, 5, 6])  
s_b = set([4, 5, 6, 7])
# 并集
print(s_a | s_b)
print(s_a.union(s_b))
# 交集
print(s_a & s_b)
print(s_a.intersection(s_b))  # 通过 intersection 生成一个新的 set
# 差集 a-a&b
print(s_a - s_b)
print(s_a.difference(s_b))
# 对称差 （A|B）-(A&B) 把两个集合相同的部分去除  这应该就是异或吧
print(s_a ^ s_b)
print(s_a.symmetric_difference(s_b))
```

输出：

```
{1, 2, 3, 4, 5, 6, 7}
{1, 2, 3, 4, 5, 6, 7}
{4, 5, 6}
{4, 5, 6}
{1, 2, 3}
{1, 2, 3}
{1, 2, 3, 7}
{1, 2, 3, 7}
```

注意：

- **对于比较大的 set，`|` 比 `union` 效率要高。以此类推。**



## 修改或删除集合中元素

举例：

```py
s_a = set([1, 2, 2, 3, 4, 5, 6])
s_a.add('x')
s_a.update([4, 5, 6, 9])
print(s_a)


s_a = set([1, 2, 2, 'x', 4, 5, 6])
s_a.remove('x')
s_a.remove(88)
```

输出：

```
{1, 2, 3, 4, 5, 6, 'x', 9}
Traceback (most recent call last):
  File "E:\11.ProgramFiles\Anaconda3\lib\site-packages\IPython\core\interactiveshell.py", line 2881, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "<iPython-input-5-4a1d9e03e895>", line 4, in <module>
    s_a.remove(88)
KeyError: 88
```

注意：

- 必须知道这个元素的值，不然只能用 try catch
