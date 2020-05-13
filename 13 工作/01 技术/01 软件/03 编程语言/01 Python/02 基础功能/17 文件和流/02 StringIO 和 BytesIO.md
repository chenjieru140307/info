# StringIO 和 BytesIO

像 `open()` 函数返回的这种有个 `read()` 方法的对象，在 Python 中统称为 file-like Object。

除了 file 外，还可以是内存的字节流，网络流，自定义流等等。file-like Object 不要求从特定类继承，只要写个`read()`方法就行。

如：

- `StringIO` 在内存中读写 str，常用作临时缓冲。
- `BytesIO` 在内存中读写 bytes，常用作临时缓冲。

# StringIO 


举例：

```py
from io import StringIO

f = StringIO()
f.write('hello')
f.write(' ')
f.write('world!')
print(f.getvalue())


f = StringIO('Hello!\nHi!\nGoodbye!')
while True:
    s = f.readline()
    if s == '':
        break
    print(s.strip())

```

输出：

```txt
hello world!
Hello!
Hi!
Goodbye!
```

说明：

- `getvalue()` 方法用于获得写入后的 str。


## BytesIO


举例：

```py
from io import BytesIO

f = BytesIO()
f.write('中文'.encode('utf-8'))
print(f.getvalue())

f = BytesIO(b'\xe4\xb8\xad\xe6\x96\x87')
print(f.read())
```

输出：

```txt
b'\xe4\xb8\xad\xe6\x96\x87'
b'\xe4\xb8\xad\xe6\x96\x87'
```

说明：

- 注意，写入的不是 str，而是经过 UTF-8 编码的 bytes。


疑问：

- `f.getvalue()` 与 `f.read()` 有什么区别？什么时候使用这个 BytesIO？

