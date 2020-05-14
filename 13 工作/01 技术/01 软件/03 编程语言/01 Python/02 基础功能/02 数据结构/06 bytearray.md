# bytearray

（没怎么用过这个）

举例：

```py
ba = bytearray(b'\x00\x0F')

ba[0] = 255
ba.append(255)
print(ba)

b = bytes(ba)
print(b)
```

输出：

```txt
bytearray(b'\xff\x0f\xff')
b'\xff\x0f\xff'
```

