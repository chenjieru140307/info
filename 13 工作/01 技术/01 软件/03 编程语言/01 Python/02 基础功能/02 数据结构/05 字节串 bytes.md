# 字节串 bytes

举例：

```py
b1 = bytes()
b2 = b''
# 通过b前缀指定hello是bytes类型的值
b3 = b'Python'
print(b3)
print(b3[0])
print(b3[2:4])


# 调用bytes方法将字符串转成bytes对象
b4 = bytes('Python 编程',encoding='utf-8')
print(b4)

# 利用字符串的encode()方法编码成bytes，默认使用utf-8字符集
b5 = "Python 编程".encode('utf-8')
print(b5)
```

输出：


```
b'Python'
80
b'th'
b'Python \xe7\xbc\x96\xe7\xa8\x8b'
b'Python \xe7\xbc\x96\xe7\xa8\x8b'
```

说明：

- b2、b3 都是直接在 ASCII 字符串前添加b前缀来得到字节串的：
- b4 调用 bytes() 函数来构建字节串；
- b5 则调用字符串的 encode 方法来构建字节串。


## 字节串与 bytes 转换

举例：

```py
a = "Python 编程".encode('utf-8')
print(a)
b = a.decode('utf-8')  # 默认使用UTF-8进行解码
print(b)
```

输出：

```
b'Python \xe7\xbc\x96\xe7\xa8\x8b'
Python 编程
```

## 数字与 bytes 转换

<span style="color:red;">这个一般什么时候会使用？</span>

举例：

```py
# 2 是字节长度
# to_bytes 就是转换为二进制字符串
print((1024).to_bytes(2, byteorder='big'))  # 高位在前
print((1024).to_bytes(2, byteorder='little'))  # 低位在前
print((-1024).to_bytes(2, byteorder='big', signed=True))  # 高位在前
print((-1024).to_bytes(2, byteorder='little', signed=True))  # 低位在前

print((1024).to_bytes(4, byteorder='big'))  # 高位在前
print((1024).to_bytes(4, byteorder='little'))  # 低位在前
print((-1024).to_bytes(4, byteorder='big', signed=True))  # 高位在前
print((-1024).to_bytes(4, byteorder='little', signed=True))  # 低位在前
```

输出：


```
b'\x04\x00'
b'\x00\x04'
b'\xfc\x00'
b'\x00\xfc'
b'\x00\x00\x04\x00'
b'\x00\x04\x00\x00'
b'\xff\xff\xfc\x00'
b'\x00\xfc\xff\xff'
```


举例：

```py
print((3124).to_bytes(2,byteorder="big"))
print('%x'%3124)
print('%d'%0x0c34)
```


输出：

```
b'\x0c4'
c34
3124
```

注意：

- 之所以第一行输出的是 `\0xc4` 是因为 `4` 的 ASICC 码是 `34`，所以当输出为`\x0c\x34` 的时候，Python直接把`\x34` 打印成 `4`。


