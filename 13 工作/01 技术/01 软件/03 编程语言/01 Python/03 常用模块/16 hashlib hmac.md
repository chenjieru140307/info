# hashlib hmac

哈希算法：

- 它通过一个函数，把任意长度的数据转换为一个长度固定的数据串（通常用16进制的字符串表示）。
  - 举例：你写了一篇文章，内容是一个字符串`'how to use python hashlib - by Michael'`，并附上这篇文章的摘要是`'2d73d4f15c0db7f5ecb321b6a65e5d6d'`。如果有人篡改了你的文章，并发表为`'how to use python hashlib - by Bob'`，你可以一下子指出Bob篡改了你的文章，因为根据`'how to use python hashlib - by Bob'`计算出的摘要不同于原始文章的摘要。
- 可见，摘要算法就是通过摘要函数`f()`对任意长度的数据`data`计算出固定长度的摘要`digest`，目的是为了发现原始数据是否被人篡改过。
- 摘要算法之所以能指出数据是否被篡改过，就是因为摘要函数是一个单向函数，计算`f(data)`很容易，但通过`digest`反推`data`却非常困难。而且，对原始数据做一个bit的修改，都会导致计算出的摘要完全不同。
- 有没有可能两个不同的数据通过某个摘要算法得到了相同的摘要？完全有可能，因为任何摘要算法都是把无限多的数据集合映射到一个有限的集合中。这种情况称为碰撞，比如Bob试图根据你的摘要反推出一篇文章`'how to learn hashlib in python - by Bob'`，并且这篇文章的摘要恰好和你的文章完全一致，这种情况也并非不可能出现，但是非常非常困难。
- 摘要算法在很多地方都有广泛的应用。要注意摘要算法不是加密算法，不能用于加密（因为无法通过摘要反推明文），只能用于防篡改，但是它的单向计算特性决定了可以在不存储明文口令的情况下验证用户口令。


## hashlib

作用：

- 提供了常见的哈希算法，如MD5，SHA1等等。

### MD5

举例：

- 以常见的摘要算法MD5为例，计算出一个字符串的MD5值：

```py
import hashlib

md5 = hashlib.md5()
md5.update('how to use md5 in python hashlib?'.encode('utf-8'))
print(md5.hexdigest())

md5 = hashlib.md5()
md5.update('how to use md5 in '.encode('utf-8'))
md5.update('python hashlib?'.encode('utf-8'))
print(md5.hexdigest())
```

输出：

```
d26a53750bc40b38b65a520292f69306
```

说明：

- 如果数据量很大，可以分块多次调用`update()`，最后计算的结果是一样的：
- 试试改动一个字母，看看计算的结果是否完全不同。
- MD5是最常见的摘要算法，速度很快，生成结果是固定的128 bit字节，通常用一个32位的16进制字符串表示。


### SHA1

举例：

```py
import hashlib

sha1 = hashlib.sha1()
sha1.update('how to use sha1 in '.encode('utf-8'))
sha1.update('python hashlib?'.encode('utf-8'))
print(sha1.hexdigest())
```

输出：

```txt
2c76b57293ce30acef38d98f6046927161b46a44
```

说明：

- SHA1的结果是160 bit字节，通常用一个40位的16进制字符串表示。
- 比SHA1更安全的算法是SHA256和SHA512，不过越安全的算法不仅越慢，而且摘要长度更长。


### 应用


网站用户密码，可以保存为MD5：

| username | password                         |
| :------- | :------------------------------- |
| michael  | e10adc3949ba59abbe56e057f20f883e |
| bob      | 878ef96e86145580c38c87f0410ad153 |
| alice    | 99b1c2188db85afee403b1536010c2c9 |

当用户登录时，首先计算用户输入的明文密码的MD5，然后和数据库存储的MD5对比，如果一致，说明密码输入正确，如果不一致，密码肯定错误。

采用MD5存储密码是否就一定安全呢？

也不一定。假设你是一个黑客，已经拿到了存储MD5密码的数据库，如何通过MD5反推用户的明文密码呢？暴力破解费事费力，真正的黑客不会这么干。

考虑这么个情况，很多用户喜欢用`123456`，`888888`，`password`这些简单的密码，于是，黑客可以事先计算出这些常用密码的MD5值，得到一个反推表：

```
'e10adc3949ba59abbe56e057f20f883e': '123456'
'21218cca77804d2ba1922c33e0151105': '888888'
'5f4dcc3b5aa765d61d8327deb882cf99': 'password'
```

这样，无需破解，只需要对比数据库的MD5，黑客就获得了使用常用密码的用户账号。

### 加盐

对于用户来讲，当然不要使用过于简单的密码。但是，我们能否在程序设计上对简单密码加强保护呢？

由于常用密码的MD5值很容易被计算出来，所以，要确保存储的用户密码不是那些已经被计算出来的常用密码的MD5，这一方法通过对原始密码加一个复杂字符串来实现，俗称“加盐”：

```py
def calc_md5(password):
    return get_md5(password + 'the-Salt')
```

经过Salt处理的MD5密码，只要Salt不被黑客知道，即使用户输入简单密码，也很难通过MD5反推明文密码。

但是如果有两个用户都使用了相同的简单密码比如`123456`，在数据库中，将存储两条相同的MD5值，这说明这两个用户的密码是一样的。有没有办法让使用相同密码的用户存储不同的MD5呢？

如果假定用户无法修改登录名，就可以通过把登录名作为Salt的一部分来计算MD5，从而实现相同密码的用户也存储不同的MD5。


## hmac

作用：

- 和我们自定义的加salt算法不同，Hmac算法针对所有哈希算法都通用，无论是MD5还是SHA-1。采用Hmac替代我们自己的salt算法，可以使程序算法更标准化，也更安全。

举例：

```py
import hmac

message = b'Hello, world!'
key = b'secret'
h = hmac.new(key, message, digestmod='MD5')
h.update(b'msgmsgmsg')
print(h.hexdigest())
```

输出：

```txt
afd02506fb7f784859cd1b1c61cf03d1
```

说明：

- 如果消息很长，可以多次调用h.update(msg)
- 使用hmac和普通hash算法非常类似。hmac输出的长度和原始哈希算法的长度一致。需要注意传入的key和message都是`bytes`类型，`str`类型需要首先编码为`bytes`。
- Hmac算法利用一个key对message计算“杂凑”后的hash，使用hmac算法比标准hash算法更安全，因为针对相同的message，不同的key会产生不同的hash。