# chardet

作用：

- 对于未知编码的`bytes`，要把它转换成`str`，需要先“猜测”编码。chardet 可以用来检测编码，简单易用。

文档：

- [文档](https://chardet.readthedocs.io/en/latest/supported-encodings.html)

安装：（anaconda 附带了）

- `pip install chardet`


举例：

```py
import chardet

print(chardet.detect(b'Hello, world!'))
print(chardet.detect('离离原上草，一岁一枯荣'.encode('gbk')))
print(chardet.detect('离离原上草，一岁一枯荣'.encode('utf-8')))
print(chardet.detect('最新の主要ニュース'.encode('euc-jp')))
```

输出：

```txt
{'encoding': 'ascii', 'confidence': 1.0, 'language': ''}
{'encoding': 'GB2312', 'confidence': 0.7407407407407407, 'language': 'Chinese'}
{'encoding': 'utf-8', 'confidence': 0.99, 'language': ''}
{'encoding': 'EUC-JP', 'confidence': 0.99, 'language': 'Japanese'}
```


说明：

- 对于第二个检测，检测的编码是`GB2312`，注意到GBK是GB2312的超集，两者是同一种编码，检测正确的概率是74%，`language`字段指出的语言是`'Chinese'`。
- 可见，可以先用 chardet 检测编码，获取到编码后，再对 `bytes`做`decode()`。
