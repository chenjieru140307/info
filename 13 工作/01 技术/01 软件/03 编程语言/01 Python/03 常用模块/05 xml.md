# XML


使用：

- 如果要生成复杂的XML，建议不要用XML，改成JSON。
- xml 在Web中应用也不如以前多了，不过仍有很多地方在用，所以，有必要了解如何操作XML。

操作 XML 有两种方法：DOM和SAX。

- DOM会把整个XML读入内存，解析为树，因此占用内存大，解析慢，优点是可以任意遍历树的节点。
- SAX是流模式，边读边解析，占用内存小，解析快，缺点是我们需要自己处理事件。


正常情况下，优先考虑SAX，因为 DOM 实在太占内存。

在Python中使用SAX解析XML非常简洁，通常我们关心的事件是`start_element`，`end_element`和`char_data`，准备好这3个函数，然后就可以解析xml了。

举个例子，当SAX解析器读到一个节点时：

```
<a href="/">python</a>
```

会产生3个事件：

1. start_element事件，在读取`[`]()时；
2. char_data事件，在读取`python`时；
3. end_element事件，在读取``时。

用代码实验一下：

```py
from xml.parsers.expat import ParserCreate

class DefaultSaxHandler(object):
    def start_element(self, name, attrs):
        print('sax:start_element: %s, attrs: %s' % (name, str(attrs)))

    def end_element(self, name):
        print('sax:end_element: %s' % name)

    def char_data(self, text):
        print('sax:char_data: %s' % text)

xml = r'''<?xml version="1.0"?>
<ol>
    <li><a href="/python">Python</a></li>
    <li><a href="/ruby">Ruby</a></li>
</ol>
'''

handler = DefaultSaxHandler()
parser = ParserCreate()

# 为解析器设置自定义的回调函数
parser.StartElementHandler = handler.start_element
parser.EndElementHandler = handler.end_element
parser.CharacterDataHandler = handler.char_data
parser.Parse(xml)
```

说明：

- 利用SAX解析XML文档牵涉到两个部分: 解析器和事件处理器。
  - 解析器 `parser=ParserCreate()` 负责读取XML文档，并向事件处理器发送事件，如元素开始跟元素结束事件。
  - 事件处理器 `handler=DefualtSaxHandler()` 则负责对事件作出响应，对传递的XML数据进行处理
- name表示节点名称，attrs表示节点属性（字典）
- text表示节点数据

需要注意的是读取一大段字符串时，`CharacterDataHandler`可能被多次调用，所以需要自己保存起来，在`EndElementHandler`里面再合并。

解析XML时，注意找出自己感兴趣的节点，响应事件时，把节点数据保存起来。解析完毕后，就可以处理数据。


解析过程：

（没有看)

```py
#一旦expat解析器遇到xml的 元素开始，元素结束，元素值 事件时
#会回分别调用start_element, end_element, char_data函数

#关于XMLParser Objects的方法介绍下
#详见python文档：xml.parsers.expat
#xmlparser.StartElementHandler(name, attributes)
#遇到XML开始标签时调用，name是标签的名字，attrs是标签的属性值字典
#xmlparser.EndElementHandler(name)
#遇到XML结束标签时调用。
#xmlparser.CharacterDataHandler(data) 
#调用时机：
#从行开始，遇到标签之前，存在字符，content 的值为这些字符串。
#从一个标签，遇到下一个标签之前， 存在字符，content 的值为这些字符串。
#从一个标签，遇到行结束符之前，存在字符，content 的值为这些字符串。
#标签可以是开始标签，也可以是结束标签。

#为了方便理解，我已经在下面还原来解析过程，
#标出何时调用，分别用S：表示开始；E：表示结束；D：表示data

如果看不明白，请配合脚本输出结果一起看
S<ol>C
C   S<li>S<a href="/python">CPython</a>E</li>EC
C   S<li>S<a href="/ruby">CRuby</a>E</li>EC
S</ol>E
```




## 生成 XML

除了解析XML外，如何生成XML呢？99%的情况下需要生成的XML结构都是非常简单的，因此，最简单也是最有效的生成XML的方法是拼接字符串：

```py
L = []
L.append(r'<?xml version="1.0"?>')
L.append(r'<root>')
L.append(encode('some & data'))
L.append(r'</root>')
return ''.join(L)
```




