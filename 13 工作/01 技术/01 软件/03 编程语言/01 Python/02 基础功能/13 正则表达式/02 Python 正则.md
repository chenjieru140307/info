# Python 正则表达式

Python提供 `re` 模块，包含所有正则表达式的功能。

注意：

- 由于 Python 的字符串本身也用`\`转义，所以建议使用 Python 的 `r` 前缀，就不用考虑转义的问题了。`r'ABC\-001'`


## re.match

- 从字符串的起始位置匹配一个模式，如果不是起始位置匹配成功的话，返回none。

注意：

- 只匹配一次。`re.findall` 是匹配所有。

语法：

- `re.match(pattern, string, flags=0)`

举例：

```py
import re
print(re.match('www', 'www.runoob.com').span())  # 在起始位置匹配
print(re.match('com', 'www.runoob.com'))         # 不在起始位置匹配
```

输出：

```txt
(0, 3)
None
```



## re.search

- 扫描整个字符串，匹配成功返回一个匹配的对象，否则返回None。

语法：

- `re.search(pattern, string, flags=0)`

举例：

```py
import re

print(re.search('www', 'Www.runoob.com',flags=re.I).span())  # 在起始位置匹配
print(re.search('com', 'www.runoob.com').span())  # 不在起始位置匹配

line = "Cats are smarter than dogs"
searchObj = re.search(r'(.*) are (.*?) .*', line, re.M | re.I)
if searchObj:
    print("searchObj.group() : ", searchObj.group())
    print("searchObj.group(1) : ", searchObj.group(1))
    print("searchObj.group(2) : ", searchObj.group(2))
```

输出：

```
(0, 3)
(11, 14)
searchObj.group() :  Cats are smarter than dogs
searchObj.group(1) :  Cats
searchObj.group(2) :  smarter
```

re.match 与 re.search的区别：

- re.match 只匹配字符串的开始，如果字符串开始不符合正则表达式，则匹配失败，函数返回None；
- re.search 匹配整个字符串，直到找到一个匹配。



## group

- 用 `()` 来表示要提取的分组（Group）。

`^(\d{3})-(\d{3,8})$`分别定义了两个组，可以直接从匹配的字符串中提取出区号和本地号码：

```py
import re

m = re.match(r'^(\d{3})-(\d{3,8})$', '010-12345')
print(m)
print(m.group(0))
print(m.group(1))
print(m.group(2))

t = '19:05:30'
m = re.match(
    r'^(0[0-9]|1[0-9]|2[0-3]|[0-9])\:(0[0-9]|1[0-9]|2[0-9]|3[0-9]|4[0-9]|5[0-9]|[0-9])\:(0[0-9]|1[0-9]|2[0-9]|3[0-9]|4[0-9]|5[0-9]|[0-9])$',
    t)
print(m.groups())
print(m.group(0))
print(m.group(1))
print(m.group(2))
print(m.group(3))
```

输出：

```txt
<re.Match object; span=(0, 9), match='010-12345'>
010-12345
010
12345
('19', '05', '30')
19:05:30
19
05
30
```

说明：

- 如果正则表达式中定义了组，就可以在`Match`对象上用 `group()` 方法提取出子串来。
- 注意：`group(0)` 永远是原始字符串，`group(1)`、`group(2)`……表示第 1、2、……个子串。
- 第二个正则表达式可以识别合法的时间，但是对于日期，比如 `2-30`，`4-31`这样的非法日期，用正则还是识别不了，或者说写出来非常困难，这时就需要程序配合识别了。



## 贪婪与非贪婪

正则匹配默认是贪婪匹配，也就是匹配尽可能多的字符。

举例：

```py
import re

print(re.match(r'^(\d+)(0*)$', '102300').groups())
print(re.match(r'^(\d+?)(0*)$', '102300').groups())
```

输出：

```txt
('102300', '')
('1023', '00')
```

说明：

- 由于`\d+`采用贪婪匹配，直接把后面的`0`全部匹配了，结果`0*`只能匹配空字符串了。
- 必须让`\d+`采用非贪婪匹配（也就是尽可能少匹配），才能把后面的`0`匹配出来，加个`?`就可以让`\d+`采用非贪婪匹配：



## re.sub


- 替换字符串中的匹配项。

语法：

- `re.sub(pattern, repl, string, count=0, flags=0)`
  - pattern : 正则中的模式字符串。
  - repl : 替换的字符串，也可为一个函数。
  - string : 要被查找替换的原始字符串。
  - count : 模式匹配后替换的最大次数，默认 0 表示替换所有的匹配。

举例：


```py
import re

phone = "2004-959-559 # 这是一个国外电话号码"
num = re.sub(r'#.*$', "", phone)  # 删除字符串中的 Python注释
print(num)
num = re.sub(r'\D', "", phone)  # 删除非数字(-)的字符串
print(num)


# 将匹配的数字乘于 2
def double(matched):
    value = int(matched.group('value'))
    return str(value * 2)
s = 'A23G4HFD567'
print(re.sub(r'(?P<value>\d+)', repl=double, string=s))
```

输出:

```
2004-959-559 
2004959559
A46G8HFD1134
```


疑问：

- 1134 是怎么算出来的？
- `r'(?P<value>\d+)'` 是什么？

## re.compile

- 生成一个正则表达式（ Pattern ）对象，供 match() 和 search() 这两个函数使用。

语法：

- `re.compile(pattern[, flags])`

举例：

```py
import re

pattern = re.compile(r'\d+')  # 用于匹配至少一个数字
m = pattern.match('one12twothree34four')  # 查找头部，没有匹配
print(m)
m = pattern.match('one12twothree34four', pos=2, endpos=10)  # 从'e'的位置开始匹配，没有匹配
print(m)
m = pattern.match('one12twothree34four', pos=3, endpos=16)  # 从'1'的位置开始匹配，正好匹配
print(m)
print(m.group(0))  # 可省略 0
print(m.start(0))  # 可省略 0
print(m.end(0))  # 可省略 0
print(m.span(0))  # 可省略 0

print('')

pattern = re.compile(r'([a-z]+) ([a-z]+)', re.I)  # re.I 表示忽略大小写
m = pattern.match('Hello World Wide Web')
print(m)  # 匹配成功，返回一个 Match 对象
print(m.group(0))  # 返回匹配成功的整个子串
print(m.span(0))  # 返回匹配成功的整个子串的索引
print(m.group(1))  # 返回第一个分组匹配成功的子串
print(m.span(1))  # 返回第一个分组匹配成功的子串的索引
print(m.group(2))  # 返回第二个分组匹配成功的子串
print(m.span(2))  # 返回第二个分组匹配成功的子串索引
print(m.groups())  # 等价于 (m.group(1), m.group(2), ...)
print(m.group(3))  # 不存在第三个分组
```

输出：

```txt
None
None
<re.Match object; span=(3, 5), match='12'>
12
3
5
(3, 5)

Hello World
(0, 11)
Hello
(0, 5)
World
(6, 11)
('Hello', 'World')
Traceback (most recent call last):
<re.Match object; span=(0, 11), match='Hello World'>
  File "D:/21.Practice/demo/t.py", line 13, in <module>
    print(m.group(3))  # 不存在第三个分组
IndexError: no such group
```
说明：

- `group([group1, …])` 方法用于获得一个或多个分组匹配的字符串，当要获得整个匹配的子串时，可直接使用 `group()` 或 `group(0)`；
- `start([group])` 方法用于获取分组匹配的子串在整个字符串中的起始位置（子串第一个字符的索引），参数默认值为 0；
- `end([group])` 方法用于获取分组匹配的子串在整个字符串中的结束位置（子串最后一个字符的索引+1），参数默认值为 0；
- `span([group])` 方法返回 `(start(group), end(group))`。



## re.findall

- 在字符串中找到所匹配的所有子串，并返回一个列表，如果没有找到匹配的，则返回空列表。

**注意：**

- match 和 search 是匹配一次。findall 匹配所有。

语法：

- `findall(string[, pos[, endpos]])`
  - **pos** : 可选参数，指定字符串的起始位置，默认为 0。
  - **endpos** : 可选参数，指定字符串的结束位置，默认为字符串的长度。

举例：

```py
import re

pattern = re.compile(r'\d+')  # 查找数字
result1 = pattern.findall('runoob 123 google 456')
result2 = pattern.findall('run88oob123google456', 0, 10)

print(result1)
print(result2)
```

输出：

```
['123', '456']
['88', '12']
```

## re.finditer

- 和 `re.findall` 类似，在字符串中找到正则表达式所匹配的所有子串，并把它们作为一个迭代器返回。

语法：

- `re.finditer(pattern, string, flags=0)`

举例：

```py
import re

it = re.finditer(r"\d+", "12a32bc43jf3")
for match in it:
    print(match.group())
```

输出：

```
12 
32 
43 
3
```

## re.split

- 按照能够匹配的子串，将字符串分割后返回列表

语法：

- re.split(pattern, string[, maxsplit=0, flags=0])
  - maxsplit 分隔次数，maxsplit=1 分隔一次，默认为 0，不限制次数。

举例：

```py
import re

print(re.split('\W+', 'runoob, runoob, runoob.'))
print(re.split('(\W+)', ' runoob, runoob, runoob.'))
print(re.split('\W+', ' runoob, runoob, runoob.', 1))
print(re.split('a*', 'hello world'))  # 对于一个找不到匹配的字符串而言，split 不会对其作出分割
```

输出：

```txt
['runoob', 'runoob', 'runoob', '']
['', ' ', 'runoob', ', ', 'runoob', ', ', 'runoob', '.', '']
['', 'runoob, runoob, runoob.']
['', 'h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', '']
```

疑问：

- 还有点没没明白。


## 正则表达式对象

- re.RegexObject
  - re.compile() 返回 RegexObject 对象。
- re.MatchObject
  - group() 返回被 RE 匹配的字符串。
    - **start()** 返回匹配开始的位置
    - **end()** 返回匹配结束的位置
    - **span()** 返回一个元组包含匹配 (开始,结束) 的位置



## 可选标志

- 控制匹配的模式。
- 多个标志可以通过按位 OR(|) 它们来指定。如 re.I | re.M 被设置成 I 和 M 标志：

| 修饰符 | 描述                                                         |
| :----- | :----------------------------------------------------------- |
| re.I   | 使匹配对大小写不敏感                                         |
| re.L   | 做本地化识别（locale-aware）匹配，表示特殊字符集 \w, \W, \b, \B, \s, \S 依赖于当前环境                             |
| re.M   | 多行匹配，影响 ^ 和 $                                        |
| re.S   | 使 . 匹配包括换行在内的所有字符                              |
| re.U   | 根据Unicode字符集解析字符。表示特殊字符集 \w, \W, \b, \B, \d, \D, \s, \S 依赖于 Unicode 字符属性数据库      |
| re.X   | 该标志通过给予你更灵活的格式以便你将正则表达式写得更易于理解。为了增加可读性，忽略空格和 **#** 后面的注释 |


## 一些字符

如果你使用模式的同时提供了可选的标志参数，某些模式元素的含义会改变。

| 模式        | 描述                                                         |
| :---------- | :----------------------------------------------------------- |
| ^           | 匹配字符串的开头                                             |
| $           | 匹配字符串的末尾。                                           |
| .           | 匹配任意字符，除了换行符，当re.DOTALL标记被指定时，则可以匹配包括换行符的任意字符。 |
| [...]       | 用来表示一组字符,单独列出：[amk] 匹配 'a'，'m'或'k'          |
| [^...]      | 不在[]中的字符：[^abc] 匹配除了a,b,c之外的字符。             |
| re*         | 匹配0个或多个的表达式。                                      |
| re+         | 匹配1个或多个的表达式。                                      |
| re?         | 匹配0个或1个由前面的正则表达式定义的片段，非贪婪方式         |
| re{ n}      | 精确匹配 n 个前面表达式。例如， **o{2}** 不能匹配 "Bob" 中的 "o"，但是能匹配 "food" 中的两个 o。 |
| re{ n,}     | 匹配 n 个前面表达式。例如， o{2,} 不能匹配"Bob"中的"o"，但能匹配 "foooood"中的所有 o。"o{1,}" 等价于 "o+"。"o{0,}" 则等价于 "o*"。 |
| re{ n, m}   | 匹配 n 到 m 次由前面的正则表达式定义的片段，贪婪方式         |
| a\| b       | 匹配a或b                                                     |
| (re)        | 对正则表达式分组并记住匹配的文本                             |
| (?imx)      | 正则表达式包含三种可选标志：i, m, 或 x 。只影响括号中的区域。 |
| (?-imx)     | 正则表达式关闭 i, m, 或 x 可选标志。只影响括号中的区域。     |
| (?: re)     | 类似 (...), 但是不表示一个组                                 |
| (?imx: re)  | 在括号中使用i, m, 或 x 可选标志                              |
| (?-imx: re) | 在括号中不使用i, m, 或 x 可选标志                            |
| (?#...)     | 注释.                                                        |
| (?= re)     | 前向肯定界定符。如果所含正则表达式，以 ... 表示，在当前位置成功匹配时成功，否则失败。但一旦所含表达式已经尝试，匹配引擎根本没有提高；模式的剩余部分还要尝试界定符的右边。 |
| (?! re)     | 前向否定界定符。与肯定界定符相反；当所含表达式不能在字符串当前位置匹配时成功 |
| (?> re)     | 匹配的独立模式，省去回溯。                                   |
| \w          | 匹配字母数字及下划线                                         |
| \W          | 匹配非字母数字及下划线                                       |
| \s          | 匹配任意空白字符，等价于 [\t\n\r\f].                         |
| \S          | 匹配任意非空字符                                             |
| \d          | 匹配任意数字，等价于 [0-9].                                  |
| \D          | 匹配任意非数字                                               |
| \A          | 匹配字符串开始                                               |
| \Z          | 匹配字符串结束，如果是存在换行，只匹配到换行前的结束字符串。 |
| \z          | 匹配字符串结束                                               |
| \G          | 匹配最后匹配完成的位置。                                     |
| \b          | 匹配一个单词边界，也就是指单词和空格间的位置。例如， 'er\b' 可以匹配"never" 中的 'er'，但不能匹配 "verb" 中的 'er'。 |
| \B          | 匹配非单词边界。'er\B' 能匹配 "verb" 中的 'er'，但不能匹配 "never" 中的 'er'。 |
| \n, \t, 等. | 匹配一个换行符。匹配一个制表符。等                           |
| \1...\9     | 匹配第n个分组的内容。                                        |
| \10         | 匹配第n个分组的内容，如果它经匹配。否则指的是八进制字符码的表达式。 |

特殊字符：

| 实例 | 描述                                                         |
| :--- | :----------------------------------------------------------- |
| .    | 匹配除 "\n" 之外的任何单个字符。要匹配包括 '\n' 在内的任何字符，请使用象 '[.\n]' 的模式。 |
| \d   | 匹配一个数字字符。等价于 [0-9]。                             |
| \D   | 匹配一个非数字字符。等价于 [^0-9]。                          |
| \s   | 匹配任何空白字符，包括空格、制表符、换页符等等。等价于 [ \f\n\r\t\v]。 |
| \S   | 匹配任何非空白字符。等价于 [^ \f\n\r\t\v]。                  |
| \w   | 匹配包括下划线的任何单词字符。等价于'[A-Za-z0-9_]'。         |
| \W   | 匹配任何非单词字符。等价于 '[^A-Za-z0-9_]'。                 |




## 举例

**举例1：**

（这个没怎么看）

- 处理 email 条目


```py
import re
import pandas as pd
import email

emails = []

fh = open(r"test_emails.txt", "r").read()

contents = re.split(r"From r",fh)
contents.pop(0)

for item in contents:
    emails_dict = {}

    sender = re.search(r"From:.*", item)

    if sender is not None:
        s_email = re.search(r"\w\S*@.*\w", sender.group())
        s_name = re.search(r":.*<", sender.group())
    else:
        s_email = None
        s_name = None

    if s_email is not None:
        sender_email = s_email.group()
    else:
        sender_email = None

    emails_dict["sender_email"] = sender_email

    if s_name is not None:
        sender_name = re.sub("\s*<", "", re.sub(":\s*", "", s_name.group()))
    else:
        sender_name = None

    emails_dict["sender_name"] = sender_name

    recipient = re.search(r"To:.*", item)

    if recipient is not None:
        r_email = re.search(r"\w\S*@.*\w", recipient.group())
        r_name = re.search(r":.*<", recipient.group())
    else:
        r_email = None
        r_name = None

    if r_email is not None:
        recipient_email = r_email.group()
    else:
        recipient_email = None

    emails_dict["recipient_email"] = recipient_email

    if r_name is not None:
        recipient_name = re.sub("\s*<", "", re.sub(":\s*", "", r_name.group()))
    else:
        recipient_name = None

    emails_dict["recipient_name"] = recipient_name

    date_field = re.search(r"Date:.*", item)

    if date_field is not None:
        date = re.search(r"\d+\s\w+\s\d+", date_field.group())
    else:
        date = None

    if date is not None:
        date_sent = date.group()
    else:
        date_sent = None

    emails_dict["date_sent"] = date_sent

    subject_field = re.search(r"Subject: .*", item)

    if subject_field is not None:
        subject = re.sub(r"Subject: ", "", subject_field.group())
    else:
        subject = None

    emails_dict["subject"] = subject

    # "item" substituted with "email content here" so full email not displayed.

    full_email = email.message_from_string(item)
    body = full_email.get_payload()
    emails_dict["email_body"] = "email body here"

    emails.append(emails_dict)

    # Print number of dictionaries, and hence, emails, in the list.
    print("Number of emails: " + str(len(emails_dict)))

    print("\n")

    # Print first item in the emails list to see how it looks.
    for key, value in emails[0].items():
        print(str(key) + ": " + str(emails[0][key]))
```
