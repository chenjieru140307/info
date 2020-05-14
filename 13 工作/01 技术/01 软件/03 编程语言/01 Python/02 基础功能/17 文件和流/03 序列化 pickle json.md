
# 序列化

- 把变量从内存中变成可存储或传输的过程称之为序列化

## pickle

举例：

```py
import pickle

d = dict(name='Bob', age=20, score=88)
print(pickle.dumps(d))

with open('dump.txt', 'wb') as f:
    pickle.dump(d, f)

with open('dump.txt', 'rb') as f:
    d = pickle.load(f)
    print(d)
```

输出:

```txt
b'\x80\x03}q\x00(X\x04\x00\x00\x00nameq\x01X\x03\x00\x00\x00Bobq\x02X\x03\x00\x00\x00ageq\x03K\x14X\x05\x00\x00\x00scoreq\x04KXu.'
{'name': 'Bob', 'age': 20, 'score': 88}
```


说明:

- `pickle.dumps(d)` 把任意对象序列化成一个 `bytes`
- `pickle.dump(d, f)` 可以把对象序列化后写入一个 file-like Object。
- `pickle.load(f)` 从一个 `file-like Object` 中直接反序列化出对象。
- 注意：是 `rb` 和 `wb`，因为会转化为 bytes 类型。

pickle 的问题：

- 只能用于 Python，并且可能不同版本的 Python 彼此都不兼容。

## json

json 优点：

- 可以被所有语言读取，也可以方便地存储到磁盘或者通过网络传输。
- 比 XML 更快，而且可以直接在 Web 页面中读取，非常方便。

json 表示的对象就是标准的 JavaScript 语言的对象。

JSON和 Python 内置的数据类型对应如下：

| JSON类型   | Python类型 |
| ---------- | ---------- |
| {}         | dict       |
| []         | list       |
| "string"   | str        |
| 1234.56    | int或 float |
| true/false | True/False |
| null       | None       |

举例：


```py
import json

d = dict(name='Bob', age=20, score=88)
print(json.dumps(d),type(json.dumps(d)))
print(json.dumps(d, sort_keys=True, indent=4, separators=(',', ':')))
json_str = '{"age": 20, "score": 88, "name": "Bob"}'
print(json.loads(json_str),type(json.loads(json_str)))


with open('dump.json', 'w') as f:
    json.dump(d, f)

with open('dump.json', 'r') as f:
    d = json.load(f)
    print(d)

print('')



class Student(object):
    def __init__(self, name, age, score):
        self.name = name
        self.age = age
        self.score = score

s = Student('Bob', 20, 88)
print(json.dumps(s))


def student2dict(std):
    return {
        'name': std.name,
        'age': std.age,
        'score': std.score
    }


print(json.dumps(s, default=student2dict))
print(json.dumps(s, default=lambda obj: obj.__dict__)) # 推荐


def dict2student(d):
    return Student(d['name'], d['age'], d['score'])

json_str = '{"age": 20, "score": 88, "name": "Bob"}'
print(json.loads(json_str, object_hook=dict2student)) # 推荐
```

输出：

```txt
{"name": "Bob", "age": 20, "score": 88} <class 'str'>
{
    "age":20,
    "name":"Bob",
    "score":88
}
{'age': 20, 'score': 88, 'name': 'Bob'} <class 'dict'>
{'name': 'Bob', 'age': 20, 'score': 88}

Traceback (most recent call last):
  File "D:/21.Practice/demo/t.py", line 26, in <module>
    print(json.dumps(s))
  File "D:\01.ProgramFiles\Anaconda3\envs\tensorflow2\lib\json\__init__.py", line 231, in dumps
    return _default_encoder.encode(obj)
  File "D:\01.ProgramFiles\Anaconda3\envs\tensorflow2\lib\json\encoder.py", line 199, in encode
    chunks = self.iterencode(o, _one_shot=True)
  File "D:\01.ProgramFiles\Anaconda3\envs\tensorflow2\lib\json\encoder.py", line 257, in iterencode
    return _iterencode(o, 0)
  File "D:\01.ProgramFiles\Anaconda3\envs\tensorflow2\lib\json\encoder.py", line 179, in default
    raise TypeError(f'Object of type {o.__class__.__name__} '
TypeError: Object of type Student is not JSON serializable
{"name": "Bob", "age": 20, "score": 88}
{"name": "Bob", "age": 20, "score": 88}
<__main__.Student object at 0x0000024F2E3F29C8>
```

说明:

- `json.dumps()` 方法返回一个 `str`，内容就是标准的 JSON。`json.dumps(dic, sort_keys=True, indent=4, separators=(',', ':'))` 可以对输出格式进行调整。
  - sort_keys：是否按照字典排序（a-z）输出，True代表是，False代表否。
  - indent=4：设置缩进格数，一般由于 Linux 的习惯，这里会设置为 4。
  - separators：设置分隔符，在 d = {'a': 1, 'b': 2, 'c': 3}这行代码里可以看到冒号和逗号后面都带了个空格，这也是因为 Python 的默认格式也是如此，如果不想后面带有空格输出，那就可以设置成 separators=(',', ':')，如果想保持原样，可以写成 separators=(', ', ': ')。
- `dump()` 方法可以直接把 JSON 写入一个 `file-like Object`。
- 可以用 `loads()`或者 `load()`方法把 json 反序列化为 python 对象，`loads()` 从字符串反序列化，`load()` 从 `file-like Object` 中读取字符串并反序列化
- 由于 JSON 标准规定 JSON 编码是 UTF-8，所以我们总是能正确地在 Python 的 `str` 与 JSON 的字符串之间转换。
- 一个 `Student` 对象没法直接序列化为 json，可以用转换函数进行转换。也可以用 `default=lambda obj: obj.__dict__` ，这本身就是 对象里面的 dict。
- 把 json 反序列化为一个 `Student` 对象实例，`loads()` 方法首先转换出一个 `dict` 对象，然后，我们传入的 `object_hook` 函数负责把 `dict` 转换为 `Student` 实例：



备注：

- `json` 模块的 `dumps()` 和 `loads()` 函数是定义得非常好的接口的典范。当我们使用时，只需要传入一个必须的参数。但是，当默认的序列化或反序列机制不满足我们的要求时，我们又可以传入更多的参数来定制序列化或反序列化的规则，既做到了接口简单易用，又做到了充分的扩展性和灵活性。

