"plistlib" --- 生成与解析 Mac OS X ".plist" 文件
************************************************

**源代码:** Lib/plistlib.py

======================================================================

此模块提供了读写主要用于 Mac OS X 的 "property list" 文件的接口，并同
时支持二进制和 XML plist 文件。

property list (".plist") 文件格式是一种简单的序列化格式，它支持一些基
本对象类型，例如字典、列表、数字和字符串等。 通常使用一个字典作为最高
层级对象。

要写入和解析 plist 文件，请使用 "dump()" 和 "load()" 函数。

要以字节串对象形式操作 plist 数据，请使用 "dumps()" 和 "loads()"。

值可以为字符串、整数、浮点数、布尔值、元组、列表、字典（但只允许用字符
串作为键）、"Data"、"bytes"、"bytesarray" 或 "datetime.datetime" 对象
。

在 3.4 版更改: 新版 API，旧版 API 已被弃用。 添加了对二进制 plist 格式
的支持。

在 3.8 版更改: 添加了在二进制 plist 中读写 "UID" 令牌的支持，例如用于
NSKeyedArchiver 和 NSKeyedUnarchiver。

参见:

  PList 指南页面
     针对该文件格式的 Apple 文档。

这个模块定义了以下函数：

plistlib.load(fp, *, fmt=None, use_builtin_types=True, dict_type=dict)

   读取 plist 文件。 *fp* 应当可读并且为二进制文件对象。 返回已解包的
   根对象（通常是一个字典）。

   *fmt* 为文件的格式，有效的值如下:

   * "None": 自动检测文件格式

   * "FMT_XML": XML 文件格式

   * "FMT_BINARY": 二进制 plist 格式

   如果 *use_builtin_types* 为真值（默认）则将以 "bytes" 实例的形式返
   回二进制数据，否则将以 "Data" 实例的形式返回。

   *dict_type* 为字典用来从 plist 文件读取的类型。

   "FMT_XML" 格式的 XML 数据 会使用来自 "xml.parsers.expat" 的 Expat
   解析器 -- 请参阅其文档了解错误格式 XML 可能引发的异常。 未知元素将
   被 plist 解析器直接略过。

   当文件无法被解析时二进制格式的解析器将引发 "InvalidFileException"。

   3.4 新版功能.

plistlib.loads(data, *, fmt=None, use_builtin_types=True, dict_type=dict)

   从一个 bytes 对象加载 plist。 参阅 "load()" 获取相应关键字参数的说
   明。

   3.4 新版功能.

plistlib.dump(value, fp, *, fmt=FMT_XML, sort_keys=True, skipkeys=False)

   将 *value* 写入 plist 文件。 *Fp* 应当可写并且为二进制文件对象。

   *fmt* 参数指定 plist 文件的格式，可以是以下值之一:

   * "FMT_XML": XML 格式的 plist 文件

   * "FMT_BINARY": 二进制格式的 plist 文件

   当 *sort_keys* 为真值（默认）时字典的键将经过排序再写入 plist，否则
   将按字典的迭代顺序写入。

   当 *skipkeys* 为假值（默认）时该函数将在字典的键不为字符串时引发
   "TypeError"，否则将跳过这样的键。

   如果对象是不受支持的类型或者是包含不受支持类型的对象的容器则将引发
   "TypeError"。

   对于无法在（二进制）plist 文件中表示的整数值，将会引发
   "OverflowError"。

   3.4 新版功能.

plistlib.dumps(value, *, fmt=FMT_XML, sort_keys=True, skipkeys=False)

   将 *value* 以 plist 格式字节串对象的形式返回。 参阅 "dump()" 的文档
   获取此函数的关键字参数的说明。

   3.4 新版功能.

以下函数已被弃用:

plistlib.readPlist(pathOrFile)

   读取 plist 文件。 *pathOrFile* 可以是文件名或（可读并且为二进制的）
   文件对象。 返回已解包的根对象（通常是一个字典）。

   此函数会调用 "load()" 来完成实际操作，请参阅 "该函数" 的文档获取相
   应关键字参数的说明。

   3.4 版后已移除: 使用 "load()" 来代替。

   在 3.7 版更改: 结果中的字典值现在是普通字典。 你不能再使用属性引用
   形式来访问这些字典中的项。

plistlib.writePlist(rootObject, pathOrFile)

   将 *rootObject* 写入 XML plist 文件。 *pathOrFile* 可以是文件名或（
   可写并且为二进制的）文件对象。

   3.4 版后已移除: 使用 "dump()" 来代替。

plistlib.readPlistFromBytes(data)

   从字节串对象读取 plist 数据。 返回根对象。

   参阅 "load()" 获取相应关键字参数的说明。

   3.4 版后已移除: 使用 "loads()" 来代替。

   在 3.7 版更改: 结果中的字典值现在是普通字典。 你不能再使用属性引用
   形式来访问这些字典中的项。

plistlib.writePlistToBytes(rootObject)

   将 *rootObject* 作为 XML plist 格式的字节串对象返回。

   3.4 版后已移除: 使用 "dumps()" 来代替。

可以使用以下的类:

class plistlib.Data(data)

   返回一个包含字节串对象 *data* 的 "data" 包装器对象。 该类将在对
   plist 进行双向转换的函数中被用来代表 plist 中可用的 "<data>" 类型。

   它具有一个属性 "data"，可被用来提取其中所保存的 Python 字节串对象。

   3.4 版后已移除: 使用 "bytes" 对象来代替。

class plistlib.UID(data)

   包装一个 "int"。 该类将在读取或写入 NSKeyedArchiver 编码的数据时被
   使用，其中包含 UID（参见 PList 指南）。

   它具有一个属性 "data"，可被用来提取 UID 的 int 值。 "data" 的取值范
   围必须为 *0 <= data < 2**64*。

   3.8 新版功能.

可以使用以下的常量:

plistlib.FMT_XML

   用于 plist 文件的 XML 格式。

   3.4 新版功能.

plistlib.FMT_BINARY

   用于 plist 文件的二进制格式。

   3.4 新版功能.


示例
====

生成一个 plist:

   pl = dict(
       aString = "Doodah",
       aList = ["A", "B", 12, 32.1, [1, 2, 3]],
       aFloat = 0.1,
       anInt = 728,
       aDict = dict(
           anotherString = "<hello & hi there!>",
           aThirdString = "M\xe4ssig, Ma\xdf",
           aTrueValue = True,
           aFalseValue = False,
       ),
       someData = b"<binary gunk>",
       someMoreData = b"<lots of binary gunk>" * 10,
       aDate = datetime.datetime.fromtimestamp(time.mktime(time.gmtime())),
   )
   with open(fileName, 'wb') as fp:
       dump(pl, fp)

解析一个 plist:

   with open(fileName, 'rb') as fp:
       pl = load(fp)
   print(pl["aKey"])
