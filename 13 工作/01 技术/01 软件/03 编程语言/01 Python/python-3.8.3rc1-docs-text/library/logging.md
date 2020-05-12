"logging" --- Python 的日志记录工具
***********************************

**源代码：** Lib/logging/__init__.py


Important
^^^^^^^^^

此页面仅包含 API 参考信息。教程信息和更多高级用法的讨论，请参阅

* 基础教程

* 进阶教程

* 日志操作手册

======================================================================

这个模块为应用与库实现了灵活的事件日志系统的函数与类。

使用标准库提提供的 logging API 最主要的好处是，所有的 Python 模块都可
能参与日志输出，包括你自己的日志消息和第三方模块的日志消息。

这个模块提供许多强大而灵活的功能。如果你对 logging 不太熟悉的话， 掌握
它最好的方式就是查看它对应的教程（详见右侧的链接）。

该模块定义的基础类和函数都列在下面。

* 记录器暴露了应用程序代码直接使用的接口。

* 处理器将日志记录（由记录器创建）发送到适当的目标。

* 过滤器提供了更精细的设施，用于确定要输出的日志记录。

* 格式器指定最终输出中日志记录的样式。


记录器对象
==========

记录器有以下的属性和方法。注意 *永远* 不要直接实例化记录器，应当通过模
块级别的函数 "logging.getLogger(name)" 。多次使用相同的名字调用
"getLogger()" 会一直返回相同的 Logger 对象的引用。

"name" 一般是句点分割的层级值, 像``foo.bar.baz`` (尽管也可以只是普通的
"foo")。层次结构列表中位于下方的记录器是列表中较高位置的记录器的子级。
例如，有个名叫 "foo" 的记录器，而名字是 "foo.bar"，"foo.bar.baz"，和
"foo.bam" 的记录器都是 "foo" 的子级。记录器的名字分级类似 Python 包的
层级，如果您使用建议的结构 "logging.getLogger(__name__)" 在每个模块的
基础上组织记录器，则与之完全相同。这是因为在模块里，"__name__" 是该模
块在 Python 包命名空间中的名字。

class logging.Logger

   propagate

      如果这个属性为真，记录到这个记录器的事件除了会发送到此记录器的所
      有处理程序外，还会传递给更高级别（祖先）记录器的处理器，此外任何
      关联到这个记录器的处理器。消息会直接传递给祖先记录器的处理器 ——
      不考虑祖先记录器的级别和过滤器。

      如果为假，记录消息将不会传递给当前记录器的祖先记录器的处理器。

      构造器将这个属性初始化为 "True"。

      注解:

        如果你将一个处理器附加到一个记录器 *和* 其一个或多个祖先记录器
        ，它可能发出多次相同的记录。通常，您不需要将一个处理器附加到一
        个以上的记录器上 —— 如果您将它附加到记录器层次结构中最高的适当
        记录器上，则它将看到所有后代记录器记录的所有事件，前提是它们的
        传播设置保留为 "True"。一种常见的方案是仅将处理程序附加到根记
        录器，通过传播来处理其余部分。

   setLevel(level)

      给记录器设置阈值为 *level* 。日志等级小于 *level* 会被忽略。严重
      性为 *level* 或更高的日志消息将由该记录器的任何一个或多个处理器
      发出，除非将处理器的级别设置为比 *level* 更高的级别。

      创建记录器时，级别默认设置为 "NOTSET" （当记录器是根记录器时，将
      处理所有消息；如果记录器不是根记录器，则将委托给父级）。请注意，
      根记录器的默认级别为 "WARNING"。

      委派给父级的意思是如果一个记录器的级别设置为 NOTSET，将遍历其祖
      先记录器，直到找到级别不是 NOTSET 的记录器，或者到根记录器为止。

      如果发现某个父级的级别不是 NOTSET ，那么该父级的级别将被视为发起
      搜索的记录器的有效级别，并用于确定如何处理日志事件。

      如果搜索到达根记录器，并且其级别为 NOTSET，则将处理所有消息。否
      则，将使用根记录器的级别作为有效级别。

      参见 日志级别 级别列表。

      在 3.2 版更改: 现在 *level* 参数可以接受形如 'INFO' 的级别字符串
      表示形式，以代替形如 "INFO" 的整数常量。 但是请注意，级别在内部
      存储为整数，并且 "getEffectiveLevel()" 和 "isEnabledFor()" 等方
      法的传入/返回值也为整数。

   isEnabledFor(level)

      指示此记录器是否将处理级别为 *level* 的消息。此方法首先检查由
      "logging.disable(level)" 设置的模块级的级别，然后检查由
      "getEffectiveLevel()" 确定的记录器的有效级别。

   getEffectiveLevel()

      指示此记录器的有效级别。如果通过 "setLevel()" 设置了除 "NOTSET"
      以外的值，则返回该值。否则，将层次结构遍历到根，直到找到除
      "NOTSET" 以外的其他值，然后返回该值。返回的值是一个整数，通常为
      "logging.DEBUG"、 "logging.INFO" 等等。

   getChild(suffix)

      返回由后缀确定的，是该记录器的后代的记录器。 因此，
      "logging.getLogger('abc').getChild('def.ghi')" 与
      "logging.getLogger('abc.def.ghi')" 将返回相同的记录器。 这是一个
      便捷方法，当使用如 "__name__" 而不是字符串字面值命名父记录器时很
      有用。

      3.2 新版功能.

   debug(msg, *args, **kwargs)

      在此纪录器上记录 "DEBUG" 级别的消息。 *msg* 是消息格式字符串，而
      *args* 是用于字符串格式化操作合并到 *msg* 的参数。（请注意，这意
      味着您可以在格式字符串中使用关键字以及单个字典参数。）当未提供
      *args* 时，不会对 *msg* 执行 ％ 格式化操作。

      在 *kwargs* 中会检查四个关键字参数： *exc_info*，*stack_info*，
      *stacklevel*和*extra*。

      如果 *exc_info* 的求值结果不为 false，则它将异常信息添加到日志消
      息中。如果提供了一个异常元组（按照 "sys.exc_info()" 返回的格式）
      或一个异常实例，则将其使用；否则，调用 "sys.exc_info()" 以获取异
      常信息。

      第二个可选关键字参数是 *stack_info*，默认为 "False"。如果为
      True，则将堆栈信息添加到日志消息中，包括实际的日志调用。请注意，
      这与通过指定 *exc_info* 显示的堆栈信息不同：前者是从堆栈底部到当
      前线程中的日志记录调用的堆栈帧，而后者是在搜索异常处理程序时，跟
      踪异常而打开的堆栈帧的信息。

      您可以独立于 *exc_info* 来指定 *stack_info*，例如，即使在未引发
      任何异常的情况下，也可以显示如何到达代码中的特定点。堆栈帧在标题
      行之后打印：

         Stack (most recent call last):

      这模仿了显示异常帧时所使用的 "Traceback (most recent call
      last):" 。

      第三个可选关键字参数是 *stacklevel*，默认为 "1"。如果大于 1，则
      在为日志记录事件创建的 "LogRecord" 中计算行号和函数名时，将跳过
      相应数量的堆栈帧。可以在记录帮助器时使用它，以便记录的函数名称，
      文件名和行号不是帮助器的函数/方法的信息，而是其调用方的信息。此
      参数是 "warnings" 模块中的同名等效参数。

      第四个关键字参数是 *extra* ，传递一个字典，该字典用于填充为日志
      记录事件创建的、带有用户自定义属性的 "LogRecord" 中的__dict__ 。
      然后可以按照需求使用这些自定义属性。例如，可以将它们合并到已记录
      的消息中：

         FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
         logging.basicConfig(format=FORMAT)
         d = {'clientip': '192.168.0.1', 'user': 'fbloggs'}
         logger = logging.getLogger('tcpserver')
         logger.warning('Protocol problem: %s', 'connection reset', extra=d)

      输出类似于

         2006-02-08 22:20:02,165 192.168.0.1 fbloggs  Protocol problem: connection reset

      The keys in the dictionary passed in *extra* should not clash
      with the keys used by the logging system. (See the "Formatter"
      documentation for more information on which keys are used by the
      logging system.)

      如果在已记录的消息中使用这些属性，则需要格外小心。例如，在上面的
      示例中，"Formatter" 已设置了格式字符串，其在 "LogRecord" 的属性
      字典中键值为 “clientip” 和 “user”。如果缺少这些内容，则将不会记
      录该消息，因为会引发字符串格式化异常。因此，在这种情况下，您始终
      需要使用这些键传递 *extra* 字典。

      尽管这可能很烦人，但此功能旨在用于特殊情况，例如在多个上下文中执
      行相同代码的多线程服务器，并且出现的有趣条件取决于此上下文（例如
      在上面的示例中就是远程客户端IP地址和已验证用户名）。在这种情况下
      ，很可能将专门的 "Formatter" 与特定的 "Handler" 一起使用。

      在 3.2 版更改: 增加了 *stack_info* 参数。

      在 3.5 版更改: The *exc_info* parameter can now accept exception
      instances.

      在 3.8 版更改: 增加了 *stacklevel* 参数。

   info(msg, *args, **kwargs)

      在此记录器上记录 "INFO" 级别的消息。参数解释同 "debug()"。

   warning(msg, *args, **kwargs)

      在此记录器上记录 "WARNING" 级别的消息。参数解释同 "debug()"。

      注解:

        有一个功能上与 "warning" 一致的方法 "warn"。由于 "warn" 已被弃
        用，请不要使用它 —— 改为使用 "warning"。

   error(msg, *args, **kwargs)

      在此记录器上记录 "ERROR" 级别的消息。参数解释同 "debug()"。

   critical(msg, *args, **kwargs)

      在此记录器上记录 "CRITICAL" 级别的消息。参数解释同 "debug()"。

   log(level, msg, *args, **kwargs)

      在此记录器上记录 *level* 整数代表的级别的消息。参数解释同
      "debug()"。

   exception(msg, *args, **kwargs)

      在此记录器上记录 "ERROR" 级别的消息。参数解释同 "debug()"。异常
      信息将添加到日志消息中。仅应从异常处理程序中调用此方法。

   addFilter(filter)

      将指定的过滤器 *filter* 添加到此记录器。

   removeFilter(filter)

      从此记录器中删除指定的过滤器 *filter*。

   filter(record)

      将此记录器的过滤器应用于记录，如果记录能被处理则返回 "True"。过
      滤器会被依次使用，直到其中一个返回假值为止。如果它们都不返回假值
      ，则记录将被处理（传递给处理器）。如果返回任一为假值，则不会对该
      记录做进一步处理。

   addHandler(hdlr)

      将指定的处理器 *hdlr* 添加到此记录器。

   removeHandler(hdlr)

      从此记录器中删除指定的处理器 *hdlr*。

   findCaller(stack_info=False, stacklevel=1)

      查找调用源的文件名和行号，以 文件名，行号，函数名称和堆栈信息 4
      元素元组的形式返回。堆栈信息将返回 "None"，除非 *stack_info* 为
      "True"。

      *stacklevel* 参数用于调用 "debug()" 和其他 API。如果大于 1，则多
      余部分将用于跳过堆栈帧，然后再确定要返回的值。当从帮助器/包装器
      代码调用日志记录 API 时，这通常很有用，以便事件日志中的信息不是
      来自帮助器/包装器代码，而是来自调用它的代码。

   handle(record)

      通过将记录传递给与此记录器及其祖先关联的所有处理器来处理（直到某
      个 *propagate* 值为 false）。此方法用于从套接字接收的未序列化的
      以及在本地创建的记录。使用 "filter()" 进行记录器级别过滤。

   makeRecord(name, level, fn, lno, msg, args, exc_info, func=None, extra=None, sinfo=None)

      这是一种工厂方法，可以在子类中对其进行重写以创建专门的
      "LogRecord" 实例。

   hasHandlers()

      检查此记录器是否配置了任何处理器。通过在此记录器及其记录器层次结
      构中的父级中查找处理器完成此操作。如果找到处理器则返回 "True"，
      否则返回 "False"。只要找到 “propagate” 属性设置为假值的记录器，
      该方法就会停止搜索层次结构 —— 其将是最后一个检查处理器是否存在的
      记录器。

      3.2 新版功能.

   在 3.7 版更改: 现在可以对处理器进行序列化和反序列化。


日志级别
========

日志记录级别的数值在下表中给出。如果你想要定义自己的级别，并且需要它们
具有相对于预定义级别的特定值，那么这些内容可能是你感兴趣的。如果你定义
具有相同数值的级别，它将覆盖预定义的值; 预定义的名称丢失。

+----------------+-----------------+
| 级别           | 数值            |
|================|=================|
| "CRITICAL"     | 50              |
+----------------+-----------------+
| "ERROR"        | 40              |
+----------------+-----------------+
| "WARNING"      | 30              |
+----------------+-----------------+
| "INFO"         | 20              |
+----------------+-----------------+
| "DEBUG"        | 10              |
+----------------+-----------------+
| "NOTSET"       | 0               |
+----------------+-----------------+


处理器对象
==========

Handler 有以下属性和方法。注意不要直接实例化 "Handler" ；这个类用来派
生其他更有用的子类。但是，子类的 "__init__()" 方法需要调用
"Handler.__init__()" 。

class logging.Handler

   __init__(level=NOTSET)

      初始化 "Handler" 实例时，需要设置它的级别，将过滤列表置为空，并
      且创建锁（通过 "createLock()" ）来序列化对 I/O 的访问。

   createLock()

      初始化一个线程锁，用来序列化对底层的 I/O 功能的访问，底层的 I/O
      功能可能不是线程安全的。

   acquire()

      使用 "createLock()" 获取线程锁。

   release()

      使用 "acquire()" 来释放线程锁。

   setLevel(level)

      给处理器设置阈值为 *level* 。日志级别小于 *level* 将被忽略。创建
      处理器时，日志级别被设置为 "NOTSET" （所有的消息都会被处理）。

      参见 日志级别 级别列表。

      在 3.2 版更改: *level* 形参现在接受像 'INFO' 这样的字符串形式的
      级别表达方式，也可以使用像 "INFO" 这样的整数常量。

   setFormatter(fmt)

      将此处理器的 "Formatter" 设置为 *fmt*。

   addFilter(filter)

      将指定的过滤器 *filter* 添加到此处理器。

   removeFilter(filter)

      从此处理器中删除指定的过滤器 *filter* 。

   filter(record)

      将此处理器的过滤器应用于记录，在要处理记录时返回 "True" 。依次查
      询过滤器，直到其中一个返回假值为止。如果它们都不返回假值，则将发
      出记录。如果返回一个假值，则处理器将不会发出记录。

   flush()

      确保所有日志记录从缓存输出。此版本不执行任何操作，并且应由子类实
      现。

   close()

      整理处理器使用的所有资源。此版本不输出，但从内部处理器列表中删除
      处理器，内部处理器在 "shutdown()" 被调用时关闭 。子类应确保从重
      写的 "close()" 方法中调用此方法。

   handle(record)

      经已添加到处理器的过滤器过滤后，有条件地发出指定的日志记录。用获
      取/释放 I/O 线程锁包装记录的实际发出行为。

   handleError(record)

      调用 "emit()" 期间遇到异常时，应从处理器中调用此方法。如果模块级
      属性 "raiseExceptions" 是 "False"，则异常将被静默忽略。这是大多
      数情况下日志系统需要的 —— 大多数用户不会关心日志系统中的错误，他
      们对应用程序错误更感兴趣。但是，你可以根据需要将其替换为自定义处
      理器。指定的记录是发生异常时正在处理的记录。（"raiseExceptions"
      的默认值是 "True"，因为这在开发过程中是比较有用的）。

   format(record)

      如果设置了格式器则用其对记录进行格式化。否则，使用模块的默认格式
      器。

   emit(record)

      执行实际记录给定日志记录所需的操作。这个版本应由子类实现，因此这
      里直接引发 "NotImplementedError" 异常。

有关作为标准随附的处理程序列表，请参见 "logging.handlers"。


格式器对象
==========

"Formatter" 对象拥有以下的属性和方法。一般情况下，它们负责将
"LogRecord" 转换为可由人或外部系统解释的字符串。基础的 "Formatter" 允
许指定格式字符串。如果未提供任何值，则使用默认值 "'%(message)s'" ，它
仅将消息包括在日志记录调用中。要在格式化输出中包含其他信息（如时间戳）
，请阅读下文。

格式器可以使用格式化字符串来初始化，该字符串利用 "LogRecord" 的属性 ——
例如上述默认值，用户的消息和参数预先格式化为 "LogRecord" 的 *message*
属性后被使用。此格式字符串包含标准的 Python %-s 样式映射键。有关字符串
格式的更多信息，请参见 printf 风格的字符串格式化。

The useful mapping keys in a "LogRecord" are given in the section on
LogRecord 属性.

class logging.Formatter(fmt=None, datefmt=None, style='%')

   返回 "Formatter" 类的新实例。实例将使用整个消息的格式字符串以及消息
   的日期/时间部分的格式字符串进行初始化。如果未指定 *fmt* ，则使用
   "'%(message)s'"。如果未指定 *datefmt*，则使用  "formatTime()" 文档
   中描述的格式。

   The *style* parameter can be one of '%', '{' or '$' and determines
   how the format string will be merged with its data: using one of
   %-formatting, "str.format()" or "string.Template". See Using
   particular formatting styles throughout your application for more
   information on using {- and $-formatting for log messages.

   在 3.2 版更改: The *style* parameter was added.

   在 3.8 版更改: The *validate* parameter was added. Incorrect or
   mismatched style and fmt will raise a "ValueError". For example:
   "logging.Formatter('%(asctime)s - %(message)s', style='{')".

   format(record)

      The record's attribute dictionary is used as the operand to a
      string formatting operation. Returns the resulting string.
      Before formatting the dictionary, a couple of preparatory steps
      are carried out. The *message* attribute of the record is
      computed using *msg* % *args*. If the formatting string contains
      "'(asctime)'", "formatTime()" is called to format the event
      time. If there is exception information, it is formatted using
      "formatException()" and appended to the message. Note that the
      formatted exception information is cached in attribute
      *exc_text*. This is useful because the exception information can
      be pickled and sent across the wire, but you should be careful
      if you have more than one "Formatter" subclass which customizes
      the formatting of exception information. In this case, you will
      have to clear the cached value after a formatter has done its
      formatting, so that the next formatter to handle the event
      doesn't use the cached value but recalculates it afresh.

      If stack information is available, it's appended after the
      exception information, using "formatStack()" to transform it if
      necessary.

   formatTime(record, datefmt=None)

      This method should be called from "format()" by a formatter
      which wants to make use of a formatted time. This method can be
      overridden in formatters to provide for any specific
      requirement, but the basic behavior is as follows: if *datefmt*
      (a string) is specified, it is used with "time.strftime()" to
      format the creation time of the record. Otherwise, the format
      '%Y-%m-%d %H:%M:%S,uuu' is used, where the uuu part is a
      millisecond value and the other letters are as per the
      "time.strftime()" documentation.  An example time in this format
      is "2003-01-23 00:29:50,411".  The resulting string is returned.

      This function uses a user-configurable function to convert the
      creation time to a tuple. By default, "time.localtime()" is
      used; to change this for a particular formatter instance, set
      the "converter" attribute to a function with the same signature
      as "time.localtime()" or "time.gmtime()". To change it for all
      formatters, for example if you want all logging times to be
      shown in GMT, set the "converter" attribute in the "Formatter"
      class.

      在 3.3 版更改: Previously, the default format was hard-coded as
      in this example: "2010-09-06 22:38:15,292" where the part before
      the comma is handled by a strptime format string ("'%Y-%m-%d
      %H:%M:%S'"), and the part after the comma is a millisecond
      value. Because strptime does not have a format placeholder for
      milliseconds, the millisecond value is appended using another
      format string, "'%s,%03d'" --- and both of these format strings
      have been hardcoded into this method. With the change, these
      strings are defined as class-level attributes which can be
      overridden at the instance level when desired. The names of the
      attributes are "default_time_format" (for the strptime format
      string) and "default_msec_format" (for appending the millisecond
      value).

   formatException(exc_info)

      Formats the specified exception information (a standard
      exception tuple as returned by "sys.exc_info()") as a string.
      This default implementation just uses
      "traceback.print_exception()". The resulting string is returned.

   formatStack(stack_info)

      Formats the specified stack information (a string as returned by
      "traceback.print_stack()", but with the last newline removed) as
      a string. This default implementation just returns the input
      value.


Filter Objects
==============

"Filters" can be used by "Handlers" and "Loggers" for more
sophisticated filtering than is provided by levels. The base filter
class only allows events which are below a certain point in the logger
hierarchy. For example, a filter initialized with 'A.B' will allow
events logged by loggers 'A.B', 'A.B.C', 'A.B.C.D', 'A.B.D' etc. but
not 'A.BB', 'B.A.B' etc. If initialized with the empty string, all
events are passed.

class logging.Filter(name='')

   Returns an instance of the "Filter" class. If *name* is specified,
   it names a logger which, together with its children, will have its
   events allowed through the filter. If *name* is the empty string,
   allows every event.

   filter(record)

      Is the specified record to be logged? Returns zero for no,
      nonzero for yes. If deemed appropriate, the record may be
      modified in-place by this method.

Note that filters attached to handlers are consulted before an event
is emitted by the handler, whereas filters attached to loggers are
consulted whenever an event is logged (using "debug()", "info()",
etc.), before sending an event to handlers. This means that events
which have been generated by descendant loggers will not be filtered
by a logger's filter setting, unless the filter has also been applied
to those descendant loggers.

You don't actually need to subclass "Filter": you can pass any
instance which has a "filter" method with the same semantics.

在 3.2 版更改: You don't need to create specialized "Filter" classes,
or use other classes with a "filter" method: you can use a function
(or other callable) as a filter. The filtering logic will check to see
if the filter object has a "filter" attribute: if it does, it's
assumed to be a "Filter" and its "filter()" method is called.
Otherwise, it's assumed to be a callable and called with the record as
the single parameter. The returned value should conform to that
returned by "filter()".

Although filters are used primarily to filter records based on more
sophisticated criteria than levels, they get to see every record which
is processed by the handler or logger they're attached to: this can be
useful if you want to do things like counting how many records were
processed by a particular logger or handler, or adding, changing or
removing attributes in the "LogRecord" being processed. Obviously
changing the LogRecord needs to be done with some care, but it does
allow the injection of contextual information into logs (see 使用过滤
器传递上下文信息).


LogRecord Objects
=================

"LogRecord" instances are created automatically by the "Logger" every
time something is logged, and can be created manually via
"makeLogRecord()" (for example, from a pickled event received over the
wire).

class logging.LogRecord(name, level, pathname, lineno, msg, args, exc_info, func=None, sinfo=None)

   Contains all the information pertinent to the event being logged.

   The primary information is passed in "msg" and "args", which are
   combined using "msg % args" to create the "message" field of the
   record.

   参数:
      * **name** -- The name of the logger used to log the event
        represented by this LogRecord. Note that this name will always
        have this value, even though it may be emitted by a handler
        attached to a different (ancestor) logger.

      * **level** -- The numeric level of the logging event (one of
        DEBUG, INFO etc.) Note that this is converted to *two*
        attributes of the LogRecord: "levelno" for the numeric value
        and "levelname" for the corresponding level name.

      * **pathname** -- The full pathname of the source file where the
        logging call was made.

      * **lineno** -- The line number in the source file where the
        logging call was made.

      * **msg** -- The event description message, possibly a format
        string with placeholders for variable data.

      * **args** -- Variable data to merge into the *msg* argument to
        obtain the event description.

      * **exc_info** -- An exception tuple with the current exception
        information, or "None" if no exception information is
        available.

      * **func** -- The name of the function or method from which the
        logging call was invoked.

      * **sinfo** -- A text string representing stack information from
        the base of the stack in the current thread, up to the logging
        call.

   getMessage()

      Returns the message for this "LogRecord" instance after merging
      any user-supplied arguments with the message. If the user-
      supplied message argument to the logging call is not a string,
      "str()" is called on it to convert it to a string. This allows
      use of user-defined classes as messages, whose "__str__" method
      can return the actual format string to be used.

   在 3.2 版更改: The creation of a "LogRecord" has been made more
   configurable by providing a factory which is used to create the
   record. The factory can be set using "getLogRecordFactory()" and
   "setLogRecordFactory()" (see this for the factory's signature).

   This functionality can be used to inject your own values into a
   "LogRecord" at creation time. You can use the following pattern:

      old_factory = logging.getLogRecordFactory()

      def record_factory(*args, **kwargs):
          record = old_factory(*args, **kwargs)
          record.custom_attribute = 0xdecafbad
          return record

      logging.setLogRecordFactory(record_factory)

   With this pattern, multiple factories could be chained, and as long
   as they don't overwrite each other's attributes or unintentionally
   overwrite the standard attributes listed above, there should be no
   surprises.


LogRecord 属性
==============

The LogRecord has a number of attributes, most of which are derived
from the parameters to the constructor. (Note that the names do not
always correspond exactly between the LogRecord constructor parameters
and the LogRecord attributes.) These attributes can be used to merge
data from the record into the format string. The following table lists
(in alphabetical order) the attribute names, their meanings and the
corresponding placeholder in a %-style format string.

If you are using {}-formatting ("str.format()"), you can use
"{attrname}" as the placeholder in the format string. If you are using
$-formatting ("string.Template"), use the form "${attrname}". In both
cases, of course, replace "attrname" with the actual attribute name
you want to use.

In the case of {}-formatting, you can specify formatting flags by
placing them after the attribute name, separated from it with a colon.
For example: a placeholder of "{msecs:03d}" would format a millisecond
value of "4" as "004". Refer to the "str.format()" documentation for
full details on the options available to you.

+------------------+---------------------------+-------------------------------------------------+
| 属性名称         | 格式                      | 描述                                            |
|==================|===========================|=================================================|
| args             | 不需要格式化。            | The tuple of arguments merged into "msg" to     |
|                  |                           | produce "message", or a dict whose values are   |
|                  |                           | used for the merge (when there is only one      |
|                  |                           | argument, and it is a dictionary).              |
+------------------+---------------------------+-------------------------------------------------+
| asctime          | "%(asctime)s"             | Human-readable time when the "LogRecord" was    |
|                  |                           | created.  By default this is of the form        |
|                  |                           | '2003-07-08 16:49:45,896' (the numbers after    |
|                  |                           | the comma are millisecond portion of the time). |
+------------------+---------------------------+-------------------------------------------------+
| created          | "%(created)f"             | Time when the "LogRecord" was created (as       |
|                  |                           | returned by "time.time()").                     |
+------------------+---------------------------+-------------------------------------------------+
| exc_info         | 不需要格式化。            | Exception tuple (à la "sys.exc_info") or, if no |
|                  |                           | exception has occurred, "None".                 |
+------------------+---------------------------+-------------------------------------------------+
| filename         | "%(filename)s"            | Filename portion of "pathname".                 |
+------------------+---------------------------+-------------------------------------------------+
| funcName         | "%(funcName)s"            | Name of function containing the logging call.   |
+------------------+---------------------------+-------------------------------------------------+
| levelname        | "%(levelname)s"           | Text logging level for the message ("'DEBUG'",  |
|                  |                           | "'INFO'", "'WARNING'", "'ERROR'",               |
|                  |                           | "'CRITICAL'").                                  |
+------------------+---------------------------+-------------------------------------------------+
| levelno          | "%(levelno)s"             | Numeric logging level for the message ("DEBUG", |
|                  |                           | "INFO", "WARNING", "ERROR", "CRITICAL").        |
+------------------+---------------------------+-------------------------------------------------+
| lineno           | "%(lineno)d"              | Source line number where the logging call was   |
|                  |                           | issued (if available).                          |
+------------------+---------------------------+-------------------------------------------------+
| message          | "%(message)s"             | The logged message, computed as "msg % args".   |
|                  |                           | This is set when "Formatter.format()" is        |
|                  |                           | invoked.                                        |
+------------------+---------------------------+-------------------------------------------------+
| module           | "%(module)s"              | 模块 ("filename" 的名称部分)。                  |
+------------------+---------------------------+-------------------------------------------------+
| msecs            | "%(msecs)d"               | Millisecond portion of the time when the        |
|                  |                           | "LogRecord" was created.                        |
+------------------+---------------------------+-------------------------------------------------+
| msg              | 不需要格式化。            | The format string passed in the original        |
|                  |                           | logging call. Merged with "args" to produce     |
|                  |                           | "message", or an arbitrary object (see 使用任意 |
|                  |                           | 对象 作为消息).                                 |
+------------------+---------------------------+-------------------------------------------------+
| name             | "%(name)s"                | Name of the logger used to log the call.        |
+------------------+---------------------------+-------------------------------------------------+
| pathname         | "%(pathname)s"            | Full pathname of the source file where the      |
|                  |                           | logging call was issued (if available).         |
+------------------+---------------------------+-------------------------------------------------+
| process          | "%(process)d"             | 进程ID（如果可用）                              |
+------------------+---------------------------+-------------------------------------------------+
| processName      | "%(processName)s"         | 进程名（如果可用）                              |
+------------------+---------------------------+-------------------------------------------------+
| relativeCreated  | "%(relativeCreated)d"     | Time in milliseconds when the LogRecord was     |
|                  |                           | created, relative to the time the logging       |
|                  |                           | module was loaded.                              |
+------------------+---------------------------+-------------------------------------------------+
| stack_info       | 不需要格式化。            | Stack frame information (where available) from  |
|                  |                           | the bottom of the stack in the current thread,  |
|                  |                           | up to and including the stack frame of the      |
|                  |                           | logging call which resulted in the creation of  |
|                  |                           | this record.                                    |
+------------------+---------------------------+-------------------------------------------------+
| thread           | "%(thread)d"              | 线程ID（如果可用）                              |
+------------------+---------------------------+-------------------------------------------------+
| threadName       | "%(threadName)s"          | 线程名（如果可用）                              |
+------------------+---------------------------+-------------------------------------------------+

在 3.1 版更改: 添加了 *processName*


LoggerAdapter 对象
==================

"LoggerAdapter" instances are used to conveniently pass contextual
information into logging calls. For a usage example, see the section
on adding contextual information to your logging output.

class logging.LoggerAdapter(logger, extra)

   Returns an instance of "LoggerAdapter" initialized with an
   underlying "Logger" instance and a dict-like object.

   process(msg, kwargs)

      Modifies the message and/or keyword arguments passed to a
      logging call in order to insert contextual information. This
      implementation takes the object passed as *extra* to the
      constructor and adds it to *kwargs* using key 'extra'. The
      return value is a (*msg*, *kwargs*) tuple which has the
      (possibly modified) versions of the arguments passed in.

In addition to the above, "LoggerAdapter" supports the following
methods of "Logger": "debug()", "info()", "warning()", "error()",
"exception()", "critical()", "log()", "isEnabledFor()",
"getEffectiveLevel()", "setLevel()" and "hasHandlers()". These methods
have the same signatures as their counterparts in "Logger", so you can
use the two types of instances interchangeably.

在 3.2 版更改: The "isEnabledFor()", "getEffectiveLevel()",
"setLevel()" and "hasHandlers()" methods were added to
"LoggerAdapter".  These methods delegate to the underlying logger.


线程安全
========

The logging module is intended to be thread-safe without any special
work needing to be done by its clients. It achieves this though using
threading locks; there is one lock to serialize access to the module's
shared data, and each handler also creates a lock to serialize access
to its underlying I/O.

If you are implementing asynchronous signal handlers using the
"signal" module, you may not be able to use logging from within such
handlers. This is because lock implementations in the "threading"
module are not always re-entrant, and so cannot be invoked from such
signal handlers.


模块级别函数
============

In addition to the classes described above, there are a number of
module-level functions.

logging.getLogger(name=None)

   Return a logger with the specified name or, if name is "None",
   return a logger which is the root logger of the hierarchy. If
   specified, the name is typically a dot-separated hierarchical name
   like *'a'*, *'a.b'* or *'a.b.c.d'*. Choice of these names is
   entirely up to the developer who is using logging.

   All calls to this function with a given name return the same logger
   instance. This means that logger instances never need to be passed
   between different parts of an application.

logging.getLoggerClass()

   Return either the standard "Logger" class, or the last class passed
   to "setLoggerClass()". This function may be called from within a
   new class definition, to ensure that installing a customized
   "Logger" class will not undo customizations already applied by
   other code. For example:

      class MyLogger(logging.getLoggerClass()):
          # ... override behaviour here

logging.getLogRecordFactory()

   Return a callable which is used to create a "LogRecord".

   3.2 新版功能: This function has been provided, along with
   "setLogRecordFactory()", to allow developers more control over how
   the "LogRecord" representing a logging event is constructed.

   See "setLogRecordFactory()" for more information about the how the
   factory is called.

logging.debug(msg, *args, **kwargs)

   Logs a message with level "DEBUG" on the root logger. The *msg* is
   the message format string, and the *args* are the arguments which
   are merged into *msg* using the string formatting operator. (Note
   that this means that you can use keywords in the format string,
   together with a single dictionary argument.)

   There are three keyword arguments in *kwargs* which are inspected:
   *exc_info* which, if it does not evaluate as false, causes
   exception information to be added to the logging message. If an
   exception tuple (in the format returned by "sys.exc_info()") or an
   exception instance is provided, it is used; otherwise,
   "sys.exc_info()" is called to get the exception information.

   第二个可选关键字参数是 *stack_info*，默认为 "False"。如果为  True，
   则将堆栈信息添加到日志消息中，包括实际的日志调用。请注意，这与通过
   指定 *exc_info* 显示的堆栈信息不同：前者是从堆栈底部到当前线程中的
   日志记录调用的堆栈帧，而后者是在搜索异常处理程序时，跟踪异常而打开
   的堆栈帧的信息。

   您可以独立于 *exc_info* 来指定 *stack_info*，例如，即使在未引发任何
   异常的情况下，也可以显示如何到达代码中的特定点。堆栈帧在标题行之后
   打印：

      Stack (most recent call last):

   这模仿了显示异常帧时所使用的 "Traceback (most recent call last):"
   。

   The third optional keyword argument is *extra* which can be used to
   pass a dictionary which is used to populate the __dict__ of the
   LogRecord created for the logging event with user-defined
   attributes. These custom attributes can then be used as you like.
   For example, they could be incorporated into logged messages. For
   example:

      FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
      logging.basicConfig(format=FORMAT)
      d = {'clientip': '192.168.0.1', 'user': 'fbloggs'}
      logging.warning('Protocol problem: %s', 'connection reset', extra=d)

   would print something like:

      2006-02-08 22:20:02,165 192.168.0.1 fbloggs  Protocol problem: connection reset

   The keys in the dictionary passed in *extra* should not clash with
   the keys used by the logging system. (See the "Formatter"
   documentation for more information on which keys are used by the
   logging system.)

   If you choose to use these attributes in logged messages, you need
   to exercise some care. In the above example, for instance, the
   "Formatter" has been set up with a format string which expects
   'clientip' and 'user' in the attribute dictionary of the LogRecord.
   If these are missing, the message will not be logged because a
   string formatting exception will occur. So in this case, you always
   need to pass the *extra* dictionary with these keys.

   尽管这可能很烦人，但此功能旨在用于特殊情况，例如在多个上下文中执行
   相同代码的多线程服务器，并且出现的有趣条件取决于此上下文（例如在上
   面的示例中就是远程客户端IP地址和已验证用户名）。在这种情况下，很可
   能将专门的 "Formatter" 与特定的 "Handler" 一起使用。

   在 3.2 版更改: 增加了 *stack_info* 参数。

logging.info(msg, *args, **kwargs)

   Logs a message with level "INFO" on the root logger. The arguments
   are interpreted as for "debug()".

logging.warning(msg, *args, **kwargs)

   Logs a message with level "WARNING" on the root logger. The
   arguments are interpreted as for "debug()".

   注解:

     There is an obsolete function "warn" which is functionally
     identical to "warning". As "warn" is deprecated, please do not
     use it - use "warning" instead.

logging.error(msg, *args, **kwargs)

   Logs a message with level "ERROR" on the root logger. The arguments
   are interpreted as for "debug()".

logging.critical(msg, *args, **kwargs)

   Logs a message with level "CRITICAL" on the root logger. The
   arguments are interpreted as for "debug()".

logging.exception(msg, *args, **kwargs)

   Logs a message with level "ERROR" on the root logger. The arguments
   are interpreted as for "debug()". Exception info is added to the
   logging message. This function should only be called from an
   exception handler.

logging.log(level, msg, *args, **kwargs)

   Logs a message with level *level* on the root logger. The other
   arguments are interpreted as for "debug()".

   注解:

     The above module-level convenience functions, which delegate to
     the root logger, call "basicConfig()" to ensure that at least one
     handler is available. Because of this, they should *not* be used
     in threads, in versions of Python earlier than 2.7.1 and 3.2,
     unless at least one handler has been added to the root logger
     *before* the threads are started. In earlier versions of Python,
     due to a thread safety shortcoming in "basicConfig()", this can
     (under rare circumstances) lead to handlers being added multiple
     times to the root logger, which can in turn lead to multiple
     messages for the same event.

logging.disable(level=CRITICAL)

   Provides an overriding level *level* for all loggers which takes
   precedence over the logger's own level. When the need arises to
   temporarily throttle logging output down across the whole
   application, this function can be useful. Its effect is to disable
   all logging calls of severity *level* and below, so that if you
   call it with a value of INFO, then all INFO and DEBUG events would
   be discarded, whereas those of severity WARNING and above would be
   processed according to the logger's effective level. If
   "logging.disable(logging.NOTSET)" is called, it effectively removes
   this overriding level, so that logging output again depends on the
   effective levels of individual loggers.

   Note that if you have defined any custom logging level higher than
   "CRITICAL" (this is not recommended), you won't be able to rely on
   the default value for the *level* parameter, but will have to
   explicitly supply a suitable value.

   在 3.7 版更改: The *level* parameter was defaulted to level
   "CRITICAL". See Issue #28524 for more information about this
   change.

logging.addLevelName(level, levelName)

   Associates level *level* with text *levelName* in an internal
   dictionary, which is used to map numeric levels to a textual
   representation, for example when a "Formatter" formats a message.
   This function can also be used to define your own levels. The only
   constraints are that all levels used must be registered using this
   function, levels should be positive integers and they should
   increase in increasing order of severity.

   注解:

     If you are thinking of defining your own levels, please see the
     section on 自定义级别.

logging.getLevelName(level)

   Returns the textual representation of logging level *level*. If the
   level is one of the predefined levels "CRITICAL", "ERROR",
   "WARNING", "INFO" or "DEBUG" then you get the corresponding string.
   If you have associated levels with names using "addLevelName()"
   then the name you have associated with *level* is returned. If a
   numeric value corresponding to one of the defined levels is passed
   in, the corresponding string representation is returned. Otherwise,
   the string 'Level %s' % level is returned.

   注解:

     Levels are internally integers (as they need to be compared in
     the logging logic). This function is used to convert between an
     integer level and the level name displayed in the formatted log
     output by means of the "%(levelname)s" format specifier (see
     LogRecord 属性).

   在 3.4 版更改: In Python versions earlier than 3.4, this function
   could also be passed a text level, and would return the
   corresponding numeric value of the level. This undocumented
   behaviour was considered a mistake, and was removed in Python 3.4,
   but reinstated in 3.4.2 due to retain backward compatibility.

logging.makeLogRecord(attrdict)

   Creates and returns a new "LogRecord" instance whose attributes are
   defined by *attrdict*. This function is useful for taking a pickled
   "LogRecord" attribute dictionary, sent over a socket, and
   reconstituting it as a "LogRecord" instance at the receiving end.

logging.basicConfig(**kwargs)

   Does basic configuration for the logging system by creating a
   "StreamHandler" with a default "Formatter" and adding it to the
   root logger. The functions "debug()", "info()", "warning()",
   "error()" and "critical()" will call "basicConfig()" automatically
   if no handlers are defined for the root logger.

   This function does nothing if the root logger already has handlers
   configured, unless the keyword argument *force* is set to "True".

   注解:

     This function should be called from the main thread before other
     threads are started. In versions of Python prior to 2.7.1 and
     3.2, if this function is called from multiple threads, it is
     possible (in rare circumstances) that a handler will be added to
     the root logger more than once, leading to unexpected results
     such as messages being duplicated in the log.

   支持以下关键字参数。

   +----------------+-----------------------------------------------+
   | 格式           | 描述                                          |
   |================|===============================================|
   | *filename*     | 使用指定的文件名而不是 StreamHandler 创建     |
   |                | FileHandler。                                 |
   +----------------+-----------------------------------------------+
   | *filemode*     | If *filename* is specified, open the file in  |
   |                | this mode. Defaults to "'a'".                 |
   +----------------+-----------------------------------------------+
   | *format*       | 处理器使用的指定格式字符串。                  |
   +----------------+-----------------------------------------------+
   | *datefmt*      | Use the specified date/time format, as        |
   |                | accepted by "time.strftime()".                |
   +----------------+-----------------------------------------------+
   | *style*        | If *format* is specified, use this style for  |
   |                | the format string. One of "'%'", "'{'" or     |
   |                | "'$'" for printf-style, "str.format()" or     |
   |                | "string.Template" respectively. Defaults to   |
   |                | "'%'".                                        |
   +----------------+-----------------------------------------------+
   | *level*        | Set the root logger level to the specified    |
   |                | level.                                        |
   +----------------+-----------------------------------------------+
   | *stream*       | Use the specified stream to initialize the    |
   |                | StreamHandler. Note that this argument is     |
   |                | incompatible with *filename* - if both are    |
   |                | present, a "ValueError" is raised.            |
   +----------------+-----------------------------------------------+
   | *handlers*     | If specified, this should be an iterable of   |
   |                | already created handlers to add to the root   |
   |                | logger. Any handlers which don't already have |
   |                | a formatter set will be assigned the default  |
   |                | formatter created in this function. Note that |
   |                | this argument is incompatible with *filename* |
   |                | or *stream* - if both are present, a          |
   |                | "ValueError" is raised.                       |
   +----------------+-----------------------------------------------+
   | *force*        | 如果将此关键字参数指定为 true，则在执行其他参 |
   |                | 数指定的配置之前，将移 除并关闭附加到根记录器 |
   |                | 的所有现有处理器。                            |
   +----------------+-----------------------------------------------+

   在 3.2 版更改: 增加了 *style* 参数。

   在 3.3 版更改: The *handlers* argument was added. Additional checks
   were added to catch situations where incompatible arguments are
   specified (e.g. *handlers* together with *stream* or *filename*, or
   *stream* together with *filename*).

   在 3.8 版更改: 增加了 *force* 参数。

logging.shutdown()

   Informs the logging system to perform an orderly shutdown by
   flushing and closing all handlers. This should be called at
   application exit and no further use of the logging system should be
   made after this call.

   When the logging module is imported, it registers this function as
   an exit handler (see "atexit"), so normally there's no need to do
   that manually.

logging.setLoggerClass(klass)

   Tells the logging system to use the class *klass* when
   instantiating a logger. The class should define "__init__()" such
   that only a name argument is required, and the "__init__()" should
   call "Logger.__init__()". This function is typically called before
   any loggers are instantiated by applications which need to use
   custom logger behavior. After this call, as at any other time, do
   not instantiate loggers directly using the subclass: continue to
   use the "logging.getLogger()" API to get your loggers.

logging.setLogRecordFactory(factory)

   Set a callable which is used to create a "LogRecord".

   参数:
      **factory** -- The factory callable to be used to instantiate a
      log record.

   3.2 新版功能: This function has been provided, along with
   "getLogRecordFactory()", to allow developers more control over how
   the "LogRecord" representing a logging event is constructed.

   The factory has the following signature:

   "factory(name, level, fn, lno, msg, args, exc_info, func=None,
   sinfo=None, **kwargs)"

      name:
         日志记录器名称

      level:
         日志记录级别（数字）。

      fn:
         进行日志记录调用的文件的完整路径名。

      lno:
         记录调用所在文件中的行号。

      msg:
         日志消息。

      args:
         日志记录消息的参数。

      exc_info:
         异常元组，或 "None" 。

      func:
         调用日志记录调用的函数或方法的名称。

      sinfo:
         A stack traceback such as is provided by
         "traceback.print_stack()", showing the call hierarchy.

      kwargs:
         其他关键字参数。


模块级属性
==========

logging.lastResort

   A "handler of last resort" is available through this attribute.
   This is a "StreamHandler" writing to "sys.stderr" with a level of
   "WARNING", and is used to handle logging events in the absence of
   any logging configuration. The end result is to just print the
   message to "sys.stderr". This replaces the earlier error message
   saying that "no handlers could be found for logger XYZ". If you
   need the earlier behaviour for some reason, "lastResort" can be set
   to "None".

   3.2 新版功能.


与警告模块集成
==============

"captureWarnings()" 函数可用来将 "logging" 和 "warnings" 模块集成。

logging.captureWarnings(capture)

   此函数用于打开和关闭日志系统对警告的捕获。

   如果 *capture* 是 "True"，则 "warnings" 模块发出的警告将重定向到日
   志记录系统。具体来说，将使用 "warnings.formatwarning()" 格式化警告
   信息，并将结果字符串使用 "WARNING" 等级记录到名为 "'py.warnings'"
   的记录器中。

   如果 *capture* 是 "False"，则将停止将警告重定向到日志记录系统，并且
   将警告重定向到其原始目标（即在  "captureWarnings(True)"  调用之前的
   有效目标）。

参见:

  "logging.config" 模块
     日志记录模块的配置 API 。

  "logging.handlers" 模块
     日志记录模块附带的有用处理器。

  **PEP 282** - Logging 系统
     该提案描述了Python标准库中包含的这个特性。

  Original Python logging package
     这是该 "logging" 包的原始来源。该站点提供的软件包版本适用于
     Python 1.5.2、2.1.x 和 2.2.x，它们不被 "logging" 包含在标准库中。
