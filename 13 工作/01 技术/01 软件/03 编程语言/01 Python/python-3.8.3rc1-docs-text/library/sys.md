"sys" --- 系统相关的参数和函数
******************************

======================================================================

该模块提供了一些变量和函数。这些变量可能被解释器使用，也可能由解释器提
供。这些函数会影响解释器。本模块总是可用的。

sys.abiflags

   在POSIX系统上，以标准的 "configure" 脚本构建的 Python 中，这个变量
   会包含 **PEP 3149** 中定义的ABI标签。

   在 3.8 版更改: 默认的 flags 变为了空字符串（用于 pymalloc 的 "m" 旗
   标已经移除）

   3.2 新版功能.

sys.addaudithook(hook)

   将可调用的 *hook* 附加到当前解释器的活动的审核钩子列表中。

   通过 "sys.audit()" 函数引发审计事件时，将按照加入钩子的先后顺序调用
   每个钩子，调用时将带有事件名称和参数元组。首先调用由
   "PySys_AddAuditHook()" 添加的静态钩子，然后调用添加到当前解释器中的
   钩子。

   引发一个 审计事件 "sys.addaudithook"，没有附带参数。

   3.8 新版功能.

   在 3.8.1 版更改: 派生自 "Exception" （而非 "RuntimeError" ）的异常
   不会被抑制。

   **CPython implementation detail:** 启用跟踪时（参阅 "settrace()" ）
   ，仅当可调用对象（钩子）的 "__cantrace__" 成员设置为 true 时，才会
   跟踪该钩子。否则，跟踪功能将跳过该钩子。

sys.argv

   一个列表，其中包含了被传递给 Python 脚本的命令行参数。 "argv[0]" 为
   脚本的名称（是否是完整的路径名取决于操作系统）。如果是通过 Python
   解释器的命令行参数 "-c" 来执行的， "argv[0]" 会被设置成字符串
   "'-c'" 。如果没有脚本名被传递给 Python 解释器， "argv[0]" 为空字符
   串。

   为了遍历标准输入，或者通过命令行传递的文件列表，参照 "fileinput" 模
   块

   注解:

     在 Unix 上，系统传递的命令行参数是字节类型的。Python 使用文件系统
     编码和 "surrogateescape" 错误处理程序对它们进行解码。当需要原始字
     节时，可以通过 "[os.fsencode(arg) for arg in sys.argv]" 来获取。

sys.audit(event, *args)

   利用某些活动的钩子引发审计事件。事件名称是一个字符串，用于标记事件
   及其关联的架构（即参数的数量和类型）。给定事件的架构应视作公开且稳
   定的 API，并且不应在版本之间进行修改。

   任何钩子抛出第一个异常时，此函数也将抛出该异常。通常不应处理这些错
   误，而应尽快终止该进程。

   钩子程序由 "sys.addaudithook()" 或 "PySys_AddAuditHook()" 函数添加
   。

   与本函数相等效的原生函数是 "PySys_Audit()"，应尽量使用原生函数。

   参阅 审计事件表 以获取 CPython 定义的所有审计事件。

   3.8 新版功能.

sys.base_exec_prefix

   在 "site.py" 运行之前， Python 启动的时候被设置为跟 "exec_prefix"
   同样的值。如果不是运行在 虚拟环境 中，两个值会保持相同；如果
   "site.py" 发现处于一个虚拟环境中， "prefix" 和 "exec_prefix" 将会指
   向虚拟环境。然而 "base_prefix" 和 "base_exec_prefix" 将仍然会指向基
   础的 Python 环境（用来创建虚拟环境的 Python 环境）

   3.3 新版功能.

sys.base_prefix

   在 "site.py" 运行之前， Python 启动的时候被设置为跟 "prefix" 同样的
   值。如果不是运行在 虚拟环境 中, 两个值会保持相同；如果 "site.py" 发
   现处于一个虚拟环境中， "prefix" 和 "exec_prefix" 将会指向虚拟环境。
   然而 "base_prefix" 和 "base_exec_prefix" 将仍然会指向基础的 Python
   环境（用来创建虚拟环境的 Python 环境）

   3.3 新版功能.

sys.byteorder

   本地字节顺序的指示符。在大端序（最高有效位优先）操作系统上值为
   "'big'" ，在小端序（最低有效位优先）操作系统上为 "'little'" 。

sys.builtin_module_names

   一个元素为字符串的元组。包含了所有的被编译进 Python 解释器的模块。
   （这个信息无法通过其他的办法获取， "modules.keys()" 只包括被导入过
   的模块。）

sys.call_tracing(func, args)

   在启用跟踪时调用 "func(*args)"  来保存跟踪状态，然后恢复跟踪状态。
   这将从检查点的调试器调用，以便递归地调试其他的一些代码。

sys.copyright

   一个字符串，包含了 Python 解释器有关的版权信息

sys._clear_type_cache()

   清除内部的类型缓存。类型缓存是为了加速查找方法和属性的。在调试引用
   泄漏的时候调用这个函数 *只会* 清除不必要的引用。

   这个函数应该只在内部为了一些特定的目的使用。

sys._current_frames()

   返回一个字典，存放着每个线程的标识符与（调用本函数时）该线程栈顶的
   帧（当前活动的帧）之间的映射。注意 "traceback" 模块中的函数可以在给
   定某一帧的情况下构建调用堆栈。

   这对于调试死锁最有用：本函数不需要死锁线程的配合，并且只要这些线程
   的调用栈保持死锁，它们就是冻结的。在调用本代码来检查栈顶的帧的那一
   刻，非死锁线程返回的帧可能与该线程当前活动的帧没有任何关系。

   这个函数应该只在内部为了一些特定的目的使用。

   引发一个 审计事件 "sys._current_frames"，没有附带参数。

sys.breakpointhook()

   本钩子函数由内建函数 "breakpoint()" 调用。默认情况下，它将进入
   "pdb" 调试器，但可以将其改为任何其他函数，以选择使用哪个调试器。

   该函数的特征取决于其调用的函数。例如，默认绑定（即
   "pdb.set_trace()" ）不要求提供参数，但可以将绑定换成要求提供附加参
   数（位置参数/关键字参数）的函数。内建函数 "breakpoint()" 直接将其
   "*args" 和 "**kws" 传入。"breakpointhooks()" 返回的所有内容都会从
   "breakpoint()" 返回。

   默认的实现首先会查询环境变量 "PYTHONBREAKPOINT"。如果将该变量设置为
   ""0""，则本函数立即返回，表示在断点处无操作。如果未设置该环境变量或
   将其设置为空字符串，则调用 "pdb.set_trace()"。否则，此变量应指定要
   运行的函数，指定函数时应使用 Python 的点导入命名法，如
   "package.subpackage.module.function"。这种情况下将导入
   "package.subpackage.module"，且导入的模块必须有一个名为
   "function()" 的可调用对象。该可调用对象会运行，"*args" 和 "**kws"
   会传入，且无论 "function()" 返回什么，"sys.breakpointhook()" 都将返
   回到內建函数 "breakpoint()"。

   Note that if anything goes wrong while importing the callable named
   by "PYTHONBREAKPOINT", a "RuntimeWarning" is reported and the
   breakpoint is ignored.

   Also note that if "sys.breakpointhook()" is overridden
   programmatically, "PYTHONBREAKPOINT" is *not* consulted.

   3.7 新版功能.

sys._debugmallocstats()

   Print low-level information to stderr about the state of CPython's
   memory allocator.

   If Python is configured --with-pydebug, it also performs some
   expensive internal consistency checks.

   3.3 新版功能.

   **CPython implementation detail:** This function is specific to
   CPython.  The exact output format is not defined here, and may
   change.

sys.dllhandle

   Integer specifying the handle of the Python DLL.

   可用性: Windows。

sys.displayhook(value)

   If *value* is not "None", this function prints "repr(value)" to
   "sys.stdout", and saves *value* in "builtins._". If "repr(value)"
   is not encodable to "sys.stdout.encoding" with "sys.stdout.errors"
   error handler (which is probably "'strict'"), encode it to
   "sys.stdout.encoding" with "'backslashreplace'" error handler.

   "sys.displayhook" is called on the result of evaluating an
   *expression* entered in an interactive Python session.  The display
   of these values can be customized by assigning another one-argument
   function to "sys.displayhook".

   伪代码:

      def displayhook(value):
          if value is None:
              return
          # Set '_' to None to avoid recursion
          builtins._ = None
          text = repr(value)
          try:
              sys.stdout.write(text)
          except UnicodeEncodeError:
              bytes = text.encode(sys.stdout.encoding, 'backslashreplace')
              if hasattr(sys.stdout, 'buffer'):
                  sys.stdout.buffer.write(bytes)
              else:
                  text = bytes.decode(sys.stdout.encoding, 'strict')
                  sys.stdout.write(text)
          sys.stdout.write("\n")
          builtins._ = value

   在 3.2 版更改: Use "'backslashreplace'" error handler on
   "UnicodeEncodeError".

sys.dont_write_bytecode

   If this is true, Python won't try to write ".pyc" files on the
   import of source modules.  This value is initially set to "True" or
   "False" depending on the "-B" command line option and the
   "PYTHONDONTWRITEBYTECODE" environment variable, but you can set it
   yourself to control bytecode file generation.

sys.pycache_prefix

   If this is set (not "None"), Python will write bytecode-cache
   ".pyc" files to (and read them from) a parallel directory tree
   rooted at this directory, rather than from "__pycache__"
   directories in the source code tree. Any "__pycache__" directories
   in the source code tree will be ignored and new *.pyc* files
   written within the pycache prefix. Thus if you use "compileall" as
   a pre-build step, you must ensure you run it with the same pycache
   prefix (if any) that you will use at runtime.

   A relative path is interpreted relative to the current working
   directory.

   This value is initially set based on the value of the "-X"
   "pycache_prefix=PATH" command-line option or the
   "PYTHONPYCACHEPREFIX" environment variable (command-line takes
   precedence). If neither are set, it is "None".

   3.8 新版功能.

sys.excepthook(type, value, traceback)

   This function prints out a given traceback and exception to
   "sys.stderr".

   When an exception is raised and uncaught, the interpreter calls
   "sys.excepthook" with three arguments, the exception class,
   exception instance, and a traceback object.  In an interactive
   session this happens just before control is returned to the prompt;
   in a Python program this happens just before the program exits.
   The handling of such top-level exceptions can be customized by
   assigning another three-argument function to "sys.excepthook".

   Raise an auditing event "sys.excepthook" with arguments "hook",
   "type", "value", "traceback" when an uncaught exception occurs. If
   no hook has been set, "hook" may be "None". If any hook raises an
   exception derived from "RuntimeError" the call to the hook will be
   suppressed. Otherwise, the audit hook exception will be reported as
   unraisable and "sys.excepthook" will be called.

   参见:

     The "sys.unraisablehook()" function handles unraisable exceptions
     and the "threading.excepthook()" function handles exception
     raised by "threading.Thread.run()".

sys.__breakpointhook__
sys.__displayhook__
sys.__excepthook__
sys.__unraisablehook__

   These objects contain the original values of "breakpointhook",
   "displayhook", "excepthook", and "unraisablehook" at the start of
   the program.  They are saved so that "breakpointhook",
   "displayhook" and "excepthook", "unraisablehook" can be restored in
   case they happen to get replaced with broken or alternative
   objects.

   3.7 新版功能: __breakpointhook__

   3.8 新版功能: __unraisablehook__

sys.exc_info()

   This function returns a tuple of three values that give information
   about the exception that is currently being handled.  The
   information returned is specific both to the current thread and to
   the current stack frame.  If the current stack frame is not
   handling an exception, the information is taken from the calling
   stack frame, or its caller, and so on until a stack frame is found
   that is handling an exception.  Here, "handling an exception" is
   defined as "executing an except clause."  For any stack frame, only
   information about the exception being currently handled is
   accessible.

   If no exception is being handled anywhere on the stack, a tuple
   containing three "None" values is returned.  Otherwise, the values
   returned are "(type, value, traceback)".  Their meaning is: *type*
   gets the type of the exception being handled (a subclass of
   "BaseException"); *value* gets the exception instance (an instance
   of the exception type); *traceback* gets a traceback object which
   encapsulates the call stack at the point where the exception
   originally occurred.

sys.exec_prefix

   A string giving the site-specific directory prefix where the
   platform-dependent Python files are installed; by default, this is
   also "'/usr/local'".  This can be set at build time with the "--
   exec-prefix" argument to the **configure** script.  Specifically,
   all configuration files (e.g. the "pyconfig.h" header file) are
   installed in the directory "*exec_prefix*/lib/python*X.Y*/config",
   and shared library modules are installed in
   "*exec_prefix*/lib/python*X.Y*/lib-dynload", where *X.Y* is the
   version number of Python, for example "3.2".

   注解:

     If a virtual environment is in effect, this value will be changed
     in "site.py" to point to the virtual environment. The value for
     the Python installation will still be available, via
     "base_exec_prefix".

sys.executable

   A string giving the absolute path of the executable binary for the
   Python interpreter, on systems where this makes sense. If Python is
   unable to retrieve the real path to its executable,
   "sys.executable" will be an empty string or "None".

sys.exit([arg])

   从Python中退出。实现方式是抛出一个 "SystemExit" 异常。异常抛出后
   "try" 声明的 finally 分支语句的清除动作将被出发。此动作有可能打断更
   外层的退出尝试。

   The optional argument *arg* can be an integer giving the exit
   status (defaulting to zero), or another type of object.  If it is
   an integer, zero is considered "successful termination" and any
   nonzero value is considered "abnormal termination" by shells and
   the like.  Most systems require it to be in the range 0--127, and
   produce undefined results otherwise.  Some systems have a
   convention for assigning specific meanings to specific exit codes,
   but these are generally underdeveloped; Unix programs generally use
   2 for command line syntax errors and 1 for all other kind of
   errors.  If another type of object is passed, "None" is equivalent
   to passing zero, and any other object is printed to "stderr" and
   results in an exit code of 1.  In particular, "sys.exit("some error
   message")" is a quick way to exit a program when an error occurs.

   由于 "exit()" 最终“只是”抛出一个异常，因此当从主线程调用时，只会从
   进程退出；而异常不会因此被打断。

   在 3.6 版更改: If an error occurs in the cleanup after the Python
   interpreter has caught "SystemExit" (such as an error flushing
   buffered data in the standard streams), the exit status is changed
   to 120.

sys.flags

   The *named tuple* *flags* exposes the status of command line flags.
   The attributes are read only.

   +-------------------------------+-------------------------------+
   | 属性                          | 标志                          |
   |===============================|===============================|
   | "debug"                       | "-d"                          |
   +-------------------------------+-------------------------------+
   | "inspect"                     | "-i"                          |
   +-------------------------------+-------------------------------+
   | "interactive"                 | "-i"                          |
   +-------------------------------+-------------------------------+
   | "isolated"                    | "-I"                          |
   +-------------------------------+-------------------------------+
   | "optimize"                    | "-O" 或 "-OO"                 |
   +-------------------------------+-------------------------------+
   | "dont_write_bytecode"         | "-B"                          |
   +-------------------------------+-------------------------------+
   | "no_user_site"                | "-s"                          |
   +-------------------------------+-------------------------------+
   | "no_site"                     | "-S"                          |
   +-------------------------------+-------------------------------+
   | "ignore_environment"          | "-E"                          |
   +-------------------------------+-------------------------------+
   | "verbose"                     | "-v"                          |
   +-------------------------------+-------------------------------+
   | "bytes_warning"               | "-b"                          |
   +-------------------------------+-------------------------------+
   | "quiet"                       | "-q"                          |
   +-------------------------------+-------------------------------+
   | "hash_randomization"          | "-R"                          |
   +-------------------------------+-------------------------------+
   | "dev_mode"                    | "-X" "dev"                    |
   +-------------------------------+-------------------------------+
   | "utf8_mode"                   | "-X" "utf8"                   |
   +-------------------------------+-------------------------------+

   在 3.2 版更改: 为新的 "-q" 标志添加了 "quiet" 属性

   3.2.3 新版功能: "hash_randomization" 属性

   在 3.3 版更改: 删除了过时的 "division_warning" 属性

   在 3.4 版更改: 为 "-I" "isolated" 标志添加了 "isolated" 属性。

   在 3.7 版更改: Added "dev_mode" attribute for the new "-X" "dev"
   flag and "utf8_mode" attribute for the new  "-X" "utf8" flag.

sys.float_info

   A *named tuple* holding information about the float type. It
   contains low level information about the precision and internal
   representation.  The values correspond to the various floating-
   point constants defined in the standard header file "float.h" for
   the 'C' programming language; see section 5.2.4.2.2 of the 1999
   ISO/IEC C standard [C99], 'Characteristics of floating types', for
   details.

   +-----------------------+------------------+----------------------------------------------------+
   | 属性                  | float.h 宏       | 说明                                               |
   |=======================|==================|====================================================|
   | "epsilon"             | DBL_EPSILON      | difference between 1.0 and the least value greater |
   |                       |                  | than 1.0 that is representable as a float          |
   +-----------------------+------------------+----------------------------------------------------+
   | "dig"                 | DBL_DIG          | maximum number of decimal digits that can be       |
   |                       |                  | faithfully represented in a float;  see below      |
   +-----------------------+------------------+----------------------------------------------------+
   | "mant_dig"            | DBL_MANT_DIG     | float precision: the number of base-"radix" digits |
   |                       |                  | in the significand of a float                      |
   +-----------------------+------------------+----------------------------------------------------+
   | "max"                 | DBL_MAX          | maximum representable positive finite float        |
   +-----------------------+------------------+----------------------------------------------------+
   | "max_exp"             | DBL_MAX_EXP      | maximum integer *e* such that "radix**(e-1)" is a  |
   |                       |                  | representable finite float                         |
   +-----------------------+------------------+----------------------------------------------------+
   | "max_10_exp"          | DBL_MAX_10_EXP   | maximum integer *e* such that "10**e" is in the    |
   |                       |                  | range of representable finite floats               |
   +-----------------------+------------------+----------------------------------------------------+
   | "min"                 | DBL_MIN          | minimum representable positive *normalized* float  |
   +-----------------------+------------------+----------------------------------------------------+
   | "min_exp"             | DBL_MIN_EXP      | minimum integer *e* such that "radix**(e-1)" is a  |
   |                       |                  | normalized float                                   |
   +-----------------------+------------------+----------------------------------------------------+
   | "min_10_exp"          | DBL_MIN_10_EXP   | minimum integer *e* such that "10**e" is a         |
   |                       |                  | normalized float                                   |
   +-----------------------+------------------+----------------------------------------------------+
   | "radix"               | FLT_RADIX        | radix of exponent representation                   |
   +-----------------------+------------------+----------------------------------------------------+
   | "rounds"              | FLT_ROUNDS       | integer constant representing the rounding mode    |
   |                       |                  | used for arithmetic operations.  This reflects the |
   |                       |                  | value of the system FLT_ROUNDS macro at            |
   |                       |                  | interpreter startup time.  See section 5.2.4.2.2   |
   |                       |                  | of the C99 standard for an explanation of the      |
   |                       |                  | possible values and their meanings.                |
   +-----------------------+------------------+----------------------------------------------------+

   The attribute "sys.float_info.dig" needs further explanation.  If
   "s" is any string representing a decimal number with at most
   "sys.float_info.dig" significant digits, then converting "s" to a
   float and back again will recover a string representing the same
   decimal value:

      >>> import sys
      >>> sys.float_info.dig
      15
      >>> s = '3.14159265358979'    # decimal string with 15 significant digits
      >>> format(float(s), '.15g')  # convert to float and back -> same value
      '3.14159265358979'

   But for strings with more than "sys.float_info.dig" significant
   digits, this isn't always true:

      >>> s = '9876543211234567'    # 16 significant digits is too many!
      >>> format(float(s), '.16g')  # conversion changes value
      '9876543211234568'

sys.float_repr_style

   A string indicating how the "repr()" function behaves for floats.
   If the string has value "'short'" then for a finite float "x",
   "repr(x)" aims to produce a short string with the property that
   "float(repr(x)) == x".  This is the usual behaviour in Python 3.1
   and later.  Otherwise, "float_repr_style" has value "'legacy'" and
   "repr(x)" behaves in the same way as it did in versions of Python
   prior to 3.1.

   3.1 新版功能.

sys.getallocatedblocks()

   Return the number of memory blocks currently allocated by the
   interpreter, regardless of their size.  This function is mainly
   useful for tracking and debugging memory leaks.  Because of the
   interpreter's internal caches, the result can vary from call to
   call; you may have to call "_clear_type_cache()" and "gc.collect()"
   to get more predictable results.

   If a Python build or implementation cannot reasonably compute this
   information, "getallocatedblocks()" is allowed to return 0 instead.

   3.4 新版功能.

sys.getandroidapilevel()

   Return the build time API version of Android as an integer.

   Availability: Android.

   3.7 新版功能.

sys.getcheckinterval()

   Return the interpreter's "check interval"; see
   "setcheckinterval()".

   3.2 版后已移除: Use "getswitchinterval()" instead.

sys.getdefaultencoding()

   Return the name of the current default string encoding used by the
   Unicode implementation.

sys.getdlopenflags()

   Return the current value of the flags that are used for "dlopen()"
   calls.  Symbolic names for the flag values can be found in the "os"
   module ("RTLD_xxx" constants, e.g. "os.RTLD_LAZY").

   可用性: Unix。

sys.getfilesystemencoding()

   Return the name of the encoding used to convert between Unicode
   filenames and bytes filenames. For best compatibility, str should
   be used for filenames in all cases, although representing filenames
   as bytes is also supported. Functions accepting or returning
   filenames should support either str or bytes and internally convert
   to the system's preferred representation.

   This encoding is always ASCII-compatible.

   "os.fsencode()" and "os.fsdecode()" should be used to ensure that
   the correct encoding and errors mode are used.

   * 在UTF-8模式下，任何平台上的编码均为 "utf-8" 。

   * 在macOS上，编码为 "'utf-8'"。

   * 在Unix上，编码是语言环境编码。

   * 在Windows上取决于用户配置，编码可能是 "'utf-8'" 或 "'mbcs'"。

   * 在Android上，编码为 "'utf-8'"。

   * 在VxWorks上，编码为 "'utf-8'"。

   在 3.2 版更改: "getfilesystemencoding()" result cannot be "None"
   anymore.

   在 3.6 版更改: Windows is no longer guaranteed to return "'mbcs'".
   See **PEP 529** and "_enablelegacywindowsfsencoding()" for more
   information.

   在 3.7 版更改: 在UTF-8模式下返回 'utf-8' 。

sys.getfilesystemencodeerrors()

   Return the name of the error mode used to convert between Unicode
   filenames and bytes filenames. The encoding name is returned from
   "getfilesystemencoding()".

   "os.fsencode()" and "os.fsdecode()" should be used to ensure that
   the correct encoding and errors mode are used.

   3.6 新版功能.

sys.getrefcount(object)

   Return the reference count of the *object*.  The count returned is
   generally one higher than you might expect, because it includes the
   (temporary) reference as an argument to "getrefcount()".

sys.getrecursionlimit()

   Return the current value of the recursion limit, the maximum depth
   of the Python interpreter stack.  This limit prevents infinite
   recursion from causing an overflow of the C stack and crashing
   Python.  It can be set by "setrecursionlimit()".

sys.getsizeof(object[, default])

   Return the size of an object in bytes. The object can be any type
   of object. All built-in objects will return correct results, but
   this does not have to hold true for third-party extensions as it is
   implementation specific.

   Only the memory consumption directly attributed to the object is
   accounted for, not the memory consumption of objects it refers to.

   If given, *default* will be returned if the object does not provide
   means to retrieve the size.  Otherwise a "TypeError" will be
   raised.

   "getsizeof()" calls the object's "__sizeof__" method and adds an
   additional garbage collector overhead if the object is managed by
   the garbage collector.

   See recursive sizeof recipe for an example of using "getsizeof()"
   recursively to find the size of containers and all their contents.

sys.getswitchinterval()

   Return the interpreter's "thread switch interval"; see
   "setswitchinterval()".

   3.2 新版功能.

sys._getframe([depth])

   Return a frame object from the call stack.  If optional integer
   *depth* is given, return the frame object that many calls below the
   top of the stack.  If that is deeper than the call stack,
   "ValueError" is raised.  The default for *depth* is zero, returning
   the frame at the top of the call stack.

   Raises an auditing event "sys._getframe" with no arguments.

   **CPython implementation detail:** This function should be used for
   internal and specialized purposes only. It is not guaranteed to
   exist in all implementations of Python.

sys.getprofile()

   Get the profiler function as set by "setprofile()".

sys.gettrace()

   Get the trace function as set by "settrace()".

   **CPython implementation detail:** The "gettrace()" function is
   intended only for implementing debuggers, profilers, coverage tools
   and the like.  Its behavior is part of the implementation platform,
   rather than part of the language definition, and thus may not be
   available in all Python implementations.

sys.getwindowsversion()

   Return a named tuple describing the Windows version currently
   running.  The named elements are *major*, *minor*, *build*,
   *platform*, *service_pack*, *service_pack_minor*,
   *service_pack_major*, *suite_mask*, *product_type* and
   *platform_version*. *service_pack* contains a string,
   *platform_version* a 3-tuple and all other values are integers. The
   components can also be accessed by name, so
   "sys.getwindowsversion()[0]" is equivalent to
   "sys.getwindowsversion().major". For compatibility with prior
   versions, only the first 5 elements are retrievable by indexing.

   *platform* will be "2 (VER_PLATFORM_WIN32_NT)".

   *product_type* may be one of the following values:

   +-----------------------------------------+-----------------------------------+
   | 常数                                    | 含义                              |
   |=========================================|===================================|
   | "1 (VER_NT_WORKSTATION)"                | 系统是工作站。                    |
   +-----------------------------------------+-----------------------------------+
   | "2 (VER_NT_DOMAIN_CONTROLLER)"          | 系统是域控制器。                  |
   +-----------------------------------------+-----------------------------------+
   | "3 (VER_NT_SERVER)"                     | 系统是服务器，但不是域控制器。    |
   +-----------------------------------------+-----------------------------------+

   This function wraps the Win32 "GetVersionEx()" function; see the
   Microsoft documentation on "OSVERSIONINFOEX()" for more information
   about these fields.

   *platform_version* returns the accurate major version, minor
   version and build number of the current operating system, rather
   than the version that is being emulated for the process. It is
   intended for use in logging rather than for feature detection.

   可用性: Windows。

   在 3.2 版更改: Changed to a named tuple and added
   *service_pack_minor*, *service_pack_major*, *suite_mask*, and
   *product_type*.

   在 3.6 版更改: 添加了 *platform_version*

sys.get_asyncgen_hooks()

   Returns an *asyncgen_hooks* object, which is similar to a
   "namedtuple" of the form *(firstiter, finalizer)*, where
   *firstiter* and *finalizer* are expected to be either "None" or
   functions which take an *asynchronous generator iterator* as an
   argument, and are used to schedule finalization of an asynchronous
   generator by an event loop.

   3.6 新版功能: See **PEP 525** for more details.

   注解:

     This function has been added on a provisional basis (see **PEP
     411** for details.)

sys.get_coroutine_origin_tracking_depth()

   Get the current coroutine origin tracking depth, as set by
   "set_coroutine_origin_tracking_depth()".

   3.7 新版功能.

   注解:

     This function has been added on a provisional basis (see **PEP
     411** for details.)  Use it only for debugging purposes.

sys.hash_info

   A *named tuple* giving parameters of the numeric hash
   implementation.  For more details about hashing of numeric types,
   see 数字类型的哈希运算.

   +-----------------------+----------------------------------------------------+
   | 属性                  | 说明                                               |
   |=======================|====================================================|
   | "width"               | 用于哈希值的位宽度                                 |
   +-----------------------+----------------------------------------------------+
   | "modulus"             | 用于数字散列方案的素数模数P。                      |
   +-----------------------+----------------------------------------------------+
   | "inf"                 | 为正无穷大返回的哈希值                             |
   +-----------------------+----------------------------------------------------+
   | "nan"                 | 为nan返回的哈希值                                  |
   +-----------------------+----------------------------------------------------+
   | "imag"                | 用于复数虚部的乘数                                 |
   +-----------------------+----------------------------------------------------+
   | "algorithm"           | 字符串、字节和内存视图的哈希算法的名称             |
   +-----------------------+----------------------------------------------------+
   | "hash_bits"           | 哈希算法的内部输出大小。                           |
   +-----------------------+----------------------------------------------------+
   | "seed_bits"           | 散列算法的种子密钥的大小                           |
   +-----------------------+----------------------------------------------------+

   3.2 新版功能.

   在 3.4 版更改: 添加了 *algorithm*, *hash_bits* 和 *seed_bits*

sys.hexversion

   The version number encoded as a single integer.  This is guaranteed
   to increase with each version, including proper support for non-
   production releases.  For example, to test that the Python
   interpreter is at least version 1.5.2, use:

      if sys.hexversion >= 0x010502F0:
          # use some advanced feature
          ...
      else:
          # use an alternative implementation or warn the user
          ...

   This is called "hexversion" since it only really looks meaningful
   when viewed as the result of passing it to the built-in "hex()"
   function.  The *named tuple*  "sys.version_info" may be used for a
   more human-friendly encoding of the same information.

   More details of "hexversion" can be found at API 和 ABI 版本管理.

sys.implementation

   An object containing information about the implementation of the
   currently running Python interpreter.  The following attributes are
   required to exist in all Python implementations.

   *name* is the implementation's identifier, e.g. "'cpython'".  The
   actual string is defined by the Python implementation, but it is
   guaranteed to be lower case.

   *version* is a named tuple, in the same format as
   "sys.version_info".  It represents the version of the Python
   *implementation*.  This has a distinct meaning from the specific
   version of the Python *language* to which the currently running
   interpreter conforms, which "sys.version_info" represents.  For
   example, for PyPy 1.8 "sys.implementation.version" might be
   "sys.version_info(1, 8, 0, 'final', 0)", whereas "sys.version_info"
   would be "sys.version_info(2, 7, 2, 'final', 0)".  For CPython they
   are the same value, since it is the reference implementation.

   *hexversion* is the implementation version in hexadecimal format,
   like "sys.hexversion".

   *cache_tag* is the tag used by the import machinery in the
   filenames of cached modules.  By convention, it would be a
   composite of the implementation's name and version, like
   "'cpython-33'".  However, a Python implementation may use some
   other value if appropriate.  If "cache_tag" is set to "None", it
   indicates that module caching should be disabled.

   "sys.implementation" may contain additional attributes specific to
   the Python implementation.  These non-standard attributes must
   start with an underscore, and are not described here.  Regardless
   of its contents, "sys.implementation" will not change during a run
   of the interpreter, nor between implementation versions.  (It may
   change between Python language versions, however.)  See **PEP 421**
   for more information.

   3.3 新版功能.

   注解:

     The addition of new required attributes must go through the
     normal PEP process. See **PEP 421** for more information.

sys.int_info

   A *named tuple* that holds information about Python's internal
   representation of integers.  The attributes are read only.

   +---------------------------+------------------------------------------------+
   | 属性                      | 说明                                           |
   |===========================|================================================|
   | "bits_per_digit"          | number of bits held in each digit.  Python     |
   |                           | integers are stored internally in base         |
   |                           | "2**int_info.bits_per_digit"                   |
   +---------------------------+------------------------------------------------+
   | "sizeof_digit"            | 用于表示数字的C类型的字节大小                  |
   +---------------------------+------------------------------------------------+

   3.1 新版功能.

sys.__interactivehook__

   When this attribute exists, its value is automatically called (with
   no arguments) when the interpreter is launched in interactive mode.
   This is done after the "PYTHONSTARTUP" file is read, so that you
   can set this hook there.  The "site" module sets this.

   Raises an auditing event "cpython.run_interactivehook" with the
   hook object as the argument when the hook is called on startup.

   3.4 新版功能.

sys.intern(string)

   Enter *string* in the table of "interned" strings and return the
   interned string -- which is *string* itself or a copy. Interning
   strings is useful to gain a little performance on dictionary lookup
   -- if the keys in a dictionary are interned, and the lookup key is
   interned, the key comparisons (after hashing) can be done by a
   pointer compare instead of a string compare.  Normally, the names
   used in Python programs are automatically interned, and the
   dictionaries used to hold module, class or instance attributes have
   interned keys.

   Interned strings are not immortal; you must keep a reference to the
   return value of "intern()" around to benefit from it.

sys.is_finalizing()

   Return "True" if the Python interpreter is *shutting down*, "False"
   otherwise.

   3.5 新版功能.

sys.last_type
sys.last_value
sys.last_traceback

   These three variables are not always defined; they are set when an
   exception is not handled and the interpreter prints an error
   message and a stack traceback. Their intended use is to allow an
   interactive user to import a debugger module and engage in post-
   mortem debugging without having to re-execute the command that
   caused the error.  (Typical use is "import pdb; pdb.pm()" to enter
   the post-mortem debugger; see "pdb" module for more information.)

   The meaning of the variables is the same as that of the return
   values from "exc_info()" above.

sys.maxsize

   An integer giving the maximum value a variable of type "Py_ssize_t"
   can take.  It's usually "2**31 - 1" on a 32-bit platform and "2**63
   - 1" on a 64-bit platform.

sys.maxunicode

   An integer giving the value of the largest Unicode code point, i.e.
   "1114111" ("0x10FFFF" in hexadecimal).

   在 3.3 版更改: Before **PEP 393**, "sys.maxunicode" used to be
   either "0xFFFF" or "0x10FFFF", depending on the configuration
   option that specified whether Unicode characters were stored as
   UCS-2 or UCS-4.

sys.meta_path

   A list of *meta path finder* objects that have their "find_spec()"
   methods called to see if one of the objects can find the module to
   be imported. The "find_spec()" method is called with at least the
   absolute name of the module being imported. If the module to be
   imported is contained in a package, then the parent package's
   "__path__" attribute is passed in as a second argument. The method
   returns a *module spec*, or "None" if the module cannot be found.

   参见:

     "importlib.abc.MetaPathFinder"
        The abstract base class defining the interface of finder
        objects on "meta_path".

     "importlib.machinery.ModuleSpec"
        The concrete class which "find_spec()" should return instances
        of.

   在 3.4 版更改: *Module specs* were introduced in Python 3.4, by
   **PEP 451**. Earlier versions of Python looked for a method called
   "find_module()". This is still called as a fallback if a
   "meta_path" entry doesn't have a "find_spec()" method.

sys.modules

   This is a dictionary that maps module names to modules which have
   already been loaded.  This can be manipulated to force reloading of
   modules and other tricks. However, replacing the dictionary will
   not necessarily work as expected and deleting essential items from
   the dictionary may cause Python to fail.

sys.path

   A list of strings that specifies the search path for modules.
   Initialized from the environment variable "PYTHONPATH", plus an
   installation-dependent default.

   As initialized upon program startup, the first item of this list,
   "path[0]", is the directory containing the script that was used to
   invoke the Python interpreter.  If the script directory is not
   available (e.g.  if the interpreter is invoked interactively or if
   the script is read from standard input), "path[0]" is the empty
   string, which directs Python to search modules in the current
   directory first.  Notice that the script directory is inserted
   *before* the entries inserted as a result of "PYTHONPATH".

   A program is free to modify this list for its own purposes.  Only
   strings and bytes should be added to "sys.path"; all other data
   types are ignored during import.

   参见:

     Module "site" This describes how to use .pth files to extend
     "sys.path".

sys.path_hooks

   A list of callables that take a path argument to try to create a
   *finder* for the path. If a finder can be created, it is to be
   returned by the callable, else raise "ImportError".

   Originally specified in **PEP 302**.

sys.path_importer_cache

   A dictionary acting as a cache for *finder* objects. The keys are
   paths that have been passed to "sys.path_hooks" and the values are
   the finders that are found. If a path is a valid file system path
   but no finder is found on "sys.path_hooks" then "None" is stored.

   Originally specified in **PEP 302**.

   在 3.3 版更改: "None" is stored instead of "imp.NullImporter" when
   no finder is found.

sys.platform

   This string contains a platform identifier that can be used to
   append platform-specific components to "sys.path", for instance.

   For Unix systems, except on Linux and AIX, this is the lowercased
   OS name as returned by "uname -s" with the first part of the
   version as returned by "uname -r" appended, e.g. "'sunos5'" or
   "'freebsd8'", *at the time when Python was built*.  Unless you want
   to test for a specific system version, it is therefore recommended
   to use the following idiom:

      if sys.platform.startswith('freebsd'):
          # FreeBSD-specific code here...
      elif sys.platform.startswith('linux'):
          # Linux-specific code here...
      elif sys.platform.startswith('aix'):
          # AIX-specific code here...

   对于其他系统，值是：

   +------------------+-----------------------------+
   | 系统             | "平台" 值                   |
   |==================|=============================|
   | AIX              | "'aix'"                     |
   +------------------+-----------------------------+
   | Linux            | "'linux'"                   |
   +------------------+-----------------------------+
   | Windows          | "'win32'"                   |
   +------------------+-----------------------------+
   | Windows/Cygwin   | "'cygwin'"                  |
   +------------------+-----------------------------+
   | macOS            | "'darwin'"                  |
   +------------------+-----------------------------+

   在 3.3 版更改: On Linux, "sys.platform" doesn't contain the major
   version anymore. It is always "'linux'", instead of "'linux2'" or
   "'linux3'".  Since older Python versions include the version
   number, it is recommended to always use the "startswith" idiom
   presented above.

   在 3.8 版更改: On AIX, "sys.platform" doesn't contain the major
   version anymore. It is always "'aix'", instead of "'aix5'" or
   "'aix7'".  Since older Python versions include the version number,
   it is recommended to always use the "startswith" idiom presented
   above.

   参见:

     "os.name" has a coarser granularity.  "os.uname()" gives system-
     dependent version information.

     "platform" 模块提供了对系统标识更详细的检查。

sys.prefix

   A string giving the site-specific directory prefix where the
   platform independent Python files are installed; by default, this
   is the string "'/usr/local'".  This can be set at build time with
   the "--prefix" argument to the **configure** script.  The main
   collection of Python library modules is installed in the directory
   "*prefix*/lib/python*X.Y*" while the platform independent header
   files (all except "pyconfig.h") are stored in
   "*prefix*/include/python*X.Y*", where *X.Y* is the version number
   of Python, for example "3.2".

   注解:

     If a virtual environment is in effect, this value will be changed
     in "site.py" to point to the virtual environment. The value for
     the Python installation will still be available, via
     "base_prefix".

sys.ps1
sys.ps2

   Strings specifying the primary and secondary prompt of the
   interpreter.  These are only defined if the interpreter is in
   interactive mode.  Their initial values in this case are "'>>> '"
   and "'... '".  If a non-string object is assigned to either
   variable, its "str()" is re-evaluated each time the interpreter
   prepares to read a new interactive command; this can be used to
   implement a dynamic prompt.

sys.setcheckinterval(interval)

   Set the interpreter's "check interval".  This integer value
   determines how often the interpreter checks for periodic things
   such as thread switches and signal handlers.  The default is "100",
   meaning the check is performed every 100 Python virtual
   instructions. Setting it to a larger value may increase performance
   for programs using threads.  Setting it to a value "<=" 0 checks
   every virtual instruction, maximizing responsiveness as well as
   overhead.

   3.2 版后已移除: This function doesn't have an effect anymore, as
   the internal logic for thread switching and asynchronous tasks has
   been rewritten.  Use "setswitchinterval()" instead.

sys.setdlopenflags(n)

   Set the flags used by the interpreter for "dlopen()" calls, such as
   when the interpreter loads extension modules.  Among other things,
   this will enable a lazy resolving of symbols when importing a
   module, if called as "sys.setdlopenflags(0)".  To share symbols
   across extension modules, call as
   "sys.setdlopenflags(os.RTLD_GLOBAL)".  Symbolic names for the flag
   values can be found in the "os" module ("RTLD_xxx" constants, e.g.
   "os.RTLD_LAZY").

   可用性: Unix。

sys.setprofile(profilefunc)

   Set the system's profile function, which allows you to implement a
   Python source code profiler in Python.  See chapter Python
   Profilers 分析器 for more information on the Python profiler.  The
   system's profile function is called similarly to the system's trace
   function (see "settrace()"), but it is called with different
   events, for example it isn't called for each executed line of code
   (only on call and return, but the return event is reported even
   when an exception has been set). The function is thread-specific,
   but there is no way for the profiler to know about context switches
   between threads, so it does not make sense to use this in the
   presence of multiple threads. Also, its return value is not used,
   so it can simply return "None".  Error in the profile function will
   cause itself unset.

   Profile functions should have three arguments: *frame*, *event*,
   and *arg*. *frame* is the current stack frame.  *event* is a
   string: "'call'", "'return'", "'c_call'", "'c_return'", or
   "'c_exception'". *arg* depends on the event type.

   Raises an auditing event "sys.setprofile" with no arguments.

   这些事件具有以下含义：

   "'call'"
      A function is called (or some other code block entered).  The
      profile function is called; *arg* is "None".

   "'return'"
      A function (or other code block) is about to return.  The
      profile function is called; *arg* is the value that will be
      returned, or "None" if the event is caused by an exception being
      raised.

   "'c_call'"
      A C function is about to be called.  This may be an extension
      function or a built-in.  *arg* is the C function object.

   "'c_return'"
      A C function has returned. *arg* is the C function object.

   "'c_exception'"
      A C function has raised an exception.  *arg* is the C function
      object.

sys.setrecursionlimit(limit)

   Set the maximum depth of the Python interpreter stack to *limit*.
   This limit prevents infinite recursion from causing an overflow of
   the C stack and crashing Python.

   The highest possible limit is platform-dependent.  A user may need
   to set the limit higher when they have a program that requires deep
   recursion and a platform that supports a higher limit.  This should
   be done with care, because a too-high limit can lead to a crash.

   If the new limit is too low at the current recursion depth, a
   "RecursionError" exception is raised.

   在 3.5.1 版更改: A "RecursionError" exception is now raised if the
   new limit is too low at the current recursion depth.

sys.setswitchinterval(interval)

   Set the interpreter's thread switch interval (in seconds).  This
   floating-point value determines the ideal duration of the
   "timeslices" allocated to concurrently running Python threads.
   Please note that the actual value can be higher, especially if
   long-running internal functions or methods are used.  Also, which
   thread becomes scheduled at the end of the interval is the
   operating system's decision.  The interpreter doesn't have its own
   scheduler.

   3.2 新版功能.

sys.settrace(tracefunc)

   Set the system's trace function, which allows you to implement a
   Python source code debugger in Python.  The function is thread-
   specific; for a debugger to support multiple threads, it must
   register a trace function using "settrace()" for each thread being
   debugged or use "threading.settrace()".

   Trace functions should have three arguments: *frame*, *event*, and
   *arg*. *frame* is the current stack frame.  *event* is a string:
   "'call'", "'line'", "'return'", "'exception'" or "'opcode'".  *arg*
   depends on the event type.

   The trace function is invoked (with *event* set to "'call'")
   whenever a new local scope is entered; it should return a reference
   to a local trace function to be used for the new scope, or "None"
   if the scope shouldn't be traced.

   The local trace function should return a reference to itself (or to
   another function for further tracing in that scope), or "None" to
   turn off tracing in that scope.

   If there is any error occurred in the trace function, it will be
   unset, just like "settrace(None)" is called.

   这些事件具有以下含义：

   "'call'"
      A function is called (or some other code block entered).  The
      global trace function is called; *arg* is "None"; the return
      value specifies the local trace function.

   "'line'"
      The interpreter is about to execute a new line of code or re-
      execute the condition of a loop.  The local trace function is
      called; *arg* is "None"; the return value specifies the new
      local trace function.  See "Objects/lnotab_notes.txt" for a
      detailed explanation of how this works. Per-line events may be
      disabled for a frame by setting "f_trace_lines" to "False" on
      that frame.

   "'return'"
      A function (or other code block) is about to return.  The local
      trace function is called; *arg* is the value that will be
      returned, or "None" if the event is caused by an exception being
      raised.  The trace function's return value is ignored.

   "'exception'"
      An exception has occurred.  The local trace function is called;
      *arg* is a tuple "(exception, value, traceback)"; the return
      value specifies the new local trace function.

   "'opcode'"
      The interpreter is about to execute a new opcode (see "dis" for
      opcode details).  The local trace function is called; *arg* is
      "None"; the return value specifies the new local trace function.
      Per-opcode events are not emitted by default: they must be
      explicitly requested by setting "f_trace_opcodes" to "True" on
      the frame.

   Note that as an exception is propagated down the chain of callers,
   an "'exception'" event is generated at each level.

   For more fine-grained usage, it's possible to set a trace function
   by assigning "frame.f_trace = tracefunc" explicitly, rather than
   relying on it being set indirectly via the return value from an
   already installed trace function. This is also required for
   activating the trace function on the current frame, which
   "settrace()" doesn't do. Note that in order for this to work, a
   global tracing function must have been installed with "settrace()"
   in order to enable the runtime tracing machinery, but it doesn't
   need to be the same tracing function (e.g. it could be a low
   overhead tracing function that simply returns "None" to disable
   itself immediately on each frame).

   For more information on code and frame objects, refer to 标准类型层
   级结构.

   Raises an auditing event "sys.settrace" with no arguments.

   **CPython implementation detail:** The "settrace()" function is
   intended only for implementing debuggers, profilers, coverage tools
   and the like.  Its behavior is part of the implementation platform,
   rather than part of the language definition, and thus may not be
   available in all Python implementations.

   在 3.7 版更改: "'opcode'" event type added; "f_trace_lines" and
   "f_trace_opcodes" attributes added to frames

sys.set_asyncgen_hooks(firstiter, finalizer)

   Accepts two optional keyword arguments which are callables that
   accept an *asynchronous generator iterator* as an argument. The
   *firstiter* callable will be called when an asynchronous generator
   is iterated for the first time. The *finalizer* will be called when
   an asynchronous generator is about to be garbage collected.

   Raises an auditing event "sys.set_asyncgen_hooks_firstiter" with no
   arguments.

   Raises an auditing event "sys.set_asyncgen_hooks_finalizer" with no
   arguments.

   Two auditing events are raised because the underlying API consists
   of two calls, each of which must raise its own event.

   3.6 新版功能: See **PEP 525** for more details, and for a reference
   example of a *finalizer* method see the implementation of
   "asyncio.Loop.shutdown_asyncgens" in Lib/asyncio/base_events.py

   注解:

     This function has been added on a provisional basis (see **PEP
     411** for details.)

sys.set_coroutine_origin_tracking_depth(depth)

   Allows enabling or disabling coroutine origin tracking. When
   enabled, the "cr_origin" attribute on coroutine objects will
   contain a tuple of (filename, line number, function name) tuples
   describing the traceback where the coroutine object was created,
   with the most recent call first. When disabled, "cr_origin" will be
   None.

   To enable, pass a *depth* value greater than zero; this sets the
   number of frames whose information will be captured. To disable,
   pass set *depth* to zero.

   This setting is thread-specific.

   3.7 新版功能.

   注解:

     This function has been added on a provisional basis (see **PEP
     411** for details.)  Use it only for debugging purposes.

sys._enablelegacywindowsfsencoding()

   Changes the default filesystem encoding and errors mode to 'mbcs'
   and 'replace' respectively, for consistency with versions of Python
   prior to 3.6.

   This is equivalent to defining the "PYTHONLEGACYWINDOWSFSENCODING"
   environment variable before launching Python.

   可用性: Windows。

   3.6 新版功能: 有关更多详细信息，请参阅 **PEP 529**。

sys.stdin
sys.stdout
sys.stderr

   *File objects* used by the interpreter for standard input, output
   and errors:

   * "stdin" is used for all interactive input (including calls to
     "input()");

   * "stdout" is used for the output of "print()" and *expression*
     statements and for the prompts of "input()";

   * The interpreter's own prompts and its error messages go to
     "stderr".

   These streams are regular *text files* like those returned by the
   "open()" function.  Their parameters are chosen as follows:

   * The character encoding is platform-dependent.  Non-Windows
     platforms use the locale encoding (see
     "locale.getpreferredencoding()").

     On Windows, UTF-8 is used for the console device.  Non-character
     devices such as disk files and pipes use the system locale
     encoding (i.e. the ANSI codepage).  Non-console character devices
     such as NUL (i.e. where "isatty()" returns "True") use the value
     of the console input and output codepages at startup,
     respectively for stdin and stdout/stderr. This defaults to the
     system locale encoding if the process is not initially attached
     to a console.

     The special behaviour of the console can be overridden by setting
     the environment variable PYTHONLEGACYWINDOWSSTDIO before starting
     Python. In that case, the console codepages are used as for any
     other character device.

     Under all platforms, you can override the character encoding by
     setting the "PYTHONIOENCODING" environment variable before
     starting Python or by using the new "-X" "utf8" command line
     option and "PYTHONUTF8" environment variable.  However, for the
     Windows console, this only applies when
     "PYTHONLEGACYWINDOWSSTDIO" is also set.

   * When interactive, "stdout" and "stderr" streams are line-
     buffered. Otherwise, they are block-buffered like regular text
     files.  You can override this value with the "-u" command-line
     option.

   注解:

     To write or read binary data from/to the standard streams, use
     the underlying binary "buffer" object.  For example, to write
     bytes to "stdout", use "sys.stdout.buffer.write(b'abc')".However,
     if you are writing a library (and do not control in which context
     its code will be executed), be aware that the standard streams
     may be replaced with file-like objects like "io.StringIO" which
     do not support the "buffer" attribute.

sys.__stdin__
sys.__stdout__
sys.__stderr__

   These objects contain the original values of "stdin", "stderr" and
   "stdout" at the start of the program.  They are used during
   finalization, and could be useful to print to the actual standard
   stream no matter if the "sys.std*" object has been redirected.

   It can also be used to restore the actual files to known working
   file objects in case they have been overwritten with a broken
   object.  However, the preferred way to do this is to explicitly
   save the previous stream before replacing it, and restore the saved
   object.

   注解:

     Under some conditions "stdin", "stdout" and "stderr" as well as
     the original values "__stdin__", "__stdout__" and "__stderr__"
     can be "None". It is usually the case for Windows GUI apps that
     aren't connected to a console and Python apps started with
     **pythonw**.

sys.thread_info

   A *named tuple* holding information about the thread
   implementation.

   +--------------------+-----------------------------------------------------------+
   | 属性               | 说明                                                      |
   |====================|===========================================================|
   | "name"             | 线程实现的名称：  * "'nt'": Windows 线程  * "'pthread'":  |
   |                    | POSIX 线程  * "'solaris'": Solaris 线程                   |
   +--------------------+-----------------------------------------------------------+
   | "lock"             | 锁实现的名称：  * "'semaphore'": 锁使用信号量  *          |
   |                    | "'mutex+cond'": 锁使用互斥和条件变量  * "None" 如果此信息 |
   |                    | 未知                                                      |
   +--------------------+-----------------------------------------------------------+
   | "version"          | 线程库的名称和版本。它是一个字符串，如果此信息未知，则为  |
   |                    | "None" 。                                                 |
   +--------------------+-----------------------------------------------------------+

   3.3 新版功能.

sys.tracebacklimit

   When this variable is set to an integer value, it determines the
   maximum number of levels of traceback information printed when an
   unhandled exception occurs. The default is "1000".  When set to "0"
   or less, all traceback information is suppressed and only the
   exception type and value are printed.

sys.unraisablehook(unraisable, /)

   Handle an unraisable exception.

   Called when an exception has occurred but there is no way for
   Python to handle it. For example, when a destructor raises an
   exception or during garbage collection ("gc.collect()").

   The *unraisable* argument has the following attributes:

   * *exc_type*: 异常类型

   * *exc_value*: 异常值，可以是 "None".

   * *exc_traceback*: 异常回溯，可以是 "None".

   * *err_msg*: 错误信息，可以是 "None".

   * *object*: 导致异常的对象，可以为 "None".

   The default hook formats *err_msg* and *object* as: "f'{err_msg}:
   {object!r}'"; use "Exception ignored in" error message if *err_msg*
   is "None".

   "sys.unraisablehook()" can be overridden to control how unraisable
   exceptions are handled.

   使用定制钩子存放 *exc_value* 可能会创建引用循环。 它应当在不再需要
   异常时被显式地清空以打破引用循环。

   Storing *object* using a custom hook can resurrect it if it is set
   to an object which is being finalized. Avoid storing *object* after
   the custom hook completes to avoid resurrecting objects.

   See also "excepthook()" which handles uncaught exceptions.

   Raise an auditing event "sys.unraisablehook" with arguments "hook",
   "unraisable" when an exception that cannot be handled occurs. The
   "unraisable" object is the same as what will be passed to the hook.
   If no hook has been set, "hook" may be "None".

   3.8 新版功能.

sys.version

   A string containing the version number of the Python interpreter
   plus additional information on the build number and compiler used.
   This string is displayed when the interactive interpreter is
   started.  Do not extract version information out of it, rather, use
   "version_info" and the functions provided by the "platform" module.

sys.api_version

   The C API version for this interpreter.  Programmers may find this
   useful when debugging version conflicts between Python and
   extension modules.

sys.version_info

   A tuple containing the five components of the version number:
   *major*, *minor*, *micro*, *releaselevel*, and *serial*.  All
   values except *releaselevel* are integers; the release level is
   "'alpha'", "'beta'", "'candidate'", or "'final'".  The
   "version_info" value corresponding to the Python version 2.0 is
   "(2, 0, 0, 'final', 0)".  The components can also be accessed by
   name, so "sys.version_info[0]" is equivalent to
   "sys.version_info.major" and so on.

   在 3.1 版更改: Added named component attributes.

sys.warnoptions

   This is an implementation detail of the warnings framework; do not
   modify this value.  Refer to the "warnings" module for more
   information on the warnings framework.

sys.winver

   The version number used to form registry keys on Windows platforms.
   This is stored as string resource 1000 in the Python DLL.  The
   value is normally the first three characters of "version".  It is
   provided in the "sys" module for informational purposes; modifying
   this value has no effect on the registry keys used by Python.

   可用性: Windows。

sys._xoptions

   A dictionary of the various implementation-specific flags passed
   through the "-X" command-line option.  Option names are either
   mapped to their values, if given explicitly, or to "True".
   Example:

      $ ./python -Xa=b -Xc
      Python 3.2a3+ (py3k, Oct 16 2010, 20:14:50)
      [GCC 4.4.3] on linux2
      Type "help", "copyright", "credits" or "license" for more information.
      >>> import sys
      >>> sys._xoptions
      {'a': 'b', 'c': True}

   **CPython implementation detail:** This is a CPython-specific way
   of accessing options passed through "-X".  Other implementations
   may export them through other means, or not at all.

   3.2 新版功能.

-[ Citations ]-

[C99] ISO/IEC 9899:1999.  "Programming languages -- C."  A public
      draft of this standard is available at http://www.open-
      std.org/jtc1/sc22/wg14/www/docs/n1256.pdf.
