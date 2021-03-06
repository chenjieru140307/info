Python Profilers 分析器
***********************

**源代码：** Lib/profile.py 和 Lib/pstats.py

======================================================================


profile分析器简介
=================

"cProfile" 和 "profile" 提供了 Python 程序的 *确定性性能分析* 。
*profile* 是一组统计数据，描述程序的各个部分执行的频率和时间。这些统计
数据可以通过 "pstats" 模块格式化为报表。

Python 标准库提供了同一分析接口的两种不同实现：

1. 对于大多数用户，建议使用 "cProfile" ；这是一个 C 扩展插件，因为其合
   理的运行开销，所以适合于分析长时间运行的程序。该插件基于 "lsprof"
   ，由 Brett Rosen 和 Ted Chaotter 贡献。

2. "profile" 是一个纯 Python 模块（"cProfile" 就是模拟其接口的 C 语言
   实现），但它会显著增加配置程序的开销。如果你正在尝试以某种方式扩展
   分析器，则使用此模块可能会更容易完成任务。该模块最初由 Jim Roskind
   设计和编写。

注解:

  profiler 分析器模块被设计为给指定的程序提供执行概要文件，而不是用于
  基准测试目的（ "timeit" 才是用于此目标的，它能获得合理准确的结果）。
  这特别适用于将 Python 代码与 C 代码进行基准测试：分析器为Python 代码
  引入开销，但不会为 C级别的函数引入开销，因此 C 代码似乎比任何Python
  代码都更快。


实时用户手册
============

本节是为 “不想阅读手册” 的用户提供的。它提供了非常简短的概述，并允许用
户快速对现有应用程序执行评测。

要分析采用单个参数的函数，可以执行以下操作：

   import cProfile
   import re
   cProfile.run('re.compile("foo|bar")')

（如果 "cProfile" 在您的系统上不可用，请使用 "profile" 。）

上述操作将运行 "re.compile()" 并打印分析结果，如下所示：

         197 function calls (192 primitive calls) in 0.002 seconds

   Ordered by: standard name

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.001    0.001 <string>:1(<module>)
        1    0.000    0.000    0.001    0.001 re.py:212(compile)
        1    0.000    0.000    0.001    0.001 re.py:268(_compile)
        1    0.000    0.000    0.000    0.000 sre_compile.py:172(_compile_charset)
        1    0.000    0.000    0.000    0.000 sre_compile.py:201(_optimize_charset)
        4    0.000    0.000    0.000    0.000 sre_compile.py:25(_identityfunction)
      3/1    0.000    0.000    0.000    0.000 sre_compile.py:33(_compile)

第一行显示监听了197个调用。在这些调用中，有192个是 *原始的* ，这意味着
调用不是通过递归引发的。下一行: "Ordered by: standard name" ，表示最右
边列中的文本字符串用于对输出进行排序。列标题包括：

ncalls
   调用次数

tottime
   在指定函数中消耗的总时间（不包括调用子函数的时间）

percall
   是 "tottime" 除以 "ncalls" 的商

cumtime
   指定的函数及其所有子函数（从调用到退出）消耗的累积时间。这个数字对
   于递归函数来说是准确的。

percall
   是 "cumtime" 除以原始调用（次数）的商（即：函数运行一次的平均时间）

filename:lineno(function)
   提供相应数据的每个函数

如果第一列中有两个数字（例如3/1），则表示函数递归。第二个值是原始调用
次数，第一个是调用的总次数。请注意，当函数不递归时，这两个值是相同的，
并且只打印单个数字。

profile 运行结束时，打印输出不是必须的。也可以通过为 "run()" 函数指定
文件名，将结果保存到文件中：

   import cProfile
   import re
   cProfile.run('re.compile("foo|bar")', 'restats')

"pstats.Stats" 类从文件中读取 profile 结果，并以各种方式对其进行格式化
。

"cProfile" 和 "profile" 文件也可以作为脚本调用，以分析另一个脚本。例如
：

   python -m cProfile [-o output_file] [-s sort_order] (-m module | myscript.py)

"-o" 将profile 结果写入文件而不是标准输出

"-s" 指定 "sort_stats()" 排序值之一以对输出进行排序。这仅适用于未提供
"-o" 的情况

"-m" 指定要分析的是模块而不是脚本。

   3.7 新版功能: "cProfile" 添加 "-m" 选项

   3.8 新版功能: "profile" 添加 "-m" 选项

The "pstats" module's "Stats" class has a variety of methods for
manipulating and printing the data saved into a profile results file:

   import pstats
   from pstats import SortKey
   p = pstats.Stats('restats')
   p.strip_dirs().sort_stats(-1).print_stats()

The "strip_dirs()" method removed the extraneous path from all the
module names. The "sort_stats()" method sorted all the entries
according to the standard module/line/name string that is printed. The
"print_stats()" method printed out all the statistics.  You might try
the following sort calls:

   p.sort_stats(SortKey.NAME)
   p.print_stats()

The first call will actually sort the list by function name, and the
second call will print out the statistics.  The following are some
interesting calls to experiment with:

   p.sort_stats(SortKey.CUMULATIVE).print_stats(10)

This sorts the profile by cumulative time in a function, and then only
prints the ten most significant lines.  If you want to understand what
algorithms are taking time, the above line is what you would use.

If you were looking to see what functions were looping a lot, and
taking a lot of time, you would do:

   p.sort_stats(SortKey.TIME).print_stats(10)

to sort according to time spent within each function, and then print
the statistics for the top ten functions.

你也可以尝试：

   p.sort_stats(SortKey.FILENAME).print_stats('__init__')

This will sort all the statistics by file name, and then print out
statistics for only the class init methods (since they are spelled
with "__init__" in them).  As one final example, you could try:

   p.sort_stats(SortKey.TIME, SortKey.CUMULATIVE).print_stats(.5, 'init')

This line sorts statistics with a primary key of time, and a secondary
key of cumulative time, and then prints out some of the statistics. To
be specific, the list is first culled down to 50% (re: ".5") of its
original size, then only lines containing "init" are maintained, and
that sub-sub-list is printed.

If you wondered what functions called the above functions, you could
now ("p" is still sorted according to the last criteria) do:

   p.print_callers(.5, 'init')

and you would get a list of callers for each of the listed functions.

If you want more functionality, you're going to have to read the
manual, or guess what the following functions do:

   p.print_callees()
   p.add('restats')

Invoked as a script, the "pstats" module is a statistics browser for
reading and examining profile dumps.  It has a simple line-oriented
interface (implemented using "cmd") and interactive help.


"profile" 和 "cProfile" 模块参考
================================

"profile" 和 "cProfile" 模块都提供下列函数：

profile.run(command, filename=None, sort=-1)

   This function takes a single argument that can be passed to the
   "exec()" function, and an optional file name.  In all cases this
   routine executes:

      exec(command, __main__.__dict__, __main__.__dict__)

   and gathers profiling statistics from the execution. If no file
   name is present, then this function automatically creates a "Stats"
   instance and prints a simple profiling report. If the sort value is
   specified, it is passed to this "Stats" instance to control how the
   results are sorted.

profile.runctx(command, globals, locals, filename=None, sort=-1)

   This function is similar to "run()", with added arguments to supply
   the globals and locals dictionaries for the *command* string. This
   routine executes:

      exec(command, globals, locals)

   and gathers profiling statistics as in the "run()" function above.

class profile.Profile(timer=None, timeunit=0.0, subcalls=True, builtins=True)

   This class is normally only used if more precise control over
   profiling is needed than what the "cProfile.run()" function
   provides.

   A custom timer can be supplied for measuring how long code takes to
   run via the *timer* argument. This must be a function that returns
   a single number representing the current time. If the number is an
   integer, the *timeunit* specifies a multiplier that specifies the
   duration of each unit of time. For example, if the timer returns
   times measured in thousands of seconds, the time unit would be
   ".001".

   Directly using the "Profile" class allows formatting profile
   results without writing the profile data to a file:

      import cProfile, pstats, io
      from pstats import SortKey
      pr = cProfile.Profile()
      pr.enable()
      # ... do something ...
      pr.disable()
      s = io.StringIO()
      sortby = SortKey.CUMULATIVE
      ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
      ps.print_stats()
      print(s.getvalue())

   The "Profile" class can also be used as a context manager
   (supported only in "cProfile" module. see 上下文管理器类型):

      import cProfile

      with cProfile.Profile() as pr:
          # ... do something ...

      pr.print_stats()

   在 3.8 版更改: 添加了上下文管理器支持。

   enable()

      开始收集分析数据。仅在 "cProfile" 可用。

   disable()

      停止收集分析数据。仅在 "cProfile" 可用。

   create_stats()

      停止收集分析数据，并在内部将结果记录为当前 profile。

   print_stats(sort=-1)

      Create a "Stats" object based on the current profile and print
      the results to stdout.

   dump_stats(filename)

      将当前profile 的结果写入 *filename* 。

   run(cmd)

      Profile the cmd via "exec()".

   runctx(cmd, globals, locals)

      Profile the cmd via "exec()" with the specified global and local
      environment.

   runcall(func, *args, **kwargs)

      Profile "func(*args, **kwargs)"

Note that profiling will only work if the called command/function
actually returns.  If the interpreter is terminated (e.g. via a
"sys.exit()" call during the called command/function execution) no
profiling results will be printed.


"Stats" 类
==========

Analysis of the profiler data is done using the "Stats" class.

class pstats.Stats(*filenames or profile, stream=sys.stdout)

   This class constructor creates an instance of a "statistics object"
   from a *filename* (or list of filenames) or from a "Profile"
   instance. Output will be printed to the stream specified by
   *stream*.

   The file selected by the above constructor must have been created
   by the corresponding version of "profile" or "cProfile".  To be
   specific, there is *no* file compatibility guaranteed with future
   versions of this profiler, and there is no compatibility with files
   produced by other profilers, or the same profiler run on a
   different operating system.  If several files are provided, all the
   statistics for identical functions will be coalesced, so that an
   overall view of several processes can be considered in a single
   report.  If additional files need to be combined with data in an
   existing "Stats" object, the "add()" method can be used.

   Instead of reading the profile data from a file, a
   "cProfile.Profile" or "profile.Profile" object can be used as the
   profile data source.

   "Stats" 对象有以下方法:

   strip_dirs()

      This method for the "Stats" class removes all leading path
      information from file names.  It is very useful in reducing the
      size of the printout to fit within (close to) 80 columns.  This
      method modifies the object, and the stripped information is
      lost.  After performing a strip operation, the object is
      considered to have its entries in a "random" order, as it was
      just after object initialization and loading. If "strip_dirs()"
      causes two function names to be indistinguishable (they are on
      the same line of the same filename, and have the same function
      name), then the statistics for these two entries are accumulated
      into a single entry.

   add(*filenames)

      This method of the "Stats" class accumulates additional
      profiling information into the current profiling object.  Its
      arguments should refer to filenames created by the corresponding
      version of "profile.run()" or "cProfile.run()". Statistics for
      identically named (re: file, line, name) functions are
      automatically accumulated into single function statistics.

   dump_stats(filename)

      Save the data loaded into the "Stats" object to a file named
      *filename*.  The file is created if it does not exist, and is
      overwritten if it already exists.  This is equivalent to the
      method of the same name on the "profile.Profile" and
      "cProfile.Profile" classes.

   sort_stats(*keys)

      This method modifies the "Stats" object by sorting it according
      to the supplied criteria.  The argument can be either a string
      or a SortKey enum identifying the basis of a sort (example:
      "'time'", "'name'", "SortKey.TIME" or "SortKey.NAME"). The
      SortKey enums argument have advantage over the string argument
      in that it is more robust and less error prone.

      When more than one key is provided, then additional keys are
      used as secondary criteria when there is equality in all keys
      selected before them.  For example, "sort_stats(SortKey.NAME,
      SortKey.FILE)" will sort all the entries according to their
      function name, and resolve all ties (identical function names)
      by sorting by file name.

      For the string argument, abbreviations can be used for any key
      names, as long as the abbreviation is unambiguous.

      The following are the valid string and SortKey:

      +--------------------+-----------------------+------------------------+
      | 有效字符串参数     | 有效枚举参数          | 含义                   |
      |====================|=======================|========================|
      | "'calls'"          | SortKey.CALLS         | 调用次数               |
      +--------------------+-----------------------+------------------------+
      | "'cumulative'"     | SortKey.CUMULATIVE    | 累积时间               |
      +--------------------+-----------------------+------------------------+
      | "'cumtime'"        | N/A                   | 累积时间               |
      +--------------------+-----------------------+------------------------+
      | "'file'"           | N/A                   | 文件名                 |
      +--------------------+-----------------------+------------------------+
      | "'filename'"       | SortKey.FILENAME      | 文件名                 |
      +--------------------+-----------------------+------------------------+
      | "'module'"         | N/A                   | 文件名                 |
      +--------------------+-----------------------+------------------------+
      | "'ncalls'"         | N/A                   | 调用次数               |
      +--------------------+-----------------------+------------------------+
      | "'pcalls'"         | SortKey.PCALLS        | 原始调用计数           |
      +--------------------+-----------------------+------------------------+
      | "'line'"           | SortKey.LINE          | 行号                   |
      +--------------------+-----------------------+------------------------+
      | "'name'"           | SortKey.NAME          | 函数名称               |
      +--------------------+-----------------------+------------------------+
      | "'nfl'"            | SortKey.NFL           | 名称/文件/行           |
      +--------------------+-----------------------+------------------------+
      | "'stdname'"        | SortKey.STDNAME       | 标准名称               |
      +--------------------+-----------------------+------------------------+
      | "'time'"           | SortKey.TIME          | 内部时间               |
      +--------------------+-----------------------+------------------------+
      | "'tottime'"        | N/A                   | 内部时间               |
      +--------------------+-----------------------+------------------------+

      Note that all sorts on statistics are in descending order
      (placing most time consuming items first), where as name, file,
      and line number searches are in ascending order (alphabetical).
      The subtle distinction between "SortKey.NFL" and
      "SortKey.STDNAME" is that the standard name is a sort of the
      name as printed, which means that the embedded line numbers get
      compared in an odd way.  For example, lines 3, 20, and 40 would
      (if the file names were the same) appear in the string order 20,
      3 and 40. In contrast, "SortKey.NFL" does a numeric compare of
      the line numbers. In fact, "sort_stats(SortKey.NFL)" is the same
      as "sort_stats(SortKey.NAME, SortKey.FILENAME, SortKey.LINE)".

      For backward-compatibility reasons, the numeric arguments "-1",
      "0", "1", and "2" are permitted.  They are interpreted as
      "'stdname'", "'calls'", "'time'", and "'cumulative'"
      respectively.  If this old style format (numeric) is used, only
      one sort key (the numeric key) will be used, and additional
      arguments will be silently ignored.

      3.7 新版功能: Added the SortKey enum.

   reverse_order()

      This method for the "Stats" class reverses the ordering of the
      basic list within the object.  Note that by default ascending vs
      descending order is properly selected based on the sort key of
      choice.

   print_stats(*restrictions)

      This method for the "Stats" class prints out a report as
      described in the "profile.run()" definition.

      The order of the printing is based on the last "sort_stats()"
      operation done on the object (subject to caveats in "add()" and
      "strip_dirs()").

      The arguments provided (if any) can be used to limit the list
      down to the significant entries.  Initially, the list is taken
      to be the complete set of profiled functions.  Each restriction
      is either an integer (to select a count of lines), or a decimal
      fraction between 0.0 and 1.0 inclusive (to select a percentage
      of lines), or a string that will interpreted as a regular
      expression (to pattern match the standard name that is printed).
      If several restrictions are provided, then they are applied
      sequentially. For example:

         print_stats(.1, 'foo:')

      would first limit the printing to first 10% of list, and then
      only print functions that were part of filename ".*foo:".  In
      contrast, the command:

         print_stats('foo:', .1)

      would limit the list to all functions having file names
      ".*foo:", and then proceed to only print the first 10% of them.

   print_callers(*restrictions)

      This method for the "Stats" class prints a list of all functions
      that called each function in the profiled database.  The
      ordering is identical to that provided by "print_stats()", and
      the definition of the restricting argument is also identical.
      Each caller is reported on its own line.  The format differs
      slightly depending on the profiler that produced the stats:

      * With "profile", a number is shown in parentheses after each
        caller to show how many times this specific call was made.
        For convenience, a second non-parenthesized number repeats the
        cumulative time spent in the function at the right.

      * With "cProfile", each caller is preceded by three numbers: the
        number of times this specific call was made, and the total and
        cumulative times spent in the current function while it was
        invoked by this specific caller.

   print_callees(*restrictions)

      This method for the "Stats" class prints a list of all function
      that were called by the indicated function.  Aside from this
      reversal of direction of calls (re: called vs was called by),
      the arguments and ordering are identical to the
      "print_callers()" method.


什么是确定性性能分析？
======================

*确定性性能分析* 旨在反映这样一个事实：即所有 *函数调用* 、 *函数返回*
和 *异常* 事件都被监控，并且对这些事件之间的间隔（在此期间用户的代码正
在执行）进行精确计时。相反，统计分析（不是由该模块完成）随机采样有效指
令指针，并推断时间花费在哪里。后一种技术传统上涉及较少的开销（因为代码
不需要检测），但只提供了时间花在哪里的相对指示。

在Python中，由于在执行过程中总有一个活动的解释器，因此执行确定性评测不
需要插入指令的代码。Python 自动为每个事件提供一个:dfn:*钩子* （可选回
调）。此外，Python 的解释特性往往会给执行增加太多开销，以至于在典型的
应用程序中，确定性分析往往只会增加很小的处理开销。结果是，确定性分析并
没有那么代价高昂，但是它提供了有关 Python 程序执行的大量运行时统计信息
。

调用计数统计信息可用于识别代码中的错误（意外计数），并识别可能的内联扩
展点（高频调用）。内部时间统计可用于识别应仔细优化的 "热循环" 。累积时
间统计可用于识别算法选择上的高级别错误。请注意，该分析器中对累积时间的
异常处理，允许直接比较算法的递归实现与迭代实现的统计信息。


局限性
======

一个限制是关于时间信息的准确性。确定性性能分析存在一个涉及精度的基本问
题。最明显的限制是，底层的 "时钟" 周期大约为0.001秒（通常）。因此，没
有什么测量会比底层时钟更精确。如果进行了足够的测量，那么 "误差" 将趋于
平均。不幸的是，删除第一个错误会导致第二个错误来源。

第二个问题是，从调度事件到分析器调用获取时间函数实际 *获取* 时钟状态，
这需要 "一段时间" 。类似地，从获取时钟值（然后保存）开始，直到再次执行
用户代码为止，退出分析器事件句柄时也存在一定的延迟。因此，多次调用单个
函数或调用多个函数通常会累积此错误。尽管这种方式的误差通常小于时钟的精
度（小于一个时钟周期），但它 *可以* 累积并变得非常可观。

与开销较低的 "cProfile" 相比， "profile" 的问题更为严重。出于这个原因
， "profile" 提供了一种针对指定平台的自我校准方法，以便可以在很大程度
上（平均地）消除此误差。  校准后，结果将更准确（在最小二乘意义上），但
它有时会产生负数（当调用计数异常低，且概率之神对您不利时：-）。因此 *
不要* 对产生的负数感到惊慌。它们应该只在你手工校准分析器的情况下才会出
现，实际上结果比没有校准的情况要好。


准确估量
========

"profile" 模块的 profiler 会从每个事件处理时间中减去一个常量，以补偿调
用 time 函数和存储结果的开销。默认情况下，常数为0。对于特定的平台，可
用以下程序获得更好修正常数（ 局限性 ）。

   import profile
   pr = profile.Profile()
   for i in range(5):
       print(pr.calibrate(10000))

The method executes the number of Python calls given by the argument,
directly and again under the profiler, measuring the time for both. It
then computes the hidden overhead per profiler event, and returns that
as a float.  For example, on a 1.8Ghz Intel Core i5 running Mac OS X,
and using Python's time.process_time() as the timer, the magical
number is about 4.04e-6.

The object of this exercise is to get a fairly consistent result. If
your computer is *very* fast, or your timer function has poor
resolution, you might have to pass 100000, or even 1000000, to get
consistent results.

当你有一个一致的答案时，有三种方法可以使用：

   import profile

   # 1. Apply computed bias to all Profile instances created hereafter.
   profile.Profile.bias = your_computed_bias

   # 2. Apply computed bias to a specific Profile instance.
   pr = profile.Profile()
   pr.bias = your_computed_bias

   # 3. Specify computed bias in instance constructor.
   pr = profile.Profile(bias=your_computed_bias)

If you have a choice, you are better off choosing a smaller constant,
and then your results will "less often" show up as negative in profile
statistics.


使用自定义计时器
================

If you want to change how current time is determined (for example, to
force use of wall-clock time or elapsed process time), pass the timing
function you want to the "Profile" class constructor:

   pr = profile.Profile(your_time_func)

The resulting profiler will then call "your_time_func". Depending on
whether you are using "profile.Profile" or "cProfile.Profile",
"your_time_func"'s return value will be interpreted differently:

"profile.Profile"
   "your_time_func" should return a single number, or a list of
   numbers whose sum is the current time (like what "os.times()"
   returns).  If the function returns a single time number, or the
   list of returned numbers has length 2, then you will get an
   especially fast version of the dispatch routine.

   Be warned that you should calibrate the profiler class for the
   timer function that you choose (see 准确估量).  For most machines,
   a timer that returns a lone integer value will provide the best
   results in terms of low overhead during profiling.  ("os.times()"
   is *pretty* bad, as it returns a tuple of floating point values).
   If you want to substitute a better timer in the cleanest fashion,
   derive a class and hardwire a replacement dispatch method that best
   handles your timer call, along with the appropriate calibration
   constant.

"cProfile.Profile"
   "your_time_func" should return a single number.  If it returns
   integers, you can also invoke the class constructor with a second
   argument specifying the real duration of one unit of time.  For
   example, if "your_integer_time_func" returns times measured in
   thousands of seconds, you would construct the "Profile" instance as
   follows:

      pr = cProfile.Profile(your_integer_time_func, 0.001)

   As the "cProfile.Profile" class cannot be calibrated, custom timer
   functions should be used with care and should be as fast as
   possible.  For the best results with a custom timer, it might be
   necessary to hard-code it in the C source of the internal "_lsprof"
   module.

Python 3.3 adds several new functions in "time" that can be used to
make precise measurements of process or wall-clock time. For example,
see "time.perf_counter()".
