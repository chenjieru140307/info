Python文档内容
**************

* Python 有什么新变化？

  * Python 3.8 有什么新变化

    * 摘要 - 发布重点

    * 新的特性

      * 赋值表达式

      * 仅限位置形参

      * 用于已编译字节码文件的并行文件系统缓存

      * 调试构建使用与发布构建相同的 ABI

      * f-字符串支持 "=" 用于自动记录表达式和调试文档

      * PEP 578: Python 运行时审核钩子

      * PEP 587: Python 初始化配置

      * Vectorcall: 用于 CPython 的快速调用协议

      * 具有外部数据缓冲区的 pickle 协议 5

    * 其他语言特性修改

    * 新增模块

    * 改进的模块

      * ast

      * asyncio

      * builtins

      * collections

      * cProfile

      * csv

      * curses

      * ctypes

      * datetime

      * functools

      * gc

      * gettext

      * gzip

      * IDLE 与 idlelib

      * inspect

      * io

      * itertools

      * json.tool

      * logging

      * math

      * mmap

      * multiprocessing

      * os

      * os.path

      * pathlib

      * pickle

      * plistlib

      * pprint

      * py_compile

      * shlex

      * shutil

      * socket

      * ssl

      * statistics

      * sys

      * tarfile

      * threading

      * tokenize

      * tkinter

      * time

      * typing

      * unicodedata

      * unittest

      * venv

      * weakref

      * xml

      * xmlrpc

    * 性能优化

    * 构建和 C API 的改变

    * 弃用

    * API 与特性的移除

    * 移植到 Python 3.8

      * Python 行为的改变

      * 更改的Python API

      * C API 中的改变

      * CPython 字节码的改变

      * 演示和工具

    * Python 3.8.1 中的重要变化

    * Python 3.8.2 中的重要变化

    * Python 3.8.3 中的重要变化

  * Python 3.7 有什么新变化

    * 摘要 - 发布重点

    * 新的特性

      * PEP 563：延迟的标注求值

      * PEP 538: 传统 C 区域强制转换

      * PEP 540: 强制 UTF-8 运行时模式

      * PEP 553: 内置的 "breakpoint()"

      * PEP 539: 用于线程局部存储的新 C API

      * PEP 562: 定制对模块属性的访问

      * PEP 564: 具有纳秒级精度的新时间函数

      * PEP 565: 在 "__main__" 中显示 DeprecationWarning

      * PEP 560: 对 "typing" 模块和泛型类型的核心支持

      * PEP 552: 基于哈希值的 .pyc 文件

      * PEP 545: Python 文档翻译

      * 开发运行时模式: -X dev

    * 其他语言特性修改

    * 新增模块

      * contextvars

      * dataclasses

      * importlib.resources

    * 改进的模块

      * argparse

      * asyncio

      * binascii

      * calendar

      * collections

      * compileall

      * concurrent.futures

      * contextlib

      * cProfile

      * crypt

      * datetime

      * dbm

      * decimal

      * dis

      * distutils

      * enum

      * functools

      * gc

      * hmac

      * http.client

      * http.server

      * idlelib 与 IDLE

      * importlib

      * io

      * ipaddress

      * itertools

      * locale

      * logging

      * math

      * mimetypes

      * msilib

      * multiprocessing

      * os

      * pathlib

      * pdb

      * py_compile

      * pydoc

      * queue

      * re

      * signal

      * socket

      * socketserver

      * sqlite3

      * ssl

      * string

      * subprocess

      * sys

      * time

      * tkinter

      * tracemalloc

      * types

      * unicodedata

      * unittest

      * unittest.mock

      * urllib.parse

      * uu

      * uuid

      * warnings

      * xml.etree

      * xmlrpc.server

      * zipapp

      * zipfile

    * C API 的改变

    * 构建的改变

    * 性能优化

    * 其他 CPython 实现的改变

    * 已弃用的 Python 行为

    * 已弃用的 Python 模块、函数和方法

      * aifc

      * asyncio

      * collections

      * dbm

      * enum

      * gettext

      * importlib

      * locale

      * macpath

      * threading

      * socket

      * ssl

      * sunau

      * sys

      * wave

    * 已弃用的 C API 函数和类型

    * 平台支持的移除

    * API 与特性的移除

    * 移除的模块

    * Windows 专属的改变

    * 移植到 Python 3.7

      * Python 行为的更改

      * 更改的Python API

      * C API 中的改变

      * CPython 字节码的改变

      * Windows 专属的改变

      * 其他 CPython 实现的改变

    * Python 3.7.1 中的重要变化

    * Python 3.7.2 中的重要变化

    * Python 3.7.6 中的重要变化

  * Python 3.6 有什么新变化

    * 摘要 - 发布重点

    * 新的特性

      * PEP 498: 格式化的字符串文字

      * PEP 526: 变量注释的语法

      * PEP 515: 数字文字中的下划线。

      * PEP 525: 异步生成器

      * PEP 530: 异步推导

      * PEP 487: Simpler customization of class creation

      * PEP 487: Descriptor Protocol Enhancements

      * PEP 519: 添加文件系统路径协议

      * PEP 495: 消除本地时间的歧义

      * PEP 529: 将Windows文件系统编码更改为UTF-8

      * PEP 528: 将Windows控制台编码更改为UTF-8

      * PEP 520: 保留类属性定义顺序

      * PEP 468: 保留关键字参数顺序

      * 新的 *dict* 实现

      * PEP 523: 向CPython 添加框架评估API

      * PYTHONMALLOC 环境变量

      * DTrace 和 SystemTap 探测支持

    * 其他语言特性修改

    * 新增模块

      * secrets

    * 改进的模块

      * array

      * ast

      * asyncio

      * binascii

      * cmath

      * collections

      * concurrent.futures

      * contextlib

      * datetime

      * decimal

      * distutils

      * email

      * encodings

      * enum

      * faulthandler

      * fileinput

      * hashlib

      * http.client

      * idlelib 与 IDLE

      * importlib

      * inspect

      * json

      * logging

      * math

      * multiprocessing

      * os

      * pathlib

      * pdb

      * pickle

      * pickletools

      * pydoc

      * random

      * re

      * readline

      * rlcompleter

      * shlex

      * site

      * sqlite3

      * socket

      * socketserver

      * ssl

      * statistics

      * struct

      * subprocess

      * sys

      * telnetlib

      * time

      * timeit

      * tkinter

      * traceback

      * tracemalloc

      * typing

      * unicodedata

      * unittest.mock

      * urllib.request

      * urllib.robotparser

      * venv

      * warnings

      * winreg

      * winsound

      * xmlrpc.client

      * zipfile

      * zlib

    * 性能优化

    * 构建和 C API 的改变

    * 其他改进

    * 弃用

      * 新关键字

      * 已弃用的 Python 行为

      * 已弃用的 Python 模块、函数和方法

        * asynchat

        * asyncore

        * dbm

        * distutils

        * grp

        * importlib

        * os

        * re

        * ssl

        * tkinter

        * venv

      * 已弃用的 C API 函数和类型

      * 弃用的构建选项

    * 移除

      * API 与特性的移除

    * 移植到Python 3.6

      *  'python' 命令行为的变化

      * Python API 的更改

      * C API 中的改变

      * CPython 字节码的改变

    * Python 3.6.2 中的重要变化

      * New "make regen-all" build target

      * Removal of "make touch" build target

    * Python 3.6.4 中的重要变化

    * Python 3.6.5 中的重要变化

    * Python 3.6.7 中的重要变化

    * Python 3.6.10 中的重要变化

  * Python 3.5 有什么新变化

    * 摘要 - 发布重点

    * 新的特性

      * PEP 492 - 使用 async 和 await 语法实现协程

      * PEP 465 - 用于矩阵乘法的专用中缀运算符

      * PEP 448 - Additional Unpacking Generalizations

      * PEP 461 - percent formatting support for bytes and bytearray

      * PEP 484 - 类型提示

      * PEP 471 - os.scandir() function -- a better and faster
        directory iterator

      * PEP 475: Retry system calls failing with EINTR

      * PEP 479: Change StopIteration handling inside generators

      * PEP 485: A function for testing approximate equality

      * PEP 486: Make the Python Launcher aware of virtual
        environments

      * PEP 488: Elimination of PYO files

      * PEP 489: Multi-phase extension module initialization

    * 其他语言特性修改

    * 新增模块

      * typing

      * zipapp

    * 改进的模块

      * argparse

      * asyncio

      * bz2

      * cgi

      * cmath

      * code

      * collections

      * collections.abc

      * compileall

      * concurrent.futures

      * configparser

      * contextlib

      * csv

      * curses

      * dbm

      * difflib

      * distutils

      * doctest

      * email

      * enum

      * faulthandler

      * functools

      * glob

      * gzip

      * heapq

      * http

      * http.client

      * idlelib 与 IDLE

      * imaplib

      * imghdr

      * importlib

      * inspect

      * io

      * ipaddress

      * json

      * linecache

      * locale

      * logging

      * lzma

      * math

      * multiprocessing

      * operator

      * os

      * pathlib

      * pickle

      * poplib

      * re

      * readline

      * selectors

      * shutil

      * signal

      * smtpd

      * smtplib

      * sndhdr

      * socket

      * ssl

        * Memory BIO Support

        * Application-Layer Protocol Negotiation Support

        * Other Changes

      * sqlite3

      * subprocess

      * sys

      * sysconfig

      * tarfile

      * threading

      * time

      * timeit

      * tkinter

      * traceback

      * types

      * unicodedata

      * unittest

      * unittest.mock

      * urllib

      * wsgiref

      * xmlrpc

      * xml.sax

      * zipfile

    * 其他模块级更改

    * 性能优化

    * 构建和 C API 的改变

    * 弃用

      * 新关键字

      * 已弃用的 Python 行为

      * 不支持的操作系统

      * 已弃用的 Python 模块、函数和方法

    * 移除

      * API 与特性的移除

    * 移植到Python 3.5

      * Python 行为的改变

      * 改变了的Python API

      * C API 中的改变

    * Python 3.5.4 的显著变化

      * New "make regen-all" build target

      * Removal of "make touch" build target

  * Python 3.4 有什么新变化

    * 摘要 - 发布重点

    * 新的特性

      * PEP 453: Explicit Bootstrapping of PIP in Python Installations

        * Bootstrapping pip By Default

        * 文档更改

      * PEP 446: Newly Created File Descriptors Are Non-Inheritable

      * Improvements to Codec Handling

      * PEP 451: A ModuleSpec Type for the Import System

      * 其他语言特性修改

    * 新增模块

      * asyncio

      * ensurepip

      * enum

      * pathlib

      * selectors

      * statistics

      * tracemalloc

    * 改进的模块

      * abc

      * aifc

      * argparse

      * audioop

      * base64

      * collections

      * colorsys

      * contextlib

      * dbm

      * dis

      * doctest

      * email

      * filecmp

      * functools

      * gc

      * glob

      * hashlib

      * hmac

      * html

      * http

      * idlelib 与 IDLE

      * importlib

      * inspect

      * ipaddress

      * logging

      * marshal

      * mmap

      * multiprocessing

      * operator

      * os

      * pdb

      * pickle

      * plistlib

      * poplib

      * pprint

      * pty

      * pydoc

      * re

      * resource

      * select

      * shelve

      * shutil

      * smtpd

      * smtplib

      * socket

      * sqlite3

      * ssl

      * stat

      * struct

      * subprocess

      * sunau

      * sys

      * tarfile

      * textwrap

      * threading

      * traceback

      * types

      * urllib

      * unittest

      * venv

      * wave

      * weakref

      * xml.etree

      * zipfile

    * CPython Implementation Changes

      * PEP 445: Customization of CPython Memory Allocators

      * PEP 442: Safe Object Finalization

      * PEP 456: Secure and Interchangeable Hash Algorithm

      * PEP 436: Argument Clinic

      * Other Build and C API Changes

      * 其他改进

      * Significant Optimizations

    * 弃用

      * Deprecations in the Python API

      * Deprecated Features

    * 移除

      * 不再支持的操作系统

      * API 与特性的移除

      * Code Cleanups

    * 移植到 Python 3.4

      *  'python' 命令行为的变化

      * Python API 的变化

      * C API 的变化

    * 3.4.3 的变化

      * PEP 476: Enabling certificate verification by default for
        stdlib http clients

  * Python 3.3 有什么新变化

    * 摘要 - 发布重点

    * PEP 405: 虚拟环境

    * PEP 420: 隐式命名空间包

    * PEP 3118: 新的内存视图实现和缓冲协议文档

      * 相关特性

      * API changes

    * PEP 393: 灵活的字符串表示

      * Functionality

      * Performance and resource usage

    * PEP 397: 适用于Windows的Python启动器

    * PEP 3151: 重写 OS 和 IO 异常的层次结构

    * PEP 380: 委托给子生成器的语法

    * PEP 409: 清除异常上下文

    * PEP 414: 显式的Unicode文本

    * PEP 3155: 类和函数的限定名称

    * PEP 412: Key-Sharing Dictionary

    * PEP 362: 函数签名对象

    * PEP 421: 添加 sys.implementation

      * SimpleNamespace

    * Using importlib as the Implementation of Import

      * New APIs

      * Visible Changes

    * 其他语言特性修改

    * A Finer-Grained Import Lock

    * Builtin functions and types

    * 新增模块

      * faulthandler

      * ipaddress

      * lzma

    * 改进的模块

      * abc

      * array

      * base64

      * binascii

      * bz2

      * codecs

      * collections

      * contextlib

      * crypt

      * curses

      * datetime

      * decimal

        * 相关特性

        * API changes

      * email

        * Policy Framework

        * Provisional Policy with New Header API

        * Other API Changes

      * ftplib

      * functools

      * gc

      * hmac

      * http

      * html

      * imaplib

      * inspect

      * io

      * itertools

      * logging

      * math

      * mmap

      * multiprocessing

      * nntplib

      * os

      * pdb

      * pickle

      * pydoc

      * re

      * sched

      * select

      * shlex

      * shutil

      * signal

      * smtpd

      * smtplib

      * socket

      * socketserver

      * sqlite3

      * ssl

      * stat

      * struct

      * subprocess

      * sys

      * tarfile

      * tempfile

      * textwrap

      * threading

      * time

      * types

      * unittest

      * urllib

      * webbrowser

      * xml.etree.ElementTree

      * zlib

    * 性能优化

    * 构建和 C API 的改变

    * 弃用

      * 不支持的操作系统

      * 已弃用的 Python 模块、函数和方法

      * 已弃用的 C API 函数和类型

      * 弃用的功能

    * 移植到 Python 3.3

      * Porting Python code

      * Porting C code

      * Building C extensions

      * Command Line Switch Changes

  * Python 3.2 有什么新变化

    * PEP 384: 定义稳定的ABI

    * PEP 389: Argparse 命令行解析模块

    * PEP 391:  基于字典的日志配置

    * PEP 3148:  "concurrent.futures" 模块

    * PEP 3147:  PYC 仓库目录

    * PEP 3149: ABI Version Tagged .so Files

    * PEP 3333: Python Web服务器网关接口v1.0.1

    * 其他语言特性修改

    * 新增，改进和弃用的模块

      * email

      * elementtree

      * functools

      * itertools

      * collections

      * threading

      * datetime 和 time

      * math

      * abc

      * io

      * reprlib

      * logging

      * csv

      * contextlib

      * decimal and fractions

      * ftp

      * popen

      * select

      * gzip 和 zipfile

      * tarfile

      * hashlib

      * ast

      * os

      * shutil

      * sqlite3

      * html

      * socket

      * ssl

      * nntp

      * certificates

      * imaplib

      * http.client

      * unittest

      * random

      * poplib

      * asyncore

      * tempfile

      * inspect

      * pydoc

      * dis

      * dbm

      * ctypes

      * site

      * sysconfig

      * pdb

      * configparser

      * urllib.parse

      * mailbox

      * turtledemo

    * 多线程

    * 性能优化

    * Unicode

    * 编解码器

    * 文档

    * IDLE

    * 代码库

    * 构建和 C API 的改变

    * 移植到 Python 3.2

  * Python 3.1 有什么新变化

    * PEP 372: 有序字典

    * PEP 378: 千位分隔符的格式说明符

    * 其他语言特性修改

    * 新增，改进和弃用的模块

    * 性能优化

    * IDLE

    * 构建和 C API 的改变

    * 移植到 Python 3.1

  * Python 3.0 有什么新变化

    * 常见的绊脚石

      * Print Is A Function

      * Views And Iterators Instead Of Lists

      * Ordering Comparisons

      * 整数

      * Text Vs. Data Instead Of Unicode Vs. 8-bit

    * Overview Of Syntax Changes

      * 新语法

      * 修改的语法

      * 移除的语法

    * Changes Already Present In Python 2.6

    * Library Changes

    * **PEP 3101**: A New Approach To String Formatting

    * Changes To Exceptions

    * Miscellaneous Other Changes

      * Operators And Special Methods

      * Builtins

    * 构建和 C API 的改变

    * 性能

    * 移植到 Python 3.0

  * Python 2.7 有什么新变化

    * Python 2.x的未来

    * Changes to the Handling of Deprecation Warnings

    * Python 3.1 Features

    * PEP 372: Adding an Ordered Dictionary to collections

    * PEP 378: 千位分隔符的格式说明符

    * PEP 389: The argparse Module for Parsing Command Lines

    * PEP 391: Dictionary-Based Configuration For Logging

    * PEP 3106: Dictionary Views

    * PEP 3137: The memoryview Object

    * 其他语言特性修改

      * Interpreter Changes

      * 性能优化

    * 新增和改进的模块

      * 新模块：importlib

      * 新模块：sysconfig

      * ttk: Themed Widgets for Tk

      * 更新的模块：unittest

      * 更新的模块：ElementTree 1.3

    * 构建和 C API 的改变

      * 胶囊

      * 特定于 Windows 的更改：

      * 特定于 Mac OS X 的更改：

      * 特定于 FreeBSD 的更改：

    * Other Changes and Fixes

    * 移植到 Python 2.7

    * New Features Added to Python 2.7 Maintenance Releases

      * Two new environment variables for debug mode

      * PEP 434: IDLE Enhancement Exception for All Branches

      * PEP 466: Network Security Enhancements for Python 2.7

      * PEP 477: Backport ensurepip (PEP 453) to Python 2.7

        * Bootstrapping pip By Default

        * 文档更改

      * PEP 476: Enabling certificate verification by default for
        stdlib http clients

      * PEP 493：适用于Python 2.7 的 HTTPS 验证迁移工具

      * New "make regen-all" build target

      * Removal of "make touch" build target

    * 致谢

  * Python 2.6 有什么新变化

    * Python 3.0

    * 开发过程的变化

      * New Issue Tracker: Roundup

      * 新的文档格式：使用 Sphinx 的 reStructuredText

    * PEP 343: "with" 语句

      * Writing Context Managers

      * contextlib 模块

    * PEP 366: 从主模块显式相对导入

    * PEP 370: 分用户的 site-packages 目录

    * PEP 371: 多任务处理包

    * PEP 3101: 高级字符串格式

    * PEP 3105: "print" 改为函数

    * PEP 3110: 异常处理的变更

    * PEP 3112: 字节字面值

    * PEP 3116: 新 I/O 库

    * PEP 3118: 修改缓冲区协议

    * PEP 3119: 抽象基类

    * PEP 3127: 整型文字支持和语法

    * PEP 3129: 类装饰器

    * PEP 3141: A Type Hierarchy for Numbers

      * "fractions" 模块

    * 其他语言特性修改

      * 性能优化

      * Interpreter Changes

    * 新增和改进的模块

      * "ast" 模块

      * "future_builtins" 模块

      * The "json" module: JavaScript Object Notation

      * "plistlib" 模块：属性列表解析器

      * ctypes Enhancements

      * Improved SSL Support

    * Deprecations and Removals

    * 构建和 C API 的改变

      * 特定于 Windows 的更改：

      * 特定于 Mac OS X 的更改：Mac OS X

      * 特定于 IRIX 的更改：

    * 移植到Python 2.6

    * 致谢

  * Python 2.5 有什么新变化

    * PEP 308: 条件表达式

    * PEP 309: Partial Function Application

    * PEP 314: Python软件包的元数据 v1.1

    * PEP 328: 绝对导入和相对导入

    * PEP 338: 将模块作为脚本执行

    * PEP 341: 统一 try/except/finally

    * PEP 342: 生成器的新特性

    * PEP 343: "with" 语句

      * Writing Context Managers

      * contextlib 模块

    * PEP 352: 异常作为新型的类

    * PEP 353: 使用ssize_t作为索引类型

    * PEP 357: '__index__' 方法

    * 其他语言特性修改

      * 交互解释器变更

      * 性能优化

    * 新增，改进和删除的模块

      * ctypes 包

      * ElementTree 包

      * hashlib 包

      * sqlite3 包

      * wsgiref 包

    * 构建和 C API 的改变

      * Port-Specific Changes

    * 移植到Python 2.5

    * 致谢

  * Python 2.4 有什么新变化

    * PEP 218: 内置集合对象

    * PEP 237: 统一长整数和整数

    * PEP 289: 生成器表达式

    * PEP 292: Simpler String Substitutions

    * PEP 318: Decorators for Functions and Methods

    * PEP 322: 反向迭代

    * PEP 324: 新的子进程模块

    * PEP 327: 十进制数据类型

      * 为什么需要十进制？

      * "Decimal" 类型

      * "Context" 类型

    * PEP 328: 多行导入

    * PEP 331: Locale-Independent Float/String Conversions

    * 其他语言特性修改

      * 性能优化

    * 新增，改进和弃用的模块

      * cookielib

      * doctest

    * 构建和 C API 的改变

      * Port-Specific Changes

    * 移植到 Python 2.4

    * 致谢

  * Python 2.3 有什么新变化

    * PEP 218: A Standard Set Datatype

    * PEP 255: Simple Generators

    * PEP 263: Source Code Encodings

    * PEP 273: 从ZIP压缩包导入模块

    * PEP 277: Unicode file name support for Windows NT

    * PEP 278: 通用换行支持

    * PEP 279: enumerate()

    * PEP 282: logging 包

    * PEP 285: 布尔类型

    * PEP 293: Codec Error Handling Callbacks

    * PEP 301: Distutils的软件包索引和元数据

    * PEP 302: New Import Hooks

    * PEP 305: 逗号分隔文件

    * PEP 307: Pickle Enhancements

    * 扩展切片

    * 其他语言特性修改

      * String Changes

      * 性能优化

    * 新增，改进和弃用的模块

      * Date/Time 类型

      * optparse 模块

    * Pymalloc: A Specialized Object Allocator

    * 构建和 C API 的改变

      * Port-Specific Changes

    * Other Changes and Fixes

    * 移植到 Python 2.3

    * 致谢

  * Python 2.2 有什么新变化

    * 概述

    * PEPs 252 and 253: Type and Class Changes

      * Old and New Classes

      * Descriptors

      * Multiple Inheritance: The Diamond Rule

      * Attribute Access

      * Related Links

    * PEP 234: Iterators

    * PEP 255: Simple Generators

    * PEP 237: 统一长整数和整数

    * PEP 238: Changing the Division Operator

    * Unicode Changes

    * PEP 227: Nested Scopes

    * 新增和改进的模块

    * Interpreter Changes and Fixes

    * Other Changes and Fixes

    * 致谢

  * Python 2.1 有什么新变化

    * 概述

    * PEP 227: Nested Scopes

    * PEP 236: __future__ Directives

    * PEP 207: Rich Comparisons

    * PEP 230: Warning Framework

    * PEP 229: New Build System

    * PEP 205: Weak References

    * PEP 232: Function Attributes

    * PEP 235: Importing Modules on Case-Insensitive Platforms

    * PEP 217: Interactive Display Hook

    * PEP 208: New Coercion Model

    * PEP 241: Metadata in Python Packages

    * 新增和改进的模块

    * Other Changes and Fixes

    * 致谢

  * Python 2.0 有什么新变化

    * 概述

    * What About Python 1.6?

    * 新开发流程

    * Unicode

    * 列表推导式

    * Augmented Assignment

    * 字符串的方法

    * Garbage Collection of Cycles

    * 其他核心变化

      * Minor Language Changes

      * Changes to Built-in Functions

    * 移植 Python 2.0

    * 扩展/嵌入更改

    * Distutils：使模块易于安装

    * XML 模块

      * SAX2 Support

      * DOM Support

      * Relationship to PyXML

    * 模块更改

    * 新增模块

    * IDLE 改进

    * 删除和弃用的模块

    * 致谢

  * 更新日志

    * Python 下一版

      * 核心与内置

      * 库

      * 文档

      * Windows

      * C API

    * Python 3.8.3 release candidate 1

      * 安全

      * 核心与内置

      * 库

      * 文档

      * 测试

      * 构建

      * Windows

      * macOS

      * IDLE

      * 工具/示例

      * C API

    * Python 3.8.2 正式版

      * 核心与内置

      * 库

      * 文档

      * IDLE

    * Python 3.8.2 rc2

      * 安全

      * 核心与内置

      * 库

      * IDLE

    * Python 3.8.2 rc1

      * 安全

      * 核心与内置

      * 库

      * 文档

      * 测试

      * 构建

      * Windows

      * IDLE

    * Python 3.8.1 正式版

      * 核心与内置

      * 库

      * 测试

      * Windows

      * macOS

      * IDLE

    * Python 3.8.1 rc1

      * 安全

      * 核心与内置

      * 库

      * 文档

      * 测试

      * 构建

      * Windows

      * macOS

      * IDLE

      * C API

    * Python 3.8.0 正式版

      * 核心与内置

      * 库

      * 文档

      * 测试

      * Windows

      * IDLE

      * 工具/示例

      * C API

    * Python 3.8.0 rc1

      * 安全

      * 核心与内置

      * 库

      * 文档

      * 测试

      * 构建

      * Windows

      * macOS

      * IDLE

      * 工具/示例

      * C API

    * Python 3.8.0 beta 4

      * 安全

      * 核心与内置

      * 库

      * 文档

      * 测试

      * 构建

      * Windows

      * macOS

      * IDLE

      * 工具/示例

      * C API

    * Python 3.8.0 beta 3

      * 安全

      * 核心与内置

      * 库

      * 文档

      * 测试

      * 构建

      * Windows

      * IDLE

      * 工具/示例

    * Python 3.8.0 beta 2

      * 安全

      * 核心与内置

      * 库

      * 文档

      * 测试

      * 构建

      * Windows

      * macOS

      * IDLE

      * C API

    * Python 3.8.0 beta 1

      * 安全

      * 核心与内置

      * 库

      * 文档

      * 测试

      * 构建

      * Windows

      * macOS

      * IDLE

      * 工具/示例

      * C API

    * Python 3.8.0 alpha 4

      * 安全

      * 核心与内置

      * 库

      * 文档

      * 测试

      * 构建

      * Windows

      * macOS

      * IDLE

      * 工具/示例

      * C API

    * Python 3.8.0 alpha 3

      * 安全

      * 核心与内置

      * 库

      * 文档

      * 测试

      * 构建

      * Windows

      * IDLE

      * 工具/示例

      * C API

    * Python 3.8.0 alpha 2

      * 核心与内置

      * 库

      * 文档

      * 测试

      * Windows

      * IDLE

    * Python 3.8.0 alpha 1

      * 安全

      * 核心与内置

      * 库

      * 文档

      * 测试

      * 构建

      * Windows

      * macOS

      * IDLE

      * 工具/示例

      * C API

    * Python 3.7.0 正式版

      * 库

      * C API

    * Python 3.7.0 rc1

      * 核心与内置

      * 库

      * 文档

      * 构建

      * Windows

      * IDLE

    * Python 3.7.0 beta 5

      * 核心与内置

      * 库

      * 文档

      * 测试

      * 构建

      * macOS

      * IDLE

    * Python 3.7.0 beta 4

      * 核心与内置

      * 库

      * 文档

      * 测试

      * 构建

      * Windows

      * macOS

      * IDLE

      * 工具/示例

    * Python 3.7.0 beta 3

      * 安全

      * 核心与内置

      * 库

      * 文档

      * 测试

      * 构建

      * Windows

      * macOS

      * IDLE

      * 工具/示例

      * C API

    * Python 3.7.0 beta 2

      * 安全

      * 核心与内置

      * 库

      * 文档

      * 测试

      * 构建

      * Windows

      * macOS

      * IDLE

      * 工具/示例

    * Python 3.7.0 beta 1

      * 核心与内置

      * 库

      * 文档

      * 测试

      * 构建

      * Windows

      * macOS

      * C API

    * Python 3.7.0 alpha 4

      * 核心与内置

      * 库

      * 文档

      * 测试

      * Windows

      * 工具/示例

      * C API

    * Python 3.7.0 alpha 3

      * 核心与内置

      * 库

      * 文档

      * 测试

      * 构建

      * Windows

      * macOS

      * IDLE

      * 工具/示例

      * C API

    * Python 3.7.0 alpha 2

      * 核心与内置

      * 库

      * 文档

      * 构建

      * IDLE

      * C API

    * Python 3.7.0 alpha 1

      * 安全

      * 核心与内置

      * 库

      * 文档

      * 测试

      * 构建

      * Windows

      * IDLE

      * 工具/示例

      * C API

    * Python 3.6.6 正式版

    * Python 3.6.6 rc1

      * 核心与内置

      * 库

      * 文档

      * 测试

      * 构建

      * Windows

      * macOS

      * IDLE

      * 工具/示例

      * C API

    * Python 3.6.5 正式版

      * 测试

      * 构建

    * Python 3.6.5 rc1

      * 安全

      * 核心与内置

      * 库

      * 文档

      * 测试

      * 构建

      * Windows

      * macOS

      * IDLE

      * 工具/示例

      * C API

    * Python 3.6.4 正式版

    * Python 3.6.4 rc1

      * 核心与内置

      * 库

      * 文档

      * 测试

      * 构建

      * Windows

      * macOS

      * IDLE

      * 工具/示例

      * C API

    * Python 3.6.3 正式版

      * 库

      * 构建

    * Python 3.6.3 rc1

      * 安全

      * 核心与内置

      * 库

      * 文档

      * 测试

      * 构建

      * Windows

      * IDLE

      * 工具/示例

    * Python 3.6.2 正式版

    * Python 3.6.2 rc2

      * 安全

    * Python 3.6.2 rc1

      * 核心与内置

      * 库

      * 安全

      * 库

      * IDLE

      * C API

      * 构建

      * 文档

      * 工具/示例

      * 测试

      * Windows

    * Python 3.6.1 正式版

      * 核心与内置

      * 构建

    * Python 3.6.1 rc1

      * 核心与内置

      * 库

      * IDLE

      * Windows

      * C API

      * 文档

      * 测试

      * 构建

    * Python 3.6.0 正式版

    * Python 3.6.0 rc2

      * 核心与内置

      * 工具/示例

      * Windows

      * 构建

    * Python 3.6.0 rc1

      * 核心与内置

      * 库

      * C API

      * 文档

      * 工具/示例

    * Python 3.6.0 beta 4

      * 核心与内置

      * 库

      * 文档

      * 测试

      * 构建

    * Python 3.6.0 beta 3

      * 核心与内置

      * 库

      * Windows

      * 构建

      * 测试

    * Python 3.6.0 beta 2

      * 核心与内置

      * 库

      * Windows

      * C API

      * 构建

      * 测试

    * Python 3.6.0 beta 1

      * 核心与内置

      * 库

      * IDLE

      * C API

      * 测试

      * 构建

      * 工具/示例

      * Windows

    * Python 3.6.0 alpha 4

      * 核心与内置

      * 库

      * IDLE

      * 测试

      * Windows

      * 构建

    * Python 3.6.0 alpha 3

      * 核心与内置

      * 库

      * 安全

      * 库

      * 安全

      * 库

      * IDLE

      * C API

      * 构建

      * 工具/示例

      * 文档

      * 测试

    * Python 3.6.0 alpha 2

      * 核心与内置

      * 库

      * 安全

      * 库

      * 安全

      * 库

      * IDLE

      * 文档

      * 测试

      * Windows

      * 构建

      * Windows

      * C API

      * 工具/示例

    * Python 3.6.0 alpha 1

      * 核心与内置

      * 库

      * 安全

      * 库

      * 安全

      * 库

      * 安全

      * 库

      * IDLE

      * 文档

      * 测试

      * 构建

      * Windows

      * 工具/示例

      * C API

    * Python 3.5.5 正式版

    * Python 3.5.5 rc1

      * 安全

      * 核心与内置

      * 库

    * Python 3.5.4 正式版

      * 库

    * Python 3.5.4 rc1

      * 安全

      * 核心与内置

      * 库

      * 文档

      * 测试

      * 构建

      * Windows

      * C API

    * Python 3.5.3 正式版

    * Python 3.5.3 rc1

      * 核心与内置

      * 库

      * 安全

      * 库

      * 安全

      * 库

      * IDLE

      * C API

      * 文档

      * 测试

      * 工具/示例

      * Windows

      * 构建

    * Python 3.5.2 正式版

      * 核心与内置

      * 测试

      * IDLE

    * Python 3.5.2 rc1

      * 核心与内置

      * 安全

      * 库

      * 安全

      * 库

      * 安全

      * 库

      * 安全

      * 库

      * 安全

      * 库

      * IDLE

      * 文档

      * 测试

      * 构建

      * Windows

      * 工具/示例

      * Windows

    * Python 3.5.1 正式版

      * 核心与内置

      * Windows

    * Python 3.5.1 rc1

      * 核心与内置

      * 库

      * IDLE

      * 文档

      * 测试

      * 构建

      * Windows

      * 工具/示例

    * Python 3.5.0 正式版

      * 构建

    * Python 3.5.0 rc4

      * 库

      * 构建

    * Python 3.5.0 rc3

      * 核心与内置

      * 库

    * Python 3.5.0 rc2

      * 核心与内置

      * 库

    * Python 3.5.0 rc1

      * 核心与内置

      * 库

      * IDLE

      * 文档

      * 测试

    * Python 3.5.0 beta 4

      * 核心与内置

      * 库

      * 构建

    * Python 3.5.0 beta 3

      * 核心与内置

      * 库

      * 测试

      * 文档

      * 构建

    * Python 3.5.0 beta 2

      * 核心与内置

      * 库

    * Python 3.5.0 beta 1

      * 核心与内置

      * 库

      * IDLE

      * 测试

      * 文档

      * 工具/示例

    * Python 3.5.0 alpha 4

      * 核心与内置

      * 库

      * 构建

      * 测试

      * 工具/示例

      * C API

    * Python 3.5.0 alpha 3

      * 核心与内置

      * 库

      * 构建

      * 测试

      * 工具/示例

    * Python 3.5.0 alpha 2

      * 核心与内置

      * 库

      * 构建

      * C API

      * Windows

    * Python 3.5.0 alpha 1

      * 核心与内置

      * 库

      * IDLE

      * 构建

      * C API

      * 文档

      * 测试

      * 工具/示例

      * Windows

* Python 教程

  * 1. 课前甜点

  * 2. 使用 Python 解释器

    * 2.1. 调用解释器

      * 2.1.1. 传入参数

      * 2.1.2. 交互模式

    * 2.2. 解释器的运行环境

      * 2.2.1. 源文件的字符编码

  * 3. Python 的非正式介绍

    * 3.1. Python 作为计算器使用

      * 3.1.1. 数字

      * 3.1.2. 字符串

      * 3.1.3. 列表

    * 3.2. 走向编程的第一步

  * 4. 其他流程控制工具

    * 4.1. "if" 语句

    * 4.2. "for" 语句

    * 4.3. "range()" 函数

    * 4.4. "break" 和 "continue" 语句，以及循环中的 "else" 子句

    * 4.5. "pass" 语句

    * 4.6. 定义函数

    * 4.7. 函数定义的更多形式

      * 4.7.1. 参数默认值

      * 4.7.2. 关键字参数

      * 4.7.3. 特殊参数

        * 4.7.3.1. 位置或关键字参数

        * 4.7.3.2. 仅限位置参数

        * 4.7.3.3. 仅限关键字参数

        * 4.7.3.4. 函数示例

        * 4.7.3.5. 概括

      * 4.7.4. 任意的参数列表

      * 4.7.5. 解包参数列表

      * 4.7.6. Lambda 表达式

      * 4.7.7. 文档字符串

      * 4.7.8. 函数标注

    * 4.8. 小插曲：编码风格

  * 5. 数据结构

    * 5.1. 列表的更多特性

      * 5.1.1. 列表作为栈使用

      * 5.1.2. 列表作为队列使用

      * 5.1.3. 列表推导式

      * 5.1.4. 嵌套的列表推导式

    * 5.2. "del" 语句

    * 5.3. 元组和序列

    * 5.4. 集合

    * 5.5. 字典

    * 5.6. 循环的技巧

    * 5.7. 深入条件控制

    * 5.8. 序列和其它类型的比较

  * 6. 模块

    * 6.1. 有关模块的更多信息

      * 6.1.1. 以脚本的方式执行模块

      * 6.1.2. 模块搜索路径

      * 6.1.3. “编译过的”Python文件

    * 6.2. 标准模块

    * 6.3. "dir()" 函数

    * 6.4. 包

      * 6.4.1. 从包中导入 *

      * 6.4.2. 子包参考

      * 6.4.3. 多个目录中的包

  * 7. 输入输出

    * 7.1. 更漂亮的输出格式

      * 7.1.1. 格式化字符串文字

      * 7.1.2. 字符串的 format() 方法

      * 7.1.3. 手动格式化字符串

      * 7.1.4. 旧的字符串格式化方法

    * 7.2. 读写文件

      * 7.2.1. 文件对象的方法

      * 7.2.2. 使用 "json" 保存结构化数据

  * 8. 错误和异常

    * 8.1. 语法错误

    * 8.2. 异常

    * 8.3. 处理异常

    * 8.4. 抛出异常

    * 8.5. 用户自定义异常

    * 8.6. 定义清理操作

    * 8.7. 预定义的清理操作

  * 9. 类

    * 9.1. 名称和对象

    * 9.2. Python 作用域和命名空间

      * 9.2.1. 作用域和命名空间示例

    * 9.3. 初探类

      * 9.3.1. 类定义语法

      * 9.3.2. 类对象

      * 9.3.3. 实例对象

      * 9.3.4. 方法对象

      * 9.3.5. 类和实例变量

    * 9.4. 补充说明

    * 9.5. 继承

      * 9.5.1. 多重继承

    * 9.6. 私有变量

    * 9.7. 杂项说明

    * 9.8. 迭代器

    * 9.9. 生成器

    * 9.10. 生成器表达式

  * 10. 标准库简介

    * 10.1. 操作系统接口

    * 10.2. 文件通配符

    * 10.3. 命令行参数

    * 10.4. 错误输出重定向和程序终止

    * 10.5. 字符串模式匹配

    * 10.6. 数学

    * 10.7. 互联网访问

    * 10.8. 日期和时间

    * 10.9. 数据压缩

    * 10.10. 性能测量

    * 10.11. 质量控制

    * 10.12. 自带电池

  * 11. 标准库简介 —— 第二部分

    * 11.1. 格式化输出

    * 11.2. 模板

    * 11.3. 使用二进制数据记录格式

    * 11.4. 多线程

    * 11.5. 日志

    * 11.6. 弱引用

    * 11.7. 用于操作列表的工具

    * 11.8. 十进制浮点运算

  * 12. 虚拟环境和包

    * 12.1. 概述

    * 12.2. 创建虚拟环境

    * 12.3. 使用pip管理包

  * 13. 接下来？

  * 14. 交互式编辑和编辑历史

    * 14.1. Tab 补全和编辑历史

    * 14.2. 默认交互式解释器的替代品

  * 15. 浮点算术：争议和限制

    * 15.1. 表示性错误

  * 16. 附录

    * 16.1. 交互模式

      * 16.1.1. 错误处理

      * 16.1.2. 可执行的Python脚本

      * 16.1.3. 交互式启动文件

      * 16.1.4. 定制模块

* 安装和使用 Python

  * 1. 命令行与环境

    * 1.1. 命令行

      * 1.1.1. 接口选项

      * 1.1.2. 通用选项

      * 1.1.3. 其他选项

      * 1.1.4. 不应当使用的选项

    * 1.2. 环境变量

      * 1.2.1. 调试模式变量

  * 2. 在Unix平台中使用Python

    * 2.1. 获取最新版本的Python

      * 2.1.1. 在Linux中

      * 2.1.2. 在FreeBSD和OpenBSD上

      * 2.1.3. 在OpenSolaris系统上

    * 2.2. 构建Python

    * 2.3. 与Python相关的路径和文件

    * 2.4. 杂项

  * 3. 在Windows上使用 Python

    * 3.1. 完整安装程序

      * 3.1.1. 安装步骤

      * 3.1.2. 删除 MAX_PATH 限制

      * 3.1.3. 无UI 安装

      * 3.1.4. 免下载安装

      * 3.1.5. 修改安装

    * 3.2. Microsoft Store包

      * 3.2.1. 已知的问题

    * 3.3. nuget.org 安装包

    * 3.4. 可嵌入的包

      * 3.4.1. Python应用程序

      * 3.4.2. 嵌入Python

    * 3.5. 替代捆绑包

    * 3.6. 配置Python

      * 3.6.1. 附录：设置环境变量

      * 3.6.2. 查找Python可执行文件

    * 3.7. UTF-8 模式

    * 3.8. 适用于Windows的Python启动器

      * 3.8.1. 入门

        * 3.8.1.1. 从命令行

        * 3.8.1.2. 从虚拟环境

        * 3.8.1.3. 从脚本

        * 3.8.1.4. 从文件关联

      * 3.8.2. Shebang Lines

      * 3.8.3. shebang lines 的参数

      * 3.8.4. 自定义

        * 3.8.4.1. 通过INI文件自定义

        * 3.8.4.2. 自定义默认的Python版本

      * 3.8.5. 诊断

    * 3.9. 查找模块

    * 3.10. 附加模块

      * 3.10.1. PyWin32

      * 3.10.2. cx_Freeze

      * 3.10.3. WConio

    * 3.11. 在Windows上编译Python

    * 3.12. 其他平台

  * 4. 在苹果系统上使用 Python

    * 4.1. 获取和安装 MacPython

      * 4.1.1. 如何运行 Python 脚本

      * 4.1.2. 运行有图形界面的脚本

      * 4.1.3. 配置

    * 4.2. IDE

    * 4.3. 安装额外的 Python 包

    * 4.4. Mac 上的图形界面编程

    * 4.5. 在 Mac 上分发 Python 应用程序

    * 4.6. 其他资源

  * 5. 编辑器和集成开发环境

* Python 语言参考

  * 1. 概述

    * 1.1. 其他实现

    * 1.2. 标注

  * 2. 词法分析

    * 2.1. 行结构

      * 2.1.1. 逻辑行

      * 2.1.2. 物理行

      * 2.1.3. 注释

      * 2.1.4. 编码声明

      * 2.1.5. 显式的行拼接

      * 2.1.6. 隐式的行拼接

      * 2.1.7. 空白行

      * 2.1.8. 缩进

      * 2.1.9. 形符之间的空白

    * 2.2. 其他形符

    * 2.3. 标识符和关键字

      * 2.3.1. 关键字

      * 2.3.2. 保留的标识符类

    * 2.4. 字面值

      * 2.4.1. 字符串和字节串字面值

      * 2.4.2. 字符串字面值拼接

      * 2.4.3. 格式化字符串字面值

      * 2.4.4. 数字字面值

      * 2.4.5. 整型数字面值

      * 2.4.6. 浮点数字面值

      * 2.4.7. 虚数字面值

    * 2.5. 运算符

    * 2.6. 分隔符

  * 3. 数据模型

    * 3.1. 对象、值与类型

    * 3.2. 标准类型层级结构

    * 3.3. 特殊方法名称

      * 3.3.1. 基本定制

      * 3.3.2. 自定义属性访问

        * 3.3.2.1. 自定义模块属性访问

        * 3.3.2.2. 实现描述器

        * 3.3.2.3. 发起调用描述器

        * 3.3.2.4. __slots__

          * 3.3.2.4.1. 使用 *__slots__* 的注意事项

      * 3.3.3. 自定义类创建

        * 3.3.3.1. 元类

        * 3.3.3.2. 解析 MRO 条目

        * 3.3.3.3. 确定适当的元类

        * 3.3.3.4. 准备类命名空间

        * 3.3.3.5. 执行类主体

        * 3.3.3.6. 创建类对象

        * 3.3.3.7. 元类的作用

      * 3.3.4. 自定义实例及子类检查

      * 3.3.5. 模拟泛型类型

      * 3.3.6. 模拟可调用对象

      * 3.3.7. 模拟容器类型

      * 3.3.8. 模拟数字类型

      * 3.3.9. with 语句上下文管理器

      * 3.3.10. 特殊方法查找

    * 3.4. 协程

      * 3.4.1. 可等待对象

      * 3.4.2. 协程对象

      * 3.4.3. 异步迭代器

      * 3.4.4. 异步上下文管理器

  * 4. 执行模型

    * 4.1. 程序的结构

    * 4.2. 命名与绑定

      * 4.2.1. 名称的绑定

      * 4.2.2. 名称的解析

      * 4.2.3. 内置命名空间和受限的执行

      * 4.2.4. 与动态特性的交互

    * 4.3. 异常

  * 5. 导入系统

    * 5.1. "importlib"

    * 5.2. 包

      * 5.2.1. 常规包

      * 5.2.2. 命名空间包

    * 5.3. 搜索

      * 5.3.1. 模块缓存

      * 5.3.2. 查找器和加载器

      * 5.3.3. 导入钩子

      * 5.3.4. 元路径

    * 5.4. 加载

      * 5.4.1. 加载器

      * 5.4.2. 子模块

      * 5.4.3. 模块规格说明

      * 5.4.4. 导入相关的模块属性

      * 5.4.5. module.__path__

      * 5.4.6. 模块的 repr

      * 5.4.7. 已缓存字节码的失效

    * 5.5. 基于路径的查找器

      * 5.5.1. 路径条目查找器

      * 5.5.2. 路径条目查找器协议

    * 5.6. 替换标准导入系统

    * 5.7. 包相对导入

    * 5.8. 有关 __main__ 的特殊事项

      * 5.8.1. __main__.__spec__

    * 5.9. 开放问题项

    * 5.10. 参考文献

  * 6. 表达式

    * 6.1. 算术转换

    * 6.2. 原子

      * 6.2.1. 标识符（名称）

      * 6.2.2. 字面值

      * 6.2.3. 带圆括号的形式

      * 6.2.4. 列表、集合与字典的显示

      * 6.2.5. 列表显示

      * 6.2.6. 集合显示

      * 6.2.7. 字典显示

      * 6.2.8. 生成器表达式

      * 6.2.9. yield 表达式

        * 6.2.9.1. 生成器-迭代器的方法

        * 6.2.9.2. 示例

        * 6.2.9.3. 异步生成器函数

        * 6.2.9.4. 异步生成器-迭代器方法

    * 6.3. 原型

      * 6.3.1. 属性引用

      * 6.3.2. 抽取

      * 6.3.3. 切片

      * 6.3.4. 调用

    * 6.4. await 表达式

    * 6.5. 幂运算符

    * 6.6. 一元算术和位运算

    * 6.7. 二元算术运算符

    * 6.8. 移位运算

    * 6.9. 二元位运算

    * 6.10. 比较运算

      * 6.10.1. 值比较

      * 6.10.2. 成员检测运算

      * 6.10.3. 标识号比较

    * 6.11. 布尔运算

    * 6.12. 赋值表达式

    * 6.13. 条件表达式

    * 6.14. lambda 表达式

    * 6.15. 表达式列表

    * 6.16. 求值顺序

    * 6.17. 运算符优先级

  * 7. 简单语句

    * 7.1. 表达式语句

    * 7.2. 赋值语句

      * 7.2.1. 增强赋值语句

      * 7.2.2. 带标注的赋值语句

    * 7.3. "assert" 语句

    * 7.4. "pass" 语句

    * 7.5. "del" 语句

    * 7.6. "return" 语句

    * 7.7. "yield" 语句

    * 7.8. "raise" 语句

    * 7.9. "break" 语句

    * 7.10. "continue" 语句

    * 7.11. "import" 语句

      * 7.11.1. future 语句

    * 7.12. "global" 语句

    * 7.13. "nonlocal" 语句

  * 8. 复合语句

    * 8.1. "if" 语句

    * 8.2. "while" 语句

    * 8.3. "for" 语句

    * 8.4. "try" 语句

    * 8.5. "with" 语句

    * 8.6. 函数定义

    * 8.7. 类定义

    * 8.8. 协程

      * 8.8.1. 协程函数定义

      * 8.8.2. "async for" 语句

      * 8.8.3. "async with" 语句

  * 9. 最高层级组件

    * 9.1. 完整的 Python 程序

    * 9.2. 文件输入

    * 9.3. 交互式输入

    * 9.4. 表达式输入

  * 10. 完整的语法规范

* Python 标准库

  * 概述

    * 可用性注释

  * 内置函数

  * 内置常量

    * 由 "site" 模块添加的常量

  * 内置类型

    * 逻辑值检测

    * 布尔运算 --- "and", "or", "not"

    * 比较

    * 数字类型 --- "int", "float", "complex"

      * 整数类型的按位运算

      * 整数类型的附加方法

      * 浮点类型的附加方法

      * 数字类型的哈希运算

    * 迭代器类型

      * 生成器类型

    * 序列类型 --- "list", "tuple", "range"

      * 通用序列操作

      * 不可变序列类型

      * 可变序列类型

      * 列表

      * 元组

      * range 对象

    * 文本序列类型 --- "str"

      * 字符串的方法

      * "printf" 风格的字符串格式化

    * 二进制序列类型 --- "bytes", "bytearray", "memoryview"

      * bytes 对象

      * bytearray 对象

      * bytes 和 bytearray 操作

      * "printf" 风格的字节串格式化

      * 内存视图

    * 集合类型 --- "set", "frozenset"

    * 映射类型 --- "dict"

      * 字典视图对象

    * 上下文管理器类型

    * 其他内置类型

      * 模块

      * 类与类实例

      * 函数

      * 方法

      * 代码对象

      * 类型对象

      * 空对象

      * 省略符对象

      * 未实现对象

      * 布尔值

      * 内部对象

    * 特殊属性

  * 内置异常

    * 基类

    * 具体异常

      * OS 异常

    * 警告

    * 异常层次结构

  * 文本处理服务

    * "string" --- 常见的字符串操作

      * 字符串常量

      * 自定义字符串格式化

      * 格式字符串语法

        * 格式规格迷你语言

        * 格式示例

      * 模板字符串

      * 辅助函数

    * "re" --- 正则表达式操作

      * 正则表达式语法

      * 模块内容

      * 正则表达式对象 （正则对象）

      * 匹配对象

      * 正则表达式例子

        * 检查对子

        * 模拟 scanf()

        * search() vs. match()

        * 制作一个电话本

        * 文字整理

        * 查找所有副词

        * 查找所有的副词及其位置

        * 原始字符串标记

        * 写一个词法分析器

    * "difflib" --- 计算差异的辅助工具

      * SequenceMatcher 对象

      * SequenceMatcher 的示例

      * Differ 对象

      * Differ 示例

      * difflib 的命令行接口

    * "textwrap" --- 文本自动换行与填充

    * "unicodedata" --- Unicode 数据库

    * "stringprep" --- 因特网字符串预备

    * "readline" --- GNU readline 接口

      * 初始化文件

      * 行缓冲区

      * 历史文件

      * 历史列表

      * 启动钩子

      * Completion

      * 示例

    * "rlcompleter" --- GNU readline 的补全函数

      * Completer 对象

  * 二进制数据服务

    * "struct" --- 将字节串解读为打包的二进制数据

      * 函数和异常

      * 格式字符串

        * 字节顺序，大小和对齐方式

        * 格式字符

        * 示例

      * 类

    * "codecs" --- 编解码器注册和相关基类

      * 编解码器基类

        * 错误处理方案

        * 无状态的编码和解码

        * 增量式的编码和解码

          * IncrementalEncoder 对象

          * IncrementalDecoder 对象

        * 流式的编码和解码

          * StreamWriter 对象

          * StreamReader 对象

          * StreamReaderWriter 对象

          * StreamRecoder 对象

      * 编码格式与 Unicode

      * 标准编码

      * Python 专属的编码格式

        * 文字编码

        * 二进制转换

        * 文字转换

      * "encodings.idna" --- 应用程序中的国际化域名

      * "encodings.mbcs" --- Windows ANSI代码页

      * "encodings.utf_8_sig" --- 带BOM签名的UTF-8编解码器

  * 数据类型

    * "datetime" --- 基本的日期和时间类型

      * 感知型对象和简单型对象

      * 常量

      * 有效的类型

        * 通用的特征属性

        * 确定一个对象是感知型还是简单型

      * "timedelta" 类对象

        * class:*timedelta* 用法示例

      * "date" 对象

        * class:*date* 用法示例

      * "datetime" 对象

        * 用法示例: "datetime"

      * "time" 对象

        * 用法示例: "time"

      * "tzinfo" 对象

      * "timezone" 对象

      * "strftime()" 和 "strptime()" 的行为

        * "strftime()" 和 "strptime()" Format Codes

        * 技术细节

    * "calendar" --- 日历相关函数

    * "collections" --- 容器数据类型

      * "ChainMap" 对象

        * "ChainMap" 例子和方法

      * "Counter" 对象

      * "deque" 对象

        * "deque" 用法

      * "defaultdict" 对象

        * "defaultdict" 例子

      * "namedtuple()" 命名元组的工厂函数

      * "OrderedDict" 对象

        * "OrderedDict" 例子和用法

      * "UserDict" 对象

      * "UserList" 对象

      * "UserString" 对象

    * "collections.abc" --- 容器的抽象基类

      * 容器抽象基类

    * "heapq" --- 堆队列算法

      * 基本示例

      * 优先队列实现说明

      * 理论

    * "bisect" --- 数组二分查找算法

      * 搜索有序列表

      * 其他示例

    * "array" --- 高效的数值数组

    * "weakref" --- 弱引用

      * 弱引用对象

      * 示例

      * 终结器对象

      * 比较终结器与 "__del__()" 方法

    * "types" --- 动态类型创建和内置类型名称

      * 动态类型创建

      * 标准解释器类型

      * 附加工具类和函数

      * 协程工具函数

    * "copy" --- 浅层 (shallow) 和深层 (deep) 复制操作

    * "pprint" --- 数据美化输出

      * PrettyPrinter 对象

      * 示例

    * "reprlib" --- 另一种 "repr()" 实现

      * Repr 对象

      * 子类化 Repr 对象

    * "enum" --- 对枚举的支持

      * 模块内容

      * 创建一个 Enum

      * 对枚举成员及其属性的程序化访问

      * 复制枚举成员和值

      * 确保唯一的枚举值

      * 使用自动设定的值

      * 迭代

      * 比较

      * 允许的枚举成员和属性

      * 受限的 Enum 子类化

      * 封存

      * 功能性 API

      * 派生的枚举

        * IntEnum

        * IntFlag

        * Flag

        * 其他事项

      * 何时使用 "__new__()" 与 "__init__()"

      * 有趣的示例

        * 省略值

          * 使用 "auto"

          * 使用 "object"

          * 使用描述性字符串

          * 使用自定义的 "__new__()"

        * OrderedEnum

        * DuplicateFreeEnum

        * Planet

        * TimePeriod

      * 各种枚举有何区别？

        * 枚举类

        * 枚举成员（即实例）

        * 细节要点

          * 支持的 "__dunder__" 名称

          * 支持的 "_sunder_" 名称

          * "Enum" 成员类型

          * "Enum" 类和成员的布尔值

          * 带有方法的 "Enum" 类

          * 组合 "Flag" 的成员

  * 数字和数学模块

    * "numbers" --- 数字的抽象基类

      * 数字的层次

      * 类型接口注释。

        * 加入更多数字的ABC

        * 实现算数运算

    * "math" --- 数学函数

      * 数论与表示函数

      * 幂函数与对数函数

      * 三角函数

      * 角度转换

      * 双曲函数

      * 特殊函数

      * 常数

    * "cmath" --- 关于复数的数学函数

      * 到极坐标和从极坐标的转换

      * 幂函数与对数函数

      * 三角函数

      * 双曲函数

      * 分类函数

      * 常数

    * "decimal" --- 十进制定点和浮点运算

      * 快速入门教程

      * Decimal 对象

        * 逻辑操作数

      * 上下文对象

      * 常数

      * 舍入模式

      * 信号

      * 浮点数说明

        * 通过提升精度来解决舍入错误

        * 特殊的值

      * 使用线程

      * 例程

      * Decimal 常见问题

    * "fractions" --- 分数

    * "random" --- 生成伪随机数

      * 簿记功能

      * 整数用函数

      * 序列用函数

      * 实值分布

      * 替代生成器

      * 关于再现性的说明

      * 例子和配方

    * "statistics" --- 数学统计函数

      * 平均值以及对中心位置的评估

      * 对分散程度的评估

      * 函数细节

      * 异常

      * "NormalDist" 对象

        * "NormalDist" 示例和用法

  * 函数式编程模块

    * "itertools" --- 为高效循环而创建迭代器的函数

      * Itertool函数

      * itertools 配方

    * "functools" --- 高阶函数和可调用对象上的操作

      * "partial" 对象

    * "operator" --- 标准运算符替代函数

      * 将运算符映射到函数

      * 原地运算符

  * 文件和目录访问

    * "pathlib" --- 面向对象的文件系统路径

      * 基础使用

      * 纯路径

        * 通用性质

        * 运算符

        * 访问个别部分

        * 方法和特征属性

      * 具体路径

        * 方法

      * 对应的 "os" 模块的工具

    * "os.path" --- 常用路径操作

    * "fileinput" --- 迭代来自多个输入流的行

    * "stat" --- 解析 "stat()" 结果

    * "filecmp" --- 文件及目录的比较

      * "dircmp" 类

    * "tempfile" --- 生成临时文件和目录

      * 示例

      * 已弃用的函数和变量

    * "glob" --- Unix 风格路径名模式扩展

    * "fnmatch" --- Unix 文件名模式匹配

    * "linecache" --- 随机读写文本行

    * "shutil" --- 高阶文件操作

      * 目录和文件操作

        * 依赖于具体平台的高效拷贝操作

        * copytree 示例

        * rmtree 示例

      * 归档操作

        * 归档程序示例

      * 查询输出终端的尺寸

  * 数据持久化

    * "pickle" --- Python 对象序列化

      * 与其他 Python 模块间的关系

        * 与 "marshal" 间的关系

        * 与 "json" 模块的比较

      * 数据流格式

      * 模块接口

      * 可以被封存/解封的对象

      * 封存类实例

        * 持久化外部对象

        * Dispatch 表

        * 处理有状态的对象

      * 类型，函数和其他对象的自定义归约

      * 外部缓冲区

        * 提供方 API

        * 使用方 API

        * 示例

      * 限制全局变量

      * 性能

      * 示例

    * "copyreg" --- 注册配合 "pickle" 模块使用的函数

      * 示例

    * "shelve" --- Python 对象持久化

      * 限制

      * 示例

    * "marshal" --- 内部 Python 对象序列化

    * "dbm" --- Unix "数据库" 接口

      * "dbm.gnu" --- GNU 对 dbm 的重解析

      * "dbm.ndbm" --- 基于 ndbm 的接口

      * "dbm.dumb" --- 便携式 DBM 实现

    * "sqlite3" --- SQLite 数据库 DB-API 2.0 接口模块

      * 模块函数和常量

      * 连接对象（Connection）

      * Cursor 对象

      * 行对象

      * 异常

      * SQLite 与 Python 类型

        * 概述

        * 使用适配器将额外的 Python 类型保存在 SQLite 数据库中。

          * 让对象自行适配

          * 注册可调用的适配器

        * 将SQLite 值转换为自定义Python 类型

        * 默认适配器和转换器

      * 控制事务

      * 有效使用 "sqlite3"

        * 使用快捷方式

        * 通过名称而不是索引访问索引

        * 使用连接作为上下文管理器

      * 常见问题

        * 多线程

  * 数据压缩和存档

    * "zlib" --- 与 **gzip** 兼容的压缩

    * "gzip" --- 对 **gzip** 格式的支持

      * 用法示例

      * 命令行界面

        * 命令行选项

    * "bz2" --- 对 **bzip2** 压缩算法的支持

      * 文件压缩和解压

      * 增量压缩和解压

      * 一次性压缩或解压缩

      * 用法示例

    * "lzma" --- 用 LZMA 算法压缩

      * 读写压缩文件

      * 在内存中压缩和解压缩数据

      * 杂项

      * 指定自定义的过滤器链

      * 示例

    * "zipfile" --- 使用ZIP存档

      * ZipFile 对象

      * Path 对象

      * PyZipFile 对象

      * ZipInfo 对象

      * 命令行界面

        * 命令行选项

      * 解压缩的障碍

        * 由于文件本身

        * 文件系统限制

        * 资源限制

        * 中断

        * 提取的默认行为

    * "tarfile" --- 读写tar归档文件

      * TarFile 对象

      * TarInfo 对象

      * 命令行界面

        * 命令行选项

      * 示例

      * 受支持的 tar 格式

      * Unicode 问题

  * 文件格式

    * "csv" --- CSV 文件读写

      * 模块内容

      * 变种与格式参数

      * Reader 对象

      * Writer 对象

      * 示例

    * "configparser" --- 配置文件解析器

      * 快速起步

      * 支持的数据类型

      * 回退值

      * 受支持的 INI 文件结构

      * 值的插值

      * 映射协议访问

      * 定制解析器行为

      * 旧式 API 示例

      * ConfigParser 对象

      * RawConfigParser 对象

      * 异常

    * "netrc" --- netrc 文件处理

      * netrc 对象

    * "xdrlib" --- 编码与解码 XDR 数据

      * Packer 对象

      * Unpacker 对象

      * 异常

    * "plistlib" --- 生成与解析 Mac OS X ".plist" 文件

      * 示例

  * 加密服务

    * "hashlib" --- 安全哈希与消息摘要

      * 哈希算法

      * SHAKE 可变长度摘要

      * 密钥派生

      * BLAKE2

        * 创建哈希对象

        * 常数

        * 示例

          * 简单哈希

          * 使用不同的摘要大小

          * 密钥哈希

          * 随机哈希

          * 个性化

          * 树形模式

        * 开发人员

    * "hmac" --- 基于密钥的消息验证

    * "secrets" --- 生成安全随机数字用于管理密码

      * 随机数

      * 生成凭据

        * 凭据应当使用多少个字节？

      * 其他功能

      * 应用技巧与最佳实践

  * 通用操作系统服务

    * "os" --- 多种操作系统接口

      * 文件名，命令行参数，以及环境变量。

      * 进程参数

      * 创建文件对象

      * 文件描述符操作

        * 查询终端的尺寸

        * 文件描述符的继承

      * 文件和目录

        * Linux 扩展属性

      * 进程管理

      * 调度器接口

      * 其他系统信息

      * 随机数

    * "io" --- 处理流的核心工具

      * 概述

        * 文本 I/O

        * 二进制 I/O

        * 原始 I/O

      * 高阶模块接口

        * 内存中的流

      * 类的层次结构

        * I/O 基类

        * 原始文件 I/O

        * 缓冲流

        * 文本 I/O

      * 性能

        * 二进制 I/O

        * 文本 I/O

        * 多线程

        * 可重入性

    * "time" --- 时间的访问和转换

      * 函数

      * Clock ID 常量

      * 时区常量

    * "argparse" --- 命令行选项、参数和子命令解析器

      * 示例

        * 创建一个解析器

        * 添加参数

        * 解析参数

      * ArgumentParser 对象

        * prog

        * usage

        * description

        * epilog

        * parents

        * formatter_class

        * prefix_chars

        * fromfile_prefix_chars

        * argument_default

        * allow_abbrev

        * conflict_handler

        * add_help

      * add_argument() 方法

        * name or flags

        * action

        * nargs

        * const

        * default

        * type

        * choices

        * required

        * help

        * metavar

        * dest

        * Action classes

      * parse_args() 方法

        * Option value syntax

        * 无效的参数

        * 包含 "-" 的参数

        * 参数缩写（前缀匹配）

        * Beyond "sys.argv"

        * 命名空间对象

      * 其它实用工具

        * 子命令

        * FileType 对象

        * 参数组

        * Mutual exclusion

        * Parser defaults

        * 打印帮助

        * Partial parsing

        * 自定义文件解析

        * 退出方法

        * Intermixed parsing

      * 升级 optparse 代码

    * "getopt" --- C-style parser for command line options

    * "logging" --- Python 的日志记录工具

      * 记录器对象

      * 日志级别

      * 处理器对象

      * 格式器对象

      * Filter Objects

      * LogRecord Objects

      * LogRecord 属性

      * LoggerAdapter 对象

      * 线程安全

      * 模块级别函数

      * 模块级属性

      * 与警告模块集成

    * "logging.config" --- 日志记录配置

      * Configuration functions

      * Configuration dictionary schema

        * Dictionary Schema Details

        * Incremental Configuration

        * Object connections

        * User-defined objects

        * Access to external objects

        * Access to internal objects

        * Import resolution and custom importers

      * Configuration file format

    * "logging.handlers" --- 日志处理

      * StreamHandler

      * FileHandler

      * NullHandler

      * WatchedFileHandler

      * BaseRotatingHandler

      * RotatingFileHandler

      * TimedRotatingFileHandler

      * SocketHandler

      * DatagramHandler

      * SysLogHandler

      * NTEventLogHandler

      * SMTPHandler

      * MemoryHandler

      * HTTPHandler

      * QueueHandler

      * QueueListener

    * "getpass" --- 便携式密码输入工具

    * "curses" --- 终端字符单元显示的处理

      * 函数

      * Window Objects

      * 常量

    * "curses.textpad" --- Text input widget for curses programs

      * 文本框对象

    * "curses.ascii" --- Utilities for ASCII characters

    * "curses.panel" --- A panel stack extension for curses

      * 函数

      * Panel Objects

    * "platform" ---  获取底层平台的标识数据

      * 跨平台

      * Java平台

      * Windows平台

      * Mac OS平台

      * Unix Platforms

    * "errno" --- Standard errno system symbols

    * "ctypes" --- Python 的外部函数库

      * ctypes 教程

        * 载入动态连接库

        * 操作导入的动态链接库中的函数

        * 调用函数

        * 基础数据类型

        * 调用函数，继续

        * 使用自定义的数据类型调用函数

        * Specifying the required argument types (function prototypes)

        * Return types

        * Passing pointers (or: passing parameters by reference)

        * Structures and unions

        * Structure/union alignment and byte order

        * Bit fields in structures and unions

        * Arrays

        * Pointers

        * Type conversions

        * Incomplete Types

        * Callback functions

        * Accessing values exported from dlls

        * Surprises

        * Variable-sized data types

      * ctypes reference

        * Finding shared libraries

        * Loading shared libraries

        * Foreign functions

        * Function prototypes

        * Utility functions

        * Data types

        * 基础数据类型

        * Structured data types

        * Arrays and pointers

  * 并发执行

    * "threading" --- 基于线程的并行

      * 线程本地数据

      * 线程对象

      * 锁对象

      * 递归锁对象

      * 条件对象

      * 信号量对象

        * "Semaphore" 例子

      * 事件对象

      * 定时器对象

      * 栅栏对象

      * 在 "with" 语句中使用锁、条件和信号量

    * "multiprocessing" --- 基于进程的并行

      * 概述

        * "Process" 类

        * 上下文和启动方法

        * 在进程之间交换对象

        * 进程间同步

        * 进程间共享状态

        * 使用工作进程

      * 参考

        * "Process" 和异常

        * 管道和队列

        * 杂项

        * 连接（Connection）对象

        * 同步原语

        * 共享 "ctypes" 对象

          * "multiprocessing.sharedctypes" 模块

        * 管理器

          * 自定义管理器

          * 使用远程管理器

        * 代理对象

          * 清理

        * 进程池

        * 监听者及客户端

          * 地址格式

        * 认证密码

        * 日志

        * "multiprocessing.dummy" 模块

      * 编程指导

        * 所有start方法

        * *spawn* 和 *forkserver* 启动方式

      * 示例

    * "multiprocessing.shared_memory" --- 可从进程直接访问的共享内存

    * "concurrent" 包

    * "concurrent.futures" --- 启动并行任务

      * 执行器对象

      * ThreadPoolExecutor

        * ThreadPoolExecutor 例子

      * ProcessPoolExecutor

        * ProcessPoolExecutor 例子

      * 期程对象

      * 模块函数

      * Exception类

    * "subprocess" --- 子进程管理

      * 使用 "subprocess" 模块

        * 常用参数

        * Popen 构造函数

        * 异常

      * 安全考量

      * Popen 对象

      * Windows Popen 助手

        * Windows 常数

      * Older high-level API

      * Replacing Older Functions with the "subprocess" Module

        * Replacing **/bin/sh** shell command substitution

        * Replacing shell pipeline

        * Replacing "os.system()"

        * Replacing the "os.spawn" family

        * Replacing "os.popen()", "os.popen2()", "os.popen3()"

        * Replacing functions from the "popen2" module

      * Legacy Shell Invocation Functions

      * 注释

        * Converting an argument sequence to a string on Windows

    * "sched" --- 事件调度器

      * 调度器对象

    * "queue" --- 一个同步的队列类

      * Queue对象

      * SimpleQueue 对象

    * "_thread" --- 底层多线程 API

    * "_dummy_thread" --- "_thread" 的替代模块

    * "dummy_threading" ---  可直接替代 "threading" 模块。

  * "contextvars" --- Context Variables

    * Context Variables

    * Manual Context Management

    * asyncio support

  * 网络和进程间通信

    * "asyncio" --- 异步 I/O

      * 协程与任务

        * 协程

        * 可等待对象

        * 运行 asyncio 程序

        * 创建任务

        * 休眠

        * 并发运行任务

        * 屏蔽取消操作

        * 超时

        * 简单等待

        * 来自其他线程的日程安排

        * 内省

        * Task 对象

        * 基于生成器的协程

      * 流

        * StreamReader

        * StreamWriter

        * 示例

          * TCP echo client using streams

          * TCP echo server using streams

          * Get HTTP headers

          * Register an open socket to wait for data using streams

      * Synchronization Primitives

        * Lock

        * Event

        * Condition

        * Semaphore

        * BoundedSemaphore

      * 子进程

        * Creating Subprocesses

        * 常数

        * Interacting with Subprocesses

          * Subprocess and Threads

          * 示例

      * 队列集

        * 队列

        * 优先级队列

        * 后进先出队列

        * 异常

        * 示例

      * 异常

      * 事件循环

        * 事件循环方法集

          * 运行和停止循环

          * 调度回调

          * 调度延迟回调

          * 创建 Futures 和 Tasks

          * 打开网络连接

          * 创建网络服务

          * 传输文件

          * TLS 升级

          * 监控文件描述符

          * 直接使用 socket 对象

          * DNS

          * 使用管道

          * Unix 信号

          * 在线程或者进程池中执行代码。

          * 错误处理API

          * 开启调试模式

          * 运行子进程

        * 回调处理

        * Server Objects

        * 事件循环实现

        * 示例

          * call_soon() 的 Hello World 示例。

          * 使用 call_later() 来展示当前的日期

          * 监控一个文件描述符的读事件

          * 为SIGINT和SIGTERM设置信号处理器

      * Futures

        * Future 函数

        * Future 对象

      * 传输和协议

        * 传输

          * 传输层级

          * 基础传输

          * 只读传输

          * 只写传输

          * 数据报传输

          * 子进程传输

        * 协议

          * 基础协议

          * 基础协议

          * 流协议

          * 缓冲流协议

          * 数据报协议

          * 子进程协议

        * 示例

          * TCP回应服务器

          * TCP回应客户端

          * UDP回应服务器

          * UDP回应客户端

          * 链接已存在的套接字

          * loop.subprocess_exec() and SubprocessProtocol

      * 策略

        * 获取和设置策略

        * 策略对象

        * 进程监视器

        * 自定义策略

      * 平台支持

        * 所有平台

        * Windows

          * Windows的子进程支持

        * macOS

      * 高级API索引

        * Tasks

        * 队列集

        * 子进程集

        * 流

        * 同步

        * 异常

      * 底层API目录

        * 获取事件循环

        * 事件循环方法集

        * 传输

        * 协议

        * 事件循环策略

      * 用 asyncio 开发

        * Debug 模式

        * 并发性和多线程

        * 运行阻塞的代码

        * 日志

        * 检测 never-awaited 协同程序

        * 检测就再也没异常

    * "socket" --- 底层网络接口

      * 套接字协议族

      * 模块内容

        * 异常

        * 常数

        * 函数

          * 创建套接字

          * 其他功能

      * 套接字对象

      * Notes on socket timeouts

        * Timeouts and the "connect" method

        * Timeouts and the "accept" method

      * 示例

    * "ssl" --- 套接字对象的TLS/SSL封装

      * Functions, Constants, and Exceptions

        * Socket creation

        * 上下文创建

        * 异常

        * Random generation

        * Certificate handling

        * 常数

      * SSL Sockets

      * SSL Contexts

      * Certificates

        * Certificate chains

        * CA certificates

        * Combined key and certificate

        * Self-signed certificates

      * 示例

        * Testing for SSL support

        * Client-side operation

        * Server-side operation

      * Notes on non-blocking sockets

      * Memory BIO Support

      * SSL session

      * Security considerations

        * Best defaults

        * Manual settings

          * Verifying certificates

          * Protocol versions

          * Cipher selection

        * Multi-processing

      * TLS 1.3

      * LibreSSL support

    * "select" --- 等待 I/O 完成

      * "/dev/poll" 轮询对象

      * 边缘触发和水平触发的轮询 (epoll) 对象

      * Poll 对象

      * Kqueue 对象

      * Kevent 对象

    * "selectors" --- 高级 I/O 复用库

      * 概述

      * 类

      * 示例

    * "asyncore" --- 异步socket处理器

      * asyncore Example basic HTTP client

      * asyncore Example basic echo server

    * "asynchat" --- 异步 socket 指令/响应 处理器

      * asynchat Example

    * "signal" --- 设置异步事件处理程序

      * 一般规则

        * 执行 Python 信号处理程序

        * 信号与线程

      * 模块内容

      * 示例

      * Note on SIGPIPE

    * "mmap" --- 内存映射文件支持

      * MADV_* Constants

  * 互联网数据处理

    * "email" --- 电子邮件与 MIME 处理包

      * "email.message": Representing an email message

      * "email.parser": Parsing email messages

        * FeedParser API

        * Parser API

        * Additional notes

      * "email.generator": Generating MIME documents

      * "email.policy": Policy Objects

      * "email.errors": 异常和缺陷类

      * "email.headerregistry": Custom Header Objects

      * "email.contentmanager": Managing MIME Content

        * Content Manager Instances

      * "email": 示例

      * "email.message.Message": Representing an email message using
        the "compat32" API

      * "email.mime": Creating email and MIME objects from scratch

      * "email.header": Internationalized headers

      * "email.charset": Representing character sets

      * "email.encoders": 编码器

      * "email.utils": 其他工具

      * "email.iterators": 迭代器

    * "json" --- JSON 编码和解码器

      * 基本使用

      * 编码器和解码器

      * 异常

      * 标准符合性和互操作性

        * 字符编码

        * Infinite 和 NaN 数值

        * 对象中的重复名称

        * 顶级非对象，非数组值

        * 实现限制

      * 命令行界面

        * 命令行选项

    * "mailcap" --- Mailcap 文件处理

    * "mailbox" --- Manipulate mailboxes in various formats

      * "Mailbox" 对象

        * "Maildir"

        * "mbox"

        * "MH"

        * "Babyl"

        * "MMDF"

      * "Message" objects

        * "MaildirMessage"

        * "mboxMessage"

        * "MHMessage"

        * "BabylMessage"

        * "MMDFMessage"

      * 异常

      * 示例

    * "mimetypes" --- Map filenames to MIME types

      * MimeTypes Objects

    * "base64" --- Base16, Base32, Base64, Base85 数据编码

    * "binhex" --- 对binhex4文件进行编码和解码

      * 注释

    * "binascii" --- 二进制和 ASCII 码互转

    * "quopri" --- 编码与解码经过 MIME 转码的可打印数据

    * "uu" --- 对 uuencode 文件进行编码与解码

  * 结构化标记处理工具

    * "html" --- 超文本标记语言支持

    * "html.parser" --- 简单的 HTML 和 XHTML 解析器

      * HTML 解析器的示例程序

      * "HTMLParser" 方法

      * 示例

    * "html.entities" --- HTML 一般实体的定义

    * XML处理模块

      * XML 漏洞

      * "defusedxml" 和 "defusedexpat" 软件包

    * "xml.etree.ElementTree" ---  ElementTree XML API

      * 教程

        * XML树和元素

        * 解析XML

        * Pull API进行非阻塞解析

        * 查找感兴趣的元素

        * 修改XML文件

        * 构建XML文档

        * 使用命名空间解析XML

        * 其他资源

      * XPath支持

        * 示例

        * 支持的XPath语法

      * 参考引用

        * 函数

      * XInclude 支持

        * 示例

      * 参考引用

        * 函数

        * 元素对象

        * ElementTree 对象

        * QName Objects

        * TreeBuilder Objects

        * XMLParser对象

        * XMLPullParser对象

        * 异常

    * "xml.dom" --- The Document Object Model API

      * 模块内容

      * Objects in the DOM

        * DOMImplementation Objects

        * 节点对象

        * 节点列表对象

        * 文档类型对象

        * 文档对象

        * 元素对象

        * Attr 对象

        * NamedNodeMap 对象

        * 注释对象

        * Text 和 CDATASection 对象

        * ProcessingInstruction 对象

        * 异常

      * 一致性

        * 类型映射

        * Accessor Methods

    * "xml.dom.minidom" --- Minimal DOM implementation

      * DOM Objects

      * DOM Example

      * minidom and the DOM standard

    * "xml.dom.pulldom" --- Support for building partial DOM trees

      * DOMEventStream Objects

    * "xml.sax" --- Support for SAX2 parsers

      * SAXException Objects

    * "xml.sax.handler" --- Base classes for SAX handlers

      * ContentHandler 对象

      * DTDHandler 对象

      * EntityResolver 对象

      * ErrorHandler 对象

    * "xml.sax.saxutils" --- SAX 工具集

    * "xml.sax.xmlreader" --- Interface for XML parsers

      * XMLReader 对象

      * IncrementalParser 对象

      * Locator 对象

      * InputSource 对象

      * The "Attributes" Interface

      * The "AttributesNS" Interface

    * "xml.parsers.expat" --- Fast XML parsing using Expat

      * XMLParser对象

      * ExpatError Exceptions

      * 示例

      * Content Model Descriptions

      * Expat error constants

  * 互联网协议和支持

    * "webbrowser" --- 方便的Web浏览器控制器

      * 浏览器控制器对象

    * "cgi" --- Common Gateway Interface support

      * 概述

      * 使用cgi模块。

      * Higher Level Interface

      * 函数

      * Caring about security

      * Installing your CGI script on a Unix system

      * Testing your CGI script

      * Debugging CGI scripts

      * Common problems and solutions

    * "cgitb" --- 用于 CGI 脚本的回溯管理器

    * "wsgiref" --- WSGI Utilities and Reference Implementation

      * "wsgiref.util" -- WSGI environment utilities

      * "wsgiref.headers" -- WSGI response header tools

      * "wsgiref.simple_server" -- a simple WSGI HTTP server

      * "wsgiref.validate" --- WSGI conformance checker

      * "wsgiref.handlers" -- server/gateway base classes

      * 示例

    * "urllib" --- URL 处理模块

    * "urllib.request" --- 用于打开 URL 的可扩展库

      * Request 对象

      * OpenerDirector 对象

      * BaseHandler 对象

      * HTTPRedirectHandler 对象

      * HTTPCookieProcessor 对象

      * ProxyHandler 对象

      * HTTPPasswordMgr 对象

      * HTTPPasswordMgrWithPriorAuth 对象

      * AbstractBasicAuthHandler 对象

      * HTTPBasicAuthHandler 对象

      * ProxyBasicAuthHandler 对象

      * AbstractDigestAuthHandler 对象

      * HTTPDigestAuthHandler 对象

      * ProxyDigestAuthHandler 对象

      * HTTPHandler 对象

      * HTTPSHandler 对象

      * FileHandler 对象

      * DataHandler 对象

      * FTPHandler 对象

      * CacheFTPHandler 对象

      * UnknownHandler 对象

      * HTTPErrorProcessor 对象

      * 示例

      * Legacy interface

      * "urllib.request" Restrictions

    * "urllib.response" --- urllib 使用的 Response 类

    * "urllib.parse" --- Parse URLs into components

      * URL 解析

      * 解析ASCII编码字节

      * 结构化解析结果

      * URL Quoting

    * "urllib.error" --- urllib.request 引发的异常类

    * "urllib.robotparser" --- robots.txt 语法分析程序

    * "http" --- HTTP 模块

      * HTTP 状态码

    * "http.client" --- HTTP 协议客户端

      * HTTPConnection 对象

      * HTTPResponse 对象

      * 示例

      * HTTPMessage Objects

    * "ftplib" --- FTP 协议客户端

      * FTP Objects

      * FTP_TLS Objects

    * "poplib" --- POP3 protocol client

      * POP3 Objects

      * POP3 Example

    * "imaplib" --- IMAP4 protocol client

      * IMAP4 Objects

      * IMAP4 Example

    * "nntplib" --- NNTP protocol client

      * NNTP Objects

        * Attributes

        * 方法

      * Utility functions

    * "smtplib" ---SMTP协议客户端

      * SMTP Objects

      * SMTP Example

    * "smtpd" --- SMTP 服务器

      * SMTPServer 对象

      * DebuggingServer 对象

      * PureProxy对象

      * MailmanProxy 对象

      * SMTPChannel 对象

    * "telnetlib" --- Telnet client

      * Telnet Objects

      * Telnet Example

    * "uuid" --- UUID objects according to **RFC 4122**

      * 示例

    * "socketserver" --- A framework for network servers

      * Server Creation Notes

      * Server Objects

      * Request Handler Objects

      * 示例

        * "socketserver.TCPServer" Example

        * "socketserver.UDPServer" Example

        * Asynchronous Mixins

    * "http.server" --- HTTP 服务器

    * "http.cookies" --- HTTP状态管理

      * Cookie 对象

      * Morsel 对象

      * 示例

    * "http.cookiejar" —— HTTP 客户端的 Cookie 处理

      * CookieJar 和 FileCookieJar 对象

      * FileCookieJar subclasses and co-operation with web browsers

      * CookiePolicy 对象

      * DefaultCookiePolicy 对象

      * Cookie 对象

      * 示例

    * "xmlrpc" --- XMLRPC 服务端与客户端模块

    * "xmlrpc.client" --- XML-RPC client access

      * ServerProxy 对象

      * DateTime 对象

      * Binary 对象

      * Fault 对象

      * ProtocolError 对象

      * MultiCall 对象

      * Convenience Functions

      * Example of Client Usage

      * Example of Client and Server Usage

    * "xmlrpc.server" --- Basic XML-RPC servers

      * SimpleXMLRPCServer Objects

        * SimpleXMLRPCServer Example

      * CGIXMLRPCRequestHandler

      * Documenting XMLRPC server

      * DocXMLRPCServer Objects

      * DocCGIXMLRPCRequestHandler

    * "ipaddress" --- IPv4/IPv6 manipulation library

      * Convenience factory functions

      * IP Addresses

        * Address objects

        * Conversion to Strings and Integers

        * 运算符

          * Comparison operators

          * Arithmetic operators

      * IP Network definitions

        * Prefix, net mask and host mask

        * Network objects

        * 运算符

          * Logical operators

          * 迭代

          * Networks as containers of addresses

      * Interface objects

        * 运算符

          * Logical operators

      * Other Module Level Functions

      * Custom Exceptions

  * 多媒体服务

    * "audioop" --- Manipulate raw audio data

    * "aifc" --- Read and write AIFF and AIFC files

    * "sunau" --- 读写 Sun AU 文件

      * AU_read 对象

      * AU_write 对象

    * "wave" --- 读写WAV格式文件

      * Wave_read对象

      * Wave_write 对象

    * "chunk" --- 读取 IFF 分块数据

    * "colorsys" --- 颜色系统间的转换

    * "imghdr" --- 推测图像类型

    * "sndhdr" --- 推测声音文件的类型

    * "ossaudiodev" --- Access to OSS-compatible audio devices

      * Audio Device Objects

      * Mixer Device Objects

  * 国际化

    * "gettext" --- 多语种国际化服务

      * GNU **gettext** API

      * Class-based API

        * The "NullTranslations" class

        * The "GNUTranslations" class

        * Solaris message catalog support

        * The Catalog constructor

      * Internationalizing your programs and modules

        * Localizing your module

        * Localizing your application

        * Changing languages on the fly

        * Deferred translations

      * 致谢

    * "locale" --- 国际化服务

      * Background, details, hints, tips and caveats

      * For extension writers and programs that embed Python

      * Access to message catalogs

  * 程序框架

    * "turtle" --- 海龟绘图

      * 概述

      * 可用的 Turtle 和 Screen 方法概览

        * Turtle 方法

        * TurtleScreen/Screen 方法

      * RawTurtle/Turtle 方法和对应函数

        * 海龟动作

        * 获取海龟的状态

        * 度量单位设置

        * 画笔控制

          * 绘图状态

          * 颜色控制

          * 填充

          * 更多绘图控制

        * 海龟状态

          * 可见性

          * 外观

        * 使用事件

        * 特殊海龟方法

        * 复合形状

      * TurtleScreen/Screen 方法及对应函数

        * 窗口控制

        * 动画控制

        * 使用屏幕事件

        * 输入方法

        * 设置与特殊方法

        * Screen 专有方法, 而非继承自 TurtleScreen

      * 公共类

      * 帮助与配置

        * 如何使用帮助

        * 文档字符串翻译为不同的语言

        * 如何配置 Screen 和 Turtle

      * "turtledemo" --- 演示脚本集

      * Python 2.6 之后的变化

      * Python 3.0 之后的变化

    * "cmd" --- 支持面向行的命令解释器

      * Cmd 对象

      * Cmd 例子

    * "shlex" --- Simple lexical analysis

      * shlex Objects

      * Parsing Rules

      * Improved Compatibility with Shells

  * Tk图形用户界面(GUI)

    * "tkinter" --- Tcl/Tk的Python接口

      * Tkinter 模块

      * Tkinter Life Preserver

        * How To Use This Section

        * A Simple Hello World Program

      * A (Very) Quick Look at Tcl/Tk

      * Mapping Basic Tk into Tkinter

      * How Tk and Tkinter are Related

      * Handy Reference

        * Setting Options

        * The Packer

        * Packer Options

        * Coupling Widget Variables

        * The Window Manager

        * Tk Option Data Types

        * Bindings and Events

        * The index Parameter

        * Images

      * File Handlers

    * "tkinter.ttk" --- Tk主题小部件

      * 使用 Ttk

      * Ttk Widgets

      * Widget

        * 标准选项

        * Scrollable Widget Options

        * Label Options

        * Compatibility Options

        * Widget States

        * ttk.Widget

      * Combobox

        * 选项

        * Virtual events

        * ttk.Combobox

      * Spinbox

        * 选项

        * Virtual events

        * ttk.Spinbox

      * Notebook

        * 选项

        * Tab Options

        * Tab Identifiers

        * Virtual Events

        * ttk.Notebook

      * Progressbar

        * 选项

        * ttk.Progressbar

      * Separator

        * 选项

      * Sizegrip

        * Platform-specific notes

        * Bugs

      * Treeview

        * 选项

        * Item Options

        * Tag Options

        * Column Identifiers

        * Virtual Events

        * ttk.Treeview

      * Ttk Styling

        * Layouts

    * "tkinter.tix" --- Extension widgets for Tk

      * Using Tix

      * Tix Widgets

        * Basic Widgets

        * File Selectors

        * Hierarchical ListBox

        * Tabular ListBox

        * Manager Widgets

        * Image Types

        * Miscellaneous Widgets

        * Form Geometry Manager

      * Tix Commands

    * "tkinter.scrolledtext" --- 滚动文字控件

    * IDLE

      * 目录

        * 文件菜单 （命令行和编辑器）

        * 编辑菜单（命令行和编辑器）

        * 格式菜单（仅 window 编辑器）

        * 运行菜单（仅 window 编辑器）

        * Shell 菜单（仅 window 编辑器）

        * 调试菜单（仅 window 编辑器）

        * 选项菜单（命令行和编辑器）

        * Window 菜单（命令行和编辑器）

        * 帮助菜单（命令行和编辑器）

        * 上下文菜单

      * 编辑和导航

        * 编辑窗口

        * 按键绑定

        * 自动缩进

        * 完成

        * 提示

        * 代码上下文

        * Python Shell 窗口

        * 文本颜色

      * 启动和代码执行

        * 命令行语法

        * 启动失败

        * 运行用户代码

        * Shell中的用户输出

        * 开发 tkinter 应用程序

        * 在没有子进程的情况下运行

      * 帮助和偏好

        * 帮助源

        * 首选项设置

        * macOS 上的IDLE

        * 扩展

    * 其他图形用户界面（GUI）包

  * 开发工具

    * "typing" --- 类型标注支持

      * 类型别名

      * NewType

      * Callable

      * 泛型(Generic)

      * 用户定义的泛型类型

      * "Any" 类型

      * Nominal vs structural subtyping

      * 类,函数和修饰器.

    * "pydoc" --- Documentation generator and online help system

    * "doctest" --- 测试交互性的Python示例

      * 简单用法：检查Docstrings中的示例

      * Simple Usage: Checking Examples in a Text File

      * How It Works

        * Which Docstrings Are Examined?

        * How are Docstring Examples Recognized?

        * What's the Execution Context?

        * What About Exceptions?

        * Option Flags

        * Directives

        * 警告

      * Basic API

      * Unittest API

      * Advanced API

        * DocTest 对象

        * Example Objects

        * DocTestFinder 对象

        * DocTestParser 对象

        * DocTestRunner 对象

        * OutputChecker 对象

      * 调试

      * Soapbox

    * "unittest" --- 单元测试框架

      * 基本实例

      * 命令行界面

        * 命令行选项

      * 探索性测试

      * 组织你的测试代码

      * 复用已有的测试代码

      * 跳过测试与预计的失败

      * Distinguishing test iterations using subtests

      * 类与函数

        * 测试用例

          * Deprecated aliases

        * Grouping tests

        * Loading and running tests

          * load_tests Protocol

      * Class and Module Fixtures

        * setUpClass and tearDownClass

        * setUpModule and tearDownModule

      * Signal Handling

    * "unittest.mock" --- mock对象库

      * Quick Guide

      * The Mock Class

        * Calling

        * Deleting Attributes

        * Mock names and the name attribute

        * Attaching Mocks as Attributes

      * The patchers

        * patch

        * patch.object

        * patch.dict

        * patch.multiple

        * patch methods: start and stop

        * patch builtins

        * TEST_PREFIX

        * Nesting Patch Decorators

        * Where to patch

        * Patching Descriptors and Proxy Objects

      * MagicMock and magic method support

        * Mocking Magic Methods

        * Magic Mock

      * Helpers

        * sentinel

        * DEFAULT

        * call

        * create_autospec

        * ANY

        * FILTER_DIR

        * mock_open

        * Autospeccing

        * Sealing mocks

    * "unittest.mock" 上手指南

      * 使用 mock

        * 模拟方法调用

        * 对象上的方法调用的 mock

        * Mocking Classes

        * Naming your mocks

        * Tracking all Calls

        * Setting Return Values and Attributes

        * Raising exceptions with mocks

        * Side effect functions and iterables

        * Mocking asynchronous iterators

        * Mocking asynchronous context manager

        * Creating a Mock from an Existing Object

      * Patch Decorators

      * Further Examples

        * Mocking chained calls

        * Partial mocking

        * Mocking a Generator Method

        * Applying the same patch to every test method

        * Mocking Unbound Methods

        * Checking multiple calls with mock

        * Coping with mutable arguments

        * Nesting Patches

        * Mocking a dictionary with MagicMock

        * Mock subclasses and their attributes

        * Mocking imports with patch.dict

        * Tracking order of calls and less verbose call assertions

        * More complex argument matching

    * 2to3 - 自动将 Python 2 代码转为 Python 3 代码

      * 使用 2to3

      * 修复器

      * "lib2to3" —— 2to3 支持库

    * "test" --- Regression tests package for Python

      * Writing Unit Tests for the "test" package

      * Running tests using the command-line interface

    * "test.support" --- Utilities for the Python test suite

    * "test.support.script_helper" --- Utilities for the Python
      execution tests

  * 调试和分析

    * 审计事件表

    * "bdb" --- Debugger framework

    * "faulthandler" --- Dump the Python traceback

      * Dumping the traceback

      * Fault handler state

      * Dumping the tracebacks after a timeout

      * Dumping the traceback on a user signal

      * Issue with file descriptors

      * 示例

    * "pdb" --- Python的调试器

      * Debugger Commands

    * Python Profilers 分析器

      * profile分析器简介

      * 实时用户手册

      * "profile" 和 "cProfile" 模块参考

      * "Stats" 类

      * 什么是确定性性能分析？

      * 局限性

      * 准确估量

      * 使用自定义计时器

    * "timeit" --- 测量小代码片段的执行时间

      * 基本示例

      * Python 接口

      * 命令行界面

      * 示例

    * "trace" --- Trace or track Python statement execution

      * Command-Line Usage

        * Main options

        * Modifiers

        * Filters

      * 编程接口

    * "tracemalloc" --- 跟踪内存分配

      * 示例

        * 显示前10项

        * 计算差异

        * Get the traceback of a memory block

        * Pretty top

      * API

        * 函数

        * 域过滤器

        * 过滤器

        * Frame

        * 快照

        * 统计

        * StatisticDiff

        * 跟踪

        * 回溯

  * 软件打包和分发

    * "distutils" --- 构建和安装 Python 模块

    * "ensurepip" --- Bootstrapping the "pip" installer

      * Command line interface

      * Module API

    * "venv" --- 创建虚拟环境

      * 创建虚拟环境

      * API

      * 一个扩展 "EnvBuilder" 的例子

    * "zipapp" --- Manage executable Python zip archives

      * Basic Example

      * 命令行界面

      * Python API

      * 示例

      * Specifying the Interpreter

      * Creating Standalone Applications with zipapp

        * Making a Windows executable

        * Caveats

      * The Python Zip Application Archive Format

  * Python运行时服务

    * "sys" --- 系统相关的参数和函数

    * "sysconfig" --- Provide access to Python's configuration
      information

      * 配置变量

      * 安装路径

      * 其他功能

      * Using "sysconfig" as a script

    * "builtins" --- 内建对象

    * "__main__" --- 顶层脚本环境

    * "warnings" --- Warning control

      * 警告类别

      * The Warnings Filter

        * Describing Warning Filters

        * 默认警告过滤器

        * Overriding the default filter

      * 暂时禁止警告

      * 测试警告

      * Updating Code For New Versions of Dependencies

      * Available Functions

      * Available Context Managers

    * "dataclasses" --- 数据类

      * 模块级装饰器、类和函数

      * 初始化后处理

      * 类变量

      * 仅初始化变量

      * 冻结的实例

      * 继承

      * 默认工厂函数

      * 可变的默认值

      * 异常

    * "contextlib" --- Utilities for "with"-statement contexts

      * 工具

      * 例子和配方

        * Supporting a variable number of context managers

        * Catching exceptions from "__enter__" methods

        * Cleaning up in an "__enter__" implementation

        * Replacing any use of "try-finally" and flag variables

        * Using a context manager as a function decorator

      * Single use, reusable and reentrant context managers

        * Reentrant context managers

        * Reusable context managers

    * "abc" --- 抽象基类

    * "atexit" --- 退出处理器

      * "atexit" 示例

    * "traceback" --- 打印或检索堆栈回溯

      * "TracebackException" Objects

      * "StackSummary" Objects

      * "FrameSummary" Objects

      * Traceback Examples

    * "__future__" --- Future 语句定义

    * "gc" --- 垃圾回收器接口

    * "inspect" --- 检查对象

      * 类型和成员

      * Retrieving source code

      * Introspecting callables with the Signature object

      * 类与函数

      * The interpreter stack

      * Fetching attributes statically

      * Current State of Generators and Coroutines

      * Code Objects Bit Flags

      * 命令行界面

    * "site" —— 指定域的配置钩子

      * Readline configuration

      * 模块内容

      * 命令行界面

  * 自定义 Python 解释器

    * "code" --- 解释器基类

      * 交互解释器对象

      * 交互式控制台对象

    * "codeop" --- 编译Python代码

  * 导入模块

    * "zipimport" --- 从 Zip 存档中导入模块

      * zipimporter 对象

      * 示例

    * "pkgutil" --- 包扩展工具

    * "modulefinder" --- 查找脚本使用的模块

      * "ModuleFinder" 的示例用法

    * "runpy" --- Locating and executing Python modules

    * "importlib" --- "import" 的实现

      * 概述

      * 函数

      * "importlib.abc" —— 关于导入的抽象基类

      * "importlib.resources" -- 资源

      * "importlib.machinery" -- Importers and path hooks

      * "importlib.util" -- Utility code for importers

      * 示例

        * Importing programmatically

        * Checking if a module can be imported

        * Importing a source file directly

        * Setting up an importer

        * Approximating "importlib.import_module()"

    * Using importlib.metadata

      * 概述

      * 可用 API

        * Entry points

        * Distribution metadata

        * Distribution versions

        * Distribution files

        * Distribution requirements

      * Distributions

      * Extending the search algorithm

  * Python 语言服务

    * "parser" --- Access Python parse trees

      * Creating ST Objects

      * Converting ST Objects

      * Queries on ST Objects

      * Exceptions and Error Handling

      * ST Objects

      * Example: Emulation of "compile()"

    * "ast" --- 抽象语法树

      * 节点类

      * 抽象文法

      * "ast" 中的辅助函数

    * "symtable" --- Access to the compiler's symbol tables

      * Generating Symbol Tables

      * Examining Symbol Tables

    * "symbol" --- 与 Python 解析树一起使用的常量

    * "token" --- 与Python解析树一起使用的常量

    * "keyword" --- 检验Python关键字

    * "tokenize" --- 对 Python 代码使用的标记解析器

      * 对输入进行解析标记

      * Command-Line Usage

      * 示例

    * "tabnanny" --- 模糊缩进检测

    * "pyclbr" --- Python module browser support

      * 函数对象

      * 类对象

    * "py_compile" --- Compile Python source files

    * "compileall" --- Byte-compile Python libraries

      * Command-line use

      * Public functions

    * "dis" --- Python 字节码反汇编器

      * 字节码分析

      * 分析函数

      * Python字节码说明

      * 操作码集合

    * "pickletools" --- pickle 开发者工具集

      * 命令行语法

        * 命令行选项

      * 编程接口

  * 杂项服务

    * "formatter" --- 通用格式化输出

      * The Formatter Interface

      * Formatter Implementations

      * The Writer Interface

      * Writer Implementations

  * Windows系统相关模块

    * "msilib" --- Read and write Microsoft Installer files

      * Database Objects

      * View Objects

      * Summary Information Objects

      * Record Objects

      * Errors

      * CAB Objects

      * Directory Objects

      * 相关特性

      * GUI classes

      * Precomputed tables

    * "msvcrt" --- Useful routines from the MS VC++ runtime

      * File Operations

      * Console I/O

      * Other Functions

    * "winreg" --- Windows 注册表访问

      * 函数

      * 常数

        * HKEY_* Constants

        * Access Rights

          * 64-bit Specific

        * Value Types

      * Registry Handle Objects

    * "winsound" --- Sound-playing interface for Windows

  * Unix 专有服务

    * "posix" --- 最常见的 POSIX 系统调用

      * 大文件支持

      * 重要的模块内容

    * "pwd" --- 用户密码数据库

    * "spwd" --- The shadow password database

    * "grp" --- The group database

    * "crypt" --- Function to check Unix passwords

      * Hashing Methods

      * Module Attributes

      * 模块函数

      * 示例

    * "termios" --- POSIX 风格的 tty 控制

      * 示例

    * "tty" --- 终端控制功能

    * "pty" --- 伪终端工具

      * 示例

    * "fcntl" --- The "fcntl" and "ioctl" system calls

    * "pipes" --- 终端管道接口

      * 模板对象

    * "resource" --- Resource usage information

      * Resource Limits

      * Resource Usage

    * "nis" --- Sun 的 NIS (黄页) 接口

    * Unix syslog 库例程

      * 示例

        * Simple example

  * 被取代的模块

    * "optparse" --- 解析器的命令行选项

      * 背景

        * 术语

        * What are options for?

        * 位置位置

      * 教程

        * Understanding option actions

        * The store action

        * Handling boolean (flag) options

        * Other actions

        * 默认值

        * Generating help

          * Grouping Options

        * Printing a version string

        * How "optparse" handles errors

        * Putting it all together

      * 参考指南

        * 创建解析器

        * 填充解析器

        * 定义选项

        * Option attributes

        * Standard option actions

        * Standard option types

        * 解析参数

        * Querying and manipulating your option parser

        * Conflicts between options

        * 清理

        * Other methods

      * Option Callbacks

        * Defining a callback option

        * How callbacks are called

        * Raising errors in a callback

        * Callback example 1: trivial callback

        * Callback example 2: check option order

        * Callback example 3: check option order (generalized)

        * Callback example 4: check arbitrary condition

        * Callback example 5: fixed arguments

        * Callback example 6: variable arguments

      * Extending "optparse"

        * Adding new types

        * Adding new actions

    * "imp" --- Access the *import* internals

      * 示例

  * 未创建文档的模块

    * 平台特定模块

* 扩展和嵌入 Python 解释器

  * 推荐的第三方工具

  * 不使用第三方工具创建扩展

    * 1. 使用 C 或 C++ 扩展 Python

      * 1.1. 一个简单的例子

      * 1.2. 关于错误和异常

      * 1.3. 回到例子

      * 1.4. 模块方法表和初始化函数

      * 1.5. 编译和链接

      * 1.6. 在C中调用Python函数

      * 1.7. 提取扩展函数的参数

      * 1.8. 给扩展函数的关键字参数

      * 1.9. 构造任意值

      * 1.10. 引用计数

        * 1.10.1. Python中的引用计数

        * 1.10.2. 拥有规则

        * 1.10.3. 危险的薄冰

        * 1.10.4. NULL指针

      * 1.11. 在C++中编写扩展

      * 1.12. 给扩展模块提供C API

    * 2. 自定义扩展类型：教程

      * 2.1. 基础

      * 2.2. Adding data and methods to the Basic example

      * 2.3. Providing finer control over data attributes

      * 2.4. Supporting cyclic garbage collection

      * 2.5. Subclassing other types

    * 3. 定义扩展类型：已分类主题

      * 3.1. 终结和内存释放

      * 3.2. 对象展示

      * 3.3. Attribute Management

        * 3.3.1. 泛型属性管理

        * 3.3.2. Type-specific Attribute Management

      * 3.4. Object Comparison

      * 3.5. Abstract Protocol Support

      * 3.6. Weak Reference Support

      * 3.7. 更多建议

    * 4. 构建C/C++扩展

      * 4.1. 使用distutils构建C和C++扩展

      * 4.2. 发布你的扩展模块

    * 5. 在Windows平台编译C和C++扩展

      * 5.1. A Cookbook Approach

      * 5.2. Differences Between Unix and Windows

      * 5.3. Using DLLs in Practice

  * 在更大的应用程序中嵌入 CPython 运行时

    * 1. 在其它应用程序嵌入 Python

      * 1.1. Very High Level Embedding

      * 1.2. Beyond Very High Level Embedding: An overview

      * 1.3. 纯嵌入

      * 1.4. Extending Embedded Python

      * 1.5. 在 C++ 中嵌入 Python

      * 1.6. 在类 Unix 系统中编译和链接

* Python/C API 参考手册

  * 概述

    * 代码标准

    * 包含文件

    * 有用的宏

    * 对象、类型和引用计数

      * 引用计数

        * Reference Count Details

      * 类型

    * 异常

    * 嵌入Python

    * 调试构建

  * 稳定的应用程序二进制接口

  * The Very High Level Layer

  * 引用计数

  * 异常处理

    * Printing and clearing

    * 抛出异常

    * Issuing warnings

    * Querying the error indicator

    * Signal Handling

    * Exception Classes

    * Exception Objects

    * Unicode Exception Objects

    * Recursion Control

    * 标准异常

    * 标准警告类别

  * 工具

    * 操作系统实用程序

    * 系统功能

    * 过程控制

    * 导入模块

    * 数据 marshal 操作支持

    * 语句解释及变量编译

      * 解析参数

        * 字符串和缓存区

        * 数字

        * 其他对象

        * API 函数

      * 创建变量

    * 字符串转换与格式化

    * 反射

    * 编解码器注册与支持功能

      * Codec 查找API

      * 用于Unicode编码错误处理程序的注册表API

  * 抽象对象层

    * 对象协议

    * 数字协议

    * 序列协议

    * 映射协议

    * 迭代器协议

    * 缓冲协议

      * 缓冲区结构

      * Buffer request types

        * request-independent fields

        * readonly, format

        * shape, strides, suboffsets

        * 连续性的请求

        * 复合请求

      * 复杂数组

        * NumPy-style: shape and strides

        * PIL-style: shape, strides and suboffsets

      * Buffer-related functions

    * 旧缓冲协议

  * 具体的对象层

    * 基本对象

      * Type 对象

        * Creating Heap-Allocated Types

      * "None" 对象

    * 数值对象

      * 整数型对象

      * 布尔对象

      * 浮点数对象

      * 复数对象

        * 表示复数的C结构体

        * 表示复数的Python对象

    * 序列对象

      * 字节对象

      * 字节数组对象

        * 类型检查宏

        * 直接 API 函数

        * 宏

      * Unicode Objects and Codecs

        * Unicode对象

          * Unicode类型

          * Unicode字符属性

          * Creating and accessing Unicode strings

          * Deprecated Py_UNICODE APIs

          * Locale Encoding

          * File System Encoding

          * wchar_t Support

        * Built-in Codecs

          * Generic Codecs

          * UTF-8 Codecs

          * UTF-32 Codecs

          * UTF-16 Codecs

          * UTF-7 Codecs

          * Unicode-Escape Codecs

          * Raw-Unicode-Escape Codecs

          * Latin-1 Codecs

          * ASCII Codecs

          * Character Map Codecs

          * MBCS codecs for Windows

          * Methods & Slots

        * Methods and Slot Functions

      * 元组对象

      * Struct Sequence Objects

      * 列表对象

    * 容器对象

      * 字典对象

      * 集合对象

    * 函数对象

      * 函数对象

      * 实例方法对象

      * 方法对象

      * Cell 对象

      * 代码对象

    * 其他对象

      * 文件对象

      * 模块对象

        * Initializing C modules

          * Single-phase initialization

          * Multi-phase initialization

          * Low-level module creation functions

          * Support functions

        * Module lookup

      * 迭代器对象

      * 描述符对象

      * 切片对象

      * Ellipsis Object

      * MemoryView 对象

      * 弱引用对象

      * 胶囊

      * 生成器对象

      * 协程对象

      * 上下文变量对象

      * DateTime 对象

  * Initialization, Finalization, and Threads

    * 在Python初始化之前

    * 全局配置变量

    * Initializing and finalizing the interpreter

    * Process-wide parameters

    * Thread State and the Global Interpreter Lock

      * Releasing the GIL from extension code

      * 非Python创建的线程

      * Cautions about fork()

      * 高阶 API

      * Low-level API

    * Sub-interpreter support

      * 错误和警告

    * 异步通知

    * 分析和跟踪

    * 高级调试器支持

    * Thread Local Storage Support

      * Thread Specific Storage (TSS) API

        * Dynamic Allocation

        * 方法

      * Thread Local Storage (TLS) API

  * Python初始化配置

    * PyWideStringList

    * PyStatus

    * PyPreConfig

    * Preinitialization with PyPreConfig

    * PyConfig

    * Initialization with PyConfig

    * Isolated Configuration

    * Python Configuration

    * 路径配置

    * Py_RunMain()

    * Multi-Phase Initialization Private Provisional API

  * 内存管理

    * 概述

    * 原始内存接口

    * 内存接口

    * 对象分配器

    * 默认内存分配器

    * Customize Memory Allocators

    * The pymalloc allocator

      * Customize pymalloc Arena Allocator

    * tracemalloc C API

    * 示例

  * 对象实现支持

    * 在堆中分配对象

    * Common Object Structures

    * Type 对象

      * 快速参考

        * "tp 槽"

        * sub-slots

        * slot typedefs

      * PyTypeObject Definition

      * PyObject Slots

      * PyVarObject Slots

      * PyTypeObject Slots

      * Heap Types

    * Number Object Structures

    * Mapping Object Structures

    * Sequence Object Structures

    * Buffer Object Structures

    * Async Object Structures

    * Slot Type typedefs

    * 例子

    * 使对象类型支持循环垃圾回收

  * API 和 ABI 版本管理

* 分发 Python 模块

  * 关键术语

  * 开源许可与协作

  * 安装工具

  * 阅读Python包用户指南

  * 我该如何...？

    * ...为我的项目选择一个名字？

    * ...创建和分发二进制扩展？

* 安装 Python 模块

  * 关键术语

  * 基本使用

  * 我应如何 ...？

    * ... 在 Python 3.4 之前的 Python 版本中安装 "pip" ？

    * ... 只为当前用户安装软件包？

    * ... 安装科学计算类 Python 软件包？

    * ... 使用并行安装的多个 Python 版本？

  * 常见的安装问题

    * 在 Linux 的系统 Python 版本上安装

    * 未安装 pip

    * 安装二进制编译扩展

* Python 常用指引

  * 将 Python 2 代码迁移到 Python 3

    * 简要说明

    * 详情

      * 删除对Python 2.6及更早版本的支持

      * Make sure you specify the proper version support in your
        "setup.py" file

      * 良好的测试覆盖率

      * 了解Python 2 和 3之间的区别

      * 更新代码

        * 除法

        * 文本与二进制数据

        * Use feature detection instead of version detection

      * Prevent compatibility regressions

      * Check which dependencies block your transition

      * Update your "setup.py" file to denote Python 3 compatibility

      * Use continuous integration to stay compatible

      * 考虑使用可选的静态类型检查

  * 将扩展模块移植到 Python 3

  * 用 Python 进行 Curses 编程

    * curses 是什么？

      * Python 的 curses 模块

    * 开始和结束curses应用程序

    * 窗口和面板

    * 显示文字

      * 属性和颜色

    * 用户输入

    * 更多的信息

  * 描述器使用指南

    * 摘要

    * 定义和简介

    * 描述器协议

    * 调用描述器

    * 描述器示例

    * 属性

    * 函数和方法

    * 静态方法和类方法

  * 函数式编程指引

    * 概述

      * 形式证明

      * 模块化

      * 易于调试和测试

      * 组合性

    * 迭代器

      * 支持迭代器的数据类型

    * 生成器表达式和列表推导式

    * 生成器

      * 向生成器传递值

    * 内置函数

    * itertools 模块

      * 创建新的迭代器

      * 对元素使用函数

      * 选择元素

      * 组合函数

      * 为元素分组

    * functools 模块

      * operator 模块

    * 小函数和 lambda 表达式

    * 修订记录和致谢

    * 引用文献

      * 通用文献

      * Python 相关

      * Python 文档

  * 日志 HOWTO

    * 日志基础教程

      * 什么时候使用日志

      * 一个简单的例子

      * 记录日志到文件

      * 从多个模块记录日志

      * 记录变量数据

      * 更改显示消息的格式

      * 在消息中显示日期/时间

      * 后续步骤

    * 进阶日志教程

      * 记录流程

      * 记录器

      * 处理程序

      * 格式化程序

      * 配置日志记录

      * 如果没有提供配置会发生什么

      * 配置库的日志记录

    * 日志级别

      * 自定义级别

    * 有用的处理程序

    * 记录日志中引发的异常

    * 使用任意对象作为消息

    * 优化

  * 日志操作手册

    * 在多个模块中使用日志

    * 在多线程中使用日志

    * 使用多个日志处理器和多种格式化

    * 在多个地方记录日志

    * 日志服务器配置示例

    * 处理日志处理器的阻塞

    * 通过网络发送和接收日志

    * 在日志记录中添加上下文信息

      * 使用日志适配器传递上下文信息

        * 使用除字典之外的其它对象传递上下文信息

      * 使用过滤器传递上下文信息

    * 从多个进程记录至单个文件

      * Using concurrent.futures.ProcessPoolExecutor

    * 轮换日志文件

    * 使用其他日志格式化方式

    * Customizing "LogRecord"

    * Subclassing QueueHandler - a ZeroMQ example

    * Subclassing QueueListener - a ZeroMQ example

    * An example dictionary-based configuration

    * Using a rotator and namer to customize log rotation processing

    * A more elaborate multiprocessing example

    * Inserting a BOM into messages sent to a SysLogHandler

    * Implementing structured logging

    * Customizing handlers with "dictConfig()"

    * Using particular formatting styles throughout your application

      * Using LogRecord factories

      * Using custom message objects

    * Configuring filters with "dictConfig()"

    * Customized exception formatting

    * Speaking logging messages

    * 缓冲日志消息并有条件地输出它们

    * 通过配置使用UTC (GMT) 格式化时间

    * 使用上下文管理器进行选择性记录

    * A CLI application starter template

    * A Qt GUI for logging

  * 正则表达式HOWTO

    * 概述

    * 简单模式

      * 匹配字符

      * 重复

    * 使用正则表达式

      * 编译正则表达式

      * 反斜杠灾难

      * 应用匹配

      * 模块级别函数

      * 编译标志

    * 更多模式能力

      * 更多元字符

      * 分组

      * 非捕获和命名组

      * 前向断言

    * 修改字符串

      * 分割字符串

      * 搜索和替换

    * 常见问题

      * 使用字符串方法

      * match() 和 search()

      * 贪婪与非贪婪

      * 使用 re.VERBOSE

    * 反馈

  * 套接字编程指南

    * 套接字

      * 历史

    * 创建套接字

      * 进程间通信

    * 使用一个套接字

      * 二进制数据

    * 断开连接

      * 套接字何时销毁

    * 非阻塞的套接字

  * 排序指南

    * 基本排序

    * 关键函数

    * Operator 模块函数

    * 升序和降序

    * 排序稳定性和排序复杂度

    * 使用装饰-排序-去装饰的旧方法

    * 使用 *cmp* 参数的旧方法

    * 其它

  * Unicode 指南

    * Unicode 概述

      * 定义

      * 编码

      * 引用文献

    * Python's Unicode Support

      * The String Type

      * Converting to Bytes

      * Unicode Literals in Python Source Code

      * Unicode Properties

      * Comparing Strings

      * Unicode Regular Expressions

      * 引用文献

    * Reading and Writing Unicode Data

      * Unicode filenames

      * Tips for Writing Unicode-aware Programs

        * Converting Between File Encodings

        * Files in an Unknown Encoding

      * 引用文献

    * 致谢

  * HOWTO 使用 urllib 包获取网络资源

    * 概述

    * 提取URL

      * 数据

      * Headers

    * 处理异常

      * URLError

      * HTTPError

        * 错误代码

      * 包装起来

        * 数字1

        * Number 2

    * info and geturl

    * Openers and Handlers

    * 基本认证

    * 代理

    * Sockets and Layers

    * 脚注

  * Argparse 教程

    * 概念

    * 基础

    * 位置参数介绍

    * 可选参数介绍

      * 短选项

    * 结合位置参数和可选参数

    * 进行一些小小的改进

      * 矛盾的选项

    * 后记

  * ipaddress模块介绍

    * 创建 Address/Network/Interface 对象

      * 关于IP版本的说明

      * IP主机地址

      * 定义网络

      * 主机接口

    * 审查 Address/Network/Interface 对象

    * Network 作为 Address 列表

    * 比较

    * 将IP地址与其他模块一起使用

    * 实例创建失败时获取更多详细信息

  * Argument Clinic How-To

    * The Goals Of Argument Clinic

    * Basic Concepts And Usage

    * Converting Your First Function

    * Advanced Topics

      * Symbolic default values

      * Renaming the C functions and variables generated by Argument
        Clinic

      * Converting functions using PyArg_UnpackTuple

      * Optional Groups

      * Using real Argument Clinic converters, instead of "legacy
        converters"

      * Py_buffer

      * Advanced converters

      * Parameter default values

      * The "NULL" default value

      * Expressions specified as default values

      * Using a return converter

      * Cloning existing functions

      * Calling Python code

      * Using a "self converter"

      * Writing a custom converter

      * Writing a custom return converter

      * METH_O and METH_NOARGS

      * tp_new and tp_init functions

      * Changing and redirecting Clinic's output

      * The #ifdef trick

      * Using Argument Clinic in Python files

  * 使用 DTrace 和 SystemTap 检测CPython

    * 启用静态标记

    * 静态DTrace探针

    * Static SystemTap markers

    * Available static markers

    * SystemTap Tapsets

    * 示例

* Python 常见问题

  * Python常见问题

    * 一般信息

    * 现实世界中的 Python

  * 编程常见问题

    * 一般问题

    * 核心语言

    * 数字和字符串

    * 性能

    * 序列（元组/列表）

    * 对象

    * 模块

  * 设计和历史常见问题

    * 为什么Python使用缩进来分组语句？

    * 为什么简单的算术运算得到奇怪的结果？

    * 为什么浮点计算不准确？

    * 为什么Python字符串是不可变的？

    * 为什么必须在方法定义和调用中显式使用“self”？

    * 为什么不能在表达式中赋值？

    * 为什么Python对某些功能（例如list.index()）使用方法来实现，而其他
      功能（例如len(List)）使用函数实现？

    * 为什么 join()是一个字符串方法而不是列表或元组方法？

    * 异常有多快？

    * 为什么Python中没有switch或case语句？

    * 难道不能在解释器中模拟线程，而非得依赖特定于操作系统的线程实现吗
      ？

    * 为什么lambda表达式不能包含语句？

    * 可以将Python编译为机器代码，C或其他语言吗？

    * Python如何管理内存？

    * 为什么CPython不使用更传统的垃圾回收方案？

    * CPython退出时为什么不释放所有内存？

    * 为什么有单独的元组和列表数据类型？

    * 列表是如何在CPython中实现的？

    * 字典是如何在CPython中实现的？

    * 为什么字典key必须是不可变的？

    * 为什么 list.sort() 没有返回排序列表？

    * 如何在Python中指定和实施接口规范？

    * 为什么没有goto？

    * 为什么原始字符串（r-strings）不能以反斜杠结尾？

    * 为什么Python没有属性赋值的“with”语句？

    * 为什么 if/while/def/class语句需要冒号？

    * 为什么Python在列表和元组的末尾允许使用逗号？

  * 代码库和插件 FAQ

    * 通用的代码库问题

    * 通用任务

    * 线程相关

    * 输入输出

    * 网络 / Internet 编程

    * 数据库

    * 数学和数字

  * 扩展/嵌入常见问题

    * 可以使用 C 语言创建自己的函数吗？

    * 可以使用 C++ 语言创建自己的函数吗？

    * C很难写，有没有其他选择？

    * 如何在 C 中执行任意 Python 语句？

    * 如何在 C 中对任意 Python 表达式求值？

    * 如何从Python对象中提取C的值？

    * 如何使用Py_BuildValue()创建任意长度的元组？

    * 如何从C调用对象的方法？

    * 如何捕获PyErr_Print()（或打印到stdout / stderr的任何内容）的输出
      ？

    * 如何从C访问用Python编写的模块？

    * 如何在 Python 中对接 C ++ 对象？

    * 我使用Setup文件添加了一个模块，为什么make失败了？

    * 如何调试扩展？

    * 我想在Linux系统上编译一个Python模块，但是缺少一些文件。为什么?

    * 如何区分“输入不完整”和“输入无效”？

    * 如何找到未定义的g++符号__builtin_new或__pure_virtual？

    * 能否创建一个对象类，其中部分方法在C中实现，而其他方法在Python中
      实现（例如通过继承）？

  * Python在Windows上的常见问题

    * 我怎样在Windows下运行一个Python程序？

    * 我怎么让 Python 脚本可执行？

    * 为什么有时候 Python 程序会启动缓慢？

    * 我怎样使用Python脚本制作可执行文件？

    * "*.pyd" 文件和DLL文件相同吗？

    * 我怎样将Python嵌入一个Windows程序？

    * 如何让编辑器不要在我的 Python 源代码中插入 tab ？

    * 如何在不阻塞的情况下检查按键？

  * 图形用户界面（GUI）常见问题

    * 图形界面常见问题

    * Python 是否有平台无关的图形界面工具包？

    * 有哪些Python的GUI工具是某个平台专用的？

    * 有关Tkinter的问题

  * “为什么我的电脑上安装了 Python ？”

    * 什么是Python？

    * 为什么我的电脑上安装了 Python ？

    * 我能删除 Python 吗？

* 术语对照表

* 文档说明

  * Python 文档贡献者

* 解决 Bug

  * 文档错误

  * 使用 Python 的问题追踪系统

  * 开始为 Python 贡献您的知识

* 版权

* 历史和许可证

  * 该软件的历史

  * 获取或以其他方式使用 Python 的条款和条件

    * 用于 PYTHON 3.8.3rc1 的 PSF 许可协议

    * 用于 PYTHON 2.0 的 BEOPEN.COM 许可协议

    * 用于 PYTHON 1.6.1 的 CNRI 许可协议

    * 用于 PYTHON 0.9.0 至 1.2 的 CWI 许可协议

  * 被收录软件的许可证与鸣谢

    * Mersenne Twister

    * 套接字

    * 异步套接字服务

    * Cookie 管理

    * 执行追踪

    * UUencode 与 UUdecode 函数

    * XML 远程过程调用

    * test_epoll

    * Select kqueue

    * SipHash24

    * strtod 和 dtoa

    * OpenSSL

    * expat

    * libffi

    * zlib

    * cfuhash

    * libmpdec

    * W3C C14N 测试套件
