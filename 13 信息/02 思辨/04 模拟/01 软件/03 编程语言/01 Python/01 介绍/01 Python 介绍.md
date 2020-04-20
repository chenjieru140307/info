# Python 介绍

一般在操控硬件的场合使用 C++，在快速开发时候使用 Python。


## 用途

- 可以做日常任务，比如自动备份你的 MP3；
- 可以做网站，很多著名的网站包括 YouTube 就是 Python 写的；
- 可以做网络游戏的后台，很多在线游戏的后台都是 Python 开发的。
- 等等


## 优缺点

- 缺点
  - **代码不能加密。**如果要发布你的 Python 程序，实际上就是发布源代码，这一点跟 C 语言不同，C语言不用发布源代码，只需要把编译后的机器码（也就是你在 Windows 上常见的 xxx.exe文件）发布出去。要从机器码反推出 C 代码是不可能的，所以，凡是编译型的语言，都没有这个问题，而解释型的语言，则必须把源码发布出去。这个缺点仅限于你要编写的软件需要卖给别人挣钱的时候。好消息是目前的互联网时代，靠卖软件授权的商业模式越来越少了，靠网站和移动应用卖服务的模式越来越多了，后一种模式不需要把源码给别人。
- 优点
  - 用 Python 写程序时无需考虑如何管理程序使用的内存等底层细节。

## 开源程度

Python是 FLOSS（自由/开放源码软件）之一。使用者可以自由地发布这个软件的拷贝、阅读它的源代码、对它做改动、把它的一部分用于新的自由软件中。FLOSS是基于一个团体分享知识的概念。

## 速度

C 程序运行 1 秒钟，Java程序可能需要 2 秒，而 Python 程序可能就需要 10 秒。不过，Python 的底层是用 C 语言写的，很多标准库和第三方库也都是用 C 写的，运行速度非常快。

Python开发人员尽量避开不成熟或者不重要的优化。一些针对非重要部位的加快运行速度的补丁通常不会被合并到 Python 内。所以很多人认为 Python 很慢。不过，根据二八定律，大多数程序对速度要求不高。在某些对运行速度要求很高的情况，Python设计师倾向于使用 JIT 技术，或者用使用 C/C++语言改写这部分程序。可用的 JIT 技术是 PyPy。

Psyco:一个 Python 代码加速度器，可使 Python 代码的执行速度提高到与编译语言一样的水平。

不清楚的：

- <span style="color:red;">PyPy 是什么？什么时候使用？怎么使用？</span>
- <span style="color:red;">Psyco 是什么？什么时候使用？怎么使用？</span>


## 多平台支持

由于它的开源本质，Python 已经被移植在许多平台上（经过改动使它能够工作在不同平台上）。

这些平台包括 Linux、Windows、FreeBSD、Macintosh、Solaris、OS/2、Amiga、AROS、AS/400、BeOS、OS/390、z/OS、Palm OS、QNX、VMS、Psion、Acom RISC OS、VxWorks、PlayStation、Sharp Zaurus、Windows CE、PocketPC、Symbian以及 Google 基于 linux 开发的 android 平台。

<span style="color:red;">在 android 平台上是什么样的？可以使用 python 开发 app吗？</span>

# Python 标准库

Python 拥有一个强大的标准库。

Python 语言的核心只包含数字、字符串、列表、字典、文件等常见类型和函数，而由 Python 标准库提供了系统管理、网络通信、文本处理、数据库接口、图形系统、XML处理等额外的功能。


Python标准库的主要功能有：

1. 文本处理，包含文本格式化、正则表达式匹配、文本差异计算与合并、Unicode支持，二进制数据处理等功能
2. 文件处理，包含文件操作、创建临时文件、文件压缩与归档、操作配置文件等功能
3. 操作系统功能，包含线程与进程支持、IO复用、日期与时间处理、调用系统函数、写日记(logging)等功能
4. 网络通信，包含网络套接字，SSL加密通信、异步网络通信等功能
5. 网络协议，支持 HTTP，FTP，SMTP，POP，IMAP，NNTP，XMLRPC等多种网络协议，并提供了编写网络服务器的框架 W3C 格式支持，包含 HTML，SGML，XML的处理。
6. 其它功能，包括国际化支持、数学运算、HASH、Tkinter等

# Python 执行过程介绍


Python 在执行时：

1. 首先会将.py 文件中的源代码编译成 Python 的 byte code（字节码）
2. 然后再由 Python Virtual Machine（Python虚拟机）来执行这些编译好的 byte code。


这种机制的基本思想跟 Java，.NET是一致的。然而，Python Virtual Machine与 Java 或.NET的 Virtual Machine不同的是，Python的 Virtual Machine是一种更高级的 Virtual Machine。这里的高级并不是通常意义上的高级，不是说 Python 的 Virtual Machine比 Java 或.NET的功能更强大，而是说和 Java 或.NET相比，Python的 Virtual Machine距离真实机器的距离更远。或者可以这么说，Python的 Virtual Machine是一种抽象层次更高的 Virtual Machine。

基于 C 的 Python 编译出的字节码文件，通常是.pyc格式。<span style="color:red;">没明白？</span>

一个用编译性语言比如 C 或 C++写的程序可以从源文件（即 C 或 C++语言）转换到一个你的计算机使用的语言（二进制代码，即 0 和 1）。这个过程通过编译器和不同的标记、选项完成。运行程序的时候，连接/转载器软件把你的程序从硬盘复制到内存中并且运行。而 Python 语言写的程序不需要编译成二进制代码。你可以直接从源代码运行 程序。在计算机内部，Python解释器把源代码转换成称为字节码的中间形式，然后再把它翻译成计算机使用的机器语言并运行。这使得使用 Python 更加简单。也使得 Python 程序更加易于移植。

# Python 的扩充

Python本身被设计为可扩充的。并非所有的特性和功能都集成到语言核心。Python提供了丰富的 API 和工具，以便程序员能够轻松地使用 C 语言、C++、Cython来编写扩充模块。

这几年，Cython项目（http://cython.org）已经成为 Python 领域中创建编译型扩展以及对接 C/C++ 代码的一大途径。

Python编译器本身也可以被集成到其它需要脚本语言的程序内。因此，很多人还把 Python 作为一种“胶水语言”（glue language）使用。使用 Python 将其他语言编写的程序进行集成和封装。在 Google 内部的很多项目，例如 Google Engine使用 C++编写性能要求极高的部分，然后用 Python 或 Java/Go调用相应的模块。

不清楚的：

- <span style="color:red;">怎么使用 CPython 来编写扩充模块？</span>
- <span style="color:red;">Cython 到底有什么厉害的地方？要总结下</span>
- <span style="color:red;">怎么把编译器封装到别的程序里面？</span>
