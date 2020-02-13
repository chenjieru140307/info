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
