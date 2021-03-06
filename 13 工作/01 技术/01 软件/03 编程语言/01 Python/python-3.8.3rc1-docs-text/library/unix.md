Unix 专有服务
*************

本章描述的模块提供了 Unix 操作系统独有特性的接口，在某些情况下也适用于
它的某些或许多衍生版。 以下为模块概览：

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
