---
title: pandas 读取 csv 处理时报错 pandas.io.common.CParserError Error 
toc: true
date: 2018-06-12 13:08:12
---
---
author: evo
comments: true
date: 2018-05-06 01:11:55+00:00
layout: post
link: http://106.15.37.116/2018/05/06/pandas-read_csv-error/
slug: pandas-read_csv-error
title: 'tokenizing data. C error:
  Expected 1 fields in line 7, saw 2'
wordpress_id: 5312
categories:
- 随想与反思
tags:
- Python
---

<!-- more -->

[mathjax]

**注：非原创，推荐直接看原文**


# 相关






  1. [pandas读取 csv 处理时报错：ParserError: Error tokenizing data. C error: Expected 1 fields in line 29, saw 2](https://blog.csdn.net/yj928674542/article/details/75634197)




## 可以补充进来的






  * aaa




# MOTIVE






  * 在 read_csv的时候出现错误





* * *





# 错误出现的原因：


csv文件默认的是以逗号为分隔符，但是由于中文中逗号的使用率很高，所以有的时候使用 pandas 写入 csv时，会设置参数 sep，比如 sep=’\t’ ，即以 tab 为分隔符写入，毕竟 tab 在中文习惯里用的很少。

而这个时候如果你读取的时候，没有设定 sep，就会产生这种错误：

ParserError: Error tokenizing data. C error: Expected 1 fields in line 29, saw 2


# 解决的办法：


在读取 csv 进行数据处理时，要先看一下 csv 里面是以什么符号作为 sep 的，然后在读取的时候设定这个符号：比如 sep="$"，那么：


    df=pd.read_csv('path',sep="$")

























* * *





# COMMENT



