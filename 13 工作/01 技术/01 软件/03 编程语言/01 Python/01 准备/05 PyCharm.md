# PyCharm

疑问：

- pycharm 传入命令行参数有没有更好的方式。比如配置好，就按照配置的传。



- Ctrl+Alt+L 或者 Ctrl+Alt+Shift+L | 格式化程序    快捷键冲突的时候会无法起作用，之前是跟网易云音乐冲突
- Shift+F6 重命名文件。这一快捷键会和 qq 的锁定（lock）键冲突。F2 无法使用。




## 常用的快捷键

| 快捷键                           | 说明                                                                                         |
| -------------------------------- | -------------------------------------------------------------------------------------------- |
| Ctrl+R                           | 替换                                                                                         |
| Ctrl+alt+left                    | 返回上次查看（编辑）过的地方    这个按键有时候会与屏幕的旋转相冲突，可以在网上查到修改的方法 |
| Shift+F6                         | 重命名 这一快捷键会和 qq 的锁定（lock）键冲突                                                |
| Ctrl+Alt+L 或者 Ctrl+Alt+Shift+L | 格式化程序    快捷键冲突的时候会无法起作用，之前是跟网易云音乐冲突了                         |
| Ctrl+F5                          | 调试                                                                                         |
| Ctrl+F2                          | 停止调试                                                                                     |
| F9                               | 继续调试                                                                                     |
| F8                               | 单步调试                                                                                     |
| Ctrl+D                           | 复制当前行并添加在下方                                                                       |
| `Ctrl+/`                         | 多行快速注释                                                                                 |
| Ctrl+Shift+F10                   | 开始运行                                                                                     |
| Shift+Esc                        | 将左侧的栏隐藏起来                                                                           |



## 不常用的快捷键


Ctrl+F 查找

F3 查找下一个

Shift+F3 查找上一个



PyCharm学习技巧 Learning tips
/Pythoncharm/help/tip of the day:
A special variant of the Code Completion feature invoked by pressing Ctrl+Space twice allows you to complete the name of any class no matter if it was imported in the current file or not. If the class is not imported yet, the import statement is generated automatically.
You can quickly find all places where a particular class, method or variable is used in the whole project by positioning the caret at the symbol's name or at its usage in code and pressing Alt+Shift+F7 (Find Usages in the popup menu).
To navigate to the declaration of a class, method or variable used somewhere in the code, position the caret at the usage and press F12. You can also click the mouse on usages with the Ctrl key pressed to jump to declarations.
You can easily rename your local variables with automatic correction of all places where they are used.
To try it, place the caret at the variable you want to rename, and press Shift+F6 (Refactor | Rename). Type the new name in the popup window that appears, or select one of the suggested names, and press Enter.

...

切换
Use Alt+Up and Alt+Down keys to quickly move between methods in the editor.
Use Ctrl+Shift+F7 (Edit | Find | Highlight Usages in File) to quickly highlight usages of some variable in the current file.
选择
You can easily make column selection by dragging your mouse pointer while keeping the Alt key pressed.
补全
Working in the interactive consoles, you don't need to memorise the command line syntax or available functions. Instead, you can use the familiar code completion Ctrl+Space. Moreover, from within the lookup list, you can press Ctrl+Q to view the item's documentation.
显示
Use F3 and Shift+F3 keys to navigate through highlighted usages.
Press Escape to remove highlighting.
历史
Ctrl+Shift+Backspace (Navigate | Last Edit Location) brings you back to the last place where you made changes in the code.
Pressing Ctrl+Shift+Backspace a few times moves you deeper into your changes history.
Ctrl+E (View | Recent Files) brings a popup list of the recently visited files. Choose the desired file and press Enter to open it.
Use Alt+Shift+C to quickly review your recent changes to the project.
剪切板
Use the Ctrl+Shift+V shortcut to choose and insert recent clipboard contents into the text.
If nothing is selected in the editor, and you press Ctrl+C, then the whole line at caret is copied to the clipboard.
run/debug
By pressing Alt+Shift+F10 you can access the Run/Debug dropdown on the main toolbar, without the need to use your mouse.

在PyCharm安装目录 /opt/PyCharm-3.4.1/help目录下可以找到ReferenceCard.pdf快捷键英文版说明 or 打开PyCharm > help > default keymap ref

 

PyCharm3.0默认快捷键(翻译的)
PyCharm Default Keymap(mac中还不一样，ctrl可能用cmd或者cmd+ctrl代替)

1、编辑（Editing）

Ctrl + Space    基本的代码完成（类、方法、属性）
Ctrl + Alt + Space  快速导入任意类
Ctrl + Shift + Enter    语句完成
Ctrl + P    参数信息（在方法中调用参数）

Ctrl + Q    快速查看文档

F1   外部文档

Shift + F1    外部文档，进入web文档主页

Ctrl + Shift + Z --> Redo 重做

Ctrl + 悬浮/单击鼠标左键    简介/进入代码定义
Ctrl + F1    显示错误描述或警告信息
Alt + Insert    自动生成代码
Ctrl + O    重新方法
Ctrl + Alt + T    选中
Ctrl + /    行注释/取消行注释
Ctrl + Shift + /    块注释
Ctrl + W    选中增加的代码块
Ctrl + Shift + W    回到之前状态
Ctrl + Shift + ]/[     选定代码块结束、开始
Alt + Enter    快速修正
Ctrl + Alt + L     代码格式化
Ctrl + Alt + O    优化导入
Ctrl + Alt + I    自动缩进
Tab / Shift + Tab  缩进、不缩进当前行
Ctrl+X/Shift+Delete    剪切当前行或选定的代码块到剪贴板
Ctrl+C/Ctrl+Insert    复制当前行或选定的代码块到剪贴板
Ctrl+V/Shift+Insert    从剪贴板粘贴
Ctrl + Shift + V    从最近的缓冲区粘贴
Ctrl + D  复制选定的区域或行
Ctrl + Y    删除选定的行
Ctrl + Shift + J  添加智能线
Ctrl + Enter   智能线切割
Shift + Enter    另起一行
Ctrl + Shift + U  在选定的区域或代码块间切换
Ctrl + Delete   删除到字符结束
Ctrl + Backspace   删除到字符开始
Ctrl + Numpad+/-   展开/折叠代码块（当前位置的：函数，注释等）
Ctrl + shift + Numpad+/-   展开/折叠所有代码块
Ctrl + F4   关闭运行的选项卡
 2、查找/替换(Search/Replace)
F3   下一个
Shift + F3   前一个
Ctrl + R   替换
Ctrl + Shift + F  或者连续2次敲击shift   全局查找{可以在整个项目中查找某个字符串什么的，如查找某个函数名字符串看之前是怎么使用这个函数的}
Ctrl + Shift + R   全局替换
 3、运行(Running)
Alt + Shift + F10   运行模式配置
Alt + Shift + F9    调试模式配置
Shift + F10    运行
Shift + F9   调试
Ctrl + Shift + F10   运行编辑器配置
Ctrl + Alt + R   运行manage.py任务
 4、调试(Debugging)
F8   跳过
F7   进入
Shift + F8   退出
Alt + F9    运行游标
Alt + F8    验证表达式
Ctrl + Alt + F8   快速验证表达式
F9    恢复程序
Ctrl + F8   断点开关
Ctrl + Shift + F8   查看断点
 5、导航(Navigation)
Ctrl + N    跳转到类
Ctrl + Shift + N    跳转到符号

Alt + Right/Left    跳转到下一个、前一个编辑的选项卡（代码文件）（cmd+alt+right/left mac）

Alt + Up/Down跳转到上一个、下一个方法

 

F12    回到先前的工具窗口
Esc    从工具窗口回到编辑窗口
Shift + Esc   隐藏运行的、最近运行的窗口
Ctrl + Shift + F4   关闭主动运行的选项卡
Ctrl + G    查看当前行号、字符号
Ctrl + E   当前文件弹出，打开最近使用的文件列表
Ctrl+Alt+Left/Right   后退、前进

Ctrl+Shift+Backspace    导航到最近编辑区域 {差不多就是返回上次编辑的位置}

Alt + F1   查找当前文件或标识
Ctrl+B / Ctrl+Click    跳转到声明
Ctrl + Alt + B    跳转到实现
Ctrl + Shift + I查看快速定义
Ctrl + Shift + B跳转到类型声明

Ctrl + U跳转到父方法、父类

Ctrl + ]/[跳转到代码块结束、开始

Ctrl + F12弹出文件结构
Ctrl + H类型层次结构
Ctrl + Shift + H方法层次结构
Ctrl + Alt + H调用层次结构
F2 / Shift + F2下一条、前一条高亮的错误
F4 / Ctrl + Enter编辑资源、查看资源
Alt + Home显示导航条F11书签开关
Ctrl + Shift + F11书签助记开关
Ctrl + #[0-9]跳转到标识的书签
Shift + F11显示书签
 6、搜索相关(Usage Search)
Alt + F7/Ctrl + F7文件中查询用法
Ctrl + Shift + F7文件中用法高亮显示
Ctrl + Alt + F7显示用法
 7、重构(Refactoring)
F5复制F6剪切
Alt + Delete安全删除
Shift + F6重命名
Ctrl + F6更改签名
Ctrl + Alt + N内联
Ctrl + Alt + M提取方法
Ctrl + Alt + V提取属性
Ctrl + Alt + F提取字段
Ctrl + Alt + C提取常量
Ctrl + Alt + P提取参数
 8、控制VCS/Local History
Ctrl + K提交项目
Ctrl + T更新项目
Alt + Shift + C查看最近的变化
Alt + BackQuote(’)VCS快速弹出
 9、模版(Live Templates)
Ctrl + Alt + J当前行使用模版
Ctrl +Ｊ插入模版
 10、基本(General)
Alt + #[0-9]打开相应的工具窗口
Ctrl + Alt + Y同步
Ctrl + Shift + F12最大化编辑开关
Alt + Shift + F添加到最喜欢
Alt + Shift + I根据配置检查当前文件
Ctrl + BackQuote(’)快速切换当前计划
Ctrl + Alt + S　打开设置页
Ctrl + Shift + A查找编辑器里所有的动作

Ctrl + Tab在窗口间进行切换





## 设置

PyCharm 中的设置是可以导入和导出的，file>export settings可以保存当前PyCharm中的设置为jar文件，重装时可以直接import settings>jar文件，就不用重复配置了。<span style="color:red;">确认下。</span>


### 去掉波浪线


刚开始使用的时候，PyCharm 上是由很多波浪线的。


可以这样去掉：

`Editor->Color Scheme->General` 然后找 `Errors and Warnings->Typo` ，将 `effects` 去除勾选，然后再找 `Weak Warning`，将 `effects` 去除勾选。

### PyCharm 中将 tab 键设置成 4 个空格


`File->Setting->Editor->Code Style->Python`

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/181017/BLI87hc3K9.png?imageslim">
</p>



### file -> Setting ->Editor

1. 设置Python自动引入包，要先在 >general > autoimport -> Python :show popup

快捷键：Alt + Enter: 自动添加包

2. “代码自动完成”时间延时设置

> Code Completion   -> Auto code completion in (ms):0  -> Autopopup in (ms):500

3. PyCharm中默认是不能用Ctrl+滚轮改变字体大小的，可以在 >Mouse中设置

4. 显示“行号”与“空白字符”

> Appearance  -> 勾选“Show line numbers”、“Show whitespaces”、“Show method separators”

5. 设置编辑器“颜色与字体”主题

> Colors & Fonts -> Scheme name -> 选择"monokai"“Darcula”

说明：先选择“monokai”，再“Save As”为"monokai-pipi"，因为默认的主题是“只读的”，一些字体大小颜色什么的都不能修改，拷贝一份后方可修改！

修改字体大小

> Colors & Fonts -> Font -> Size -> 设置为“14”

6. 设置缩进符为制表符“Tab”

File -> Default Settings -> Code Style

-> General -> 勾选“Use tab character”

-> Python -> 勾选“Use tab character”

-> 其他的语言代码同理设置

7. 去掉默认折叠

> Code Folding -> Collapse by default -> 全部去掉勾选

8. PyCharm默认是自动保存的，习惯自己按ctrl + s  的可以进行如下设置：

> General -> Synchronization -> Save files on frame deactivation  和 Save files automatically if application is idle for .. sec 的勾去掉

> Editor Tabs -> Mark modified tabs with asterisk 打上勾

9.>file and code template>Python scripts

```
#!/usr/bin/env Python
```

```
# -*- coding: utf-8 -*-
```

```
__title__ = '$Package_name'
__author__ = '$USER'
__mtime__ = '$DATE'

# code is far away from bugs with the god animal protecting

    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
```

10. Python文件默认编码

File Encodings> IDE Encoding: UTF-8;Project Encoding: UTF-8;

11. 代码自动整理设置


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20191109/wd9RMfmKw38f.png?imageslim">
</p>

这里line breaks去掉√，否则bar, 和baz会分开在不同行，不好看。


### File -> Settings -> appearance

1. 修改IDE快捷键方案

> Keymap

execute selection in console : add keymap > ctrl + enter

系统自带了好几种快捷键方案，下拉框中有如“defaul”,“Visual Studio”，在查找Bug时非常有用，“NetBeans 6.5”，“Default for GNOME”等等可选项，

2. 设置IDE皮肤主题

> Theme -> 选择“Alloy.IDEA Theme”

 或者在setting中搜索theme可以改变主题，所有配色统一改变

File > settings > build.excution

每次打开Python控制台时自动执行代码

> console > pyconsole

```
import sys # print('Python %s on %s' % (sys.version, sys.platform)) sys.path.extend([WORKING_DIR_AND_Python_PATHS]) import os print('current workdirectory : ', os.getcwd() ) import numpy as np import scipy as sp import matplotlib as mpl
```

如果安装了iPython，则在pyconsole中使用更强大的iPython

> console

选中use iPython if available

这样每次打开pyconsole就会打开iPython

Note: 在virtualenv中安装iPython: (ubuntu_env) pika:/media/pika/files/mine/Python_workspace/ubuntu_env$pip install iPython







# Python 导入包出现的问题

## 多个文件之间有相互依赖的关系

cannot find declaration to go to||CTRL+也不起作用
同目录下，当多个文件之间有相互依赖的关系的时候，import无法识别自己写的模块，PyCharm中提示No Module；PyCharm import不能转到。

如from models.base_model import BaseModel

解决步骤：

(1). 打开File--> Setting—> 打开 Console下的Python Console，把选项（Add source roots to PythonPAT）点击勾选上

(2). 右键点击自己的工作空间文件夹（对应就是models的上级目录，假设是src），找到Mark Directory as 选择Source Root！

- [import时无法识别自己写的模块](https://blog.csdn.net/enter89/article/details/79960230)

## pycharm中进行python包管理


PyCharm中的项目中可以包含package、目录（目录名可以有空格）、等等

目录的某个包中的某个py文件要调用另一个py文件中的函数，首先要将目录设置为source root，这样才能从包中至上至上正确引入函数，否则怎么引入都出错：

SystemError: Parent module '' not loaded, cannot perform relative import

Note:目录 > 右键 > make directory as > source root

## Python脚本解释路径

ctrl + shift + f10 / f10 执行Python脚本时

当前工作目录cwd为run/debug configurations 中的working directory

可在edit configurations > project or defaults中配置

## console执行路径和当前工作目录

Python console中执行时

cwd为File > settings > build.excution > console > pyconsole中的working directory

并可在其中配置

## PyCharm配置os.environ环境

PyCharm中os.environ不能读取到terminal中的系统环境变量

PyCharm中os.environ不能读取.bashrc参数

使用PyCharm，无论在Python console还是在module中使用os.environ返回的dict中都没有~/.bashrc中的设置的变量，但是有/etc/profile中的变量配置。然而在terminal中使用Python，os.environ却可以获取~/.bashrc的内容。

### 解决方法1：

在~/.bashrc中设置的系统环境只能在terminal shell下运行spark程序才有效，因为.bashrc is only read for interactive shells.

如果要在当前用户整个系统中都有效（包括PyCharm等等IDE），就应该将系统环境变量设置在~/.profile文件中。如果是设置所有用户整个系统，修改/etc/profile或者/etc/environment吧。

如SPARK_HOME的设置[Spark：相关错误总结 ]

### 解决方法2：在代码中设置，这样不管环境有没有问题了

```
# spark environment settings

import sys, os
os.environ['SPARK_HOME'] = conf.get(SECTION, 'SPARK_HOME')
sys.path.append(os.path.join(conf.get(SECTION, 'SPARK_HOME'), 'Python'))
os.environ["PYSPARK_Python"] = conf.get(SECTION, 'PYSPARK_Python')
os.environ['SPARK_LOCAL_IP'] = conf.get(SECTION, 'SPARK_LOCAL_IP')
os.environ['JAVA_HOME'] = conf.get(SECTION, 'JAVA_HOME')
os.environ['PythonPATH'] = '$SPARK_HOME/Python/lib/py4j-0.10.3-src.zip:$PythonPATH'
```

## PyCharm配置第三方库代码自动提示

参考 [Spark安装和配置](https://blog.csdn.net/pipisorry/article/details/50924395)





# PyCharm 调试时传入命令行参数

<span style="color:red;">需要补充下。</span>





## PyCharm中清除已编译.pyc中间文件

选中你的workspace > 右键 > clean Python compiled files

还可以自己写一个清除代码

## PyCharm设置外部工具


[Python小工具 ](https://blog.csdn.net/pipisorry/article/details/46754515)针对当前PyCharm中打开的py文件对应的目录删除其中所有的pyc文件。如果是直接运行（而不是在下面的tools中运行），则删除`E:\mine\Python_workspace\WebSite`目录下的pyc文件。

## 将上面的删除代码改成外部工具

PyCharm > settings > tools > external tools > +添加

Name: DelPyc

program: $PyInterpreterDirectory$/Python Python安装路径

Parameters: $ProjectFileDir$/Oth/Utility/DelPyc.py $FileDir$

Work directory: $FileDir$

Note:Parameters后面的 $FileDir$参数是说，DelPyc是针对当前PyCharm中打开的py文件对应的目录删除其中所有的pyc文件。

之后可以通过下面的方式直接执行


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20191109/q5F8sfnnCso7.png?imageslim">
</p>

Note:再添加一个Tools名为DelPycIn

program: Python安装路径，e.g.     D:\Python3.4.2\Python.exe

Parameters: E:\mine\Python_workspace\Utility\DelPyc.py

Work directory 使用变量 $FileDir$

参数中没有$FileDir$，这样就可以直接删除常用目录r'E:\mine\Python_workspace\WebSite'了，两个一起用更方便

## 代码质量

当你在打字的时候，PyCharm会检查你的代码是否符合PEP8。它会让你知道，你是否有太多的空格或空行等等。如果你愿意，你可以配置PyCharm运行pylint作为外部工具。
