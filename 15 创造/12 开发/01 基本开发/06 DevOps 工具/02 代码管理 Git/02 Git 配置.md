

# Git 配置

## git config 介绍


Git 提供了一个叫做 git config 的工具，专门用来配置或读取相应的工作环境变量。

这些环境变量，决定了 Git 在各个环节的具体工作方式和行为。这些变量可以存放在以下三个不同的地方：

- `/etc/gitconfig` 文件：系统中对所有用户都普遍适用的配置。若使用 `git config` 时用 `--system`选项，读写的就是这个文件。
- `~/.gitconfig` 文件：用户目录下的配置文件只适用于该用户。若使用 `git config` 时用 `--global`选项，读写的就是这个文件。
- 当前项目的 Git 目录中的配置文件（也就是工作目录中的 `.git/config` 文件）：这里的配置仅仅针对当前项目有效。每一个级别的配置都会覆盖上层的相同配置，所以 `.git/config` 里的配置会覆盖 `/etc/gitconfig` 中的同名变量。


在 Windows 系统上，Git 会找寻用户主目录下的 `.gitconfig` 文件。主目录即 `$HOME` 变量指定的目录，一般都是 `C:\Documents and Settings\$USER`。

此外，Git 还会尝试找寻 `/etc/gitconfig` 文件，只不过看当初 Git 装在什么目录，就以此作为根目录来定位。


## 配置信息


```
$ git config --global user.name "w3c"
$ git config --global user.email w3c@w3cschool.cn


$ git config --global core.editor emacs
$ git config --global merge.tool vimdiff

$ git config --list
user.name=w3c
user.email=w3c@w3cschool.cn
color.status=auto
color.branch=auto
color.interactive=auto
color.diff=auto
...


$ git config user.name
Scott Chacon

```

说明：


- 如果用了 `--global` 选项，那么更改的配置文件就是位于你用户主目录下的那个，以后你所有的项目都会默认使用这里配置的用户信息。
- 如果要在某个特定的项目中使用其他名字或者电邮，只要去掉 `--global` 选项重新配置即可，新的设定保存在当前项目的 `.git/config` 文件里。
- `user.name` 与 `user.email` 是必须配置的。为个人的用户名称和电子邮件地址。
- 通过配置 `merge.tool` 来指定在解决合并冲突时使用哪种差异分析工具。也比较常用。Git 可以理解 kdiff3，tkdiff，meld，xxdiff，emerge，vimdiff，gvimdiff，ecmerge，和 opendiff 等合并工具的输出信息。
- `git config --list` 可以列出已有的配置信息。
- `git config user.name` 可以直接查阅某个环境变量的设定，只要把特定的名字跟在后面即可。

注意：

- 对于与 github 连接的项目来说，这个地方就可以写自己的 github 账号和注册 github 时候用的邮箱。
- `git config --list` 有时候会看到重复的变量名，那就说明它们来自不同的配置文件（比如 `/etc/gitconfig` 和 `~/.gitconfig`），不过最终 Git 实际采用的是最后一个。

不清楚的：

- <span style="color:red;">普通的 diff 怎么使用？</span>
- vimdiff 一定要配置吗？
