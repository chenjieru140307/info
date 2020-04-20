# VSCode

## 使用命令面板

View - Command Palette

## Markdown 对 http 图片的显示

在命令面板中输入：

```
Markdown: Change preview security settings
```

## 更换字体

Windows 下按 `Ctrl` + `,`，macOS 下按 `Cmd` + `,`，进入设定。在上方搜索框搜索 `editor.fontFamily`，在

```
Editor: Font Family
Controls the font family.
```

下方的框框填入 `font-family` 即可。

默认的是 `Consolas, 'Courier New', monospace`。

可以改为：

- `Consolas, 'Courier New', monospace,"萍方-简"`
- `Consolas, 'Courier New', monospace,DengXian`
- `Consolas, 'Courier New', monospace,"微软雅黑"`
- `Consolas, 'Courier New',"微软雅黑",monospace`


## 设置文件的 Icon 样式

1. In VS Code, open the File Icon Theme picker with **File** > **Preferences** > **File Icon Theme**. (**Code** > **Preferences** > **File Icon Theme** on macOS).
2. You can also use the **Preferences: File Icon Theme** command from the **Command Palette** (Ctrl+Shift+P).
3. Use the cursor keys to preview the icons of the theme.
4. Select the theme you want and hit Enter.

![file icon theme drop-down](https://code.visualstudio.com/assets/docs/getstarted/themes/file-icon-theme-dropdown.png)


## 设置 Snippets

打开 settings.json:

File->Preferences->Settings

在搜索中输入 quickSuggestions：

![mark](http://images.iterate.site/blog/image/20200203/T9xRTXUjOJ5B.png?imageslim)

点击 Edit in settings.json

打开 json 文件，修改为：

```json
{
    "workbench.colorTheme": "Visual Studio Light",
    "workbench.startupEditor": "newUntitledFile",
    "workbench.iconTheme": "vscode-icons",
    "git.enableSmartCommit": true,
    "git.confirmSync": false,
    "explorer.confirmDragAndDrop": false,
    "explorer.confirmDelete": false,
    "editor.fontSize": 18,
    "editor.fontFamily": "Consolas, 'Courier New', \"微软雅黑\",monospace",
    "[markdown]": {
        "editor.quickSuggestions": true,
        "editor.snippetSuggestions": "top",
    }
}
```

主要内容是：

```json
    "[markdown]": {
        "editor.quickSuggestions": true,
        "editor.snippetSuggestions": "top",
    }
```

此时，markdown 支持 snippets 的显示了。

然后，准备 snippets：

Ctrl+Shift+P，然后输入 Preferences:Configure User Snippets 选择 markdown(Markdown),此时打开一个 markdown.json，输入：

```json
{
	"red style": {
		"prefix": "re",
		"body": [
			"<span style=\"color:red;\">$1</span>",
		],
		"description": "input red style"
	},
	"blue style": {
		"prefix": "bl",
		"body": [
			"<span style=\"color:blue;\">$1</span>",
		],
		"description": "input blue style"
	},
	"center image style": {
		"prefix": "cc",
		"body": [
			"<p align=\"center\">",
			"\t<img width=\"70%\" height=\"70%\" src=\"$1\">",
			"</p>",
			"$2",
		],
		"description": "input center image style"
	},
	"code block txt": {
		"prefix": "tt",
		"body": [
			"```txt\n$1\n```",
			"$2",
		],
		"description": "input code block"
	},
	"code block py": {
		"prefix": "py",
		"body": [
			"```py\n$1\n```",
			"$2",
		],
		"description": "input code block"
	},
	"code block sh": {
		"prefix": "sh",
		"body": [
			"```sh\n$1\n```",
			"$2",
		],
		"description": "input code block"
	},
	"code block cp": {
		"prefix": "cp",
		"body": [
			"```cpp\n$1\n```",
			"$2",
		],
		"description": "input code block"
	}
}
```
