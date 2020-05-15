

# pywin32

安装：

- pip install pywin32

文档：

- [文档](http://timgolden.me.uk/pywin32-docs/contents.html)
- [github](https://github.com/mhammond/pywin32)


举例：

```py
import time
import win32gui, win32ui, win32con, win32api
import win32clipboard



# 列出窗口句柄与名称
hwnd_title = dict()
def get_all_hwnd(hwnd, mouse):
    if win32gui.IsWindow(hwnd) and win32gui.IsWindowEnabled(hwnd) and win32gui.IsWindowVisible(hwnd):
        hwnd_title.update({hwnd: win32gui.GetWindowText(hwnd)})
win32gui.EnumWindows(get_all_hwnd, 0)
print(hwnd_title.items())

win_title1='未命名• - Typora'
win_title2="Warcraft III"
hwnd = win32gui.FindWindow(0, win_title2)  # 获取句柄 # Warcraft III  #当前活跃窗口 hwnd 为0
print(hwnd)
left, top, right, bottom = win32gui.GetWindowRect(hwnd)
print(left, top, right, bottom)
win32gui.ShowWindow(hwnd, win32con.SW_SHOWNORMAL)
# win32gui.SendMessage(hwnd, win32con.WM_SYSCOMMAND, win32con.SC_RESTORE, 0) #还原最小化窗口
# win32gui.MoveWindow(hwnd, 20, 20, 405, 756, True)
# win32gui.SetBkMode(hwnd, win32con.TRANSPARENT)
# win32gui.PostMessage(hwnd,win32con.WM_CLOSE,0,0)
win32gui.SetForegroundWindow(hwnd) # 设为最前

time.sleep(1)

win32clipboard.OpenClipboard()#打开剪贴板
win32clipboard.EmptyClipboard()#清空剪贴板
msg="测试"
win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT,msg)#设置剪贴板 # msg.encode('gbk')
win32clipboard.CloseClipboard()#关闭剪贴板

win32clipboard.OpenClipboard()
text = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
win32clipboard.CloseClipboard()
print(text)


# 粘贴
win32api.keybd_event(13, 0, 0, 0)  # Enter的键位码是13
win32api.keybd_event(13, 0, win32con.KEYEVENTF_KEYUP, 0)

win32api.keybd_event(17, 0, 0, 0)  # ctrl的键位码是17
win32api.keybd_event(86, 0, 0, 0)  # v的键位码是86
win32api.keybd_event(86, 0, win32con.KEYEVENTF_KEYUP, 0)  # 释放按键
win32api.keybd_event(17, 0, win32con.KEYEVENTF_KEYUP, 0)

win32api.keybd_event(13, 0, 0, 0)  # Enter的键位码是13
win32api.keybd_event(13, 0, win32con.KEYEVENTF_KEYUP, 0)

# 好像有点问题
# win32gui.PostMessage(hwnd, win32con.WM_PASTE, 0, 0)  # 向窗口发送剪贴板内容(粘贴) QQ测试可以正常发送
# time.sleep(0.3)
# win32gui.PostMessage(hwnd, win32con.WM_KEYDOWN, win32con.VK_RETURN, 0)  # 向窗口发送 回车键
# win32gui.PostMessage(hwnd, win32con.WM_KEYUP, win32con.VK_RETURN, 0)
```

输出:

```txt
dict_items([(65772, ''), (2426046, 'Warcraft III'), ...略... (131388, 'Program Manager')])
2426046
-32000 -32000 -31840 -31972
测试
```


注意：

- 通过 `PostMessage` 进行粘贴好像是不行的，没有反应。
- 有的游戏，比如 warcraft 中，把 ctrl+v 粘贴禁用了，这时用 `keybd_event` 进行粘贴也是不行的。