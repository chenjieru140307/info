# 截图

- QImage
- mss
- win32gui

## QImage

举例：

```py
import win32gui
import win32con
import time
from PyQt5.QtWidgets import QApplication
import sys
from multiprocessing import Process, Queue
import numpy as np
import cv2
import qimage2ndarray

# 列出窗口句柄与名称
hwnd_title = dict()
def get_all_hwnd(hwnd, mouse):
    if win32gui.IsWindow(hwnd) and win32gui.IsWindowEnabled(hwnd) and win32gui.IsWindowVisible(hwnd):
        hwnd_title.update({hwnd: win32gui.GetWindowText(hwnd)})
win32gui.EnumWindows(get_all_hwnd, 0)
print(hwnd_title.items())

# 查找指定窗口并置前
hwnd = win32gui.FindWindow(0, "Warcraft III")  # 获取句柄 # Warcraft III
print(hwnd)
left, top, right, bottom = win32gui.GetWindowRect(hwnd)
print(left, top, right, bottom)
win32gui.ShowWindow(hwnd, win32con.SW_SHOWNORMAL)
win32gui.SetForegroundWindow(hwnd)
time.sleep(0.2)

app = QApplication(sys.argv)
screen = QApplication.primaryScreen()


def grab(queue):
    height = 1080
    width = 1920
    num_grab = 0
    last_time = time.time()
    for _ in range(10000):
        img = screen.grabWindow(hwnd, 0, 0, width, height).toImage()
        arr = qimage2ndarray.rgb_view(img, 'little')
        queue.put(arr)
        num_grab += 1
        print('grab ', num_grab, "fps: {}".format(num_grab / (time.time() - last_time)))
    queue.put(None)  # Tell the other worker to stop


def save(queue):
    num_save = 0
    output = "C:/Users/wanfa/Desktop/img/file_{}.jpg"
    while "there are screenshots":
        last_time = time.time()
        arr = queue.get()
        if arr is None:
            break
        cv2.imwrite(output.format(num_save), arr)
        num_save += 1
        print('save ', num_save, "fps: {}".format(1 / (time.time() - last_time)))


if __name__ == "__main__":
    queue = Queue()
    Process(target=grab, args=(queue,)).start()
    Process(target=save, args=(queue,)).start()
```


输出：

```txt
...
grab  94 fps: 28.6518598166654
save  34 fps: 11.034883384417054
...
```


## mss



文档：

- [文档](https://python-mss.readthedocs.io/index.html)

安装：

- `pip install mss`

举例：

```py
import time
import mss
import mss.tools
import numpy
import cv2
from multiprocessing import Process, Queue


def grab(queue):
    num_grab = 0
    rect = {"top": 0, "left": 0, "width": 1920, "height": 1080}
    with mss.mss() as sct:
        for _ in range(10000):
            last_time = time.time()
            queue.put(sct.grab(rect))
            num_grab += 1
            print('grab ', num_grab, "fps: {}".format(1 / (time.time() - last_time)))
    queue.put(None)  # Tell the other worker to stop


def save(queue):
    num_save = 0
    output = "C:/Users/wanfa/Desktop/img/file_{}.jpg"
    while "there are screenshots":
        last_time = time.time()
        img = queue.get()
        if img is None:
            break
        img_cv = numpy.array(img)
        cv2.imwrite(output.format(num_save), img_cv)
        num_save += 1
        print('save ', num_save, "fps: {}".format(1 / (time.time() - last_time)))


if __name__ == "__main__":
    queue = Queue()
    Process(target=grab, args=(queue,)).start()
    Process(target=save, args=(queue,)).start()
```

输出：

```txt
...
grab  30 fps: 31.062705977323056
save  30 fps: 20.856496422231395
...
```

说明：

- 可以对游戏画面截图。帧率不错。



## win32gui

举例：

```py
import time
import win32gui, win32ui, win32con, win32api


def window_capture(filename, hwnd):
    # 根据窗口句柄获取窗口的设备上下文DC（Divice Context）
    hwndDC = win32gui.GetWindowDC(hwnd)
    # 根据窗口的DC获取mfcDC
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    # mfcDC创建可兼容的DC
    saveDC = mfcDC.CreateCompatibleDC()
    # 创建bigmap准备保存图片
    saveBitMap = win32ui.CreateBitmap()
    # 获取监控器信息
    MoniterDev = win32api.EnumDisplayMonitors(None, None)
    w = MoniterDev[0][2][2]
    h = MoniterDev[0][2][3]
    # print w,h　　　#图片大小
    # 为bitmap开辟空间
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
    # 高度saveDC，将截图保存到saveBitmap中
    saveDC.SelectObject(saveBitMap)
    # 截取从左上角（0，0）长宽为（w，h）的图片
    saveDC.BitBlt((0, 0), (w, h), mfcDC, (0, 0), win32con.SRCCOPY)
    saveBitMap.SaveBitmapFile(saveDC, filename)


win_title2 = "Warcraft III"
hwnd = win32gui.FindWindow(0, win_title2)  # 获取句柄 # Warcraft III  #当前活跃窗口 hwnd 为0
print(hwnd)
win32gui.ShowWindow(hwnd, win32con.SW_SHOWNORMAL)
win32gui.SetForegroundWindow(hwnd) # 设为最前
time.sleep(0.1)
window_capture("test.jpg", hwnd)
```


