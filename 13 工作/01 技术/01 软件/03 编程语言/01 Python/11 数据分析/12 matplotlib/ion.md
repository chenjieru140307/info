# plt.ion 交互模式

<span style="color:red;">`plt.ion()` 这个还是不错的。以前没用过这个，只看到过一次。</span>

<span style="color:red;">关于这个地方的 plt 还是不错的。要整理下。如下：</span>


```py
plt.ion()  # interactive mode


fig = plt.figure()


ax = plt.subplot(1, 4, i % 4 + 1)
plt.tight_layout() # 这个是什么？
ax.clear()
ax.set_title('Sample #{}'.format(i))
ax.axis('off')
show_landmarks(**sample)


def show_landmarks(image, landmarks):
    # plt.cla() # which clears data but not axes
    # plt.clf() # which clears data and axes
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.01)  # pause a bit so that plots are updated
```
