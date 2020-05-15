# qimage2ndarray

作用：

- 将 qimage 与 numpy.array 之间互相转化。

文档：

- [文档](https://hmeine.github.io/qimage2ndarray/#)

举例：

```py
QImage img
arr = qimage2ndarray.rgb_view(img, 'little') # little -> bgr
cv2.imwrite('test.img', arr)
```
