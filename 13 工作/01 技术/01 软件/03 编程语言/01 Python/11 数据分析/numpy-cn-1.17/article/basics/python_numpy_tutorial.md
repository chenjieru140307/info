**目录**：

*   [Python](#Python)
    *   [基本数据类型](#基本数据类型)
    *   [容器(Containers)](#容器(Containers))
        *   [列表(Lists)](#列表(Lists))
        *   [字典](#字典)
        *   [集合(Sets)](#集合(Sets))
        *   [元组(Tuples)](#元组(Tuples))
    *   [函数(Functions)](#函数(Functions))
    *   [类(Classes)](#类(Classes))
*   [Numpy](#Numpy)
    *   [数组(Arrays)](#数组(Arrays))
    *   [数组索引](#数组索引)
    *   [数据类型](#数据类型)
    *   [数组中的数学](#数组中的数学)
    *   [广播(Broadcasting)](#广播(Broadcasting))
*   [SciPy](#SciPy)
    *   [图像操作](#图像操作)
    *   [MATLAB文件](#MATLAB文件)
    *   [点之间的距离](#点之间的距离)
*   [Matplotlib](#Matplotlib)
    *   [绘制](#绘制)
    *   [子图](#子图)
    *   [图片](#图片)

## Python

快速排序算法：

```py
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3,6,8,10,1,2,1]))
# Prints "[1, 1, 2, 3, 6, 8, 10]"
```










            #          [2 4]]"

# Note that taking the transpose of a rank 1 array does nothing:
v = np.array([1,2,3])
print(v)    # Prints "[1 2 3]"
print(v.T)  # Prints "[1 2 3]"
```

Numpy提供了许多用于操作数组的函数；你可以在[这篇文档](/reference/routines/array_manipulation_routines.html)中看到完整的列表。

### 广播(Broadcasting)

广播是一种强大的机制，它允许numpy在执行算术运算时使用不同形状的数组。通常，我们有一个较小的数组和一个较大的数组，我们希望多次使用较小的数组来对较大的数组执行一些操作。

例如，假设我们要向矩阵的每一行添加一个常数向量。我们可以这样做：

```python
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)   # Create an empty matrix with the same shape as x

# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v

# Now y is the following
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
print(y)
```

这会凑效; 但是当矩阵 ``x`` 非常大时，在Python中计算显式循环可能会很慢。注意，向矩阵 ``x`` 的每一行添加向量 ``v`` 等同于通过垂直堆叠多个 ``v`` 副本来形成矩阵 ``vv``，然后执行元素的求和``x`` 和 ``vv``。 我们可以像如下这样实现这种方法：

```python
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1))   # Stack 4 copies of v on top of each other
print(vv)                 # Prints "[[1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]]"
y = x + vv  # Add x and vv elementwise
print(y)  # Prints "[[ 2  2  4
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"
```

Numpy广播允许我们在不实际创建``v``的多个副本的情况下执行此计算。考虑这个需求，使用广播如下：

```python
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # Add v to each row of x using broadcasting
print(y)  # Prints "[[ 2  2  4]
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"
```

``y=x+v``行即使``x``具有形状``(4，3)``和``v``具有形状``(3,)``，但由于广播的关系，该行的工作方式就好像``v``实际上具有形状``(4，3)``，其中每一行都是``v``的副本，并且求和是按元素执行的。

将两个数组一起广播遵循以下规则：

1. 如果数组不具有相同的rank，则将较低等级数组的形状添加1，直到两个形状具有相同的长度。
1. 如果两个数组在维度上具有相同的大小，或者如果其中一个数组在该维度中的大小为1，则称这两个数组在维度上是兼容的。
1. 如果数组在所有维度上兼容，则可以一起广播。
1. 广播之后，每个数组的行为就好像它的形状等于两个输入数组的形状的元素最大值。
1. 在一个数组的大小为1且另一个数组的大小大于1的任何维度中，第一个数组的行为就像沿着该维度复制一样

如果对于以上的解释依然没有理解，请尝试阅读[这篇文档](/user_guide/numpy_basics/broadcasting.html)或[这篇解释](http://wiki.scipy.org/EricsBroadcastingDoc)中的说明。

支持广播的功能称为通用功能。你可以在[这篇文档](/reference/ufuncs/available_ufuncs.html)中找到所有通用功能的列表。

以下是广播的一些应用：

```python
import numpy as np

# Compute outer product of vectors
v = np.array([1,2,3])  # v has shape (3,)
w = np.array([4,5])    # w has shape (2,)
# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:
# [[ 4  5]
#  [ 8 10]
#  [12 15]]
print(np.reshape(v, (3, 1)) * w)

# Add a vector to each row of a matrix
x = np.array([[1,2,3], [4,5,6]])
# x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
# giving the following matrix:
# [[2 4 6]
#  [5 7 9]]
print(x + v)

# Add a vector to each column of a matrix
# x has shape (2, 3) and w has shape (2,).
# If we transpose x then it has shape (3, 2) and can be broadcast
# against w to yield a result of shape (3, 2); transposing this result
# yields the final result of shape (2, 3) which is the matrix x with
# the vector w added to each column. Gives the following matrix:
# [[ 5  6  7]
#  [ 9 10 11]]
print((x.T + w).T)
# Another solution is to reshape w to be a column vector of shape (2, 1);
# we can then broadcast it directly against x to produce the same
# output.
print(x + np.reshape(w, (2, 1)))

# Multiply a matrix by a constant:
# x has shape (2, 3). Numpy treats scalars as arrays of shape ();
# these can be broadcast together to shape (2, 3), producing the
# following array:
# [[ 2  4  6]
#  [ 8 10 12]]
print(x * 2)
```

广播通常会使你的代码更简洁，效率更高，因此你应该尽可能地使用它。

### Numpy 的文档

这个简短的概述说明了部分numpy相关的重要事项。查看[numpy参考手册](/reference/index.html)以了解有关numpy的更多信息。

## SciPy

Numpy提供了一个高性能的多维数组和基本工具来计算和操作这些数组。 而[SciPy](https://docs.scipy.org/doc/scipy/reference/)以此为基础，提供了大量在numpy数组上运行的函数，可用于不同类型的科学和工程应用程序。

熟悉SciPy的最佳方法是浏览[它的文档](https://docs.scipy.org/doc/scipy/reference/index.html)。我们将重点介绍SciPy有关的对你有价值的部分内容。

### 图像操作

SciPy提供了一些处理图像的基本函数。例如，它具有将映像从磁盘读入numpy数组、将numpy数组作为映像写入磁盘以及调整映像大小的功能。下面是一个演示这些函数的简单示例：

```python
from scipy.misc import imread, imsave, imresize

# Read an JPEG image into a numpy array
img = imread('assets/cat.jpg')
print(img.dtype, img.shape)  # Prints "uint8 (400, 248, 3)"

# We can tint the image by scaling each of the color channels
# by a different scalar constant. The image has shape (400, 248, 3);
# we multiply it by the array [1, 0.95, 0.9] of shape (3,);
# numpy broadcasting means that this leaves the red channel unchanged,
# and multiplies the green and blue channels by 0.95 and 0.9
# respectively.
img_tinted = img * [1, 0.95, 0.9]

# Resize the tinted image to be 300 by 300 pixels.
img_tinted = imresize(img_tinted, (300, 300))

# Write the tinted image back to disk
imsave('assets/cat_tinted.jpg', img_tinted)
```

![猫咪](/static/images/article/cat.jpg) ![猫咪](/static/images/article/cat_tinted.jpg)

左：原始图像。右：着色和调整大小的图像。

### MATLAB 文件

函数 ``scipy.io.loadmat`` 和 ``scipy.io.savemat`` 允许你读取和写入MATLAB文件。你可以在[这篇文档](https://docs.scipy.org/doc/scipy/reference/io.html)中学习相关操作。

### 点之间的距离

SciPy定义了一些用于计算点集之间距离的有用函数。

函数``scipy.spatial.distance.pdist``计算给定集合中所有点对之间的距离：

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Create the following array where each row is a point in 2D space:
# [[0 1]
#  [1 0]
#  [2 0]]
x = np.array([[0, 1], [1, 0], [2, 0]])
print(x)

# Compute the Euclidean distance between all rows of x.
# d[i, j] is the Euclidean distance between x[i, :] and x[j, :],
# and d is the following array:
# [[ 0.          1.41421356  2.23606798]
#  [ 1.41421356  0.          1.        ]
#  [ 2.23606798  1.          0.        ]]
d = squareform(pdist(x, 'euclidean'))
print(d)
```

你可以在[这篇文档](http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html)中阅读有关此功能的所有详细信息。

类似的函数（``scipy.spatial.distance.cdist``）计算两组点之间所有对之间的距离; 你可以在[这篇文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)中阅读它。

## Matplotlib

[Matplotlib](https://matplotlib.org/)是一个绘图库。本节简要介绍 ``matplotlib.pyplot`` 模块，该模块提供了类似于MATLAB的绘图系统。

### 绘制

matplotlib中最重要的功能是``plot``，它允许你绘制2D数据的图像。这是一个简单的例子：

```python
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

# Plot the points using matplotlib
plt.plot(x, y)
plt.show()  # You must call plt.show() to make graphics appear.
```

运行此代码会生成以下图表：

![sine](/static/images/article/sine.png)

通过一些额外的工作，我们可以轻松地一次绘制多条线，并添加标题，图例和轴标签：


```python
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()
```

![sine_cosine](/static/images/article/sine_cosine.png)

你可以在[这篇文档](https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot)中阅读有关``绘图``功能的更多信息。

### 子图

你可以使用``subplot``函数在同一个图中绘制不同的东西。 这是一个例子：

```python
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# Show the figure.
plt.show()
```

![sine_cosine_subplot](/static/images/article/sine_cosine_subplot.png)

你可以在[这篇文档](https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplot)中阅读有关``子图``功能的更多信息。

### 图片

你可以使用 ``imshow`` 函数来显示一张图片。 这是一个例子：

```python
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

img = imread('assets/cat.jpg')
img_tinted = img * [1, 0.95, 0.9]

# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(img)

# Show the tinted image
plt.subplot(1, 2, 2)

# A slight gotcha with imshow is that it might give strange results
# if presented with data that is not uint8. To work around this, we
# explicitly cast the image to uint8 before displaying it.
plt.imshow(np.uint8(img_tinted))
plt.show()
```

![cat_tinted_imshow](/static/images/article/cat_tinted_imshow.png)

## 文章出处 

由NumPy中文文档翻译，原作者为 [Justin Johnson](https://cs.stanford.edu/people/jcjohns/)，翻译至：[http://cs231n.github.io/python-numpy-tutorial/](http://cs231n.github.io/python-numpy-tutorial/)。