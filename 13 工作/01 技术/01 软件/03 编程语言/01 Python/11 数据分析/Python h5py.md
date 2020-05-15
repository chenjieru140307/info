# h5py

作用：

- h5py 文件可以用来存放 numpy 数据。

介绍：

- h5py文件是存放两类对象的容器，数据集(dataset)和组(group)
  - dataset类似数组类的数据集合，和numpy的数组差不多
  - group是像文件夹一样的容器，它好比python中的字典，有键(key)和值(value)。group中可以存放dataset或者其他的group。

文档：

- [文档](http://docs.h5py.org/en/latest/index.html)

举例：

```py
import h5py
import numpy as np

f = h5py.File("h.hdf5", "w")

d1 = f.create_dataset("dataset1", (20,), 'i')
d1[...] = np.arange(20)# 赋值
d2 = f.create_dataset("dataset2", (3, 4), 'i')
d2[...] = np.arange(12).reshape((3, 4))
f["dataset3"] = np.arange(15)
d4 = f.create_dataset("dataset4", data=np.arange(20))
for key in f.keys():
    print(key)
    print(f[key].name)
    print(f[key].shape)
    print(f[key][()])
    print()

print('---')
print()

# group
g1 = f.create_group("bar")
c1 = g1.create_group("car1")
g1["dataset1"] = np.arange(10)
g1["dataset2"] = np.arange(12).reshape((3, 4))
d = c1.create_dataset("dataset3", data=np.arange(10))
for key in c1.keys():
    print(c1[key].name)
    print(c1[key][()])
print()
for key in f.keys():
    print(f[key].name)
```

输出：

```txt

dataset1
/dataset1
(20,)
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]

dataset2
/dataset2
(3, 4)
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]

dataset3
/dataset3
(15,)
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]

dataset4
/dataset4
(20,)
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]

---

/bar/car1/dataset3
[0 1 2 3 4 5 6 7 8 9]

/bar
/dataset1
/dataset2
/dataset3
/dataset4
```





说明:

- `dataset1` 是数据集的name，`(20,)`代表数据集的shape，`i` 代表的是数据集的元素类型
- `group` 有点类似于文件夹，`dataset` 类似于文件。（不确定）