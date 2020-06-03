# Numpy学习笔记 （基于Numpy 1.11.0）

1. `Python`的列表中保存的是对象的指针。因此为了保存一个简单的列表，如`[1,2,3]`，则需要三个指针和三个整数对象。

2. ```
   numpy
   ```

   提供了两种基本的对象：

   - `ndarray`：它是存储单一数据类型的多维数组，简称数组
   - `ufunc`：它是一种能够对数组进行处理的特殊函数

## 一、 ndarray

### 1. ndarray 对象的内存结构

1. `ndarray`对象在内存中的结构如下： <p align="center">
  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200602/g6Vr2kWMyU7B.JPG">
</p>


   - `ndarray.dtype`：存储了数组保存的元素的类型。`float32`
   - `ndarray.ndim`：它是一个整数，保存了数组的维度，即多少个轴
   - `ndarray.shape`：它是一个整数的元组，每个元素一一对应地保存了数组某个维度的大小（即某个轴的长度）。
   - `ndarray.strides`：它是一个整数的元组，每个元素保存着每个轴上相邻两个元素的地址差。即当某个轴的下标增加1 时，数据存储区中的指针增加的字节数
   - `ndarray.data`：它指向数组的数据的存储区

   <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200602/Ro9fzwxmoLOP.JPG">
   </p>
    可以看到：该数组中元素类型为`float32`；该数组有2 个轴。每个轴的长度都是 3 个元素。第 0 轴增加1时，下标增加 12字节（也就是 3个元素，即一行的距离）； 第 1 轴增加 1时，下标增加 4字节（也就是一个元素的距离）。

2. 元素在数据存储区中的排列格式有两种：`C`语言格式和`Fortran`语言格式。

   - `C`语言中，多维数组的第 0 轴是最外层的。即 0 轴的下标增加 1时，元素的地址增加的字节数最多
   - `Fortran`语言中，多维数组的第 0 轴是最内层的。即 0 轴的下标增加 1时，元素的地址增加的字节数最少

   `numpy`中默认是以 `C`语言格式存储数据。如果希望改为`Fortran`格式，则只需要在创建数组时，设置`order`参数为`"F"` <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/3n0Aj1JNXzox.JPG">
   </p>
   

3. 数组的`flags`属性描述了数据存储区域的一些属性。你可以直接查看`flags`属性，也可以单独获取其中某个标志值。

   - `C_CONTIGUOUS`:数据存储区域是否是`C`语言格式的连续区域
   - `F_CONTIGUOUS`:数据存储区域是否是`F`语言格式的连续区域
   - `OWNDATA`:数组是否拥有此数据存储区域。当一个数组是其他数组的视图时，它并不拥有数据存储区域，通过视图数组的`base`属性可以获取保存数据存储区域的那个原始数组。 <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/TmkVIXk0HYEd.JPG">
   </p>
   

4. 数组的转置可以通过其`T`属性获取。转置数组可以简单的将其数据存储区域看作是`Fortran`语言格式的连续区域，并且它不拥有数据存储区域。<p align="center">
  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/nIulfxDTxC63.JPG">
</p>


5. 修改数组的内容时，会直接修改数据存储区域。所有使用该数据存储区域的数组都将被同时修改！ <p align="center">
  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/cchY3ijrMzhh.JPG">
</p>


#### 1.1 dtype

1. ```
   numpy
   ```

    

   有自己的浮点数类型：

    

   ```
   float16/float32/float64/float128
   ```

   等等。

   - 在需要指定`dtype`参数时，你可以使用`numpy.float16`，也可以传递一个表示数值类型的字符串。`numpy`中的每个数值类型都有几种字符串表示。字符串和类型之间的对应关系都存储在`numpy.typeDict`字典中。 <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/5Y0KKVEspDHC.JPG">
   </p>
   
   - `dtype`是一种对象，它不同于数值类型。只有`dtype.type`才能获取对应的数值类型 <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200602/9S8yaa3P7NdU.JPG">
   </p>
   
   - 你可以通过`np`的数值类型`np.float32`来创建数值对象。但要注意：`numpy`的数值对象的运算速度比`python`内置类型的运算速度要慢很多。所以应当尽量避免使用`numpy`的数值对象 <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/IR6dslKOLDgV.JPG">
   </p>
   

2. 使用`ndarray.astype()`方法可以对数组元素类型进行转换。 <p align="center">
  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200602/QnPnRqBcLj2z.JPG">
</p>


#### 1.2 shape

1. 你可以使用

   ```
   ndarray.reshape()
   ```

   方法调整数组的维度。

   - 你可以在某个维度设置其长度为 -1，此时该维度的长度会被自动计算 <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/qeEkQOb0UkuQ.JPG">
   </p>
   

2. 你可以直接修改

   ```
   ndarry
   ```

   的

   ```
   shape
   ```

   属性，此时直接修改原始数组。

   - 你可以在某个维度设置其长度为 -1，此时该维度的长度会被自动计算 <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/l7vU6wYPogYe.JPG">
   </p>
   

#### 1.3 view

1. 我们可以通过`ndarray.view()`方法，从同一块数据区创建不同的`dtype`数组。即使用不同的数值类型查看同一段内存中的二进制数据。它们使用的是同一块内存。<p align="center">
  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/ittebf6JfErb.JPG">
</p>

2. 如果我们直接修改原始数组的`dtype`，则同样达到这样的效果，此时直接修改原始数组。 <p align="center">
  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200602/TGBKEXtCuGRP.JPG">
</p>


#### 1.4 strides

1. 我们可以直接修改`ndarray`对象的`strides`属性。此时修改的是原始数组。<p align="center">
  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/tE8mrIM3N63X.JPG">
</p>

2. 你可以使用`np.lib.stride_tricks.as_stride()`函数创建一个不同`strides`的视图。 <p align="center">
  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200602/Lxmdvn3ODfkd.JPG">
</p>
 注意：使用`as_stride`时并不会执行内存越界检查，因此`shape`和`stride`设置不当可能会发生意想不到的错误。

#### 1.5 拷贝和视图

1. 当处理

   ```
   ndarray
   ```

   时，它的数据存储区有时被拷贝，但有时并不被拷贝。有三种情况。

   - 完全不拷贝：简单的赋值操作并不拷贝`ndarray`的任何数据，这种情况下是新的变量引用`ndarray`对象（类似于列表的简单赋值）

   - 视图和浅拷贝：不同的

     ```
     ndarray
     ```

     可能共享相同的数据存储区。如

     ```
     ndarray.view()
     ```

     方法创建一个新的

     ```
     ndarray
     ```

     但是与旧

     ```
     ndarray
     ```

     共享相同的数据存储区。新创建的那个数组称作视图数组。

     - 对于视图数组，`ndarray.base`返回的是拥有数据存储区的那个底层`ndarray`。而非视图数组的`ndarray.base`返回`None`
     - `ndarray.flags.owndata`返回数组是否拥有基础数据
     - 对于数组的分片操作返回的是一个`ndarray`的视图。对数组的索引返回的不是视图，而是含有基础数据。

   - 深拷贝：`ndarray.copy()`操作会返回一个完全的拷贝，不仅拷贝`ndarray`也拷贝数据存储区。 <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200602/y9YUwIQ7GVdd.JPG">
   </p>
   

### 2. 数组的创建

1. 这里有几个共同的参数：

   - `a`：一个`array-like`类型的实例，它不一定是数组，可以为`list`、`tuple`、`list of tuple`、`list of list`、`tuple of list`、`tuple of tuple`等等。

   - `dtype`：数组的值类型，默认为`float`。你可以指定`Python`的标准数值类型，也可以使用`numpy`的数值类型如：`numpy.int32`或者`numpy.float64`等等。

   - ```
     order
     ```

     ：指定存储多维数据的方式。

     - 可以为`'C'`，表示按行优先存储（C风格）；
     - 可以为`'F'`，表示按列优先存储（Fortran风格）。
     - 对于`**_like()`函数，`order`可以为：`'C'`，`'F'`，`'A'`（表示结果的`order`与`a`相同），`'K'`（表示结果的`order`与`a`尽可能相似）

   - `subok`：`bool`值。如果为`True`则：如果`a`为`ndarray`的子类（如`matrix`类），则结果类型与`a`类型相同。如果为`False`则：结果类型始终为`ndarray`。默认为`True`。

#### 2.1 创建全1或者全0

1. `np.empty(shape[,dtype,order])`：返回一个新的`ndarray`，指定了`shape`和`dtype`，但是没有初始化元素。因此其内容是随机的。

   - `np.empty_like(a[,dtype,order,subok])`：返回一个新的`ndarray`，`shape`与`a`相同，但是没有初始化元素。因此其内容是随机的。

   <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/6CcQoX6nCBtn.JPG">
   </p>
   

2. `np.eye(N[, M, k, dtype])`：返回一个二维数组，对角线元素为1，其余元素为0。`M`默认等于`N`。`k`默认为0表示对角线元素为1，如为正数则表示对角线上方一格的元素为1，如为负数表示对角线下方一格的元素为1.

   - `np.identity(n[, dtype])` ：返回一个单位矩阵

   <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/latXTkBFIp4Q.JPG">
   </p>
   

3. `np.ones(shape[, dtype, order])`：返回一个新的`ndarray`，指定了`shape`和`type`，每个元素初始化为1.

   - `np.ones_like(a[, dtype, order, subok])` ：返回一个新的`ndarray`，`shape`与`a`相同，每个元素初始化为1。

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/qjNDcmMf8Cm4.JPG">
  </p>
  

4. `np.zeros(shape[, dtype, order])` ：返回一个新的`ndarray`，指定了`shape`和`type`，每个元素初始化为0.

   - `np.zeros_like(a[, dtype, order, subok])`：返回一个新的`ndarray`，`shape`与`a`（另一个数组）相同，每个元素初始化为0。

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/E7Y8UYN9kac4.JPG">
  </p>
  

5. `np.full(shape, fill_value[, dtype, order])`：返回一个新的`ndarray`，指定了`shape`和`type`，每个元素初始化为`fill_value`。

   - `np.full_like(a, fill_value[, dtype, order, subok])`：返回一个新的`ndarray`，`shape`与`a`相同，每个元素初始化为`fill_value`。

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/h5HRHQunUhI0.JPG">
  </p>
  

#### 2.2 从现有数据创建

1. ```
   np.array(object[, dtype, copy, order, subok, ndmin])
   ```

   :从

   ```
   object
   ```

   创建。

   - `object`可以是一个`ndarray`，也可以是一个`array_like`的对象，也可以是一个含有返回一个序列或者`ndarray`的`__array__`方法的对象，或者一个序列。
   - `copy`：默认为`True`，表示拷贝对象
   - `order`可以为`'C'、'F'、'A'`。默认为`'A'`。
   - `subok`默认为`False`
   - `ndmin`：指定结果`ndarray`最少有多少个维度。

2. `np.asarray(a[, dtype, order])`：将`a`转换成一个`ndarray`。其中`a`是`array_like`的对象， 可以是`list`、`list of tuple`、`tuple`、`tuple of list`、`ndarray`类型。`order`默认为`C`。

3. `np.asanyarray(a[, dtype, order])`：将`a`转换成`ndarray`。

4. `np.ascontiguousarray(a[, dtype])`：返回C风格的连续`ndarray`

5. `np.asmatrix(data[, dtype])`：返回`matrix` <p align="center">
  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/N3ycipuOs5Bv.JPG">
</p>


6. `np.copy(a[, order])`：返回`ndarray`的一份深拷贝

7. `np.frombuffer(buffer[, dtype, count, offset])`：从输入数据中返回一维`ndarray`。`count`指定读取的数量，`-1`表示全部读取；`offset`指定从哪里开始读取，默认为0。创建的数组与`buffer`共享内存。`buffer`是一个提供了`buffer`接口的对象（内置的`bytes/bytearray/array.array`类型提供了该接口）。 <p align="center">
  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/9H2AeUzGQnLJ.JPG">
</p>


8. `np.fromfile(file[, dtype, count, sep])` ：从二进制文件或者文本文件中读取数据返回`ndarray`。`sep`：当从文本文件中读取时，数值之间的分隔字符串，如果`sep`是空字符串则表示文件应该作为二进制文件读取；如果`sep`为`" "`表示可以匹配0个或者多个空白字符。

9. `np.fromfunction(function, shape, **kwargs)`：返回一个`ndarray`。从函数中获取每一个坐标点的数据。假设`shape`的维度为N，那么`function`带有`N`个参数，`fn(x1,x2,...x_N)`，其返回值就是该坐标点的值。

10. `np.fromiter(iterable, dtype[, count])`：从可迭代对象中迭代获取数据创建一维`ndarray`。

11. `np.fromstring(string[, dtype, count, sep])`：从字符串或者`raw binary`中创建一维`ndarray`。如果`sep`为空字符串则`string`将按照二进制数据解释（即每个字符作为`ASCII`码值对待）。创建的数组有自己的数据存储区。 <p align="center">
  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/NqtcNAFFUufc.JPG">
</p>


12. `np.loadtxt(fname[, dtype, comments, delimiter, ...])`:从文本文件中加载数据创建`ndarray`，要求文本文件每一行都有相同数量的数值。`comments`：指示注释行的起始字符，可以为单个字符或者字符列表（默认为`#`）。`delimiter`:指定数值之间的分隔字符串，默认为空白符。`converters`：将指定列号(0,1,2...)的列数据执行转换，是一个`map`，如`{0:func1}`表示对第一列数据执行`func1(val_0)`。`skiprows`：指定跳过开头的多少行。`usecols`：指定读取那些列（0表示第一列）。

#### 2.3 从数值区间创建

1. `np.arange([start,] stop[, step,][, dtype])`:返回均匀间隔的值组成的一维`ndarray`。区间是半闭半开的`[start,stop)`，其采样行为类似Python的`range`函数。`start`为开始点，`stop`为终止点，`step`为步长，默认为1。这几个数可以为整数可以为浮点数。注意如果`step`为浮点数，则结果可能有误差，因为浮点数相等比较不准确。

2. `np.linspace(start, stop[, num, endpoint, ...])` ：返回`num`个均匀采样的数值组成的一维`ndarray`（默认为50）。区间是闭区间`[start,stop]`。`endpoint`为布尔值，如果为真则表示`stop`是最后采样的值（默认为`True`），否则结果不包含`stop`。`retstep`如果为`True`则返回结果包含采样步长`step`，默认为`True`。

3. `np.logspace(start, stop[, num, endpoint, base, ...])`：返回对数级别上均匀采样的数值组成的一维`ndarray`。采样点开始于`base^start`，结束于`base^stop`。`base`为对数的基，默认为 10。

   - 它逻辑上相当于先执行`arange`获取数组`array`，然后再执行`base^array[i]`获取采样点
   - 它没有`retstep` 关键字参数

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/NhYowOLC3qFE.JPG">
  </p>
  

### 3. 数组的索引

#### 3.1 一维数组的索引

1. 一维数组的索引和列表相同。假设

   ```
   a1
   ```

    

   是一维数组

   - 可以指定一个整数`i`作为索引下标，如`a1[i]` <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/bEatNgSsTN6D.JPG">
   </p>
   

   - 可以指定一个切片作为索引下标，如`a1[i:j]`。通过切片获得的新的数组是原始数组的一个视图，它与原始数组共享相同的一块数据存储空间。 <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/SoCMdRwRwyhX.JPG">
   </p>
   

   - 可以指定一个整数列表对数组进行存取，如`a1[[i1,i2,i3]]`。此时会将列表中的每个整数作为下标(`i1/i2/i3`)，使用列表作为下标得到的数组(为 `np.array([a1[i1],a1[i2],a1[i3]])`)不和原始数组共享数据。 <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/QnaPVdukiQhL.JPG">
   </p>
   

   - 可以指定一个整数数组作为数组下标，如

     ```
     a1[a2]
     ```

     此时会得到一个形状和下标数组

     ```
     a2
     ```

     相同的新数组。新数组的每个元素都是下标数组中对应位置的值作为下标从原始数组中获得的值。新数组不和原始数组共享数据。

     - 当下标数组是一维数组时，其结果和用列表作为下标的结果相同 <p align="center">
       <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/j2cR3xs1DjJn.JPG">
     </p>
     
     - 当下标是多维数组时，结果也是多维数组 <p align="center">
       <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/XmpIEruqIlkK.JPG">
     </p>
     

   - 可以指定一个布尔数组作为数组下标，如

     ```
     a1[b]
     ```

     。此时将获得数组

     ```
     a1
     ```

     中与数组

     ```
     b
     ```

     中的

     ```
     True
     ```

     对应的元素。新数组不和原始数组共享数据。

     - 布尔数组的形状与数组`a1` 完全相同，它就是一个`mask`

```
xxxxxxxxxx
- 如果是布尔列表，情况也相同
- 如果布尔数组的长度不够，则不够的部分作为`False`（该特性是`deprecating`，建议不要使用）
  ![index_bool](../imgs/ndarray/index_bool.JPG)
```

1. 上述介绍的一维数组的索引，既可以用于数组元素的选取，也可以用于数组元素的赋值

   - 你可以赋一个值，此时该值会填充被选取出来的每一个位置
   - 你可以赋值一个数组或者列表，此时数组或者列表的形状要跟你选取出来的位置的形状完全匹配（否则报出警告）
     - 数组不同于列表。对于列表，你无法对列表切片赋一个值，而是要赋一个形状相同的值

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/GwchSzDucUKs.JPG">
  </p>
  

#### 3.2 多维数组的索引

1. 多维数组使用元组作为数组的下标，如`a[1,2]`，当然你也可以添加圆括号为`a[(1,2)]`。

   - 元组中每个元素和数组的每个轴对应。下标元组的第 0 个元素与数组的第 0 轴对应，如第 1 个元素与数组的第 1 轴对应...

2. 多维数组的下标必须是一个长度和数组的维度`ndim`相等的元组。

   - 如果下标元组的长度大于数组的维度`ndim`，则报错
   - 如果下标元组的长度小于数组的维度`ndim`，则在元组的后面补 `:`，使得下标元组的长度等于数组维度`ndim`。
   - 如果下标对象不是元组，则`Numpy`会首先将其转换为元组。

   下面的讨论都是基于下标元组的长度等于数组维度`ndim`的条件。

3. 单独生成切片时，需要使用`slice(begin,end,step)` 来创建。其参数分别为：开始值，结束值，间隔步长。如果某些参数需要省略，则使用`None`。因此， `a[2:,2]`等价于`a[slice(2,None,None),2]`

   - 使用`python`内置的`slice()`创建下标比较麻烦(首先构造切片，再构造下标元组)，`numpy`提供了一个`numpy.s_`对象来帮助我们创建数组下标。`s_`对象实际上是`IndexExpression`类的一个对象 <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/BSNfX3yXmXGe.JPG">
   </p>
   

4. 多维数组的下标元组的元素可能为下列类型之一：整数、切片、整数数组、布尔数组。如果不是这些类型，如列表或者元组，则将其转换成整数数组。 

   - 多维数组的下标全部是整数或者切片：索引得到的是元素数组的一个视图。 <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/lnBg3DMKKEbb.JPG">
   </p>
   

   - 多维数组的下标全部是整数数组：假设多维数组为$X$。假设这些下标整数数组依次为$A_1,A_2,\cdots,A_n$。这$n$个数组必须满足广播条件。假设它们进行广播之后的维度为$M$，形状为$(d_0,d_1,\cdots,d_{M-1})$即：广播之后有$M$个轴：第 0 轴长度为$d_0$，...，第$M-1$轴长度为$d_{M-1}$。假设$A_1,A_2,\cdots,A_n$经过广播之后分别为数组$A^{\prime}_1,A^{\prime}_2,\cdots,A^{\prime}_n$

     则：索引的结果也是一个数组$R$，结果数组$R$的维度为$M$，形状为$(d_0,d_1,\cdots,d_{M-1})$。其中

    $$ R[i_0,i_1,\cdots,i_{M-1}]=\ X[A^{\prime}*1[i_0,i_1,\cdots,i*{M-1}],A^{\prime}*2[i_0,i_1,\cdots,i*{M-1}],\cdots,A^{\prime}*n[i_0,i_1,\cdots,i*{M-1}]]$$

> 结果数组的下标并不来源于$X$，而是来源于下标数组的广播之后的数组。相反，如果多维数组的下标为整数或者切片，则结果数组的下标来源于$X$

<p align="center">
  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/VGFSVpai2VMw.JPG">
</p>


- 多维数组的下标包含整数数组、切片：则切片/整数下标与整数数组下标分别处理。 <p align="center">
  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/8xqwWRWMtvuL.JPG">
</p>


- 多维数组的下标是布尔数组或者下标元组中包含了布尔数组，则相当于将布尔数组通过

  ```
  nonzero
  ```

   

  将布尔数组转换成一个整数数组的元组，然后使用整数数组进行下标运行。

  - `nonzero(a)`返回数组`a`中，值非零的元素的下标。它返回值是一个长度为`a.ndim`的元组，元组的每个元素都是一个一维的整数数组，其值为非零元素的下标在对应轴上的值。如：第 0 个元素为`a`中的非零值元素在`0`轴的下标、第 1 个元素为`a`中的非零值元素在`1`轴的下标，... <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/nQXnOCrIjxWR.JPG">
  </p>
  

当下标使用整数或者切片时，所取得的数据在数据存储区域中是等间隔分布的。因为只需要修改数组的`ndim/shape/strides`等属性以及指向数据存储区域的`data`指针就能够实现整数和切片下标的索引。所以新数组和原始数组能够共享数据存储区域。

当使用整数数组（整数元组，整数列表页转换成整数数组），布尔数组时，不能保证所取得的数据在数据存储区中是等间隔的，因此无法和原始数组共享数据，只能对数据进行复制。

索引的下标元组中：

- 如果下标元组都是切片，则索引结果的数组与原始数组的维度相同（轴的数量相等）
- 每多一个整数下标，则索引结果的数组就少一个维度（少一个轴）
- 如果所有的下标都是整数，则索引结果的维度为 0
- 如果下标元组中存在数组，则还需要考虑该下标数组广播后的维度

通过索引获取的数组元素的类型为数组的`dtype`类型 。如果你想获取标准`python`类型，可以使用数组的`item()`方法。

<p align="center">
  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/v8ML5TUaJLJD.JPG">
</p>


#### 3.3 索引的维度变换

1. 对于数组，如果我们不考虑下标数组的情况，也就是：其下标仅仅为整数、或者切片，则有：

   - 每次下标中出现一个整数下标，则索引结果的维度降 1。该维度被吸收掉
   - 每次下标中出现一个切片下标，则该维度保持不变 <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/qmOYUV5bhUrx.JPG">
   </p>
   

2. 前面提到：`多维数组的下标必须是一个长度和数组的维度 ndim 相等的元组`。但是如果下标中包含`None`，则可以突破这一限制。每多一个`None`，则索引结构维度升 1 。

   - 当数组的下标元组的长度小于等于数组的维度`ndim`时，元组中出现的`None`等价于切片`:`
   - 当数组的下标元组的长度大于数组的维度`ndim`时，元组中哪里出现`None`，索引结果就在哪里创建一个新轴，该轴长度为 1。如`c=a[0,:,None]`，索引结果的维度为 `(3,1)`；而`d=a[0,None,:]`的索引结果维度为`(1,3)`

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/MqEJiGje07PM.JPG">
  </p>
  

### 4. 操作多维数组

1. `numpy.concatenate((a1, a2, ...), axis=0)`：连接多个数组。其中`(a1,a2,...)`为数组的序列，给出了待连接的数组，它们沿着`axis`指定的轴连接。

   - 所有的这些数组的形状，除了`axis`轴之外都相同 <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200602/KT9ITumUDH8N.JPG">
   </p>
   

2. - ```
     numpy.vstack(tup)
     ```

     :等价于

     ```
     numpy.concatenate((a1, a2, ...), axis=0)
     ```

     。沿着 0 轴拼接数组

     - 沿0轴拼接（垂直拼接），增加行

   - ```
     numpy.hstack(tup)
     ```

     :等价于

     ```
     numpy.concatenate((a1, a2, ...), axis=1)
     ```

     。沿着 1 轴拼接数组

     - 沿1轴拼接（水平拼接），增加列

   - ```
     numpy.column_stack(tup)
     ```

     ：类似于

     ```
     hstack
     ```

     ，但是如果被拼接的数组是一维的，则将其形状修改为二维的

     ```
     (N,1)
     ```

     。

     - 沿列方向拼接，增加列

   - ```
     numpy.c_
     ```

     对象的

     ```
     []
     ```

     方法也可以用于按列连接数组。但是如果被拼接的数组是一维的，则将其形状修改为二维的

     ```
     (N,1)
     ```

     。

     - 沿列方向拼接，增加列

<p align="center">
  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/dcAWciaRigu0.JPG">
</p>


<p align="center">
  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/dz1JzOgA68ug.JPG">
</p>


3. `numpy.split(ary, indices_or_sections, axis=0)`用于沿着指定的轴拆分数组`ary`。`indices_or_sections`指定了拆分点：

   - 如果为整数`N`，则表示平均拆分成`N`份。如果不能平均拆分，则报错
   - 如果为序列，则该序列指定了划分区间（无需指定最开始的`0`起点和终点）。如`[1,3]`指定了区间：`[0,1],[1,3],[3:]`

   而`numpy.array_split(ary, indices_or_sections, axis=0)`的作用也是类似。唯一的区别在于：当`indices_or_sections`为整数，且无法平均拆分时，并不报错，而是尽可能的维持平均拆分。 <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/dTbPCCO3tBXb.JPG">
   </p>
   

4. `numpy.transpose(a, axes=None)`：重置轴序。如果`axes=None`，则默认重置为逆序的轴序（如原来的`shape=(1,2,3)`，逆序之后为`(3,2,1)`）。如果`axes!=None`，则要给出重置后的轴序。它获得的是原数组的视图。

   `numpy.swapaxes(a, axis1, axis2)`：交换指定的两个轴`axis1/axis2`。它获得是原数组的视图。

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/Oc3QxJkDPLOf.JPG">
  </p>
  

### 5.打印数组

1. 当打印`ndarray`时，`numpy`按照Python的嵌套`list`的格式打印输出，但是按照以下顺序打印：

   - 最底层的`axis`按照从左到右的顺序输出
   - 次底层的`axis`按照从上到下的顺序输出
   - 其他层的`axis`也是按照从上到下的顺序输出，但是每个`slice`中间间隔一条空行

   如： 一维的`ndarray`按行打印；二维的`ndarray`按照矩阵打印；三维的`ndarray`按照矩阵的`list`打印

   如果`ndarray`太大，那么`numpy`默认跳过中间部分的数据而只是输出四个角落的数据。

2. 要想任何时候都打印全部数据，可以在`print(array)`之前设置选项`numpy.set_printoptions(threshold='nan')`。这样后续的打印`ndarray`就不会省略中间数据。

### 6. Nan 和无穷大

1. 在

   ```
   numpy
   ```

   中，有几个特殊的数：

   - `numpy.nan`表示`NaN`（`Not a Number`），它并不等价于`numpy.inf`（无穷大）。
   - `numpy.inf`：正无穷
   - `numpy.PINF`：正无穷（它就引用的是`numpy.inf`）
   - `numpy.NINF`：负无穷

2. 有下列函数用于判断这几个特殊的数：

   - `numpy.isnan(x[,out])`：返回`x`是否是个`NaN`，其中`x`可以是标量，可以是数组

   - ```
     numpy.isfinite(x[, out])
     ```

     ：返回

     ```
     x
     ```

     是否是个有限大小的数，其中

     ```
     x
     ```

     可以是标量，可以是数组

     - `numpy.isfinite(np.nan)`返回`False`，因为`NaN`首先就不是一个数

   - ```
     numpy.isposinf(x[, out])
     ```

     ：返回

     ```
     x
     ```

     是否是个正无穷大的数，其中

     ```
     x
     ```

     可以是标量，可以是数组

     - `numpy.isposinf(np.nan)`返回`False`，因为`NaN`首先就不是一个数

   - ```
     numpy.isneginf(x[, out])
     ```

     ：返回

     ```
     x
     ```

     是否是个负无穷大的数，其中

     ```
     x
     ```

     可以是标量，可以是数组

     - `numpy.isneginf(np.nan)`返回`False`，因为`NaN`首先就不是一个数

   - ```
     numpy.isinf(x[, out])
     ```

     ：返回

     ```
     x
     ```

     是否是个无穷大的数，其中

     ```
     x
     ```

     可以是标量，可以是数组

     - `numpy.isinf(np.nan)`返回`False`，因为`NaN`首先就不是一个数

3. 下列函数用于对这几个特殊的数进行转换：

   - ```
     numpy.nan_to_num(x)
     ```

     ：将数组

     ```
     x
     ```

     中的下列数字替换掉，返回替换掉之后的新数组：

     - `NaN`：替换为0
     - 正无穷：替换为一个非常大的数字
     - 负无穷：替换为一个非常小的数字

## 二、 ufunc 函数

1. `ufunc` 函数是对数组的每个元素进行运算的函数。`numpy`很多内置的`ufunc`函数使用`C`语言实现的，计算速度非常快。

2. 基本上所有的

   ```
   ufunc
   ```

   函数可以指定一个

   ```
   out
   ```

   参数来保存计算结果数组，并返回

   ```
   out
   ```

   数组。同时如果未指定

   ```
   out
   ```

   参数，则创建新的数组来保存计算结果。

   - 如果你指定了`out`参数，则要求`out`数组与计算结果兼容。即：数组的尺寸要严格匹配，并且数组的`dtype`要匹配。 <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/YY8K7fv9rXzP.JPG">
   </p>
   

3. ```
   numpy
   ```

   的某些

   ```
   ufunc
   ```

   函数，如

   ```
   numpy.sin()
   ```

   ，支持计算单个数值。但是在单个数值的计算速度上，

   ```
   python
   ```

   的

   ```
   math.sin()
   ```

   要快得多。两个原因：

   - `numpy.sin()`为了同时支持数组和单个数值运算，其`C`语言的内部实现要比`math.sin()`复杂
   - 单个数值的计算上：`numpy.sin()`返回的是`numpy.float64`类型，而`math.sin()`返回的是`python`的标准`float`类型

### 1. 广播

1. 当使用`ufunc`函数对两个数组进行计算时，`ufunc`函数会对这两个数组的对应元素进行计算。这就要求这两个数组的形状相同。如果这两个数组的形状不同，就通过广播`broadcasting`进行处理：

   - 首先让所有输入数组都向其中维度最高的数组看齐。看齐方式为：在`shape`属性的左侧插入数字`1`
   - 然后输出数组的`shape`属性是输入数组的`shape`属性的各轴上的最大值
   - 如果输入数组的某个轴的长度为 1，或者与输出数组的各对应轴的长度相同，该数组能正确广播。否则计算出错
   - 当输入数组的某个轴的长度为 1时，沿着此轴运算时都用此轴上的第一组值。


2. 可以通过`numpy.broadcast_arrays()`查看广播之后的数组 <p align="center">
  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200602/XVXS4BqW0oDj.JPG">
</p>


3. 你可以通过`ndarray.repeat()`方法来手动重复某个轴上的值.其用法为`ndarray.repeat(repeats, axis=None)`，其中：

   - `repeats`为重复次数
   - `axis`指定被重复的轴，即沿着哪一轴重复。如果未指定，则将数组展平然后重复。返回的也是一个展平的数组

   被重复的是该轴的每一组值。

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/QQ9PR0urgrgG.JPG">
  </p>
  

4. `numpy`提供了`ogrid`对象，用于创建广播运算用的数组。`ogrid`对象像多维数组一样，使用切片元组作为下标，返回的是一组可以用于广播计算的数组。其切片有两种形式：

   - 开始值：结束值：步长。它指定返回数组的开始值和结束值（不包括）。默认的开始值为 0；默认的步长为 1。与`np.arange`类似
   - 开始值：结束值：长度 j。当第三个参数为虚数时，表示返回的数组的长度。与`np.linspace`类似。
   - 有多少个下标，则结果就是多少维的，同时也返回相应数量的数组。每个返回的数组只有某一维度长度大于1，其他维度的长度全部为 1。假设下标元组长度为3，则结果元组中：第一个数组的`shape=(3,1,1)`，第二个数组的`shape=(1,3,1)`，第三个数组的`shape=(1,1,3)`。 <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/7uNlPWwNwEft.JPG">
   </p>
   

5. `numpy`还提供了`mgrid`对象，它类似于`ogrid`对象。但是它返回的是广播之后的数组，而不是广播之前的数组：


<p align="center">
  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/47isOGd3f8BG.JPG">
</p>


6. `numpy`提供了`meshgrid()`函数，其用法为：`numpy.meshgrid(x1,x2,...xn)`。其中`xi`是都是一维数组。返回一个元组 `(X1,X2,...Xn)`，是广播之后的数组。假设`xi`的长度为 `li`，则返回元组的每个数组的形状都是 `(l1,l2,...ln)`。

<p align="center">
  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/5E0GjkRV6ueO.JPG">
</p>


7. `numpy.ix_()`函数可以将`N`个一维数组转换成可广播的`N`维数组。其用法为`numpy.ix_(x1,x2,x3)`，返回一个元组。元组元素分别为对应的可广播的`N`维数组。

   - 返回的是广播前的数组，而不是广播后的数组

   - 每个转换前的一维数组，对应了一个转换后的 `N` 维数组

    <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/faQVwfYbQTu8.JPG">
    </p>
    

### 2. 四则运算

1. `numpy`提供的四则运算如下，这些四则运算同时提供了函数形式以及表达式形式：

   - 加法：表达式形式`y=x1+x2`，使用`ufunc`函数的形式：`numpy.add(x1,x2[,out=y])`

   - 减法：表达式形式`y=x1-x2`，使用`ufunc`函数的形式：`numpy.subtract(x1,x2[,out=y])`

   - 乘法：表达式形式`y=x1*x2`，使用`ufunc`函数的形式：`numpy.multiply(x1,x2[,out=y])`

   - 真除法：表达式形式

     ```
     y=x1/x2
     ```

     ，使用

     ```
     ufunc
     ```

     函数的形式：

     ```
     numpy.true_divide(x1,x2[,out=y])
     ```

     - `python3` 中，`numpy.divide(x1,x2[,out=y])`也是真除法

   - 取整除法：表达式形式`y=x1//x2`，使用`ufunc`函数的形式：`numpy.floor_divide(x1,x2[,out=y])`

   - 取反：表达式形式`y=-x`，使用`ufunc`函数的形式：`numpy.negative(x[,out=y])`

   - 乘方：表达式形式`y=x1**x2`，使用`ufunc`函数的形式：`numpy.power(x1,x2[,out=y])`

   - 取余数：表达式形式`y=x1%x2`，使用`ufunc`函数的形式：`numpy.remainder(x1,x2[,out=y])`

   <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200602/LvbAszqaPqkb.JPG">
   </p>
   

2. 对于 `np.add(a,b,a)` 这种可以使用`a+=b`来表示。这些四则运算都可以采用这种方式。

3. 当表达式很复杂时，如果同时数组很大，则会因为产生大量的中间结果而降低程序的运算速度。如： `x=a*b+c`等价于：

   ```
   xxxxxxxxxx
   ```

   ```
     t=a*b
   ```

   ```
     x=t+c
   ```

   ```
     del t
   ```

   我们可以使用：

   ```
   xxxxxxxxxx
   ```

   ```
     x=a*b
   ```

   ```
     x+=c
   ```

   从而减少了一次内存分配。

### 3. 比较运算

1. `numpy`提供的比较运算如下，这些比较运算同时提供了函数形式以及表达式形式，并且产生的结果是布尔类型的数组：

   - 等于： 表达式形式`y=x1==x2`，使用`ufunc`函数的形式：`numpy.equal(x1,x2[,out=y])`
   - 不等于： 表达式形式`y=x1!=x2`，使用`ufunc`函数的形式：`numpy.not_equal(x1,x2[,out=y])`
   - 小于： 表达式形式`y=x1，使用`ufunc`函数的形式：`numpy.less(x1,x2[,out=y])`
   - 小于等于： 表达式形式`y=x1<=x2`，使用`ufunc`函数的形式：`numpy.less_equal(x1,x2[,out=y])`
   - 大于： 表达式形式`y=x1>x2`，使用`ufunc`函数的形式：`numpy.greater(x1,x2[,out=y])`
   - 大于等于： 表达式形式`y=x1>=x2`，使用`ufunc`函数的形式：`numpy.greater_equal(x1,x2[,out=y])`

   <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200602/iEyj9iEfgLmb.JPG">
   </p>
   

### 4. 逻辑运算

1. 由于`python`中的布尔运算使用`and/or/not`关键字，因此它们无法被重载。`numpy`提供的数组布尔运算只能通过`ufunc`函数进行，这些函数以`logical_`开头。进行逻辑运算时，对于数值零视作`False`；对数值非零视作`True`。运算结果也是一个布尔类型的数组：

   - 与：`ufunc`函数的形式：`numpy.logical_and(x1,x2[,out=y])`
   - 或：`ufunc`函数的形式：`numpy.logical_or(x1,x2[,out=y])`
   - 否定：`ufunc`函数的形式：`numpy.logical_not(x[,out=y])`
   - 异或：`ufunc`函数的形式：`numpy.logical_xor(x1,x2[,out=y])`

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/rJU0Sl5isxdt.JPG">
  </p>
  

2. 对于数组`x`，`numpy`定义了下面的操作：

   - `numpy.any(x)`：只要数组中有一个元素值为`True`（如果数值类型，则为非零），则结果就返回`True`；否则返回`False`
   - `numpy.all(x)`：只有数组中所有元素都为`True`（如果数值类型，则为非零），则结果才返回`True`；否则返回`False` <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200602/9Ypp1Cpi1ERS.JPG">
   </p>
   

### 5. 位运算

1. `numpy`提供的位运算如下，这些位运算同时提供了函数形式（这些函数以`bitwise_`开头）以及表达式形式。其中输入数组必须是整数或者布尔类型（如果是浮点数则报错）：

   - 按位与：表达式形式`y=x1&x2`，使用`ufunc`函数的形式：`numpy.bitwise_and(x1,x2[,out=y])`
   - 按位或：表达式形式`y=x1|x2`，使用`ufunc`函数的形式：`numpy.bitwise_or(x1,x2[,out=y])`
   - 按位取反：表达式形式`y=~x`，使用`ufunc`函数的形式：`numpy.bitwise_not(x[,out=y])`
   - 按位异或：表达式形式`y=x1^x2`，使用`ufunc`函数的形式：`numpy.bitwise_xor(x1,x2[,out=y])`

   <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200602/uGhIjmvbnbdI.JPG">
   </p>
   

2. 有几点注意：

   - 位运算符的优先级要比比较运算符高
   - 整数数组的位运算和`C`语言的位运算符相同，注意正负号

### 6. 自定义 ufunc 函数

1. 可以通过`frompyfunc()`将计算单个元素的函数转换成`ufunc`函数。调用格式为：`my_ufunc=frompyfunc(func,nin,nout)`。其中：

   - `func`：计算单个元素的函数
   - `nin`：`func`的输入参数的个数
   - `nout`：`func`返回值的个数

   调用时，使用`my_ufunc(...)`即可。 <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/Q5WPhmAp9pf3.JPG">
   </p>
   

2. 也可以通过`vectorize()`函数来实现`frompyfunc()`的功能。其原型为：`np.vectorize(func, otypes='', doc=None, excluded=None)`。其中：

   - `func`：计算单个元素的函数
   - `otypes`：可以是一个表示结果数组元素类型的字符串，也可以是一个类型列表。如果使用类型列表，可以描述多个返回数组的元素类型
   - `doc`：函数的描述字符串。若未给定，则使用`func.__doc__`
   - `excluded`：指定`func`中哪些参数未被向量化。你可以指定一个字符串和整数的集合，其中字符串代表关键字参数，整数代表位置参数。

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/AiijXStDKMSz.JPG">
  </p>
  

### 7. ufunc 对象的方法

1. `ufunc`函数对象本身还有一些方法。

   - 这些方法只对于两个输入、一个输出的`ufunc`函数函数有效。对于其他的`ufunc`函数对象调用这些方法时，会抛出`ValueError`异常。

2. `ufunc.reduce`方法：类似于`Python`的`reduce`函数，它沿着`axis`参数指定的轴，对数组进行操作。

   - 相当于将``运算符插入到沿着`axis`轴的所有元素之间：`.reduce(array,axis=0,dtype=None)`
   - 经过一次`reduce`，结果数组的维度降低一维

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/t7luBgGR8Uz5.JPG">
  </p>
  

3. `ufunc.accumulate`方法：它类似于`reduce()`的计算过程，但是它会保存所有的中间计算结果，从而使得返回数组的形状和输入数组的形状相同： <p align="center">
  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200602/V3092YSrY99y.JPG">
</p>


4. `ufunc.outer`方法：相当于将``运算符对输入数组`A`和输入数组`B`的每一对元素对`(a,b)`起作用：`.reduce(A,B)`。结果数组维度为`A.dim+B.dim`。设`A`的`shape=(4,5)`，`B`的`shape`为`(4,)`，则结果数组的`shape=(4,5,4)`

   - 一维数组和一维数组的`outer`操作为二维数组
   - 多维数组的`outer`拆分成各自的一维操作 <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/IsV1TR5L1hhM.JPG">
   </p>
   

### 8. 数学函数

1. 下面是一元的数学函数：
   - `abs/fabs`：计算整数、浮点数或者复数的绝对值。对于非复数值，可以使用更快的`fabs`
   - `sqrt` ：计算平方根，相当于`a**0.5`
   - `square`：计算平方，相当于`a**2`
   - `exp`：计算指数$e^{x}$
   - `log/log10/log2/log1p`：分别为$\log_e(a),\log_{10}(a),\log_2(a),\log_e(1+x)$
   - `sign`：计算$sign(a)$
   - `ceil`：计算各元素的`ceiling`值：大于等于该值的最小整数
   - `floor`：计算个元素的`floor`值：小于等于该值的最大整数
   - `rint`：将各元素四舍五入到最接近的整数，保留`dtype`
   - `modf`：将数组的小数和整数部分以两个独立数组的形式返回
   - `isnan`：返回一个布尔数组，该数组指示那些是`NaN`
   - `isfinite/isinf`：返回一个布尔数组，该数组指示哪些是有限的/无限数
   - `cos/cosh/sin/sinh/tan/tanh`：普通和双曲型三角函数
   - `arccos/arcsosh/arcsin/arcsinh/arctan/arctanh`:反三角函数

## 三、 函数库

### 1. 随机数库

1. `numpy`中的随机和分布函数模块有两种用法：函数式以及类式

#### 1.1 函数式

1. 随机数

   - `numpy.random.rand(d0, d1, ..., dn)`:指定形状`(d0, d1, ..., dn)`创建一个随机的`ndarray`。每个元素值来自于半闭半开区间`[0,1)`并且服从均匀分布。

     - 要求`d0, d1, ..., dn`为整数
     - 如果未提供参数，则返回一个随机的浮点数而不是`ndarray`，浮点数值来自于半闭半开区间`[0,1)`并且服从均匀分布。

   - `numpy.random.randn(d0, d1, ..., dn)`：指定形状`(d0, d1, ..., dn)`创建一个随机的`ndarray`。每个元素值服从正态分布，其中正态分布的期望为0，方差为1

     - 要求`d0, d1, ..., dn`为整数或者可以转换为整数
     - 如果`di`为浮点数，则截断成整数
     - 如果未提供参数，则返回一个随机的浮点数而不是`ndarray`，浮点数值服从正态分布，其中正态分布的期望为0，方差为1

   - `numpy.random.randint(low[, high, size])`：返回一个随机的整数`ndarray`或者一个随机的整数值。

     - 如果`high`为`None`，则表示整数值都取自`[0,low)`且服从`discrete uniform`分布
     - 如果`high`给出了值，则表示整数值都取自`[low,high)`且服从`discrete uniform`分布
     - `size`是一个整数的元组，指定了输出的`ndarray`的形状。如果为`None`则表示输出为单个整数值

   - `numpy.random.random_integers(low[, high, size])`：返回一个随机的整数`ndarray`或者一个随机的整数值。

     - 如果`high`为`None`，则表示整数值都取自`[1,low]`且服从`discrete uniform`分布
     - 如果`high`给出了值，则表示整数值都取自`[low,high]`且服从`discrete uniform`分布
     - `size`是一个整数的元组，指定了输出的`ndarray`的形状。如果为`None`则表示输出为单个整数值

     > 它与`randint`区别在于`randint`是半闭半开区间，而`random_integers`是全闭区间

   - `numpy.random.random_sample([size])`：返回一个随机的浮点`ndarray`或者一个随机的浮点值，浮点值是`[0.0,1.0)`之间均匀分布的随机数

     - `size`为整数元组或者整数，指定结果`ndarray`的形状。如果为`None`则只输出单个浮点数
     - 如果想生成`[a,b)`之间均匀分布的浮点数，那么你可以用`(b-a)*random_sample()+a`

     > 如果`size`有效，它的效果等于`numpy.random.rand(*size)`； 如果`size`无效，它的效果等于`numpy.random.rand()`

    <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/jOaPQK7D5Yht.JPG">
    </p>
    

   - `numpy.random.random([size])`：等价于`numpy.random.random_sample([size])`

   - `numpy.random.ranf([size])`：等价于`numpy.random.random_sample([size])`

   - `numpy.random.sample([size])`：等价于`numpy.random.random_sample([size])`

     <p align="center">
       <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/C9rGAy3qoOEH.JPG">
     </p>
     

   - `numpy.random.choice(a[, size, replace, p])`:从一维数组中采样产生一组随机数或者一个随机数

     - `a`为一位数组或者`int`，如果是`int`则采样数据由`numpy.arange(n)`提供，否则采用数据由`a`提供

     - `size`为整数元组或者整数，指定结果`ndarray`的形状。如果为`None`则只输单个值

     - `replace`：如果为`True`则可以重复采样（有放回的采样）；如果为`False`，则采用不放回的采样

     - `p`：为一维数组，用于指定采样数组中每个元素值的采样概率。如果为`None`则均匀采样。

     - 如果参数有问题则抛出异常：比如`a`为整数但是小于0，比如`p`不满足概率和为`1`，等等。。

      <p align="center">
        <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/ErrayDOiuu9p.JPG">
      </p>
      

   - `numpy.random.bytes(length)`：返回`length`长度的随机字节串。`length`指定字节长度。

    <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/utCphkjBJ8MU.JPG">
    </p>
    

2. 排列组合

   - `numpy.random.shuffle(x)`:原地随机混洗`x`的内容，返回`None`。`x`为`array-like`对象，原地修改它

   - `numpy.random.permutation(x)`：随机重排`x`，返回重排后的`ndarray`。`x`为`array-like`对象，不会修改它

     - 如果`x`是个整数，则重排`numpy.arange(x)`

     - 如果

       ```
       x
       ```

       是个数组，则拷贝它然后对拷贝进行混洗

       - 如果`x`是个多维数则只是混洗它的第0维

     <p align="center">
       <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/QahL0p34Qqsz.JPG">
     </p>
     

3. 概率分布函数：下面是共同参数：`size`若非`None`，则它指定输出`ndarray`的形状。如果为`None`，则输出单个值。

   - `numpy.random.beta(a, b[, size])`：Beta分布。其中`a,b`都是Beta分布的参数，要求非负浮点数。

     - 贝塔分布为：

      $$ f(x;\alpha,\beta)=\frac {1}{B(\alpha,\beta)} x^{\alpha-1}(1-x)^{\beta-1}$$

       其中：

      $$ B(\alpha,\beta)=\int_0^{1} t^{\alpha-1}(1-t)^{\beta-1},dt$$

   - `numpy.random.binomial(n, p[, size])`:二项分布。其中`n,p`都是二项分布的参数，要求`n`为大于等于0的浮点数，如果它为浮点数则截断为整数；`p`为`[0,1]`之间的浮点数。

     - 二项分布为：

      $$ P(N)=\binom{n}{N}p^{N}(1-p)^{n-N}$$

   - `numpy.random.chisquare(df[, size])`:卡方分布。其中`df`为整数，是卡方分布的自由度（若小于等于0则抛出异常）。

     - 卡方分布为：

      $$ p(x)=\frac{(1/2)^{k/2}}{\Gamma(k/2)} x^{k/2-1}e^{-x/2}$$

       其中

      $$ \Gamma(x)=\int^{\infty}_0 t^{x-1}e^{-t},dt$$

   - `numpy.random.dirichlet(alpha[, size])`:狄利克雷分布。其中`alpha`是个数组，为狄利克雷分布的参数。

   - `numpy.random.exponential([scale, size])`:指数分布。`scale`为浮点数，是参数$\beta$

     - 指数分布的概率密度函数为:

      $$ f(x;\frac {1}{\beta})=\frac{1}{\beta}\exp(-\frac{x}{\beta})$$

   - `numpy.random.f(dfnum, dfden[, size])`:`F`分布。`dfnum`为浮点数，应该大于0，是分子的自由度； `dfden`是浮点数，应该大于0，是分母的自由度。

   - `numpy.random.gamma(shape[, scale, size])`:伽玛分布。其中`shape`是个大于0的标量，表示分布的形状；`scale`是个大于0的标量，表示伽玛分布的`scale`（默认为1）。

     - 伽玛分布的概率密度函数为:

      $$ p(x)=x^{k-1} \frac {e^{-x/\theta}}{\theta^{k}\Gamma(k)}$$

       ，其中`k`为形状，$\theta$为`scale`

   - `numpy.random.geometric(p[, size])`:几何分布。其中`p`是单次试验成功的概率。

     - 几何分布为：

      $$ f(k)=(1-p)^{k-1}p$$

   - `numpy.random.gumbel([loc, scale, size])`:甘贝尔分布。其中`loc`为浮点数，是分布的`location of mode`，`scale`是浮点数，为`scale`。

     - 甘贝尔分布:

     - ```
       xxxxxxxxxx
       ```

       ```
       p(x)=\frac {e^{-(x-\mu)/\beta}}{\beta}  e^{-e-(x-\mu)/\beta}
       ```

       Preview OK

      $p(x)=\frac {e^{-(x-\mu)/\beta}}{\beta} e^{-e-(x-\mu)/\beta}$

       ，其中$\mu$为`location of mode`，$\beta$为`scale`

   - `numpy.random.hypergeometric(ngood, nbad, nsample[, size])`: 超几何分布。其中`ngood`为整数或者`array_like`，必须非负数，为好的选择；`nbad`为整数或者`array_like`，必须非负数，表示坏的选择。

     - 超级几何分布：

      $$ P(x)= \frac {\binom{m}{n} \binom{N-m}{n-x}} {\binom{N}{n}}, 0 \le x \le m \ \text{and} \ n+m-N \le x \le n$$

       ，其中`n=ngood`，`m=nbad`，`N`为样本数量。`P(x)`为`x`成功的概率

   - `numpy.random.laplace([loc, scale, size])`:拉普拉斯分布。`loc`为浮点数，`scale`为浮点数

     - 拉普拉斯分布：

      $$ f(x;\mu,\lambda)=\frac {1}{2\lambda} \exp(- \frac{|x-\mu|}{\lambda})$$

       ，其中 `loc`=$\mu$， `scale`=$\lambda$

   - `numpy.random.logistic([loc, scale, size])`:逻辑斯谛分布。其中`loc`为浮点数，`scale`为大于0的浮点数

     - 逻辑斯谛分布：

      $$ P(x)= \frac {e^{-(x-\mu)/s}}{s(1+e^{-(x-\mu)/s})^{2}}$$

       ， 其中 `loc`=$\mu$, `scale`=$s$

   - `numpy.random.lognormal([mean, sigma, size])`:对数正态分布。其中`mean`为浮点数，`sigma`为大于0的浮点数。

     - 对数正态分布：

      $$ p(x)=\frac {1}{\sigma x \sqrt{2\pi}} e^{-(\ln(x)-\mu)^{2}/(2\sigma^{2})}$$

       ，其中`mean`=$\mu$， `sigma`=$\sigma$

   - `numpy.random.logseries(p[, size])`:对数分布，其中`p`为`[0.0--1.0]`之间的浮点数。

     - 对数分布：

      $$ P(k)=\frac {-p^{k}}{k\ln(1-p)}$$

   - `numpy.random.multinomial(n, pvals[, size])`:多项式分布。`n`为执行二项分布的试验次数，`pvals`为浮点序列，要求这些序列的和为1，其长度为`n`。

   - `numpy.random.multivariate_normal(mean, cov[, size])`:多元正态分布。`mean`为一维数组，长度为`N`；`cov`为二维数组，形状为`(N,N)`

   - `numpy.random.negative_binomial(n, p[, size])`:负二项分布。`n`为整数，大于0；`p`为`[0.0--1.0]`之间的浮点数。

     - 负二项分布：

      $$ P(N;n,p)=\binom{N+n-1}{n-1}p^{n}(1-p)^{N}$$

   - `numpy.random.noncentral_chisquare(df, nonc[, size])`:非中心卡方分布。`df`为整数，必须大于0;`noc`为大于0的浮点数。

     - 非中心卡方分布：

      $$ P(x;k,\lambda)= \sum_{i=0}^{\infty } f_Y(x) \frac {e^{-\lambda/2}(-\lambda/2)^{i}}{i!}$$

       其中$Y=Y_{k+2i}$为卡方分布， `df`为`k`，`nonc`为$\lambda$

   - `numpy.random.noncentral_f(dfnum, dfden, nonc[, size])`:非中心`F`分布。其中`dfnum`为大于1的整数，`dfden`为大于1的整数，`nonc`为大于等于0的浮点数。

   - `numpy.random.normal([loc, scale, size])`:正态分布。其中`loc`为浮点数，`scale`为浮点数。

     - 正态分布：

      $$ p(x)=\frac {1}{\sqrt{2\pi\sigma^{2}}}e^{-(x-\mu)^{2}/(2\sigma^{2})}$$

       ，其中`loc`=$\mu$， `scale`=$\sigma$

   - `numpy.random.pareto(a[, size])`:帕累托分布。其中`a`为浮点数。

     - 帕累托分布：

      $$ p(x)= \frac {\alpha m ^{\alpha}}{x^{\alpha+1}}$$

       ，其中`a`=$\alpha$, `m`为`scale`

   - `numpy.random.poisson([lam, size])`:泊松分布。其中`lam`为浮点数或者一个浮点序列（浮点数大于等于0）。

     - 泊松分布：

      $$ f(k;\lambda)=\frac {\lambda^{k}e^{-\lambda}}{k!}$$

       ，其中`lam`=$\lambda$

   - `numpy.random.power(a[, size])`:幂级数分布。其中`a`为大于0的浮点数。

     - 幂级数分布：

      $$ P(x;a)=ax^{a-1},0\le x \le 1,a \gt 0$$

   - `numpy.random.rayleigh([scale, size])`: 瑞利分布。其中`scale`为大于0的浮点数。

     - 瑞利分布：

      $$ P(x;\sigma)=\frac{x}{\sigma^{2}}e^{-x^{2}/(2\sigma^{2})}$$

       ，其中`scale`=$\sigma$

   - `numpy.random.standard_cauchy([size])`:标准柯西分布。

     - 柯西分布：

      $$ P(x;x_0,\gamma)=\frac{1}{\pi\gamma[1+((x-x_0)/\gamma)^{2}]}$$

       ，其中标准柯西分布中，$x_0=1,\gamma=1$

   - `numpy.random.standard_exponential([size])`:标准指数分布。其中`scale`等于1

   - `numpy.random.standard_gamma(shape[, size])`:标准伽玛分布，其中`scale`等于1

   - `numpy.random.standard_normal([size])`:标准正态分布，其中`mean`=0，`stdev`等于1

   - `numpy.random.standard_t(df[, size])`:学生分布。其中`df`是大于0的整数。

     - 学生分布:

      $$ f(t;\nu)=\frac{\Gamma((\nu+1)/2)}{\sqrt{\nu\pi}\Gamma(\nu/2)}(1+t^{2}/\nu)^{-(\nu+1)/2}$$

       ， 其中 `df`=$\nu$

   - `numpy.random.triangular(left, mode, right[, size])`: 三角分布。其中`left`为标量，`mode`为标量，`right`为标量

     - 三角分布（其中`left`=`l`，`mode`=`m`，`right`=`r`）：

    $$ P(x;l,m,r)= \left{ \begin{matrix} \frac{2(x-l)}{(r-l)(m-l)}, & \text{for l \le x \le m*l*≤*x*≤*m*} \ \frac{2(r-x)}{(r-l)(r-m)}, & \text{for m \le x \le r*m*≤*x*≤*r*} \ 0, & \text{otherwise} \end{matrix} \right.$$

   - `numpy.random.uniform([low, high, size])`:均匀分布。其中`low`为浮点数；`high`为浮点数。

     - 均匀分布：

      $$ p(x)=\frac {1}{b-a}$$

       ，其中`low`=`a`, `high`=`b`

   - `numpy.random.vonmises(mu, kappa[, size])`:`Mises`分布。其中`mu`为浮点数，`kappa`为大于等于0的浮点数。

     - `Mises`分布：

      $$ p(x)= \frac{e^{\kappa \cos(x-\mu)}}{2\pi I_0(\kappa)}$$

       ，其中`mu`=$\mu$， `kappa`=$\kappa$，$I_0(\kappa)$是 `modified Bessel function of order 0`

   - `numpy.random.wald(mean, scale[, size])`:`Wald`分布。其中`mean`为大于0的标量，`scale`为大于等于0的标量

     - `Wald`分布：

      $$ P(x;\mu,\lambda)=\sqrt{\frac {\lambda}{2\pi x^{3}}} \exp {\frac{-\lambda(x-\mu)^{2}}{2\mu^{2}x}}$$

       ，其中`mean`=$\mu$， `scale`=$\lambda$

   - `numpy.random.weibull(a[, size])`： `Weibull`分布。其中`a`是个浮点数。

     - `Weibull`分布:

      $$ p(x)= \frac {a}{\lambda} (\frac {x}{\lambda})^{a-1} e^{-(x/\lambda)^{a}}$$

       ，其中`a`=$a$，$lambda$为`scale`

   - `numpy.random.zipf(a[, size])`:齐夫分布。其中`a`为大于1的浮点数。

     - 齐夫分布：

      $$ p(x)=\frac {x^{-a}}{\zeta(a)}$$

       ，其中 `a`=$a$，$\zeta$为 `Riemann Zeta`函数。

4. `numpy.random.seed(seed=None)`：用于设置随机数生成器的种子。`int`是个整数或者数组，要求能转化成32位无符号整数。

#### 1.2 RandomState类

1. 类式用法主要使用`numpy.random.RandomState`类，它是一个`Mersenne Twister`伪随机数生成器的容器。它提供了一些方法来生成各种各样概率分布的随机数。

   构造函数:`RandomState(seed)`。其中`seed`可以为`None`, `int`, `array_like`。这个`seed`是初始化伪随机数生成器。如果`seed`为`None`，则`RandomState`会尝试读取`/dev/urandom`或者`Windows analogure`来读取数据，或用者`clock`来做种子。

   > `Python`的`stdlib`模块`random`也提供了一个`Mersenne Twister`伪随机数生成器。但是`RandomState`提供了更多的概率分布函数。

   `RandomState`保证了通过使用同一个`seed`以及同样参数的方法序列调用会产生同样的随机数序列（除了浮点数精度上的区别）。

   `RandomState`提供了一些方法来产生各种分布的随机数。这些方法都有一个共同的参数`size`。

   - 如果`size`为`None`，则只产生一个随机数
   - 如果`size`为一个整数，则产生一个一维的随机数数组。
   - 如果`size`为一个元组，则生成一个多维的随机数数组。其中数组的形状由元组指定。

2. 生成随机数的方法

   - `.bytes(length)`：等效于`numpy.random.bytes(...)`函数
   - `.choice(a[, size, replace, p])`：等效于`numpy.random.choice(...)`函数
   - `.rand(d0, d1, ..., dn)`：等效于`numpy.random.rand(...)`函数
   - `.randint(low[, high, size])`：等效于`numpy.random.randint(...)`函数
   - `.randn(d0, d1, ..., dn)` ：等效于`numpy.random.randn(...)`函数
   - `.random_integers(low[, high, size])`：等效于`numpy.random_integers.bytes(...)`函数
   - `.random_sample([size])`：等效于`numpy.random.random_sample(...)`函数
   - `.tomaxint([size])`：等效于`numpy.random.tomaxint(...)`函数

3. 排列组合的方法

   - `.shuffle(x)`：等效于`numpy.random.shuffle(...)`函数
   - `.permutation(x)` ：等效于`numpy.random.permutation(...)`函数

4. 指定概率分布函数的方法

   - `.beta(a, b[, size])`：等效于`numpy.random.beta(...)`函数
   - `.binomial(n, p[, size])`：等效于`numpy.random.binomial(...)`函数
   - `.chisquare(df[, size])`：等效于`numpy.random.chisquare(...)`函数
   - `.dirichlet(alpha[, size])`：等效于`numpy.random.dirichlet(...)`函数
   - `.exponential([scale, size])`：等效于`numpy.random.exponential(...)`函数
   - `.f(dfnum, dfden[, size])`：等效于`numpy.random.f(...)`函数
   - `.gamma(shape[, scale, size])`：等效于`numpy.random.gamma(...)`函数
   - `.geometric(p[, size])`：等效于`numpy.random.geometric(...)`函数
   - `.gumbel([loc, scale, size])`：等效于`numpy.random.gumbel(...)`函数
   - `.hypergeometric(ngood, nbad, nsample[, size])`：等效于`numpy.random.hypergeometric(...)`函数
   - `.laplace([loc, scale, size])`：等效于`numpy.random.laplace(...)`函数
   - `.logistic([loc, scale, size])`：等效于`numpy.random.logistic(...)`函数
   - `.lognormal([mean, sigma, size])`：等效于`numpy.random.lognormal(...)`函数
   - `.logseries(p[, size])`：等效于`numpy.random.logseries(...)`函数
   - `.multinomial(n, pvals[, size])`：等效于`numpy.random.multinomial(...)`函数
   - `.multivariate_normal(mean, cov[, size])`：等效于`numpy.random.multivariate_normal(...)`函数
   - `.negative_binomial(n, p[, size])`：等效于`numpy.random.negative_binomial(...)`函数
   - `.noncentral_chisquare(df, nonc[, size])`：等效于`numpy.random.noncentral_chisquare(...)`函数
   - `.noncentral_f(dfnum, dfden, nonc[, size])`：等效于`numpy.random.noncentral_f(...)`函数
   - `.normal([loc, scale, size])`：等效于`numpy.random.normal(...)`函数
   - `.pareto(a[, size])`：等效于`numpy.random.pareto(...)`函数 -`. poisson([lam, size])`：等效于`numpy.random.poisson(...)`函数
   - `.power(a[, size])`：等效于`numpy.random.power(...)`函数
   - `.rayleigh([scale, size])`：等效于`numpy.random.rayleigh(...)`函数
   - `.standard_cauchy([size])`：等效于`numpy.random.standard_cauchy(...)`函数
   - `.standard_exponential([size])`：等效于`numpy.random.standard_exponential(...)`函数
   - `.standard_gamma(shape[, size])`：等效于`numpy.random.standard_gamma(...)`函数
   - `.standard_normal([size])`：等效于`numpy.random.standard_normal(...)`函数
   - `.standard_t(df[, size])`：等效于`numpy.random.standard_t(...)`函数
   - `.triangular(left, mode, right[, size])`：等效于`numpy.random.triangular(...)`函数
   - `.uniform([low, high, size])`：等效于`numpy.random.uniform(...)`函数
   - `.vonmises(mu, kappa[, size])`：等效于`numpy.random.vonmises(...)`函数
   - `.wald(mean, scale[, size])`：等效于`numpy.random.wald(...)`函数
   - `.weibull(a[, size])`：等效于`numpy.random.weibull(...)`函数
   - `.zipf(a[, size])`：等效于`numpy.random.zipf(...)`函数

5. 类式的其他函数

   - `seed(seed=None)`：该方法在`RandomState`被初始化时自动调用，你也可以反复调用它从而重新设置伪随机数生成器的种子。

   - ```
     get_state()
     ```

     ：该方法返回伪随机数生成器的内部状态。其结果是一个元组

     ```
     (str, ndarray of 624 uints, int, int, float)
     ```

     ，依次为：

     - 字符串`'MT19937'`
     - 一维数组，其中是624个无符号整数`key`
     - 一个整数`pos`
     - 一个整数`has_gauss`
     - 一个浮点数`cached_gaussian`

   - ```
     set_state(state)
     ```

     ：该方法设置伪随机数生成器的内部状态,如果执行成功则返回

     ```
     None
     ```

     。参数是个元组

     ```
     (str, ndarray of 624 uints, int, int, float)
     ```

     ，依次为：

     - 字符串`'MT19937'`
     - 一维数组，其中是624个无符号整数`key`
     - 一个整数`pos`
     - 一个整数`has_gauss`
     - 一个浮点数`cached_gaussian`

### 2. 统计量

1. 这里是共同的参数：

   - `a`：一个`array_like`对象

   - ```
     axis
     ```

     ：可以为为

     ```
     int
     ```

     或者

     ```
     tuple
     ```

     或者

     ```
     None
     ```

     ：

     - `None`：将`a`展平，在整个数组上操作
     - `int`：在`a`的指定轴线上操作。如果为`-1`，表示沿着最后一个轴（0轴为第一个轴）。
     - `tuple of ints`：在`a`的一组指定轴线上操作

   - `out`：可选的输出位置。必须与期望的结果形状相同

   - `keepdims`：如果为`True`，则结果数组的维度与原数组相同，从而可以与原数组进行广播运算。

2. 顺序统计：

   - `numpy.minimum(x1, x2[, out])`：返回两个数组`x1`和`x2`对应位置的最小值。要求`x1`和`x2`形状相同或者广播之后形状相同。
   - `numpy.maximum(x1, x2[, out])`：返回两个数组`x1`和`x2`对应位置的最大值。要求`x1`和`x2`形状相同或者广播之后形状相同。
   - `numpy.amin(a[, axis, out, keepdims])` ：返回`a`中指定轴线上的最小值（数组）、或者返回`a`上的最小值（标量）。
   - `numpy.amax(a[, axis, out, keepdims])` ：返回`a`中指定轴线上的最大值（数组）、或者返回`a`上的最小值（标量）。
   - `numpy.nanmin(a[, axis, out, keepdims])`: 返回`a`中指定轴线上的最小值（数组）、或者返回`a`上的最小值（标量），忽略`NaN`。
   - `numpy.nanmax(a[, axis, out, keepdims])` :返回`a`中指定轴线上的最大值（数组）、或者返回`a`上的最小值（标量）忽略`NaN`。
   - `numpy.ptp(a[, axis, out])` ：返回`a`中指定轴线上的`最大值减去最小值`（数组），即`peak to peak`
   - `numpy.argmin(a, axis=None, out=None)`：返回`a`中指定轴线上最小值的下标
   - `numpy.argmax(a, axis=None, out=None)`：返回`a`中指定轴线上最大值的下标
   - `numpy.percentile(a, q[, axis, out, ...])` ：返回`a`中指定轴线上`qth 百分比`数据。`q=50`表示 50% 分位。你可以用列表或者数组的形式一次指定多个 `q`。
   - `numpy.nanpercentile(a, q[, axis, out, ...])`：返回`a`中指定轴线上`qth 百分比`数据。`q=50`表示 50% 分位。
   - `numpy.partition(a, kth, axis=-1, kind='introselect', order=None)`：它将数组执行划分操作：第$k$位左侧的数都小于第$k$；第$k$位右侧的数都大于等于第$k$。它返回划分之后的数组
   - `numpy.argpartition(a, kth, axis=-1, kind='introselect', order=None)`：返回执行划分之后的下标（对应于数组划分之前的位置）。 <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/x6Wgv9wUiwLz.JPG">
   </p>
   <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/8Dtu72CwfKIg.JPG">
   </p>
   

3. 排序：

   - ```
     numpy.sort(a, axis=-1, kind='quicksort', order=None)
     ```

     ：返回

     ```
     a
     ```

     在指定轴上排序后的结果（并不修改原数组）。

     - `kind`：字符串指定排序算法。可以为`'quicksort'`(快速排序)，`'mergesort'`(归并排序)，`'heapsort'`(堆排序)
     - `order`：在结构化数组中排序中，用于设置排序的字段（一个字符串）

   - `numpy.argsort(a, axis=-1, kind='quicksort', order=None)`：返回`a`在指定轴上排序之后的下标（对应于数组划分之前的位置）。

   - ```
     numpy.lexsort(keys, axis=-1)
     ```

     ：

     - 如果`keys`为数组，则根据数组的最后一个轴的最后一排数值排列，并返回这些轴的排列顺序。如数组`a`的`shape=(4,5)`，则根据`a`最后一行（对应于最后一个轴的最后一排）的5列元素排列。这里`axis`指定排序的轴 。对于`argsort`，会在最后一个轴的每一排进行排列并返回一个与`a`形状相同的数组。
     - 如果`keys`为一维数组的元组，则将这些一维数组当作行向量拼接成二维数组并按照数组来操作。

   - ```
     numpy.searchsorted(a, v, side='left', sorter=None)
     ```

     ：要求

     ```
     a
     ```

     是个已排序好的一维数组。本函数将

     ```
     v
     ```

     插入到

      

     ```
     a
     ```

     中，从而使得数组

     ```
     a
     ```

     维持一个排序好的数组。函数返回的是

     ```
     v
     ```

     应该插入的位置。

     ```
     side
     ```

     指定若发现数值相等时，插入左侧

     ```
     left
     ```

     还是右侧

     ```
     right
     ```

     - 如果你想一次插入多个数值，可以将`v`设置为列表或者数组。
     - 如果`sorter=None`，则要求`a`已排序好。如果`a`未排序，则要求传入一个一维数组或者列表。这个一维数组或者列表给出了 `a`的升序排列的下标。（通常他就是`argsort`的结果）
     - 它并不执行插入操作，只是返回待插入的位置

   <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/79Vbhw2Vm9BD.JPG">
   </p>
   <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/SeJvWV8KdvBA.JPG">
   </p>
   

4. 均值和方差：

   - `numpy.sum(a, axis=None, dtype=None, out=None, keepdims=False)`：计算`a`在指定轴上的和

   - `numpy.prod(a, axis=None, dtype=None, out=None, keepdims=False)`：计算`a`在指定轴上的乘积

   - `numpy.median(a[, axis, out, overwrite_input, keepdims])`:计算`a`在指定轴上的中位数（如果有两个，则取这两个的平均值）

   - `numpy.average(a[, axis, weights, returned])`:计算`a`在指定轴上的加权平均数

   - `numpy.mean(a[, axis, dtype, out, keepdims])` :计算`a`在指定轴上的算术均值

   - `numpy.std(a[, axis, dtype, out, ddof, keepdims])`:计算`a`在指定轴上的标准差

   - `numpy.var(a[, axis, dtype, out, ddof, keepdims])` :计算`a`在指定轴上的方差。方差有两种定义：

     - 偏样本方差`biased sample variance`。计算公式为 （$\bar x$为均值）：

      $$ var=\frac 1N\sum_{i=1}^{N}(x_i-\bar x)^{2}$$

     - 无偏样本方差`unbiased sample variance`。计算公式为 （$\bar x$为均值）：

      $$ var=\frac 1{N-1}\sum_{i=1}^{N}(x_i-\bar x)^{2}$$

       当`ddof=0`时，计算偏样本方差；当`ddof=1`时，计算无偏样本方差。默认值为 0。当`ddof`为其他整数时，分母就是`N-ddof`。

   - `numpy.nanmedian(a[, axis, out, overwrite_input, ...])` :计算`a`在指定轴上的中位数，忽略`NaN`

   - `numpy.nanmean(a[, axis, dtype, out, keepdims])` :计算`a`在指定轴上的算术均值，忽略`NaN`

   - `numpy.nanstd(a[, axis, dtype, out, ddof, keepdims])`:计算`a`在指定轴上的标准差，忽略`NaN`

   - `numpy.nanvar(a[, axis, dtype, out, ddof, keepdims])`:计算`a`在指定轴上的方差，忽略`NaN`

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/XVYP1nRrv7nA.JPG">
  </p>
  

5. 相关系数：

   - `numpy.corrcoef(x[, y, rowvar, bias, ddof])` : 返回皮尔逊积差相关
   - `numpy.correlate(a, v[, mode])` ：返回两个一维数组的互相关系数
   - `numpy.cov(m[, y, rowvar, bias, ddof, fweights, ...])`：返回协方差矩阵

6. 直方图：

   - ```
     numpy.unique(ar, return_index=False, return_inverse=False, return_counts=False)
     ```

     ：返回

     ```
     ar
     ```

     中所有不同的值组成的一维数组。如果

     ```
     ar
     ```

     不是一维的，则展平为一维。

     - `return_index`：如果为`True`，则同时返回这些独一无二的数值在原始数组中的下标
     - `return_inverse`：如果为`True`，则返回元素数组的值在新返回数组中的下标（从而可以重建元素数组）
     - `return_counts`：如果为`True`，则返回每个独一无二的值在原始数组中出现的次数

   - ```
     numpy.histogram(a, bins=10, range=None, normed=False, weights=None, density=None)
     ```

     :计算一组数据的直方图。如果

     ```
     a
     ```

     不是一维的，则展平为一维。

     ```
     bins
     ```

     指定了统计的区间个数（即统计范围的等分数）。

     ```
     range
     ```

     是个长度为2的元组，表示统计范围的最小值和最大值（默认时，表示范围为数据的最小值和最大值）。当

     ```
     density
     ```

     为

     ```
     False
     ```

     时，返回

     ```
     a
     ```

     中数据在每个区间的个数；否则返回

     ```
     a
     ```

     中数据在每个区间的频率。

     ```
     weights
     ```

     设置了

     ```
     a
     ```

     中每个元素的权重，如果设置了该参数，则计数时考虑权重。它返回的是一个元组，第一个元素给出了每个直方图的计数值，第二个元素给出了直方图的统计区间的从左到右的各个闭合点 （比计数值的数量多一个）。

     - `normed`：作用与`density`相同。该参数将被废弃
     - `bins`也可以为下列字符串，此时统计区间的个数将通过计算自动得出。可选的字符串有：`'auto'`、`'fd'`、`'doane'`、`'scott'`、`'rice'`、`'sturges'`、`'sqrt'`

   - `numpy.histogram2d(x, y, bins=10, range=None, normed=False, weights=None)`：计算两组数据的二维直方图

   - `numpy.histogramdd(sample, bins=10, range=None, normed=False, weights=None)`：计算多维数据的直方图

   - `numpy.bincount(x[, weights, minlength])`：计算每个数出现的次数。它要求数组中所有元素都是非负的。其返回数组中第`i`个元素表示：整数`i`在`x`中出现的次数。要求`x`必须一维数组，否则报错。`weights`设置了`x`中每个元素的权重，如果设置了该参数，则计数时考虑权重。`minlength`指定结果的一维数组最少多长（如果未指定，则由`x`中最大的数决定）。

   - `numpy.digitize(x, bins, right=False)` ：离散化。如果`x`不是一维的，则展平为一维。它返回一个数组，该数组中元素值给出了`x`中的每个元素将对应于统计区间的哪个区间。区间由`bins`这个一维数组指定，它依次给出了统计区间的从左到右的各个闭合点。`right`为`True`，则表示统计区间为左开右闭合`(]`；为`False`，则表示统计区间为左闭合右开`[)` <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/pNjFNc64AAgw.JPG">
   </p>
   

   > 注意：`matplotlib.pyplot`也有一个建立直方图的函数（`hist(...)`），区别在于`matplotlib.pyplot.hist(...)`函数会自动绘直方图，而`numpy.histogram`仅仅产生数据

   <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/710Paq8yE0We.JPG">
   </p>
   

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/JmKpu52e0Nzn.JPG">
  </p>
  

### 3. 分段函数

1. `numpy.where(condition[, x, y])`：它类似于`python`的 `x if condition else y`。`condition/x/y`都是数组，要求形状相同或者通过广播之后形状相同。产生结果的方式为： 如果`condition`某个元素为`True`或者非零，则对应的结果元素从`x`中获取；否则对应的结果元素从`y`中获取 <p align="center">
  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/L1jApfAjtT9W.JPG">
</p>


2. 如果分段数量增加，则需要嵌套多层的

   ```
   where()
   ```

   。此时可以使用

   ```
   select()
   ```

   ：

   ```
   numpy.select(condlist, choicelist, default=0)
   ```

   。

   - 其中`condlist`为长度为 `N`的列表，列表元素为数组，给出了条件数组
   - `choicelist`为长度为`N`的列表，列表元素为数组，给出了结果被选中的候选值。
   - 所有数组的长度都形状相同，如果形状不同，则执行广播。结果数组的形状为广播之后的形状。
   - 结果筛选规则如下：
     - 从`condlist`左到右扫描，若发现第 `i` 个元素（是个数组）对应位置为`True`或者非零，则输出元素来自`choicelist` 的第 `i` 个元素（是个数组）。因此若有多个`condlist`的元素满足，则只会使用第一个遇到的。
     - 如果扫描结果是都不满足，则使用`default` <p align="center">
       <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/CVGx47tfeMGN.JPG">
     </p>
     

3. 采用

   ```
   where/select
   ```

   时，所有的参数需要在调用它们之前完成。在计算时还会产生许多保存中间结果的数组。因此如果输入数组很大，则将会发生大量内存分配和释放。 为此

   ```
   numpy
   ```

   提供了

   ```
   piecewise
   ```

   函数：

   ```
   numpy.piecewise(x, condlist, funclist, *args, **kw)
   ```

   。

   - `x`：为分段函数的自变量取值数组

   - `condlist`：为一个列表，列表元素为布尔数组，数组形状和`x`相同。

   - ```
     funclist
     ```

     ：为一个列表，列表元素为函数对象。其长度与

     ```
     condlist
     ```

     相同或者比它长1。

     - 当`condlist[i]`对应位置为 `True`时，则该位置处的输出值由 `funclist[i]`来计算。如果`funclist`长度比`condlist`长1，则当所有的`condlist`都是`False`时，则使用 `funclist[len(condlist)]`来计算。如果有多个符合条件，则使用最后一个遇到的（而不是第一个遇到的）
     - 列表元素可以为数值，表示一个返回为常数值（就是该数值）的函数。

   - `args/kw`：用于传递给函数对象`funclist[i]`的额外参数。 <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/TnKztMVGBooH.JPG">
   </p>
   

### 4. 多项式

1. 一元多项式类的构造：（注意系数按照次数从高次到低次排列）

   ```
   xxxxxxxxxx
   ```

   ```
   class numpy.poly1d(c_or_r, r=0, variable=None)
   ```

   - `c_or_r`：一个数组或者序列。其意义取决于`r`
   - `r`：布尔值。如果为`True`，则`c_or_r`指定的是多项式的根；如果为`False`，则`c_or_r`指定的是多项式的系数
   - `variable`：一个字符串，指定了打印多项式时，用什么字符代表自变量。默认为`x`

   多项式的属性有：

   - `.coeffs`属性：多项式的系数
   - `.order`属性：多项式最高次的次数
   - `.variable`属性：自变量的代表字符

   多项式的方法有：

   - `.deriv(m=1)`方法：计算多项式的微分。可以通过参数`m`指定微分次数
   - `.integ(m=1,k=0)`方法：计算多项式的积分。可以通过参数`m`指定积分次数和`k`积分常量

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/xQI6vsSRrGyi.JPG">
  </p>
  

2. 操作一元多项式类的函数：

   - 多项式对象可以像函数一样，返回多项式的值
   - 多项式对象进行加减乘除，相当于对应的多项式进行计算。也可以使用对应的`numpy.polyadd/polysub/polymul/polydiv/`函数。
   - `numpy.polyder/numpy.polyint`：进行微分/积分操作
   - `numpy.roots`函数：求多项式的根（也可以通过`p.r`方法）

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/YFdMw7b4SlPd.JPG">
  </p>
  

3. 使用`np.polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False)`函数可以对一组数据使用多项式函数进行拟合（最小均方误差）。其参数为：

   - `x`：数据点的`x`坐标序列
   - `y`：数据点的`y`坐标序列。如果某个`x`坐标由两个点，你可以传入一个二维数组。
   - `deg`：拟合多项式的次数
   - `rcond`：指定了求解过程中的条件：当`某个特征值/最大特征值时，该特征值被抛弃
   - `full`：如果为`False`，则仅仅返回拟合多项式的系数；如果为`True`，则更多的结果被返回
   - `w`：权重序列。它对`y`序列的每个位置赋予一个权重
   - `cov`：如果为`True`，则返回相关矩阵。如果`full`为`True`，则不返回。

   默认情况下，返回两个数组：一个是拟合多项式的系数；另一个是数据的相关矩阵

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/IXl5SNBUbRw5.JPG">
  </p>
  

4. `numpy`提供了更丰富的多项式函数类。注意其中的多项式的系数按照次数从小到大排列。

   - `numpy.polynomial.Polynomial`：一元多次多项式
   - `numpy.polynomial.Chebyshev`：切比雪夫多项式
   - `numpy.polynomial.Laguerre`：拉盖尔多项式
   - `numpy.polynomial.Legendre`：勒让德多项式
   - `numpy.polynomial.Hermite`：哈米特多项式
   - `numpy.polynomial.HermiteE`：`HermiteE`多项式

   所有的这些多项式的构造函数为：`XXX(coef, domain=None, window=None)`。其中`XXX`为多项式类名。`domain`为自变量取值范围，默认为`[-1,1]`。`window`指定了将`domain`映射到的范围，默认为`[-1,1]`。

   > 如切比雪夫多项式在`[-1,1]`上为正交多项式。因此只有在该区间上才能正确插值拟合多项式。为了使得对任何区域的目标函数进行插值拟合，所以在`domain`指定拟合的目标区间。

   所有的这些多项式可以使用的方法为：

   - 四则运行
   - `.basis(deg[, domain, window])`：获取转换后的一元多项式
   - `.convert(domain=None, kind=None, window=None)`：转换为另一个格式的多项式。`kind`为目标格式的多项式的类
   - `.degree()`：返回次数
   - `.fit(x, y, deg[, domain, rcond, full, w, window])`：拟合数据，返回拟合后的多项式
   - `.fromroots(roots[, domain, window])`：从根创建多项式
   - `.has_samecoef(other)`、`.has_samedomain(other)`、`.has_sametype(other)`、`.has_samewindow(other)`：判断是否有相同的系数/`domain`/类型/`window`
   - `.roots()`：返回多项式的根
   - `.trim([tol])`：将系数小于 `tol`的项截掉
   - 函数调用的方式

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/eDgj21a8lVl7.JPG">
  </p>
  

5. 切比雪夫多项式可以降低龙格现象。所谓龙格现象：等距离差值多项式在两个端点处有非常大的震荡，`n`越大，震荡越大。

### 5. 内积、外积、张量积

1. - `numpy.dot(a, b, out=None)`：计算矩阵的乘积。对于一维数组，他计算的是内积；对于二维数组，他计算的是线性代数中的矩阵乘法。

   - `numpy.vdot(a, b)`：返回一维向量之间的点积。如果`a`和`b`是多维数组，则展平成一维再点积。

    <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/01jWVz7wQ6Au.JPG">
    </p>
    

   - `numpy.inner(a, b)`：计算矩阵的内积。对于一维数组，它计算的是向量点积；对于多维数组，则它计算的是：每个数组最后轴作为向量，由此产生的内积。

   - `numpy.outer(a, b, out=None)`：计算矩阵的外积。它始终接收一维数组。如果是多维数组，则展平成一维数组。

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/GNa3vsGSoMwc.JPG">
  </p>
  

2. `numpy.tensordot(a, b, axes=2)`：计算张量乘积。

   - `axes`如果是个二元序列，则第一个元素表示`a`中的轴；第二个元素表示`b`中的轴。将这两个轴上元素相乘之后求和。其他轴不变
   - `axes`如果是个整数，则表示把`a`中的后`axes`个轴和`b`中的前`axes`个轴进行乘积之后求和。其他轴不变 <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/Cv8geVvSAjf1.JPG">
   </p>
   

3. 叉乘：`numpy.cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None)`:计算两个向量之间的叉乘。叉积用于判断两个三维空间的向量是否垂直。要求`a`和`b`都是二维向量或者三维向量，否则抛出异常。（当然他们也可以是二维向量的数组，或者三维向量的数组，此时一一叉乘）

<p align="center">
  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200602/W12OHwlfjGos.JPG">
</p>


### 6. 线性代数

1. 逆矩阵：`numpy.linalg.inv(a)`：获取`a`的逆矩阵（一个`array-like`对象）。

   - 如果传入的是多个矩阵，则依次计算这些矩阵的逆矩阵。

   - 如果`a`不是方阵，或者`a`不可逆则抛出异常

    <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/rqGgnRxuk3VK.JPG">
    </p>
    

2. 单位矩阵:`numpy.eye(N[, M, k, dtype])`：返回一个二维单位矩阵行为`N`，列为`M`，对角线元素为1，其余元素为0。`M`默认等于`N`。`k`默认为0表示对角线元素为1（单位矩阵），如为正数则表示对角线上方一格的元素为1（上单位矩阵），如为负数表示对角线下方一格的元素为1（下单位矩阵）

3. 对角线和：`numpy.trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None)`：返回对角线的和。

   - 如果`a`是二维的，则直接选取对角线的元素之和（`offsert=0`），或者对角线右侧偏移`offset`的元素之和（即选取`a[i,i+offset]`之和）

- 如果`a`不止二维，则由`axis1`和`axis2`指定的轴选取了取对角线的矩阵。

- 如果`a`少于二维，则抛出异常

 <p align="center">
   <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/LMENkRU89N1S.JPG">
 </p>
 

1. 计算线性方程的解$Ax=b$：`numpy.linalg.solve(a,b)`：计算线性方程的解`ax=b`，其中`a`为矩阵，要求为秩不为0的方阵，`b`为列向量（长度等于方阵大小）；或者`a`为标量，`b`也为标量。

   - 如果`a`不是方阵或者`a`是方阵但是行列式为0，则抛出异常

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/Qu4Hw2TqcETM.JPG">
  </p>
  

2. 特征值：`numpy.linalg.eig(a)`：计算矩阵的特征值和右特征向量。如果不是方阵则抛出异常，如果行列式为0则抛出异常。

   <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200602/GyBwfikbd1wG.JPG">
   </p>
   

3. 奇异值分解：`numpy.linalg.svd(a, full_matrices=1, compute_uv=1)`：对矩阵`a`进行奇异值分解，将它分解成`u*np.diag(s)*v`的形式，其中`u`和`v`是酉矩阵，`s`是`a`的奇异值组成的一维数组。 其中：

   - `full_matrics`:如果为`True`，则`u`形状为`(M,M)`，`v`形状为`(N,N)`；否则`u`形状为`(M,K)`，`v`形状为`(K,N)`，`K=min(M,N)`
   - `compute_uv`：如果为`True`则表示要计算`u`和`v`。默认为`True`。
   - 返回`u`、`s`、`v`的元组
   - 如果不可分解则抛出异常

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/JsAVj3lYGfLh.JPG">
  </p>
  

## 四、数组的存储和加载

### 1. 二进制

1. `numpy.save(file, arr, allow_pickle=True, fix_imports=True)`：将数组以二进制的形式存储到文件中。

   - `file`：文件名或者文件对象。如果是个文件名，则会自动添加后缀`.npy`如果没有该后缀的话
   - `arr`：被存储的数组
   - `allow_pickle`：一个布尔值，如果为`True`，则使用`Python pickle`。有时候为了安全性和可移植性而不使用`pickle`
   - `fix_imports`：用于`python3`的数组`import`到`python2`的情形

2. `numpy.savez(file, *args, **kwds)`：将多个数组以二进制的形式存储到文件中。

   - `file`：文件名或者文件对象。如果是个文件名，则会自动添加后缀`.npz`如果没有该后缀的话

   - `args`：被存储的数组。这些数组的名字将被自动命名为`arr_0/arr_1/...`

     > 如果没有名字，则完全无法知晓这些数组的区别

   - `kwds`：将被存储的数组，这些关键字参数就是键的名字

3. `numpy.load(file, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')`：将二进制文件中读取数组。

   - `file`：一个文件名或者文件对象。它存放着数组
   - `mmap_mode`：如果不是`None`，则`memory-map`该文件。此时对数组的修改会同步到文件上。当读取大文件的一小部分时很有用，因为它不必一次读取整个文件。可选值为`None/'r+'/'r'/'w+'/'c'`
   - `allow_pickle`：一个布尔值，如果为`True`，则使用`Python pickle`。有时候为了安全性和可移植性而不使用`pickle`
   - `fix_imports`：用于`python3`的数组`import`到`python2`的情形
   - `encoding`：只用于`python2`，读取`python2`字符串。

   该函数返回一个数组，元组，或者字典（当二进制文件时`savez`生成时）

### 2. 文本文件

1. `numpy.genfromtxt(fname, dtype=, comments='#', delimiter=None,``skip_header=0, skip_footer=0, converters=None, missing_values=None,``filling_values=None, usecols=None, names=None, excludelist=None,``deletechars=None, replace_space='_', autostrip=False, case_sensitive=True,``defaultfmt='f%i', unpack=None, usemask=False, loose=True,``invalid_raise=True, max_rows=None)` ：从文本文件中加载数组，通用性很强，可以处理缺失数据的情况。

   > `loadtxt()`函数只能处理数据无缺失的情况。

   - `fname`：指定的数据源。可以为：

     - 文件名字符串。如果后缀为`gz`或者`bz2`，则首先自动解压缩
     - 文件对象/字符串列表/其他可迭代对象：这些可迭代对象必须返回字符串（该字符串被视为一行）

   - `dtype`：数组的元素类型，可以提供一个序列，指定每列的数据类型

   - `comments`：一个字符串，其中每个字符都指定了注释行的第一个字符。注释行整体被放弃

   - `delimiter`：指定了分隔符。可以为：

     - 字符串：指定分隔符。默认情况下，所有连续的空白符被认为是分隔符
     - 一个整数：指定了每个字段的宽度
     - 一个整数序列：依次给出了各个字段的宽度

   - `skiprows`：被废弃，推荐使用`skip_header`

   - `skip_header`：一个整数，指定跳过文件头部多少行

   - `skip_footer`：一个整数，指定跳过文件尾部多少行

   - `converters`：用于列数据的格式转换。你可以指定一个字典，字典的键就是列号：

     ```
     xxxxxxxxxx
     ```

     ```
     converters={0: lambda s: float(s or 0),
     ```

     ```
     1: lambda s: int(s or 199),...
     ```

     ```
     }
     ```

   - `missing`：被废弃，推荐使用`missing_values`

   - `missing_values`:指定缺失数据。你可以自定一个字典，字典的键就是缺失位置的字符串，值就是缺失值。比如你可以指定`NNNN`为缺失数据，此时遇到`NNNN`时，`numpy`解析为`np.nan`

   - `filling_values`：指定缺失值的填充值。即解析到`np.nan`时，用什么值代替它

   - `usecols`：一个序列，指定了要读取那些列（列从0 计数）

   - `names`：

     - 如果为`True`，则在`skip_header`行之后第一行被视作标题行，将从该行读取每个字段的`name`。
     - 如果为序列或者一个以冒号分隔的字符串，则使用它作为各个字段的`name`
     - 如果为`None`,则每个`dtype`字段的名字被使用

   - `excludelist`：一个序列，给出了需要排除的字段的`name`。

   - `deletechars`：A string combining invalid characters that must be deleted from the names

   - `defaultfmt`：A format used to define default field names, such as “f%i” or “f_%02i”.

   - `autostrip`：一个布尔值。如果为`True`，则自动移除数据中的空白符

   - `replace_space`：一个字符。如果变量名中有空白符，如`user name`，则使用该字符来替代空白符。默认为`_`，即变量名转换为`user_name`

   - `case_sensitive`：一个布尔值或者字符串。如果为`True`，则字段名是大小写敏感的。如果为`False`或者`'upper'`，则字段名转换为大写字符。如果为`'lower'`则转换为小写字符。

   - `unpack`:If True, the returned array is transposed

   - `usemask`:If True, return a masked array

   - `loose`:If True, do not raise errors for invalid values

   - `invalid_raise`:If True, an exception is raised if an inconsistency is detected in the number of columns. If False, a warning is emitted and the offending lines are skipped

   - `max_rows`:一个整数，指定读取的最大行数。

2. `numpy.loadtxt(fname, dtype=, comments='#', delimiter=None,``converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0)`：它作用与`genfromtxt`相同，但是它只能用于规则比较简单的文件，并且它的解析速度更快。

   - `ndim`：一个整数。指定结果数组必须拥有不少于`ndim`维度。
   - 其他参数参考`genfromtxt`

3. `numpy.fromstring(string, dtype=float, count=-1, sep='')`：从`raw binary`或者字符串中创建一维数组。

   - `string`：一个字符串，给出数据源
   - `dtype`：指定数据类型
   - `count`：一个整数。从数据源（一个字符串）中读取指定数量的数值类型的数值。如果为负数，则为数据长度加上这个负值
   - `sep`：如果未提供或者为空字符串，则`string`被认为是二进制数据。如果提供了一个非空字符串，则给出了分隔符。

4. `numpy.savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='',``footer='', comments='# ')`：存储到文本文件

   - `fname`：给出了文件名或者`file`对象。如果为`.gz`后缀，则自动压缩
   - `X`：被存储的数组
   - `fmt`：一个字符串或者一个字符串序列，指定存储格式。一个字符串可以指定所有的格式化方式；一个字符串序列可以对每列指定一个格式化方式。如果是虚数，你可以通过`%.4e%+.4j`的方式指定实部和虚部。
   - `delimiter`：一个字符串，用于分隔符，分隔每个列
   - `newline`：一个字符串，指定换行符
   - `header`：一个字符串。它会写到文件的首行
   - `footer`：一个字符串。它会写到文件的末尾
   - `comments`：一个字符串。它会写到文件的中间，并且用注释符作为行首，如`#`

   注：`fmt`分隔字符串的格式为`%[flag]width[.precision]specifier`。其中：

   - `flags`：可以为`'-'`（左对齐）、`'+'`（右对齐）、`'0'`（左侧填充0）

   - `width`：最小的位宽。

   - ```
     precision
     ```

     ：

     - 对于`specifier=d/i/o/x`，指定最少的数字个数
     - 对于`specifier=e/E/f`，指定小数点后多少位
     - 对于`specifier=g/G`，指定最大的`significant digits`
     - 对于`specifier=s`，指定最大的字符数量

   - `specifier`：指定格式化类型。`c`（字符）、`d/i`（带符号整数）、`e/E`（科学计数法）、`f`（浮点数）、`g/G`（使用`shorter e/E/f`）、`o`（带符号八进制）、`s`（字符串）、`u`（无符号整数）、`x/X`（无符号十六进制）

5. `ndarray.tofile(fid, sep="", format="%s")`：保存到文件中。

   - `fid`：一个`file`对象或者文件名
   - `sep`：一个字符串，指定分隔符。如果为空或者空字符串，则按照二进制的方式写入，等价于`file.write(a.tobytes())`
   - `format`：一个字符串，指定了数值的格式化方式