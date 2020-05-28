# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # TensorFlow2教程-张量
# 
# 由命名可知，TensorFlow是一个定义和运算涉及张量的计算的框架。TensorFlow将张量表示为基本数据类型的n维数组。
# %% [markdown]
# tf.Tensor具有以下属性：
# - 数据类型（float32, int32, string等）
# - 形状
# 
# 一个张量中的每个元素都是一样的数据类型，且数据类型已知。形状可能已知也可能部分已知。某些情况下，只有执行图时才知道张量的形状。
# 
# 一些特殊的张量：
# - tf.Variable
# - tf.constant
# - tf.placeholder
# - tf.SparseTensor
# 
# 
# %% [markdown]
# ## 1 Rank
# rank是tf.Tensor对象的尺寸的数量。类似于order,degree或ｎ维。TensorFlow中的rank和数据中矩阵的秩不同。如下表所示，TensorFlow每个rank对应一个不同的数学实体。
# 
# | 秩	| 数学实体 |
# | ---- | ---- |
# | 0	| 标量（仅幅度） |
# | 1	| 向量（大小和方向） |
# | 2	| 矩阵（数字表） |
# | 3	| 3维张量（数字的立方） |
# | ñ	| n维张量 |
# %% [markdown]
# ### 1.1 rank 0
# rank0张量创建

# %%
import tensorflow as tf
mammal = tf.Variable('Hongmeng', tf.string)
ignition = tf.Variable(360, tf.int16)
floating = tf.Variable(3.141592653, tf.float64)
its_complicated = tf.Variable(12.3 - 4.56j, tf.complex64)

# %% [markdown]
# 注：在TensorFlow中，字符串被视为单个对象，而不是字符序列。可能有标量字符串，字符串向量等。
# %% [markdown]
# ### 1.2 rank 1
# 要创建rank 1的tf.Tensor对象，一般传递列表作为初始值

# %%
mystr = tf.Variable(["Hello"], tf.string)
cool_numbers  = tf.Variable([3.14159, 2.71828], tf.float32)
first_primes = tf.Variable([2, 3, 5, 7, 11], tf.int32)
its_very_complicated = tf.Variable([12.3 - 4.85j, 7.5 - 6.23j], tf.complex64)

# %% [markdown]
# ### 1.3 rank 2
# rank 2包含行跟列
# 

# %%
mymat = tf.Variable([[7],[11]], tf.int16)
myxor = tf.Variable([[False, True],[True, False]], tf.bool)
linear_squares = tf.Variable([[4], [9], [16], [25]], tf.int32)
squarish_squares = tf.Variable([ [4, 9], [16, 25] ], tf.int32)
rank_of_squares = tf.rank(squarish_squares)
mymatC = tf.Variable([[7],[11]], tf.int32)

# %% [markdown]
# ### 1.4 高阶张量
# 高阶张量由n维数组组成。例如，在图像处理期间，使用了许多等级4的张量，其尺寸对应于批中示例，图像高度，图像宽度和颜色通道。

# %%
my_image = tf.zeros([10, 299, 299, 3]) 

# %% [markdown]
# ### 1.5 获取tf.Tensor对象的rank
# 要确定tf.Tensor对象的rank，可以调用tf.rank方法。例如，以下方法以编程方式确定

# %%
r = tf.rank(my_image)
print(r)

# %% [markdown]
# ### 1.6 tf.Tensor切片
# 
# 由于 tf.Tensor是n维数组，因此要访问其中的某个元素，需要指定n个索引。
# 
# 对于等级0的张量（标量），不需要索引，因为它已经是一个数字。
# 
# 对于秩为1的张量（向量），传递单个索引可访问数字：
# 
# 

# %%
my_scalar = first_primes[2]

# %% [markdown]
# 对于2级或更高的张量，情况更加有趣。对于 tf.Tensor的rank为2的，传递两个数字将按预期返回标量：

# %%
my_scalar = mymat[1, 0]

# %% [markdown]
# 但是，传递单个数字将返回矩阵的子向量

# %%
my_row_vector = mymat[1]
my_column_vector = mymat[:, 0]

# %% [markdown]
# ## 2 形状
# 张量的形状是每个维度中元素的数量。TensorFlow在图形构建过程中自动推断形状。这些推断的形状可能具有已知或未知的等级。如果rank是已知的，则每个维度的大小可能是已知的或未知的。
# 
# TensorFlow文档使用三种符号约定来描述张量维数：rank，形状和维数。下表显示了它们之间的相互关系：
# 
# |rank	|形状	|维数	|例子|
# |---- |---- |---- |---- |
# |0	|[]	|0维	|0维张量。标量。|
# |1	|[D0]	|一维	|形状为[5]的一维张量。|
# |2	|[D0，D1]	|2维	|形状为[3，4]的二维张量。|
# |3	|[D0，D1，D2]	|3维	|形状为[1、4、3]的3-D张量。|
# |ñ	|[D0，D1，... Dn-1]	|D维	|形状为[D0，D1，... Dn-1]的张量。|
# %% [markdown]
# ### 2.1 获取tf.Tensor对象的形状
# 有两种访问形状tf.Tensor的方法。在构建图形时，询问关于张量形状的已知知识通常很有用。
# - 1、可以通过读取对象的shape属性来获取tf.Tensor。此方法返回一个TensorShape对象，这是表示部分指定的形状的便捷方式（因为在构建图形时，并非所有形状都将是完全已知的）。
# 
# - 2、也有可能在运行时获得一个代表某个张量的形状的tf.Tensor。这是通过调用tf.shape 操作来完成的。这样，可以通过构建依赖于输入的动态形状的其他张量来完成网络构建。
# 
# 例如，下面是如何制作一个零向量，其大小与给定矩阵中的列数相同：

# %%
zeros = tf.zeros(mymat.shape[1])

# %% [markdown]
# ### 2.2 改变tf.Tensor的形状
# 可以通过tf.reshape来改变张量的形状，只需确保所有形状的尺寸的乘积相同

# %%
rank_three_tensor = tf.ones([3, 4, 5])
matrix = tf.reshape(rank_three_tensor, [6, 10]) 
matrixB = tf.reshape(matrix, [3, -1])  .
matrixAlt = tf.reshape(matrixB, [4, 3, -1])   
                                            

# %% [markdown]
# ## 3 数据类型
# 除维数外，张量还具有数据类型。
# 
# 具有一个以上的数据类型是不可能的。但是，可以将任意数据结构序列化为strings并将其存储在tf.Tensors中。
# 
# 可以使用tf.cast命令将tf.Tensors从一种数据类型转换为另一种数据类型：

# %%
float_tensor = tf.cast(tf.constant([1, 2, 3]), dtype=tf.float32)

# %% [markdown]
# 要检查tf.Tensor的数据类型，请使用Tensor.dtype属性。
# 
# tf.Tensor从python对象创建时，可以选择指定数据类型。如果不设置，TensorFlow会选择一种可以代表数据的数据类型。TensorFlow将Python整数转换为tf.int32，并将python浮点数转换为tf.float32。而在将numpy转换为数组时使用的相同规则。
# %% [markdown]
# ## 4 打印张量
# 
# 出于调试目的，可能需要打印tf.Tensor的值。TensorFlow2中可以直接用print打印
# 
# 
# 

# %%
t = tf.add(3, 4)
print(t)

