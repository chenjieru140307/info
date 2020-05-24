# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# ##### Copyright 2018 The TensorFlow Authors.

# %%
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %% [markdown]
# # 动态图机制基础
# %% [markdown]
# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/tutorials/eager/eager_basics"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/eager/eager_basics.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/eager/eager_basics.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
# </table>
# %% [markdown]
# 这是一个使用 TensorFlow 的入门教程。包括以下内容：
# 
# * 导入需要的包
# * 创建和使用张量
# * 使用 GPU 加速
# * 数据集
# %% [markdown]
# ## 导入 TensorFlow
# 
# 在开始处，导入 `tensorflow` 模块，并开启动态图机制。
# 动态图机制使 TensorFlow 具有一个更具互动性的前端。具体的细节内容我们将在后面的章节讨论。

# %%
import tensorflow as tf

tf.enable_eager_execution()

# %% [markdown]
# ## 张量
# 
# 一个张量就是一个多维数组。类似于 NumPy 的 `ndarray` 对象，`Tensor` 对象也有一个数据类型属性和形状属性。除此之外，张量也可以存在于加速器内存中（比如 GPU）。TensorFlow 提供了丰富的操作库（[tf.add](https://www.tensorflow.org/api_docs/python/tf/add)、[tf.matmul](https://www.tensorflow.org/api_docs/python/tf/matmul) 和 [tf.linalg.inv](https://www.tensorflow.org/api_docs/python/tf/linalg/inv) 等等）用以使用和获得张量。这些操作会自动的转换原生的 Python 类型。例如：
# 

# %%
print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))
print(tf.encode_base64("hello world"))

# 运算符重载也支持
print(tf.square(2) + tf.square(3))

# %% [markdown]
# 每个张量都有一个形状和一个数据类型

# %%
x = tf.matmul([[1]], [[2, 3]])
print(x.shape)
print(x.dtype)

# %% [markdown]
# NumPy 数组和 TensorFlow 张量之间最显著的区别是：
# 
# 1. 加速器内存（如 GPU, TPU）支持张量的处理。
# 2. 张量是不可改变的。
# %% [markdown]
# ### 与 NumPy 的兼容性
# 
# TensorFlow 张量和 NumPy 数组间的转换非常简单，像下面：
# * TensorFlow 的操作会自动地转换 NumPy 数组为张量。
# * NumPy 操作会自动地转换张量为 NumPy 数组。
# 
# 张量可以通过调用 `.numpy()` 方法显式的转换为 NumPy 数组。
# 当数组和张量在表达形式上可以共享底层的内存时，这些转换通常很容易。然而，共享底层内存用做表达并不总是可行。因为张量可能会被托管在 GPU 内存中，而 NumPy 数组则总是由主机内存支持的，两者之间的转换因此将导致一个从 GPU 到主机内存的拷贝过程。

# %%
import numpy as np

ndarray = np.ones([3, 3])

print("TensorFlow operations convert numpy arrays to Tensors automatically")
tensor = tf.multiply(ndarray, 42)
print(tensor)


print("And NumPy operations convert Tensors to numpy arrays automatically")
print(np.add(tensor, 1))

print("The .numpy() method explicitly converts a Tensor to a numpy array")
print(tensor.numpy())

# %% [markdown]
# ## GPU 加速
# 
# 许多 TensorFlow 操作可以使用 GPU 加速计算过程。在没有任何注释的情况下，TensorFlow 会自动决定使用 GPU 或者 CPU 进行操作的处理（以及是否需要在 CPU 和 GPU 内存间进行拷贝）。由操作产生的张量通常也由执行操作的设备内存进行支持。例如：

# %%
x = tf.random_uniform([3, 3])

print("Is there a GPU available: "),
print(tf.test.is_gpu_available())

print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))

# %% [markdown]
# ### 设备名称
# 
# `Tensor.device` 属性提供托管张量内容设备的绝对名称。此名称对一串详细信息进行了编码，例如，当前正在运行程序的主机网络地址的标识符以及该主机内的设备信息。这些是以分布式的方式执行 TensorFlow 程序所需要的信息，这里我们会先略过。如果某个张量是主机的第 N 个张量，则将用以 `GPU:<N>`为结束的字符串表示。
# %% [markdown]
# 
# 
# ### 显式的设备分配
# 
# &quot;placement&quot; 在 TensorFlow 中表示的含义是怎样将独立的操作分配到对映的设备上去运行。就像之前提到的，当没有显式的指示时，TensorFlow 自动的决定哪个设备去执行某个操作，并且在需要的时候拷贝张量到相应的设备。不管怎样，TensorFlow 操作都可以通过使用上下文管理器 `tf.device` 显式的分配到指定的设配。例如：

# %%
def time_matmul(x):
  get_ipython().run_line_magic('timeit', 'tf.matmul(x, x)')

# 强制在 CPU 上运行
print("On CPU:")
with tf.device("CPU:0"):
  x = tf.random_uniform([1000, 1000])
  assert x.device.endswith("CPU:0")
  time_matmul(x)

# 如果可以找到 GPU #0，强制在它上面运行
if tf.test.is_gpu_available():
  with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
    x = tf.random_uniform([1000, 1000])
    assert x.device.endswith("GPU:0")
    time_matmul(x)

# %% [markdown]
# ## 数据集
# 
# 这部分演示使用 [`tf.data.Dataset` API](https://www.tensorflow.org/guide/datasets) 构建管道以将数据提供给模型。包括：
# 
# * 创建一个 `Dataset`。
# * 在动态图机制开启下，遍历一个 `Dataset` 。
# 
# 我们推荐使用 `Dataset` 的 API 构建有利于模型循环训练和评估的管道，该管道由简单、可重复利用模块的组成，输入虽然复杂，但是高性能。
# 
# 如果你对 TensorFlow 的图熟悉，构建 `Dataset` 对象的 API 在动态图机制开启的情况下，与之完全相同。但遍历数据集元素的过程更简单些。
# 你可以使用 Python 遍历 `tf.data.Dataset` 对象，却不用显式的创建 `tf.data.Iterator` 对象。
# 因此，在启用动态图机制时，[TensorFlow 指南](https://www.tensorflow.org/guide/datasets)关于迭代器的讨论与之无关。
# %% [markdown]
# ### 创建一个源 `Dataset`
# 
# 创建一个源数据集，使用以下其中之一的工厂函数，如 [`Dataset.from_tensors`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensors), [`Dataset.from_tensor_slices`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensor_slices) 或者使用读文件的对象，如 [`TextLineDataset`](https://www.tensorflow.org/api_docs/python/tf/data/TextLineDataset) 或者 [`TFRecordDataset`](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset)。 更多信息见 [TensorFlow Guide](https://www.tensorflow.org/guide/datasets#reading_input_data) 。

# %%
ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

# 创建一个 CSV 文件
import tempfile
_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
  f.write("""Line 1
Line 2
Line 3
  """)

ds_file = tf.data.TextLineDataset(filename)

# %% [markdown]
# ### 应用转换
# 
# 使用转换函数，如 [`map`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map), [`batch`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch), [`shuffle`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle) 等等。 应用转换到数据集记录中。更多详细信息见 [`tf.data.Dataset` API 文档](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) 。

# %%
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)

ds_file = ds_file.batch(2)

# %% [markdown]
# ### 迭代
# 
# 当启用动态图机制时，则 Dataset 对象支持迭代，
# 如果你熟悉`Dataset`在 TensorFlow 图中的使用，注意这里将不需要调用 `Dataset.make_one_shot_iterator()` 或者 `get_next()`。

# %%
print('Elements of ds_tensors:')
for x in ds_tensors:
  print(x)

print('\nElements in ds_file:')
for x in ds_file:
  print(x)

