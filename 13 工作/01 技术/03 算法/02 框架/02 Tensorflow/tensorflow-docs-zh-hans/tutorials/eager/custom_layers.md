# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
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
# # 自定义层
# %% [markdown]
# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/tutorials/eager/custom_layers"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/eager/custom_layers.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/eager/custom_layers.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
# </table>
# %% [markdown]
# 我们建议使用作为构建神经网络的高级 API `tf.keras`。也就是说，大多数 TensorFlow API 都可以在动态图机制下使用。
# 

# %%
import tensorflow as tf
tfe = tf.contrib.eager

tf.enable_eager_execution()

# %% [markdown]
# ## 图层：常用操作集合
# 
# 大多数情况下，在为机器学习模型编写代码时，您希望以比单个操作和单个变量操作更高的抽象级别进行操作。
# 
# 许多机器学习模型可以表达为相对简单的图层的组合和堆叠，TensorFlow 提供了许多常用的层,有利于使用者以一个简单的方式选择从头开始编写或是利用现有层组合的方式实现自己特定的应用层。
# 
# TensorFlow 的 tf.keras 包含全部 [Keras](https://keras.io) API，而 Keras 层在构建自己的模型时非常有用。
# 

# %%
# 在 tf.keras.layers 包中, 图层都是对象。要构造一个图层，
# 简单地构造对象。大多数图层将输出维度或者通道的数量作为第一个参数。
layer = tf.keras.layers.Dense(100)
# 输入维度的数量通常是不必要的，因为可以推断
# 第一次使用图层，但如果你愿意，可以提供
# 手动指定，这在某些复杂模型中很有用。
layer = tf.keras.layers.Dense(10, input_shape=(None, 5))

# %% [markdown]
# 可以在[文档](https://www.tensorflow.org/api_docs/python/tf/keras/layers)中查看预先存在的图层的完整列表。它包括Dense（全连接层），Conv2D，LSTM，BatchNormalization，Dropout 等等。

# %%
# 要使用图层，直接调用就行。
layer(tf.zeros([10, 5]))


# %%
# 图层有许多有用的方法。例如，你可以通过调用 layer.variables 查看
# 一层中的所有变量。在这个例子中你可以
# 查看全连接层变量的权重和偏置。
layer.variables


# %%
# 变量也可以通过访问器访问
layer.kernel, layer.bias

# %% [markdown]
# ## 实现自定义层
# 最佳的实现自定义的层的方式是继承 tf.keras.Layer 类并进行实现：
#   *  `__init__` ，你可以在其中实现所有针对与输入无关的
#   * `build`，你知道输入张量的大小，并可以进行其余的初始化
#   * `call`， 你可以进行前向传播计算
# 
# 请注意，你不必等到调用 `build` 来创建变量，您也可以在 `__init__` 中创建它们。但是，在 `build` 中创建它们的优点是它可以根据将要操作的图层输入的大小创建后期变量。另一方面，在 `__init__` 中创建变量意味着需要明确指定变量的大小。

# %%
class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(MyDenseLayer, self).__init__()
    self.num_outputs = num_outputs
    
  def build(self, input_shape):
    self.kernel = self.add_variable("kernel", 
                                    shape=[input_shape[-1].value, 
                                           self.num_outputs])
    
  def call(self, input):
    return tf.matmul(input, self.kernel)
  
layer = MyDenseLayer(10)
print(layer(tf.zeros([10, 5])))
print(layer.variables)

# %% [markdown]
# 请注意，你不必等到调用 `build` 来创建变量，你也可以在 `__init__` 中创建它们。
# 
# 如果尽可能使用标准图层，则整体代码更易于阅读和维护，因为其他读者更熟悉标准网络层的行为。 如果要使用 tf.keras.layers 或 tf.contrib.layers 中不存在的网络层，请考虑提交 [github issue](http://github.com/tensorflow/tensorflow/issues/new) 或者，更好的方式是，给我们发送 pull request！
# %% [markdown]
# ## 模型：组合图层
# 
# 机器学习模型中许多有趣的神经网络是通过组合现有层来实现的。例如，resnet 中的每个残差块是卷积，批量标准化和快捷方式的组合。
# 
# 创建神经网络模型使用的主类是 tf.keras.Model。实现是通过继承 tf.keras.Model 完成的。

# %%
class ResnetIdentityBlock(tf.keras.Model):
  def __init__(self, kernel_size, filters):
    super(ResnetIdentityBlock, self).__init__(name='')
    filters1, filters2, filters3 = filters

    self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
    self.bn2a = tf.keras.layers.BatchNormalization()

    self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
    self.bn2b = tf.keras.layers.BatchNormalization()

    self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
    self.bn2c = tf.keras.layers.BatchNormalization()

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    x += input_tensor
    return tf.nn.relu(x)

    
block = ResnetIdentityBlock(1, [1, 2, 3])
print(block(tf.zeros([1, 2, 3, 3])))
print([x.name for x in block.variables])

# %% [markdown]
# 然而，在很多时候，只是将图层一层接着一层组成一个包含许多层的模型。这可以利用 tf.keras.Sequential 以非常少的代码量完成模型。

# %%
my_seq = tf.keras.Sequential([tf.keras.layers.Conv2D(1, (1, 1)),
                              tf.keras.layers.BatchNormalization(),
                              tf.keras.layers.Conv2D(2, 1, 
                                                     padding='same'),
                              tf.keras.layers.BatchNormalization(),
                              tf.keras.layers.Conv2D(3, (1, 1)),
                              tf.keras.layers.BatchNormalization()])
my_seq(tf.zeros([1, 2, 3, 3]))

# %% [markdown]
# # 后续
# 
# 现在，你可以回到之前的笔记并调整线性回归示例，以使用更好的结构化图层和模型。
