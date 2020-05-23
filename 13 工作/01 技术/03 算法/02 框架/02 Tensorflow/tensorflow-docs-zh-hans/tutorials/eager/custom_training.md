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
# # 自定义训练：基础部分
# %% [markdown]
# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/tutorials/eager/custom_training"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/eager/custom_training.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/eager/custom_training.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
# </table>
# %% [markdown]
# 在上一个教程中，我们介绍了用于自动微分的 TensorFlow API，这是机器学习的基本模块。
# 在本教程中，我们将使用先前教程中介绍的 TensorFlow 原函数 来进行一些简单的机器学习。
# 
# TensorFlow 还包括一个更高级别的神经网络 API（`tf.keras`），利用它的高阶抽象可以减少代码引用。我们强烈建议那些训练神经网络的人使用更高级别的API。但是，在这个简短的教程中，我们从基础原理开始介绍神经网络训练，从而建立坚实的基础。
# %% [markdown]
# ## 设置

# %%
import tensorflow as tf

tfe = tf.contrib.eager

tf.enable_eager_execution()

# %% [markdown]
# ## 变量
# 
# TensorFlow 中的张量是不可变的无状态对象。但是，机器学习模型需要改变状态：当你训练模型时，同一份预测代码应该随着时间的推移而表现不同（希望损失更低！）。要表示在计算过程中不断变化的状态，你可以借助 Python 编程语言：

# %%
# 使用 Python 状态特性
x = tf.zeros([10, 10])
x += 2  # 这个等同于 x = x + 2，但是不改变原始 x 的值
print(x)

# %% [markdown]
# 但是，TensorFlow 内置了状态操作，这些操作会比一些低阶 Python 表示易用。例如，为了表示模型中的权重，使用 TensorFlow 变量更加方便。
# 
# 变量是一个存储数值的对象，当在 TensorFlow 计算中使用时，它将隐式地从存储数值中读取。有一些操作（`tf.assign_sub`，`tf.scatter_update`等）可以修改存储在 TensorFlow 变量中的值。

# %%
v = tfe.Variable(1.0)
assert v.numpy() == 1.0

# 重新赋值
v.assign(3.0)
assert v.numpy() == 3.0

# 在 TensorFlow 中对变量 `v` 使用 tf.square() 和重新赋值操作
v.assign(tf.square(v))
assert v.numpy() == 9.0

# %% [markdown]
# 在计算梯度时，会自动跟踪变量的计算。对于表示嵌入的变量，TensorFlow 默认会进行稀疏更新，这样可以提高计算效率和内存效率。
# 
# 使用变量也是一种快速让阅读代码的读者知道这段状态是可变的方法。
# %% [markdown]
# ## 案例：拟合线性模型
# 
# 现在让我们利用之前介绍的几个概念 ———“张量”，“求导”，“变量”——— 构建并训练一个简单的模型。这通常涉及几个步骤：
# 
# 1. 定义模型。
# 2. 定义损失函数。
# 3. 获取训练数据。
# 4. 运行训练数据并使用”优化器“调整变量来拟合数据。
# 
# 在本教程中，我们将介绍一个简单线性模型的简单示例：`f(x) = x * W + b`，包含两个变量 - `W` 和 `b`。最后，根据训练数据我们得到训练好的模型变量值为 `W = 3.0` and `b = 2.0`。
# %% [markdown]
# ### 定义模型
# 
# 让我们定义一个简单的类来封装变量和计算。

# %%
class Model(object):
  def __init__(self):
    # 初始化变量为（5.0, 0.0）
    # 实际上这些变量应当初始化为随机值。
    self.W = tfe.Variable(5.0)
    self.b = tfe.Variable(0.0)
    
  def __call__(self, x):
    return self.W * x + self.b
  
model = Model()

assert model(3.0).numpy() == 15.0

# %% [markdown]
# ### 定义损失函数
# 
# 损失函数衡量模型预测值与实际值的偏差。让我们使用标准的 L2 损失函数。

# %%
def loss(predicted_y, desired_y):
  return tf.reduce_mean(tf.square(predicted_y - desired_y))

# %% [markdown]
# ### 获取训练数据
# 
# 让我们合成训练数据时添入一些噪声。

# %%
TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs  = tf.random_normal(shape=[NUM_EXAMPLES])
noise   = tf.random_normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise

# %% [markdown]
# 在我们训练模型之前，让我们图示一下模型当前的情况。我们将用红色绘制模型的预测，用蓝色绘制训练数据。

# %%
import matplotlib.pyplot as plt

plt.scatter(inputs, outputs, c='b')
plt.scatter(inputs, model(inputs), c='r')
plt.show()

print('Current loss: '),
print(loss(model(inputs), outputs).numpy())

# %% [markdown]
# ### 定义循环训练
# 
# 我们现在拥有神经网络模型和训练数据。让我们训练它，例如，使用[梯度下降](https://en.wikipedia.org/wiki/Gradient_descent)方法更新模型的变量（`W` 和 `b`）减少损失。在 `tf.train.Optimizer` 实现中有许多梯度下降相关方案。我们强烈建议使用这些实现，但是本着从头开始构建的精神原则，在这个特定的例子中，我们将自己动手实现。

# %%
def train(model, inputs, outputs, learning_rate):
  with tf.GradientTape() as t:
    current_loss = loss(model(inputs), outputs)
  dW, db = t.gradient(current_loss, [model.W, model.b])
  model.W.assign_sub(learning_rate * dW)
  model.b.assign_sub(learning_rate * db)

# %% [markdown]
# 最后，让我们反复训练数据，看看 `W` 和 `b` 是如何演变的。

# %%
model = Model()

# 保存历史 W 和 b 数值用于后期绘图
Ws, bs = [], []
epochs = range(10)
for epoch in epochs:
  Ws.append(model.W.numpy())
  bs.append(model.b.numpy())
  current_loss = loss(model(inputs), outputs)

  train(model, inputs, outputs, learning_rate=0.1)
  print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %
        (epoch, Ws[-1], bs[-1], current_loss))

# 绘制图形
plt.plot(epochs, Ws, 'r',
         epochs, bs, 'b')
plt.plot([TRUE_W] * len(epochs), 'r--',
         [TRUE_b] * len(epochs), 'b--')
plt.legend(['W', 'b', 'true W', 'true_b'])
plt.show()
  

# %% [markdown]
# ## 后续
# 
# 在本教程中，我们介绍了 `变量`，并使用目前讨论的 TensorFlow 原函数构建和训练了一个简单的线性模型。
# 
# 从理论上讲，这几乎是使用 TensorFlow 进行机器学习研究所需要的全部内容。
# 在实践中，特别是对于神经网络，像 `tf.keras` 这样的高级 API 将更加方便，因为它提供了更高级别的构建块（称为“层”），用于保存和恢复状态的公用程序，一套损失函数，一套优化策略等。
