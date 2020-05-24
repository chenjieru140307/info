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
# # 自动微分法和自动求导机制
# %% [markdown]
# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/tutorials/eager/automatic_differentiation"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/eager/automatic_differentiation.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/eager/automatic_differentiation.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
# </table>
# %% [markdown]
# 在前面的教程中，我们介绍了张量和对应的操作。在这个教程中，我们将讲讲 [自动微分法](https://en.wikipedia.org/wiki/Automatic_differentiation) ，一个优化机器学习模型的关键技术。
# %% [markdown]
# ## 设置
# 

# %%
import tensorflow as tf
tf.enable_eager_execution()

tfe = tf.contrib.eager # 用符号简略表达

# %% [markdown]
# ## 函数的导数
# 
# TensorFlow 为自动微分提供了API —— 计算函数的导数。模仿数学的方式，用 `f` 概括在 Python 中函数的计算过程，并且用 tfe.gradients_function 以及对应的参数创建一个对 'f' 求导的函数。如果你对 numpy 求导函数 [autograd](https://github.com/HIPS/autograd) 熟悉，这会很相似。例如：

# %%
from math import pi

def f(x):
  return tf.square(tf.sin(x))

assert f(pi/2).numpy() == 1.0


# grad_f 将返回一个 f 的导数列表
# 来对应它的参数。因为 f() 有一个参数，
# grad_f 将返回带有一个元素的列表。
grad_f = tfe.gradients_function(f)
assert tf.abs(grad_f(pi/2)[0]).numpy() < 1e-7

# %% [markdown]
# ### 高阶梯度
# 
# 相同的 API 可以用来多次微分：
# 

# %%
def f(x):
  return tf.square(tf.sin(x))

def grad(f):
  return lambda x: tfe.gradients_function(f)(x)[0]

x = tf.lin_space(-2*pi, 2*pi, 100)  # 在-2π 和 +2π 之间生成 100 个点

import matplotlib.pyplot as plt

plt.plot(x, f(x), label="f")
plt.plot(x, grad(f)(x), label="first derivative")
plt.plot(x, grad(grad(f))(x), label="second derivative")
plt.plot(x, grad(grad(grad(f)))(x), label="third derivative")
plt.legend()
plt.show()

# %% [markdown]
# ## 自动求导机制
# 
# 每个 TensorFlow 微分操作都有一个对应的梯度函数。例如，`tf.square(x)` 的梯度函数是 `2.0 * x` 。计算用户定义函数的梯度（如上例中 `f(x)`），TensorFlow 首先记录应用于计算函数输出的所有操作。我们把这个记录叫做“求导过程”。之后将使用这个求导过程与带有原操作的梯度函数去使用[反向微分](https://en.wikipedia.org/wiki/Automatic_differentiation)计算用户定义函数的梯度。
# 
# 因为操作按照它们的执行过程进行记录，Python 控制流（例如使用 `if` 和 `while`）的自然处理方式是：
# 
# 

# %%
def f(x, y):
  output = 1
  # 当使用 TensorFlow 1.10 或更早的版本时，Python 3 环境下
  # 必须使用 range(int(y)) 替代 range(y)。在 1.11+ 版本中可以使用 range(y)。
  for i in range(int(y)):
    output = tf.multiply(output, x)
  return output

def g(x, y):
  # 返回 `f` 对应的第一个参数的梯度
  return tfe.gradients_function(f)(x, y)[0]

assert f(3.0, 2).numpy() == 9.0   # f(x, 2) 本质上是 x * x
assert g(3.0, 2).numpy() == 6.0   # 它的梯度是 2 * x
assert f(4.0, 3).numpy() == 64.0  # f(x, 3) 本质上是 x * x * x
assert g(4.0, 3).numpy() == 48.0  # 它的梯度是 3 * x * x

# %% [markdown]
# 有时，在函数中囊括计算的过程可能并不方便。例如，如果你想要输出梯度关于函数中计算的中间值。在这种情况下，略显冗长但含义明确的 [tf.GradientTape](https://www.tensorflow.org/api_docs/python/tf/GradientTape) 很有用。所有 `tf.GradientTape` 中的计算都会被记录。
# 
# 例如：

# %%
x = tf.ones((2, 2))
  
# 当问题被修复后，可以调用一个 t.gradient()。
with tf.GradientTape(persistent=True) as t:
  t.watch(x)
  y = tf.reduce_sum(x)
  z = tf.multiply(y, y)

# 使用相同的求导过程去计算 z 关于中间值 y 的导数
dz_dy = t.gradient(z, y)
assert dz_dy.numpy() == 8.0

# 关于初始输入张量 x 的 z 的导数
dz_dx = t.gradient(z, x)
for i in [0, 1]:
  for j in [0, 1]:
    assert dz_dx[i][j].numpy() == 8.0

# %% [markdown]
# ### 高阶梯度
# 
# `GradientTape` 管理器中记录的操作是为了自动微分。如果在其中计算了梯度，那么梯度计算的过程就会被记录。最后，完全相同的 API 也可以作用于高阶梯度。例如：

# %%
x = tf.constant(1.0)  # 将 Python 中的 1.0 转换为张量对象

with tf.GradientTape() as t:
  with tf.GradientTape() as t2:
    t2.watch(x)
    y = x * x * x
  # 在管理器 t 中计算梯度
  # 这意味着梯度计算也是可微分的
  dy_dx = t2.gradient(y, x)
d2y_dx2 = t.gradient(dy_dx, x)

assert dy_dx.numpy() == 3.0
assert d2y_dx2.numpy() == 6.0

# %% [markdown]
# ## 后续
# 
# 在这个教程中，我们讲了 TensorFlow 中的梯度计算。有了上面的内容，我们就有了构建和训练神经网络足够的运算基础。
