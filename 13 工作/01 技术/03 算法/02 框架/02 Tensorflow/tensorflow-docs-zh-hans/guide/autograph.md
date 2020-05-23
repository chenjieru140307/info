# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# ##### Copyright 2018 The TensorFlow Authors.
# 
# Licensed under the Apache License, Version 2.0 (the "License");

# %%
#@title Licensed under the Apache License, Version 2.0 (the "License"); { display-mode: "form" }
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
# # AutoGraph: 简易控制图模型流程
# %% [markdown]
# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/guide/autograph"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/autograph.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/guide/autograph.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
# </table>
# %% [markdown]
# [AutoGraph](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/) 可帮助你使用普通 Python 编写复杂的图形代码。在幕后，AutoGraph 会自动将代码转换为等效的[TensorFlow 图形代码](https://www.tensorflow.org/guide/graphs)。 AutoGraph 已经支持大部分 Python 语言，而且覆盖范围也在不断扩大。有关支持的 Python 语言功能的列表，请参阅[Autograph 功能和限制](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/LIMITATIONS.md)。
# %% [markdown]
# ## 安装
# 
# 要使用 AutoGraph，安装最新版本 TensorFlow：

# %%
get_ipython().system(' pip install -U tf-nightly')

# %% [markdown]
# 导入 TensorFlow，AutoGraph 和任何支持模块：

# %%
from __future__ import division, print_function, absolute_import

import tensorflow as tf
layers = tf.keras.layers
from tensorflow.contrib import autograph


import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# 我们将启用 [eager execution](https://www.tensorflow.org/guide/eager) 进行演示，但 AutoGraph 同时适用于 eager 和 [graph execution](https://www.tensorflow.org/guide/graphs) 环境：

# %%
tf.enable_eager_execution()

# %% [markdown]
# 注意：AutoGraph 转换代码旨在在图计算期间运行。启用 eager 执行时，使用显式图（如本例所示）或 `tf.contrib.eager.defun`。
# %% [markdown]
# ## 自动转换 Python 控制流
# 
# AutoGraph 会将大部分 Python语言转换为等效的 TensorFlow 图构建代码。 
# 
# 注意：在实际应用中，批处理对性能至关重要。转换为 AutoGraph 的最佳代码是在 <b>batch</b> 级别决定控制流的代码。如果在单个 <b>example</b> 级别做出决策，则必须对示例进行索引和批处理，以便在应用控制流逻辑时保持性能。 
# 
# AutoGraph 转化函数如下所示：

# %%
def square_if_positive(x):
  if x > 0:
    x = x * x
  else:
    x = 0.0
  return x

# %% [markdown]
# 对函数使用图构建方法：

# %%
print(autograph.to_code(square_if_positive))

# %% [markdown]
# 为 eager execution 编写的代码可以在 `tf.Graph` 中运行，结果相同，但具有图计算的优点：

# %%
print('Eager results: %2.2f, %2.2f' % (square_if_positive(tf.constant(9.0)), 
                                       square_if_positive(tf.constant(-9.0))))

# %% [markdown]
# 生成图并调用它：

# %%
tf_square_if_positive = autograph.to_graph(square_if_positive)

with tf.Graph().as_default():  
  # 结果像常规操作一样：将张量输入，返回张量。
  # 可以使用 tf.get_default_graph().as_graph_def() 检查图模型
  g_out1 = tf_square_if_positive(tf.constant( 9.0))
  g_out2 = tf_square_if_positive(tf.constant(-9.0))
  with tf.Session() as sess:
    print('Graph results: %2.2f, %2.2f\n' % (sess.run(g_out1), sess.run(g_out2)))

# %% [markdown]
# AutoGraph 支持常见的 Python 语句，如 `while`，`for`，`if`，`break` 和 `return`，支持嵌套。将此函数与以下代码块中显示的复杂图相比较：

# %%
# 循环跳出
def sum_even(items):
  s = 0
  for c in items:
    if c % 2 > 0:
      continue
    s += c
  return s

print('Eager result: %d' % sum_even(tf.constant([10,12,15,20])))

tf_sum_even = autograph.to_graph(sum_even)

with tf.Graph().as_default(), tf.Session() as sess:
    print('Graph result: %d\n\n' % sess.run(tf_sum_even(tf.constant([10,12,15,20]))))


# %%
print(autograph.to_code(sum_even))

# %% [markdown]
# ## 装饰器
# 
# 如果你不想通过访问原始 Python 函数调用，请使用 `convert` 装饰器：

# %%
@autograph.convert()
def fizzbuzz(i, n):
  while i < n:
    msg = ''
    if i % 3 == 0:
      msg += 'Fizz'
    if i % 5 == 0:
      msg += 'Buzz'
    if msg == '':
      msg = tf.as_string(i)
    print(msg)
    i += 1
  return i

with tf.Graph().as_default():
  final_i = fizzbuzz(tf.constant(10), tf.constant(16))
  # 结果像常规操作一样：将张量输入，返回张量。
  # 可以使用 tf.get_default_graph().as_graph_def() 检查图模型
  with tf.Session() as sess:
    sess.run(final_i)

# %% [markdown]
# ## 示例
# 
# 让我们演示一些有用的 Python 语言特性。
# 
# %% [markdown]
# ### 中断
# 
# AutoGraph 自动将 Python `assert` 语句转换为等效的 `tf.Assert` 代码：

# %%
@autograph.convert()
def inverse(x):
  assert x != 0.0, 'Do not pass zero!'
  return 1.0 / x

with tf.Graph().as_default(), tf.Session() as sess:
  try:
    print(sess.run(inverse(tf.constant(0.0))))
  except tf.errors.InvalidArgumentError as e:
    print('Got error message:\n    %s' % e.message)

# %% [markdown]
# ### 打印
# 
# 在图模型中使用 Python 函数 `print`：

# %%
@autograph.convert()
def count(n):
  i=0
  while i < n:
    print(i)
    i += 1
  return n
    
with tf.Graph().as_default(), tf.Session() as sess:
    sess.run(count(tf.constant(5)))

# %% [markdown]
# ### 列表
# 
# 在循环中向列表添加数据（自动创建张量列表操作）：

# %%
@autograph.convert()
def arange(n):
  z = []
  # 你需要指定列表的元素类型
  autograph.set_element_type(z, tf.int32)
  
  for i in tf.range(n):
    z.append(i)
  # 当你完成列表操作时，叠加它
  # （就像 np.stack）
  return autograph.stack(z) 


with tf.Graph().as_default(), tf.Session() as sess:
    sess.run(arange(tf.constant(10)))

# %% [markdown]
# ### 嵌套控制流程

# %%
@autograph.convert()
def nearest_odd_square(x):
  if x > 0:
    x = x * x
    if x % 2 == 0:
      x = x + 1
  return x

with tf.Graph().as_default():  
  with tf.Session() as sess:
    print(sess.run(nearest_odd_square(tf.constant(4))))
    print(sess.run(nearest_odd_square(tf.constant(5))))
    print(sess.run(nearest_odd_square(tf.constant(6))))

# %% [markdown]
# ### While 循环

# %%
@autograph.convert()
def square_until_stop(x, y):
  while x < y:
    x = x * x
  return x
    
with tf.Graph().as_default():  
  with tf.Session() as sess:
    print(sess.run(square_until_stop(tf.constant(4), tf.constant(100))))

# %% [markdown]
# ### For 循环

# %%
@autograph.convert()
def squares(nums):

  result = []
  autograph.set_element_type(result, tf.int64)

  for num in nums: 
    result.append(num * num)
    
  return autograph.stack(result)
    
with tf.Graph().as_default():  
  with tf.Session() as sess:
    print(sess.run(squares(tf.constant(np.arange(10)))))

# %% [markdown]
# ### 中止

# %%
@autograph.convert()
def argwhere_cumsum(x, threshold):
  current_sum = 0.0
  idx = 0
  for i in tf.range(len(x)):
    idx = i
    if current_sum >= threshold:
      break
    current_sum += x[i]
  return idx

N = 10
with tf.Graph().as_default():  
  with tf.Session() as sess:
    idx = argwhere_cumsum(tf.ones(N), tf.constant(float(N/2)))
    print(sess.run(idx))

# %% [markdown]
# ## `tf.Keras` 操作
# 
# 现在你已经掌握基础知识了，让我们用 autograph 来构建一些模型组件。
# 
# 将 `autograph` 和 `tf.keras` 融合到一起非常简单。
# 
# 
# ### 无状态函数
# 
# 对于无状态函数，如下面所示的 `collatz`，将它们包含在 keras 模型中的最简单方法是使用 `tf.keras.layers.Lambda` 将它们包装为图层。

# %%
import numpy as np

@autograph.convert()
def collatz(x):
  x = tf.reshape(x,())
  assert x > 0
  n = tf.convert_to_tensor((0,)) 
  while not tf.equal(x, 1):
    n += 1
    if tf.equal(x%2, 0):
      x = x // 2
    else:
      x = 3 * x + 1
      
  return n

with tf.Graph().as_default():
  model = tf.keras.Sequential([
    tf.keras.layers.Lambda(collatz, input_shape=(1,), output_shape=())
  ])
  
result = model.predict(np.array([6171]))
result

# %% [markdown]
# ### 自定义图层与模型
# 
# <!--TODO(markdaoust) link to full examples  or these referenced models.-->
# 
# 将 AutoGraph 与 Keras 图层和模型一起使用的最简单方法是使用`@autograph.convert()` 中 `call` 方法。有关如何构建这些类的详细信息，请参阅 [TensorFlow Keras 指南](https://tensorflow.org/guide/keras#build_advanced_models)。
# 
# 以下是实现[随机深度网络](https://arxiv.org/abs/1603.09382)的简单示例：

# %%
# `K` 用于检查我们是否处于训练或测试模式。
K = tf.keras.backend

class StochasticNetworkDepth(tf.keras.Sequential):
  def __init__(self, pfirst=1.0, plast=0.5, *args,**kwargs):
    self.pfirst = pfirst
    self.plast = plast
    super().__init__(*args,**kwargs)
        
  def build(self,input_shape):
    super().build(input_shape.as_list())
    self.depth = len(self.layers)
    self.plims = np.linspace(self.pfirst, self.plast, self.depth + 1)[:-1]
    
  @autograph.convert()
  def call(self, inputs):
    training = tf.cast(K.learning_phase(), dtype=bool)  
    if not training: 
      count = self.depth
      return super(StochasticNetworkDepth, self).call(inputs), count
    
    p = tf.random_uniform((self.depth,))
    
    keeps = (p <= self.plims)
    x = inputs
    
    count = tf.reduce_sum(tf.cast(keeps, tf.int32))
    for i in range(self.depth):
      if keeps[i]:
        x = self.layers[i](x)
      
    # 返回最后一层输出以及执行网络的层数。
    return x, count

# %% [markdown]
# 在 mnist 数据集上进行上述实验：

# %%
train_batch = np.random.randn(64, 28, 28, 1).astype(np.float32)

# %% [markdown]
# 在随机深度模型中构建一个简单的 `conv` 层堆栈：

# %%
with tf.Graph().as_default() as g:
  model = StochasticNetworkDepth(
        pfirst=1.0, plast=0.5)

  for n in range(20):
    model.add(
          layers.Conv2D(filters=16, activation=tf.nn.relu,
                        kernel_size=(3, 3), padding='same'))

  model.build(tf.TensorShape((None, None, None, 1)))
  
  init = tf.global_variables_initializer()

# %% [markdown]
# Now test it to ensure it behaves as expected in train and test modes:

# %%
# 在这里使用显式会话，以便我们切换训练/测试，以及
# 检查 `call` 返回的图层数
with tf.Session(graph=g) as sess:
  init.run()
 
  for phase, name in enumerate(['test','train']):
    K.set_learning_phase(phase)
    result, count = model(tf.convert_to_tensor(train_batch, dtype=tf.float32))

    result1, count1 = sess.run((result, count))
    result2, count2 = sess.run((result, count))

    delta = (result1 - result2)
    print(name, "sum abs delta: ", abs(delta).mean())
    print("    layers 1st call: ", count1)
    print("    layers 2nd call: ", count2)
    print()

# %% [markdown]
# ## 高级示例: An in-graph training loop
# 
# 上一节显示 AutoGraph 可以在 Keras 层和模型中使用。Keras 模型也可用于 AutoGraph 代码。
# 
# 由于在 AutoGraph 中编写控制流很容易，因此在 TensorFlow 图中循环训练也应该很容易。 
# 
# 此示例显示如何在整个训练过程中训练一个简单的 Keras 模型—批次加载，计算梯度，更新参数，计算验证集准确度，并重复直到收敛—在图中执行。
# %% [markdown]
# ### 下载数据

# %%
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# %% [markdown]
# ### 定义模型

# %%
def mlp_model(input_shape):
  model = tf.keras.Sequential((
      tf.keras.layers.Dense(100, activation='relu', input_shape=input_shape),
      tf.keras.layers.Dense(100, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')))
  model.build()
  return model


def predict(m, x, y):
  y_p = m(tf.reshape(x, (-1, 28 * 28)))
  losses = tf.keras.losses.categorical_crossentropy(y, y_p)
  l = tf.reduce_mean(losses)
  accuracies = tf.keras.metrics.categorical_accuracy(y, y_p)
  accuracy = tf.reduce_mean(accuracies)
  return l, accuracy


def fit(m, x, y, opt):
  l, accuracy = predict(m, x, y)
  # Autograph 自动添加 `tf.control_dependencies`。
  # （除了它们，其它都不依赖于 `opt.minimize`，所以它不会运行。）
  # 这使得它更像 eager-code。
  opt.minimize(l)
  return l, accuracy


def setup_mnist_data(is_training, batch_size):
  if is_training:
    ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    ds = ds.shuffle(batch_size * 10)
  else:
    ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

  ds = ds.repeat()
  ds = ds.batch(batch_size)
  return ds


def get_next_batch(ds):
  itr = ds.make_one_shot_iterator()
  image, label = itr.get_next()
  x = tf.to_float(image) / 255.0
  y = tf.one_hot(tf.squeeze(label), 10)
  return x, y 

# %% [markdown]
# ### 定义循环训练

# %%
# 使用 `recursive = True` 实现递归调用函数。
@autograph.convert(recursive=True)
def train(train_ds, test_ds, hp):
  m = mlp_model((28 * 28,))
  opt = tf.train.AdamOptimizer(hp.learning_rate)
  
  # We'd like to save our losses to a list. In order for AutoGraph
  # to convert these lists into their graph equivalent,
  # we need to specify the element type of the lists.
  train_losses = []
  autograph.set_element_type(train_losses, tf.float32)
  test_losses = []
  autograph.set_element_type(test_losses, tf.float32)
  train_accuracies = []
  autograph.set_element_type(train_accuracies, tf.float32)
  test_accuracies = []
  autograph.set_element_type(test_accuracies, tf.float32)
  
  # 在图中执行循环训练。
  i = tf.constant(0)
  while i < hp.max_steps:
    train_x, train_y = get_next_batch(train_ds)
    test_x, test_y = get_next_batch(test_ds)

    step_train_loss, step_train_accuracy = fit(m, train_x, train_y, opt)
    step_test_loss, step_test_accuracy = predict(m, test_x, test_y)
    if i % (hp.max_steps // 10) == 0:
      print('Step', i, 'train loss:', step_train_loss, 'test loss:',
            step_test_loss, 'train accuracy:', step_train_accuracy,
            'test accuracy:', step_test_accuracy)
    train_losses.append(step_train_loss)
    test_losses.append(step_test_loss)
    train_accuracies.append(step_train_accuracy)
    test_accuracies.append(step_test_accuracy)
    i += 1
  
  # 我们已经在 AutoGraph 的帮助下将我们的损失值和精度记录到图表中的列表中。
  # 为了将值作为张量返回，我们需要在返回它们之前将它们堆叠起来。
  return (autograph.stack(train_losses), autograph.stack(test_losses),  
          autograph.stack(train_accuracies), autograph.stack(test_accuracies))

# %% [markdown]
# 现在编译图模型并且执行循环训练：

# %%
with tf.Graph().as_default() as g:
  hp = tf.contrib.training.HParams(
      learning_rate=0.005,
      max_steps=500,
  )
  train_ds = setup_mnist_data(True, 50)
  test_ds = setup_mnist_data(False, 1000)
  (train_losses, test_losses, train_accuracies,
   test_accuracies) = train(train_ds, test_ds, hp)

  init = tf.global_variables_initializer()
  
with tf.Session(graph=g) as sess:
  sess.run(init)
  (train_losses, test_losses, train_accuracies,
   test_accuracies) = sess.run([train_losses, test_losses, train_accuracies,
                                test_accuracies])
  
plt.title('MNIST train/test losses')
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.xlabel('Training step')
plt.ylabel('Loss')
plt.show()
plt.title('MNIST train/test accuracies')
plt.plot(train_accuracies, label='train accuracy')
plt.plot(test_accuracies, label='test accuracy')
plt.legend(loc='lower right')
plt.xlabel('Training step')
plt.ylabel('Accuracy')
plt.show()

