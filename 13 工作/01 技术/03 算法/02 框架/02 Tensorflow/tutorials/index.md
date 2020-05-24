# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ##### Copyright 2018 The TensorFlow Authors.
# 
# Licensed under the Apache License, Version 2.0 (the "License");

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
# # 开始使用 TensorFlow
# %% [markdown]
# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/tutorials/"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/_index.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/_index.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
# </table>
# %% [markdown]
# 这是一个 [Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb) 笔记本格式文件。Python 程序可以直接在浏览器中运行 —— 一个非常好的学习和使用 TensorFlow 的方式。通过以下步骤运行 Colab 笔记本：
# 
# 1. 连接到 Python 运行平台：在菜单栏右上角，选择***连接***。
# 2. 运行所有笔记本代码单元：选择 ***代码执行程序*** > ***全部运行***。
# 
# 更多例子和教程（包括本程序的细节内容），见 [开始使用 TensorFlow](https://www.tensorflow.org/get_started/)。
# 
# 让我们开始吧，导入 TensorFlow 库到你的程序：

# %%
import tensorflow as tf

# %% [markdown]
# 加载和准备 [MNIST](http://yann.lecun.com/exdb/mnist/) 数据集。将样本从整型转换为浮点型数字：

# %%
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# %% [markdown]
# 通过层堆栈的方式，构建 `tf.keras` 模型。为训练选择一个优化器和损失函数：

# %%
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# %% [markdown]
# 训练和评估模型：

# %%
model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)

# %% [markdown]
# 你现在已经训练得到一个图片分类器，针对当前数据集的准确度在 98%。学习更多内容见 [开始使用 TensorFlow](https://www.tensorflow.org/get_started/)。
