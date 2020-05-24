# 基本分类器

[Colab notebook](https://colab.research.google.com/github/tensorflow/models/blob/master/samples/core/tutorials/keras/basic_classification.ipynb)

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


# %%
#@title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# %% [markdown]
# # 训练你的第一个神经网络：基本分类器
# %% [markdown]
# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/tutorials/keras/basic_classification"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_classification.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_classification.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
# </table>
# %% [markdown]
# 本教程训练了一个神经网络模型，用于对服装（比如运动鞋或衬衫）图像进行分类。如果你不了解各种细节也没关系，本教程是对一个完整的 TensorFlow 程序的快速概述，在过程中我们将进行详细解释
# 
# 本教程使用了 [tf.keras](https://www.tensorflow.org/guide/keras) 这一高阶 API 来构建与训练模型。

# %%
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# %% [markdown]
# ## 导入 Fashion MNIST 数据集
# %% [markdown]
# 本教程使用了 [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) 数据集，其中包括了 10 类，7 万余个灰度图片。每个图片都是一个不同的服饰，图片的分辨率为 28x28，如下图所示：
# 
# <table>
#   <tr><td>
#     <img src="https://tensorflow.org/images/fashion-mnist-sprite.png"
#          alt="Fashion MNIST sprite"  width="600">
#   </td></tr>
#   <tr><td align="center">
#     <b>Figure 1.</b> <a href="https://github.com/zalandoresearch/fashion-mnist">Fashion-MNIST samples</a> (by Zalando, MIT License).<br/>&nbsp;
#   </td></tr>
# </table>
# 
# Fashion MNIST 可以理解为是经典 [MNIST](http://yann.lecun.com/exdb/mnist/) 数据集的替代形式。经典 MNIST 数据集包括各种手写数字的图片（0、1、2 等），常被用于构建计算机视觉的“Hello World”机器学习程序。我们此处用的 Fashion MNIST 数据集的格式与此类似。
# 
# 由于 Fashion MNIST 数据集相比与经典 MNIST 数据集更具挑战性，并且为了教程的多样性，所以我们选择使用 Fashion MNIST 数据集。这个数据集相对比较小，便于验证算法是否按照预期工作，这对于测试和调试代码有利。
# 
# 我们将用 6 万张图片训练网络，1 万张图片用于评估训练后的图像分类器的准确率。你可以通过 import 直接在 TensorFlow 中加载并访问 Fashion MNIST 数据集：

# %%
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# %% [markdown]
# 加载数据集后会返回 4 个 NumPy 数组：
# 
# * `train_images` 和 `train_labels` 数组是**训练集**，也就是模型用于学习的数据。
# * 模型在与之相对的**测试集**上进行测试。测试集包括 `test_images` 和 `test_labels` 数组。
# 
# 数据集中的图片都是 28 x 28 的 Numpy 数组，像素的取值范围是 0-255。*label* 为整型数组，范围为 0-9。对应服装图片的分类 *class* 如下所示：
# 
# <table>
#   <tr>
#     <th>Label</th>
#     <th>Class</th> 
#   </tr>
#   <tr>
#     <td>0</td>
#     <td>T-shirt/top</td> 
#   </tr>
#   <tr>
#     <td>1</td>
#     <td>Trouser</td> 
#   </tr>
#     <tr>
#     <td>2</td>
#     <td>Pullover</td> 
#   </tr>
#     <tr>
#     <td>3</td>
#     <td>Dress</td> 
#   </tr>
#     <tr>
#     <td>4</td>
#     <td>Coat</td> 
#   </tr>
#     <tr>
#     <td>5</td>
#     <td>Sandal</td> 
#   </tr>
#     <tr>
#     <td>6</td>
#     <td>Shirt</td> 
#   </tr>
#     <tr>
#     <td>7</td>
#     <td>Sneaker</td> 
#   </tr>
#     <tr>
#     <td>8</td>
#     <td>Bag</td> 
#   </tr>
#     <tr>
#     <td>9</td>
#     <td>Ankle boot</td> 
#   </tr>
# </table>
# 
# 每幅图像都对应一个标签。因此类别 *class name* 没有包含在在数据集中，在这里我们直接进行存储，然后在绘制图像的时候使用：

# %%
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# %% [markdown]
# ## 探索数据
# 
# 在训练模型之前，我们先看看数据集的格式。以下代码会显示训练集中有 6 万张图片，每张图片大小是 28x28 像素：

# %%
train_images.shape

# %% [markdown]
# 同样地，训练集中有 6 万个标签：

# %%
len(train_labels)

# %% [markdown]
# 每个标签都是 0-9 的数字：

# %%
train_labels

# %% [markdown]
# 测试集中有 1 万幅图像，每幅图像大小也是 28x28 像素：

# %%
test_images.shape

# %% [markdown]
# 同样地，测试集包括 1 万个标签：

# %%
len(test_labels)

# %% [markdown]
# ## 数据预处理
# 
# 在训练网络之前，必须对数据进行预处理。如果你检查训练集的第一幅图像，你会发现像素值是 0-255 的数：

# %%
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)

# %% [markdown]
# 在将数据送入神经网络模型前，先将值缩放到 0-1 之间。将图像的数据类型从整型改为浮点，并除以 255。以下为预处理过程：
# %% [markdown]
# 请注意，必须同时对**训练集**与**测试集**都以相同的方式进行预处理：

# %%
train_images = train_images / 255.0

test_images = test_images / 255.0

# %% [markdown]
# 展示**训练集**中的前 25 幅图像，并在每幅图像下显示它对应的标签名称。此步骤的目的是验证数据格式是否正确。下面我们将构建并训练网络。

# %%
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

# %% [markdown]
# ## 构建模型
# 
# 构建神经网络需要配置好模型的各层参数，并编译模型。
# %% [markdown]
# ### 设置层
# 
# 神经网络基本的构建单元称之为**层**。各层会提炼输入数据的特征。我们希望这些特征对于要解决的问题是有意义的。
# 
# 多数的深度学习模型是将一些简单层进行堆叠。很多的层都事类似于全连接层 `tf.keras.layers.Dense` 的，另外还包括一些训练时的参数。

# %%
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# %% [markdown]
# 该网络的第一层是 `tf.keras.layers.Flatten`，可以将图片数据从 28 x 28 像素的 2d 数组转换为 28 x 28 = 784 像素的 1d 数组。可以把这一层看做是把图像每行的像素都抽出来再连在一起。这一层中没有参数需要学习，仅用于转换数据。
# 
# 在像素被展平（flatten）后，网络紧接着 2 个 `tf.keras.layers.Dense` 层。它也被称为密集层或者全连接层神经元。第一个 `Dense` 层有着 128 个节点（或称为神经元）。第二层有 10 个节点，作为 *softmax* 层，它会返回由和为 1 的 10 个概率值组成的数组。每个节点给的值就是当前图像属于 10 个类别中的那一类的概率值。
# 
# ### 编译模型
# 
# 在模型可以训练前，还要做一些设置。下面的配置可以在模型**编译**步骤中添加进网络：
# 
# * *Loss function* —损失函数，它用于在训练时度量模型的准确度。我们需要模型将这个函数值最小化，从而“引导”模型向正确的方向训练。
# * *Optimizer* —优化器，它决定了模型如何按照数据和损失函数进行参数更新。
# * *Metrics* —指标，用于在训练和测试步骤中监视模型状态。下面例子中我们会用 *accuracy* 作为指标，它由正确分类图片数除总图片数得出。

# %%
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# %% [markdown]
# ## 训练模型
# 
# 训练神经网络模型需要以下几个步骤：
# 
# 1. 将训练数据送入模型。在本例中，我们将 `train_images` 与 `train_labels` 数组传入。
# 2. 模型根据传入的图像和对应的标签进行学习
# 3. 让模型在测试集上进行预测，在本例中，我们将 `test_images` 数组传入进行预测。预测完后检查预测结果是否匹配 `test_labels` 数组中的标签。
# 
# 调用 `model.fit` 方法就能开始训练。模型将对训练数据进行“fit（拟合）”：

# %%
model.fit(train_images, train_labels, epochs=5)

# %% [markdown]
# 在模型训练时，loss 与准确率会实时展示。这个模型可以在训练集上到达约 0.88（或 88%）的准确率。
# %% [markdown]
# ## 评价准确率
# 
# 下面，比较模型在测试集上的表现：

# %%
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

# %% [markdown]
# 可以得到结果，在测试集上得到的准确率要比训练集上得到的准确率略低。训练与测试准确率存在差距是一种**过拟合**的形式。过拟合是一种机器学习模型在预测新数据时得到的结果比训练时结果差的现象。
# %% [markdown]
# ## 进行预测
# 
# 在模型训练完成后，我们可以用它来预测别的图片。

# %%
predictions = model.predict(test_images)

# %% [markdown]
# 在这而，模型对测试集中的每张图片都进行预测。让我们观察一下第一张图片的预测结果：

# %%
predictions[0]

# %% [markdown]
# 每个预测结果都是一个长度为 10 的数组。这些数字描述了模型判断图片属于 10 个不同类型的服装的“可信度”。我们可以用以下方式来看到可信度最高的值：

# %%
np.argmax(predictions[0])

# %% [markdown]
# 可以看到模型认为图片是短靴 ankle boot（`class_names[9]`）的可信度最高。然后我们检查测试集标签查看正确结果：

# %%
test_labels[0]

# %% [markdown]
# 我们可以进行如下可视化，展示所有 10 类的情况：

# %%
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# %% [markdown]
# 让我们观察第 1 张图片，预测和预测数组：

# %%
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)


# %%
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)

# %% [markdown]
# 让我们将图片和它们各自的预测结果展示出来。正确的预测标签用蓝色标记，错误的用红色标记。同时将预测时得到的百分比展示出来。请注意，即使这个百分比很大，看起来很可信，但结果也有可能是错的。

# %%
# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)

# %% [markdown]
# 最后，用训练好的的模型来对单独一张图片进行预测。

# %%
# Grab an image from the test dataset
img = test_images[0]

print(img.shape)

# %% [markdown]
# `tf.keras` 模块专门为对 *batch*，或者说一组数据进行预测而优化的。因此即使我们只处理一张图片，也需要将它放到 list 中去：

# %%
# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)

# %% [markdown]
# 现在来预测图片：

# %%
predictions_single = model.predict(img)

print(predictions_single)


# %%
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

# %% [markdown]
# `model.predict` 会返回一个 2d 列表，分别与 batch 中的图片一一对应。取出我们在这个 batch 中的唯一一张图片：

# %%
np.argmax(predictions_single[0])

# %% [markdown]
# 如上所示，模型预测分类结果为 9。
