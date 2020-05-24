# 保存与恢复模型

[Colab notebook](https://colab.research.google.com/github/tensorflow/models/blob/master/samples/core/tutorials/keras/save_and_restore_models.ipynb)

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
# # 保存和恢复模型
# %% [markdown]
# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/tutorials/keras/save_and_restore_models"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/save_and_restore_models.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/save_and_restore_models.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
# </table>
# %% [markdown]
# 模型进度可以在训练期间和训练后保存。这意味着模型可以从中断的地方恢复，从而避免长时间的训练。模型保存也意味着你可以共享你的模型，并且其他人可以复现你的工作。在公开研究模型和技术时，大多数机器学习从业者分享以下内容：
# 
# * 创建模型的代码
# * 模型训练的权重或者参数
# 
# 共享此数据有助于其他人了解模型的工作原理，并使用新数据自行尝试。
# 
# 注意：小心不受信任的 TensorFlow 模型代码。详见[安全使用 TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md)。
# 
# ### 选项
# 
# 保存 TensorFlow 模型有多种方法，取决于你使用的 API。[tf.keras](https://www.tensorflow.org/guide/keras) 使用高级 API 构建和保存模型。有关其他方法，请参阅 TensorFlow [保存与恢复模型](https://www.tensorflow.org/guide/saved_model)指南或者[使用 eager 保存模型](https://www.tensorflow.org/guide/eager#object_based_saving)。
# 
# %% [markdown]
# ## 设置
# 
# ### 安装与引用
# %% [markdown]
# 安装以及引用 TensorFlow 以及相关依赖：

# %%
get_ipython().system('pip install h5py pyyaml ')

# %% [markdown]
# ### 获取示例数据
# 
# 我们将使用 [MNIST 数据集](http://yann.lecun.com/exdb/mnist/)来训练我们的模型并演示保存权重。为了加速示例运行，请仅使用前 1000 个样本：

# %%
from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

tf.__version__


# %%
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# %% [markdown]
# ### 定义模型
# %% [markdown]
# 让我们构建一个简单的模型，我们将用它来演示保存和加载权重。

# %%
# 返回简短的序列化模型
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=tf.nn.softmax)
  ])
  
  model.compile(optimizer=tf.keras.optimizers.Adam(), 
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])
  
  return model


# 创建模型实例
model = create_model()
model.summary()

# %% [markdown]
# ## 训练中保存检查点
# %% [markdown]
# 主要用例是在训练的<b>过程中</b>和<b>结束时</b>自动保存检查点。通过这种方式，你可以使用已训练好的模型，无需重新训练，或者在模型中断的地方继续训练。
# 
# `tf.keras.callbacks.ModelCheckpoint` 执行此任务的回调。回调需要多个参数来配置检查点。
# 
# ### 检查点回调使用方法
# 
# 训练模型并传递 `ModelCheckpoint` 回调：

# %%
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 创建检查点回调
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1)

model = create_model()

model.fit(train_images, train_labels,  epochs = 10, 
          validation_data = (test_images,test_labels),
          callbacks = [cp_callback])  # pass callback to training

# %% [markdown]
# 这将创建一个 TensorFlow 检查点文件集合，这些文件在每个 epoch 结束时更新：

# %%
get_ipython().system('ls {checkpoint_dir}')

# %% [markdown]
# 创建一个新的未经训练的模型。当仅恢复模型的权重时，该模型必须与原始模型具有相同体系结构。由于它们具有相同的模型架构，我们可以共享权重，尽管它是与原模型的不同<b>实例</b>。
# 
# 现在重建一个新的未经训练的模型，并在测试集上进行评估。未经训练的模型表现一般（准确度约为 10％）：

# %%
model = create_model()

loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

# %% [markdown]
# 然后从检查点加载权重，并重新评估：

# %%
model.load_weights(checkpoint_path)
loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# %% [markdown]
# ### 检查点回调选项
# 
# 回调提供了几个选项，可以为生成的检查点提供唯一的名称，并调整检查点频率。
# 
# 训练一个新模型，每 5 个 epoch 保存一次唯一命名的检查点：

# %%
# 文件名包含 epoch 计数。（使用 `str.format`）
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5)

model = create_model()
model.fit(train_images, train_labels,
          epochs = 50, callbacks = [cp_callback],
          validation_data = (test_images,test_labels),
          verbose=0)

# %% [markdown]
# 现在，看看生成的检查点（按修改日期排序）：

# %%
import pathlib

# Sort the checkpoints by modification time.
checkpoints = pathlib.Path(checkpoint_dir).glob("*.index")
checkpoints = sorted(checkpoints, key=lambda cp:cp.stat().st_mtime)
checkpoints = [cp.with_suffix('') for cp in checkpoints]
latest = str(checkpoints[-1])
checkpoints

# %% [markdown]
# 注意：TensorFlow 默认仅保存最近的 5 个检查点。
# 
# 为了测试，请重置模型并加载最新的检查点：

# %%
model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# %% [markdown]
# ## 这些文件是什么？
# %% [markdown]
# 上述代码将权重存储到[检查点](https://www.tensorflow.org/guide/saved_model#save_and_restore_variables)格式化文件的集合中，这些文件仅包含二进制格式的训练权重。检查点包含：
# * 包含模型权重的一个或多个分片。 
# * 一个索引文件，指示权重存储在分片中的位置。  
# 
# 如果你只在一台机器上训练模型，那么你将有一个带有后缀的分片：`.data-00000-of-00001`
# %% [markdown]
# ## 手动保存权重
# 
# 上面你看到了如何将权重加载到模型中。
# 
# 手动保存权重同样简单，使用 `Model.save_weights` 方法。

# %%
# 保存权重
model.save_weights('./checkpoints/my_checkpoint')

# 恢复权重
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')

loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# %% [markdown]
# ## 保存整个模型
# 
# 整个模型可以保存到文件中，包含权重值，模型配置甚至优化器配置。这允许你检查模型并稍后从完全相同的状态恢复训练而无需运行原始代码。
# 
# 在 Keras 中保存一个功能齐全的模型非常有用，你可以在浏览器中使用 [TensorFlow.js](https://js.tensorflow.org/tutorials/import-keras.html) 加载模型，训练并运行。
# 
# Keras 提供了模型保存的标准文件格式 [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format)。出于我们的目的，可以将保存的模型视为单个二进制 blob。

# %%
model = create_model()

model.fit(train_images, train_labels, epochs=5)

# Save entire model to a HDF5 file
model.save('my_model.h5')

# %% [markdown]
# 现在从该文件重新创建模型：

# %%
# 重新构建整个模型，包含权重和优化器。
new_model = keras.models.load_model('my_model.h5')
new_model.summary()

# %% [markdown]
# 检查准确率：

# %%
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# %% [markdown]
# 该方法保存所有配置：
# 
# * 权重
# * 模型配置（架构）
# * 优化器配置
# 
# Keras 通过检查架构来保存模型。目前，它无法保存 TensorFlow 优化器（来自 `tf.train`）。使用这些时，你需要在加载后重新编译模型，并且你将失去优化器的状态。
# %% [markdown]
# ## 下一步
# 
# 本篇是使用 `tf.keras` 保存和加载的快速指南。
# 
# * [tf.keras 指南](https://www.tensorflow.org/guide/keras)介绍了使用 `tf.keras` 保存和加载模型。
# 
# * [在 eager 中保存模型](https://www.tensorflow.org/guide/eager#object_based_saving)介绍了在 eager execution 中保存模型。
# 
# * [保存与恢复](https://www.tensorflow.org/guide/saved_model)介绍了有关于 TensorFlow 保存的低级细节。
