# 基本回归模型

[Colab notebook](https://colab.research.google.com/github/tensorflow/models/blob/master/samples/core/tutorials/keras/basic_regression.ipynb)

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
# # 回归模型：房价预测
# %% [markdown]
# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/tutorials/keras/basic_regression"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_regression.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_regression.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
# </table>
# %% [markdown]
# 在一个**回归**问题中，我们希望预测一个连续的值，比如说价格或概率。而**分类**问题中，我们预测的是一个离散的标签（例如某个图片包含的是苹果还是橘子）。
# 
# 本 notebook 构建了一个模型来预测波士顿郊区在上世纪七十年代中期的房价中位数。为此，我们会给模型送入此郊区的一些特征数据，其中包括犯罪率、当地房产税税率等。
# 
# 本例使用了 `tf.keras` API，请参见[此指南](https://www.tensorflow.org/guide/keras)了解更多细节。

# %%
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)

# %% [markdown]
# ## 波士顿房价数据集
# 
# 可以在 TensorFlow 中直接访问此[数据集](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html)。通过以下方式下载及打乱训练集：

# %%
boston_housing = keras.datasets.boston_housing

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# Shuffle the training set
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

# %% [markdown]
# ### 样例与特征
# 
# 这个数据集比我们其它的数据集要小的多：它共有 506 个样例，在分割后有 404 个样例放入训练集中，有 102 个样例作为测试集：

# %%
print("Training set: {}".format(train_data.shape))  # 404 examples, 13 features
print("Testing set:  {}".format(test_data.shape))   # 102 examples, 13 features

# %% [markdown]
# 这个数据集包含了 13 个不同的特征：
# 
# 1.   人均犯罪率；
# 2.   占地面积超过 25,000 平方英尺以上住宅用地所占比例；
# 3.   城镇平均非零售商业占地面积所占比例；
# 4.   Charles 河虚拟变量 （如果地段靠近 Charles 河，则值为 1，否则为 0）；
# 5.   一氧化氮浓度（单位为千万分之一）；
# 6.   每栋住所的平均房间数；
# 7.   1940 年前建造的自住房占比；
# 8.   到 5 个波士顿工作中心的加权距离；
# 9.   辐射式高速公路的可达性指数；
# 10.  每万美元全额房产税税率；
# 11.  城镇学生-教师比例；
# 12.  1000 * (Bk - 0.63) ** 2 函数中， Bk 为城镇黑人所占比例；
# 13.  底层人口所占百分比。
# 
# 输入数据的每个特征维度都分别用不同的量纲进行存储。一些特征用 0-1 的比例来表示，还有一些特征用 1-12 的范围来表示，另外还有一些特征用 0-100 的范围表示等。这是因为它们来自于真实世界，在开发时，了解如何探索并清洗这些数据是开发中的一项重要技能。
# 
# 请注意：作为一名建模者及开发者，需要思考该如何使用这些数据，明白模型的预测会带来哪些潜在的益处或危害。一个模型可能会加大社会的不公平与偏见。一个与问题有关的特征在你手上会被用来解决不公平还是制造不公平呢？关于更多这方面的信息，请阅读：[机器学习的公平性](https://developers.google.com/machine-learning/fairness-overview/)。

# %%
print(train_data[0])  # Display sample features, notice the different scales

# %% [markdown]
# 用 [pandas](https://pandas.pydata.org) 库来对数据集的前几行进行格式优美的展示：

# %%
import pandas as pd

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']

df = pd.DataFrame(train_data, columns=column_names)
df.head()

# %% [markdown]
# ### 标签
# 
# 标签是以千美元为单位的房价。（请注意这是上世纪七十年代中期的价格。）

# %%
print(train_labels[0:10])  # Display first 10 entries

# %% [markdown]
# ## 特征标准化（Normalize）
# 
# 推荐对使用不同量纲和范围的特征进行标准化。我们对每个特征都减去各自的均值，并除以标准差（即 z-score 标准化）：

# %%
# Test data is *not* used when calculating the mean and std

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

print(train_data[0])  # First training sample, normalized

# %% [markdown]
# 虽然在没有进行特征标准化的情况下模型**可能会**收敛，但会让训练过程更加困难，并且会导致模型更加依赖于输入数据选用的单位。
# %% [markdown]
# ## 创建模型
# 
# 现在开始构建模型。我们在此处使用顺序（`Sequential`）模型，用两个全连接层作为隐藏层，并定义一个输出层，输出单个的、连续的数值。模型构建的步骤包裹在一个 `build_model` 函数中，因为稍后我们还要另外构建一个模型。

# %%
def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

model = build_model()
model.summary()

# %% [markdown]
# ## 训练模型
# 
# 将模型训练 500 个迭代，并将训练与验证准确率记录在 `history` 对象中。

# %%
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 500

# Store training stats
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()])

# %% [markdown]
# 使用存储在 `history` 对象中的状态对模型的训练过程进行可视化。我们希望用这些数据来决定模型在准确率停止提高前，何时终止训练。

# %%
import matplotlib.pyplot as plt


def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [1000$]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0, 5])

plot_history(history)

# %% [markdown]
# 根据这个图的显示，模型在大约 200 个 epoch 后提升就很小了。现在我们更新 `model.fit` 方法，让模型在验证评分不再提升时自动停止训练。我们将在每个迭代中使用 *callback* 来测试训练条件。如果在一系列迭代中都不再有提升，就自动停止训练。
# 
# 你可以阅读[此指南](https://www.tensorflow.org/versions/master/api_docs/python/tf/keras/callbacks/EarlyStopping)来了解更多有关这种 callback 的信息。

# %%
model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])

plot_history(history)

# %% [markdown]
# 这个图显示了平均误差大约在 \\$2,500 美元。这个值够好吗？并不。\$2,500 美元在部分标签仅为 $15,000 的数据中并不是微不足道的误差。
# 
# 让我们看看模型在测试集上表现如何：

# %%
[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

# %% [markdown]
# ## 预测
# 
# 最后，对测试集中的一些数据预测其房价：

# %%
test_predictions = model.predict(test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [1000$]')
plt.ylabel('Predictions [1000$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])


# %%
error = test_predictions - test_labels
plt.hist(error, bins = 50)
plt.xlabel("Prediction Error [1000$]")
_ = plt.ylabel("Count")

# %% [markdown]
# ## 总结
# 
# 本 notebook 介绍了几种用于处理回归问题的技术。
# 
# * 均方差（MSE）是一种针对回归问题（区别于分类问题）通用的损失函数。
# * 与此类似，回归问题的评价指标也与分类问题不同。平均绝对误差（MAE）是针对回归问题的一种通用评价指标。
# * 当输入数据的特征有着不同范围的值时，每个特征都要独立进行缩放。
# * 如果没有足够的训练数据，使用隐藏层较少的小型网络可以避免过拟合。
# * 尽早停止训练是一种很有用的阻止过拟合的技术。
