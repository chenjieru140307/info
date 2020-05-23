# 基础文本类

[Colab notebook](https://colab.research.google.com/github/tensorflow/models/blob/master/samples/core/tutorials/keras/basic_text_classification.ipynb)

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
# # 电影评论的文本分类
# %% [markdown]
# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/tutorials/keras/basic_text_classification"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_text_classification.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_text_classification.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
# </table>
# %% [markdown]
# 
# 在这个任务中，我们将把电影评论分为**积极**和**消极**两种，即是一个**二分类**任务，这是一个非常重要并且已经被广泛应用的机器学习问题。
# 
# 我们将使用 [IMDB 数据集](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb)，其中包括了 50000 条来自 [Internet Movie Database](https://www.imdb.com/) 的电影评论。这些评论被等分成两份，分别用于训练和测试，也就是说，训练集和测试集的样本是**平衡**的（样本数量相等）。
# 
# 接下来的代码中，我们会使用一个用于创建和训练 TensorFlow 模型的高级 API —— [tf.keras](https://www.tensorflow.org/guide/keras)。如果你希望查看 `tf.keras` 进阶版的文本分类教程，请查看 [MLCC Text Classification Guide](https://developers.google.com/machine-learning/guides/text-classification/)。

# %%
import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)

# %% [markdown]
# ## 下载 IMDB 数据集
# 
# 
# IMDB 数据集随 TensorFlow 附带，并且已经被预处理过：单词序列已经被转换成整数序列，并且每个整数对应字典中特定的一个单词。
# 
# 下面的代码将 IMDB 数据集下载到你的电脑上（如果已经下载过的话，将会使用先前的缓存）：
# 

# %%
imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# %% [markdown]
# 
# 参数 `num_words=10000` 将会保留训练集中出现频率最高的 10000 个单词，其他不怎么出现的单词将会被忽略以使得数据的大小便于训练。
# %% [markdown]
# ## 探索数据
# 
# 让我们先来看看数据的格式。数据集已经被预处理过了，其中：每个电影评论样本（一连串的单词）由一个整数数组所表示，数组中的每个整数代表一个单词。每个评论的标签是一个 0 或者 1 的整数，其中 0 代表消极的评论，1 代表积极的评论。

# %%
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# %% [markdown]
# 评论文本已经被转换成了整数数组，其中每个整数代表一个单词。其中第一个评论长这样：

# %%
print(train_data[0])

# %% [markdown]
# 电影评论的长度可能不同，下面的代码展示了第一条和第二条评论的长度。因为神经网络的输入必须是要相同长度的，所以我们接下来将会解决这个问题。

# %%
len(train_data[0]), len(train_data[1])

# %% [markdown]
# ### 将整数转换回文本
# 
# 了解如何将整数数组转换回文本是很重要的，接下来，我们将创建一个帮助函数来构建一个字典，以完成这个任务。

# %%
# 一个将单词映射成整数的字典
word_index = imdb.get_word_index()

# 前四个整数（0，1，2，3）被保留具有特殊的含义
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # 表示未知的单词
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# %% [markdown]
# 接下来我们可以使用 `decode_review` 函数来还原第一条评论：

# %%
decode_review(train_data[0])

# %% [markdown]
# ## 准备数据
# 
# 电影评论——整数数组——在喂给神经网络之前，需要被转换成 tensor 对象。我们有以下几种方式来进行这一转换：
# 
# * 将数组进行 One-hot 编码，以构成由 0 和 1 组成的向量。比如，序列 [3, 5] 会变成一个 10000 维的向量，其中除了下标 3 和 5 的位置上是 1，其他都是 0。然后，构建一个能够处理这样输入的神经网络层——一个全连接层。因为我们需要一个 `单词数量 * 评论数量` 这么大的矩阵，所以这种方式是很耗费内存的。
# 
# 
# * 我们也可以将整数数组填充（pad）成相同的长度，然后创建一个形为 `最长长度 * 评论数量` 的张量。我们可以使用 embedding 层来处理这样的输入。
# 
# 在这个教程中，我们将使用第二种方法。
# 
# 
# 我们会使用 [pad_sequences](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences) 来将评论填充成相同的长度：

# %%
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

# %% [markdown]
# 再来看看评论的长度：

# %%
len(train_data[0]), len(train_data[1])

# %% [markdown]
# 以及检查一下经过填充后的第一条评论：

# %%
print(train_data[0])

# %% [markdown]
# ## 构建模型
# 
# 神经网络由很多的层堆叠而成——所以我们需要思考两个架构问题：
# * 需要在模型中使用多少层？
# * 每一层的**隐藏单元**数量是多少？
# 
# 在我们这个问题中，输入的数据是一个整数数组，需要预测的标签是 0 或者 1，接下来，一起来构建这个模型：

# %%
# 输入的形状是词表大小
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

# %% [markdown]
# 我们的分类器由上面的几个网络层按顺序堆叠而成：
# 
# 1. 第一层是 `Embedding` （词嵌入）层。这一层会把整数编码的词表作为输入，并且把每个整数（即每个单词）映射到一个词向量。这些词向量会随着模型的训练逐渐被习得。这个向量将会在输出的数组上增加一个维度，因此得到的输出结果维度为 `(批量大小, 序列长度, embedding 大小)`
# 2. 接下来，`GlobalAveragePooling1D` 层对每个评论中的所有词向量求平均，得到一个固定长度的输出向量。这是处理不同长度的输入最简单的一种方法。
# 3. 得到的固定长度的输出向量将会经过一个全连接层（`Dense`），其中隐藏单元的数量为 16。
# 4. 最后一层将输入的向量映射为一个实数，并且通过使用 `sigmoid` 激活函数将其值变化到 0 和 1 之间，以表示概率。
# %% [markdown]
# ### 隐藏单元
# 
# 
# 上面的模型在输入和输出之间有两个中间层，或者叫“隐藏”层。输出向量的维度（单位，节点或神经元）是网络层的表示空间的维度。 换句话说，是网络在学习内部表示时所具有的自由度。
# 
# 
# 如果一个模型具有更多的隐藏单元（更高维的表示空间），或者有更多的层，那么这个网络可以学习到更加复杂的表示。但是，这也会增大网络的计算开销并且可能会导致其习得不应该习得的特征（可以视作是一种噪声）——能够提升在训练数据上的性能但是并不能在测试集上表现良好，这种现象叫做**过拟合**，我们会在后面继续讨论这一问题。
# %% [markdown]
# ### 损失函数和优化器
# 
# 
# 我们需要定义一个损失函数以及优化器用于训练神经网络模型。考虑到这是一个二分类问题，并且模型的输出是一个概率值，我们将使用 `binary_crossentropy`（二分类交叉熵） 损失函数。
# 
# 
# 除此以外，我们还可以选择 `mean_squared_error`（均方误差）作为损失函数。但是一般来说，`binary_crossentropy` 更适合输出为概率的问题：因为它能够衡量概率分布之间的距离。在我们的例子中，它可以衡量真实分布和预测分布之间的距离。
# 
# 随后，当我们探索回归问题的时候（比如说，预测一栋房子的价格），我们将会均方误差作为损失函数。
# 
# 
# 接下来，根据我们的损失函数和优化器来配置模型：

# %%
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# %% [markdown]
# ## 构建验证集
# 
# 训练的时候，我们想在模型为见过的数据上检验其准确率。通过从训练中分出 10000 个样本，我们创建了一个**验证集**。你可能会问为什么不用测试集呢？因为我们的目标是仅仅根据训练集的数据来构建和调整我们的模型，之后才会使用测试集来评估模型的准确率。

# %%
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# %% [markdown]
# ## 训练模型
# 
# 我们用每批 512 个样本来训练模型 40 轮，也就是说在所有的 `x_train` 和 `y_train` 样本上迭代 40 次。训练的时候，我们会监控模型的损失函数以及验证集上的准确率：

# %%
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# %% [markdown]
# ## 评估模型
# 
# 让我们看看模型最终表现的怎么样，我们将得到两个指标：loss（代表模型的错误，其值越低越好）以及准确率。

# %%
results = model.evaluate(test_data, test_labels)

print(results)

# %% [markdown]
# 这个非常简单的方法能够取得大约 87% 的准确率，如果使用更加先进的方法，我们将能够取得接近 95% 的准确率。
# %% [markdown]
# ## 绘制准确率和 loss 随时间变化的图表
# 
# `model.fit()` 方法会返回一个 `History` 对象，其中包括了一个带有所有训练过程中结果的字典：

# %%
history_dict = history.history
history_dict.keys()

# %% [markdown]
# 这里有四个指标，即训练过程中我们所监视的四个指标（验证集 loss，验证集准确率，训练集 loss 和 训练集准确率）。我们可以使用它们来绘制训练和验证集上的 loss 和准确率，来进行对比：

# %%
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" 代表 "蓝点"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b 代表 "蓝色实线
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# %%
plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# %% [markdown]
# 
# 在这张图中，点代表的是训练集上的指标，而实线则代表的是验证集上的 loss 和准确率。
# 
# 可以注意到，训练集上的 loss 随着训练轮数的增加在**下降**而准确率在**上升**，这正是使用梯度下降所预期的结果，因为随着迭代的进行，优化器会优化这两个指标。
# 
# 而在验证集上则不是这样，大约 20 轮之后，loss 不再快速地下降，准确率也似乎达到了顶峰。这是一个过拟合的例子：模型在训练集上表现的更好，但是在没有见过的数据上（验证集上）却并非如此。在 20 轮之后，模型过度地优化并且习得仅仅适用于训练集的**特殊表示**，而无法在测试集有良好的泛化性能。
# 
# 
# 对于这个例子，我们可以在 20 轮之后停止训练来避免过拟合。后续，你将看到如何通过一个回调函数来自动实现停止训练。
