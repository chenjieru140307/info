# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # TensorFlow2教程-掩码和填充
# %% [markdown]
# 在构建深度学习模型，特别是进行序列的相关任务时，经常会用到掩码和填充技术

# %%
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import tensorflow as tf

from tensorflow.keras import layers

# %% [markdown]
# ## 1 填充序列数据
# 处理序列数据时，常常会遇到不同长度的文本，例如处理某些句子时：
# 
# 

# %%
[
  ["The", "weather", "will", "be", "nice", "tomorrow"],
  ["How", "are", "you", "doing", "today"],
  ["Hello", "world", "!"]
]

# %% [markdown]
# 计算机无法理解字符，我们一般将词语转换为对应的id

# %%
[
  [83, 91, 1, 645, 1253, 927],
  [73, 8, 3215, 55, 927],
  [71, 1331, 4231]
]

# %% [markdown]
# 由于深度信息模型的输入一般是固定的张量，所以我们需要对短文本进行填充（或对长文本进行截断），使每个样本的长度相同。
# Keras提供了一个API，可以通过填充和截断，获取等长的数据：
# tf.keras.preprocessing.sequence.pad_sequences

# %%
raw_inputs = [
  [83, 91, 1, 645, 1253, 927],
  [73, 8, 3215, 55, 927],
  [711, 632, 71]
]
# 默认左填充
padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(raw_inputs)
print(padded_inputs)
# 右填充需要设 padding='post'
padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(raw_inputs,
                                                             padding='post')
print(padded_inputs)

# %% [markdown]
# ## 2 掩码
# 现在所有样本都获得了同样的长度，在求损失函数或输出的时候往往需要知道那些部分是填充的，需要忽略。这种忽略机制就是掩码。
# 
# keras中三种添加掩码的方法：
# - 添加一个keras.layer.Masking的网络层
# - 配置keras.layers.Embedding层的mask_zero=True
# - 在支持mark的网络层中，手动传递参数
# 
# %% [markdown]
# ### 2.1 掩码生成层：Embedding和Masking
# Embedding和Masking会创建一个掩码张量，附加到返回的输出中。

# %%
embedding = layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True)
mask_output= embedding(padded_inputs)
print(mask_output._keras_mask)


# %%
# 使用掩码层
masking_layer = layers.Masking()
unmasked_embedding = tf.cast(
tf.tile(tf.expand_dims(padded_inputs, axis=-1), [1, 1, 16]),
tf.float32)
masked_embedding = masking_layer(unmasked_embedding)
print(masked_embedding._keras_mask)

# %% [markdown]
# mark是带有2D布尔张量(batch_size, sequence_length)，其中每个单独的False指示在处理过程中应忽略相应的时间步数据。
# %% [markdown]
# ### 2.2 函数API和序列API中使用掩码
# 
# 使用函数式API或序列API时，由Embedding或Masking层生成的掩码将通过网络传播到能够使用它们的任何层（例如RNN层）。Keras将自动获取与输入相对应的掩码，并将其传递给知道如何使用该掩码的任何层。
# 
# 请注意，在子类化模型或图层的call方法中，掩码不会自动传播，因此将需要手动将mask参数传递给需要的图层。
# 
# 下面展示了LSTM层怎么自动接受掩码

# %%
# 序列式API
model = tf.keras.Sequential([
  layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True),
  layers.LSTM(32),
])


# %%
# 函数式API
inputs = tf.keras.Input(shape=(None,), dtype='int32')
x = layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True)(inputs)
outputs = layers.LSTM(32)(x)

model = tf.keras.Model(inputs, outputs)

# %% [markdown]
# #### 也可以直接在层间传递掩码参数
# 
# 可以处理掩码的图层（例如LSTM图层）在其图层中的__call__方法中有一个mask参数。
# 
# 同时，产生掩码的图层（例如Embedding）会公开compute_mask(input, previous_mask)，以获取掩码。
# 
# 因此，可以执行以下操作：

# %%
class MyLayer(layers.Layer):
  
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        self.embedding = layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True)
        self.lstm = layers.LSTM(32)

    def call(self, inputs):
        x = self.embedding(inputs)
        # 也可手动构造掩码
        mask = self.embedding.compute_mask(inputs)
        output = self.lstm(x, mask=mask)  # The layer will ignore the masked values
        return output

layer = MyLayer()
x = np.random.random((32, 10)) * 100
x = x.astype('int32')
layer(x)

# %% [markdown]
# ### 在自定义网络层中支持掩码
# 有时，可能需要编写生成掩码的图层（例如Embedding），或需要修改当前掩码的图层。
# 
# 例如，产生张量的时间维度与其输入不同的任何层，都需要修改当前掩码，以便下游层能够适当考虑掩码的时间步长。
# 
# 为此，自建网络层应实现layer.compute_mask()方法，该方法将根据输入和当前掩码生成一个新的掩码。
# 
# 大多数图层都不会修改时间维度，因此无需担心掩盖。compute_mask()在这种情况下，默认行为是仅通过当前蒙版。
# 
# #### TemporalSplit修改当前蒙版的图层的示例。
# 
# 

# %%
# 数据拆分时的掩码
class TemporalSplit(tf.keras.layers.Layer):

    def call(self, inputs):
        # 将数据沿时间维度一分为二
        return tf.split(inputs, 2, axis=1)

    def compute_mask(self, inputs, mask=None):
        # 将掩码也一分为二
        if mask is None:
            return None
        return tf.split(mask, 2, axis=1)

first_half, second_half = TemporalSplit()(masked_embedding)
print(first_half._keras_mask)
print(second_half._keras_mask)

# %% [markdown]
# 自己构建掩码，把为０的掩掉

# %%
class CustomEmbedding(tf.keras.layers.Layer):
  
    def __init__(self, input_dim, output_dim, mask_zero=False, **kwargs):
        super(CustomEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mask_zero = mask_zero

    def build(self, input_shape):
        self.embeddings = self.add_weight(
        shape=(self.input_dim, self.output_dim),
        initializer='random_normal',
        dtype='float32')

    def call(self, inputs):
        return tf.nn.embedding_lookup(self.embeddings, inputs)

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        return tf.not_equal(inputs, 0)

layer = CustomEmbedding(10, 32, mask_zero=True)
x = np.random.random((3, 10)) * 9
x = x.astype('int32')

y = layer(x)
mask = layer.compute_mask(x)

print(mask)

# %% [markdown]
# 直接用网络层实现掩码功能：call函数中接受一个mask参数，并用它来确定是否跳过某些时间步。
# 
# 要编写这样的层，只需在call函数中添加一个mask=None参数即可。只要有可用的输入，与输入关联的掩码将被传递到图层。

# %%
class MaskConsumer(tf.keras.layers.Layer):
  
    def call(self, inputs, mask=None):
        pass


# %%


