# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # TensorFlow2教程-不规则向量
# 

# %%
from __future__ import absolute_import, division, print_function, unicode_literals

import math
import tensorflow as tf

# %% [markdown]
# 数据有多种形式。张量也应该如此。 不规则的张量是嵌套可变长度列表的TensorFlow等效项。它们使存储和处理形状不均匀的数据变得容易，包括：
# 
# - 可变长度功能，例如电影中的一组演员。
# - 成批的可变长度顺序输入，例如句子或视频剪辑。
# - 分层输入，例如细分为小节，段落，句子和单词的文本文档。
# - 结构化输入中的各个字段，例如协议缓冲区。
# 
# %% [markdown]
# ## 1 不规则张量的作用
# 不规则张量受一百多个TensorFlow操作的支持，其中包括数学操作（如tf.add和tf.reduce_mean），数组操作（如tf.concat和tf.tile），字符串操作op（如 tf.substr）以及许多其他功能：

# %%
digits = tf.ragged.constant([[3,1,4,1], [], [5,9,2], [6],[]])
words = tf.ragged.constant([["So", "long"], ["thanks", "for", "all", "the", "fish"]])
print(tf.add(digits, 3))
print(tf.reduce_mean(digits, axis=1))
print(tf.concat([digits, [[5, 3]]], axis=0))
print(tf.tile(digits, [1, 2]))
print(tf.strings.substr(words, 0, 2))


# %%


