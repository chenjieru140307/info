# 使用 Estimator 构建一个线性模型

[Colab notebook](https://colab.research.google.com/github/tensorflow/models/blob/master/samples/core/tutorials/estimators/linear.ipynb)


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

# %% [markdown]
# # 使用 Estimator 构建一个线性模型
# %% [markdown]
# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/tutorials/estimators/linear"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/estimators/linear.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/estimators/linear.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
# </table>
# %% [markdown]
# 本教程使用了 TensorFlow 的 `tf.estimator` API 解决了一个基准的二分类问题。Estimator 是 TensorFlow 具有伸缩性，并且是面向生产环境的模型。更多关于 Estimator 有关详细信息，请参阅 [Estimator 指南](https://www.tensorflow.org/guide/estimators).
# 
# ## 概述
# 
# 用包含年龄、教育程度、婚姻状况和职业（**特征**）的人口普查数据，我们将尝试预测其年收入是否会超过 5 万美元（目标的**标签**）。我们将训练一个**逻辑回归**模型，根据输入的个人信息，输出 0 到 1 之间的数字，即个人年收入超过 5 万美元的概率。
# 
# 关键点：作为建模人员和开发人员，请考虑如何使用此数据以及模型预测可能带来的潜在好处和坏处。像这样的模型可能会加剧社会偏见和阶级差距。数据中的每个特征是否都与要解决的问题相关呢？是否有些特征会引入偏差呢？更多信息，请参阅 [ML 的公平性](https://developers.google.com/machine-learning/fairness-overview/).
# 
# ## 准备
# 
# 引入 TensorFlow、特征列（feature column）和其它需要的模块：

# %%
import tensorflow as tf
import tensorflow.feature_column as fc 

import os
import sys

import matplotlib.pyplot as plt
from IPython.display import clear_output

# %% [markdown]
# 在程序运行时，启用 [eager execution](https://www.tensorflow.org/guide/eager) 来检查程序：

# %%
tf.enable_eager_execution()

# %% [markdown]
# ## 下载官方实现版本
# 
# 我们使用 TensorFlow [模型库](https://github.com/tensorflow/models/)中的 [超宽超深层模型](https://github.com/tensorflow/models/tree/master/official/wide_deep/)。下载代码，将代码根目录加入 Python 的目录变量中，然后找到 `wide_deep` 目录：

# %%
get_ipython().system(' pip install requests')
get_ipython().system(' git clone --depth 1 https://github.com/tensorflow/models')

# %% [markdown]
# 将库的根目录加入 Python 路径中：

# %%
models_path = os.path.join(os.getcwd(), 'models')

sys.path.append(models_path)

# %% [markdown]
# 下载数据集：

# %%
from official.wide_deep import census_dataset
from official.wide_deep import census_main

census_dataset.download("/tmp/census_data/")

# %% [markdown]
# ### 命令行用法
# 
# 库中包含了此模型实验的完整代码。
# 
# 为了在命令行中执行此教程的代码，请先将 tensorflow 与 model 的路径加入到你的 `PYTHONPATH` 中。

# %%
#export PYTHONPATH=${PYTHONPATH}:"$(pwd)/models"
#running from python you need to set the `os.environ` or the subprocess will not see the directory.

if "PYTHONPATH" in os.environ:
  os.environ['PYTHONPATH'] += os.pathsep +  models_path
else:
  os.environ['PYTHONPATH'] = models_path

# %% [markdown]
# 可以用 `--help` 来获取命令行参数的介绍：

# %%
get_ipython().system('python -m official.wide_deep.census_main --help')

# %% [markdown]
# 现在运行模型：

# %%
get_ipython().system('python -m official.wide_deep.census_main --model_type=wide --train_epochs=2')

# %% [markdown]
# ## 读取 U.S. Census Data 数据集
# 
# 本实例使用了 1994 年与 1995 年的 [U.S Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/Census+Income)  数据集、我们提供了一个脚本：[census_dataset.py](https://github.com/tensorflow/models/tree/master/official/wide_deep/census_dataset.py) 用于下载数据集，并进行一些数据清洗。
# 
# 由于此任务是一个**二分类问题**，因此我们将收入超过 5 万美元的人标记为 1，反之标记为 0。请参考 [census_main.py](https://github.com/tensorflow/models/tree/master/official/wide_deep/census_main.py) 中的 `input_fb` 函数。
# 
# 让我们观察数据，找出用哪一列来预测目标标签：

# %%
get_ipython().system('ls  /tmp/census_data/')


# %%
train_file = "/tmp/census_data/adult.data"
test_file = "/tmp/census_data/adult.test"

# %% [markdown]
# [pandas](https://pandas.pydata.org/) 为数据分析提供了一系列好用的工具。下面将列出收入普查数据集中的列：

# %%
import pandas

train_df = pandas.read_csv(train_file, header = None, names = census_dataset._CSV_COLUMNS)
test_df = pandas.read_csv(test_file, header = None, names = census_dataset._CSV_COLUMNS)

train_df.head()

# %% [markdown]
# 这些列可以分成两组：**类别**列与**连续数值**列：
# 
# * 如果某一列的值都是属于有限的类别中的一种，那么它就是**类别**列。例如，一个人的婚姻状态（妻子、丈夫、未婚等）或者教育层次（高中、大学等）就是类别列。
# * 如果某一列的值是属于一个连续范围内的数值，那么它就是**连续数值**列。比如，个人收入（比如 14,084美元）就是一个连续数值列。
# 
# ## 将数据转换为 Tensor
# 
# 在创建 `tf.estimator` 模型时，会使用一个**输入函数**（或 `input_fn`）来输入数据。这个构造函数将返回一个批次的、按 `(features-dict, label)` 对组合的 `tf.data.Dataset`。它不会被直接调用，只有在被传入 `tf.estimator.Estimator` 后在 `train` 或 `evaluate` 时才会运行。
# 
# 输入函数返回如下数据对：
# 
# 1. `features`：一批次数据的 dict，包含了特征名、`Tensor` 或 `SparseTensors`。
# 2. `labels`：包含一批次数据标签的 `Tensor`。
# 
# `features` 的键会用于配置模型的输入层。
# 
# 注意：输入函数会在 TensorFlow 构建图时被调用，在运行图时不会调用。它会按照 TensorFlow 图操作符的形式返回输入数据的表示。
# 
# 此外，可以通过对 `pandas.DataFrame` 切片轻松构建一个 `tf.data.Dataset`：

# %%
def easy_input_function(df, label_key, num_epochs, shuffle, batch_size):
  label = df[label_key]
  ds = tf.data.Dataset.from_tensor_slices((dict(df),label))

  if shuffle:
    ds = ds.shuffle(10000)

  ds = ds.batch(batch_size).repeat(num_epochs)

  return ds

# %% [markdown]
# 由于我们开启了 eager execution，因此很容易检查数据集的结果：

# %%
ds = easy_input_function(train_df, label_key='income_bracket', num_epochs=5, shuffle=True, batch_size=10)

for feature_batch, label_batch in ds.take(1):
  print('Some feature keys:', list(feature_batch.keys())[:5])
  print()
  print('A batch of Ages  :', feature_batch['age'])
  print()
  print('A batch of Labels:', label_batch )

# %% [markdown]
# 但这种方法在大规模应用上有一些局限性。大数据集可能需要从磁盘上流式读取。`census_dataset.input_fn` 函数提供了一个样例，告诉你如何使用 `tf.decode_csv` 与 `tf.data.TextLineDataset` 读取数据：
# 
# <!-- TODO(markdaoust): This `input_fn` should use `tf.contrib.data.make_csv_dataset` -->

# %%
import inspect
print(inspect.getsource(census_dataset.input_fn))

# %% [markdown]
# 这个 `input_fn` 会返回等效于以下内容的输出：

# %%
ds = census_dataset.input_fn(train_file, num_epochs=5, shuffle=True, batch_size=10)

for feature_batch, label_batch in ds.take(1):
  print('Feature keys:', list(feature_batch.keys())[:5])
  print()
  print('Age batch   :', feature_batch['age'])
  print()
  print('Label batch :', label_batch )

# %% [markdown]
# 由于 `Estimators` 只接受没有参数的 `input_fn`，因此我们可以将输入函数包装在一个对象中。在本例中，我们配置 `train_inpf` 会遍历数据两次：

# %%
import functools

train_inpf = functools.partial(census_dataset.input_fn, train_file, num_epochs=2, shuffle=True, batch_size=64)
test_inpf = functools.partial(census_dataset.input_fn, test_file, num_epochs=1, shuffle=False, batch_size=64)

# %% [markdown]
# ## 为模型进行特征选择与特征工程
# 
# Estimator 使用名为 [特征列（feature columns）](https://www.tensorflow.org/guide/feature_columns)  的系统来向模型描述如何去解析每个原始输入特征。Estimator 需要输入的是数值向量，而特征列会告诉模型如何将每个特征转换为数值。
# 
# 选择和制作正确的特征列是学习有效模型的关键。一个**特征列**可以是原始数据 `dict` 中的原始特征（这种称为**基本特征列**），也可以是在一个或多个基本特征列上定义的转换创建的新列（称为**派生特征列**）。
# 
# 特征列可以是任何原始数据或从原始数据派生出的新数据的抽象概念，我们正是用它来预测目标标签。
# %% [markdown]
# ### 基本特征列
# %% [markdown]
# #### 数值列
# 
# 最简单的特征列 `feature_column` 就是数值列 `numeric_column` 了。数值列表明直接将数据中的数值特征送入模型。比如：

# %%
age = fc.numeric_column('age')

# %% [markdown]
# 模型将会按照特征列 `feature_column` 的定义来构建模型输入。你可以通过 `input_layer` 函数来检查其输入：

# %%
fc.input_layer(feature_batch, [age]).numpy()

# %% [markdown]
# 下列代码将会仅使用 `age` 特征训练并评价模型：

# %%
classifier = tf.estimator.LinearClassifier(feature_columns=[age])
classifier.train(train_inpf)
result = classifier.evaluate(test_inpf)

clear_output()  # used for display in notebook
print(result)

# %% [markdown]
# 与此类似，我们可以为需要在模型中用到的每个连续特征列都定义一个数值列 `NumericColumn`：

# %%
education_num = tf.feature_column.numeric_column('education_num')
capital_gain = tf.feature_column.numeric_column('capital_gain')
capital_loss = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')

my_numeric_columns = [age,education_num, capital_gain, capital_loss, hours_per_week]

fc.input_layer(feature_batch, my_numeric_columns).numpy()

# %% [markdown]
# 将 `feature_columns` 改成上述的构造函数，你就能得到对应的模型：

# %%
classifier = tf.estimator.LinearClassifier(feature_columns=my_numeric_columns)
classifier.train(train_inpf)

result = classifier.evaluate(test_inpf)

clear_output()

for key,value in sorted(result.items()):
  print('%s: %s' % (key, value))

# %% [markdown]
# #### 类别列
# 
# 用于为类别特征定义一个特征列。可以用 `tf.feature_column.categorical_column*` 中的任意一个函数来创建一个类别列 `CategoricalColumn`。
# 
# 如果你知道某一列特征中可能包含的值很少，可以使用 `categorical_column_with_vocabulary_list` 进行转换。它会从 0 开始为列表中的每一个值都自动递增地指定一个 ID。比如，`relationship` 列中将 `Husband` 设为了 0，`Not-in-family` 设为了 1。

# %%
relationship = fc.categorical_column_with_vocabulary_list(
    'relationship',
    ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'])

# %% [markdown]
# 它将根据原输入特征创建一个稀疏的 one-hot 向量。
# 
# 我们用到的 `input_layer` 函数时为 DNN 模型设计的，它期望得到稠密的输入。为了演示类别列，我们需要用 `tf.feature_column.indicator_column` 进行包装，以创建稠密的 one-hot 输出（线性 `Estimators` 可以跳过稠密化这一步）。
# 
# 请注意：另一种稀疏到稠密的可选项为：`tf.feature_column.embedding_column`。
# 
# 运行配置好 `age`、`relationship` 列的 input layer：

# %%
fc.input_layer(feature_batch, [age, fc.indicator_column(relationship)])

# %% [markdown]
# 如果我们不知道可能的值，可以用 `categorical_column_with_hash_bucket` 替代：

# %%
occupation = tf.feature_column.categorical_column_with_hash_bucket(
    'occupation', hash_bucket_size=1000)

# %% [markdown]
# 这样做，`occupation` 特征列中的每个可能的值都会在训练时通过哈希编码为一个 ID。一个样例批次中可能会有不同的职业特征：

# %%
for item in feature_batch['occupation'].numpy():
    print(item.decode())

# %% [markdown]
# 如果我们队哈希列运行 `input_layer`。可以得到形为 (batch_size, hash_bucket_size)` 的输入：

# %%
occupation_result = fc.input_layer(feature_batch, [fc.indicator_column(occupation)])

occupation_result.numpy().shape

# %% [markdown]
# 如果我们对 `hash_bucket_size` 维进行 `tf.argmax`，可以看到实际的结果。请注意任何相同的职业都会被映射到相同的伪随机 index：

# %%
tf.argmax(occupation_result, axis=1).numpy()

# %% [markdown]
# 请注意：哈希碰撞是避免不了的，但是对模型质量的影响很小。但如果使用哈希桶来压缩输入空间的话，可能会产生一定影响。请参阅 [此 notebook](https://colab.research.google.com/github/tensorflow/models/blob/master/samples/outreach/blogs/housing_prices.ipynb) 了解更多关于哈希碰撞对模型影响的可视化样例。
# 
# 毫无疑问，在定义 `SparseColumn` 时，每个特征字符串都会通过固定映射表或哈希被映射到一个 ID。在这种情况下，`LinearModel` 类会为映射创建 `tf.Variable` 来存储模型中每个特征的参数（即模型**权重**）。模型参数会在稍后模型训练过程中进行学习。
# 
# 让我们对其它类别特征也做同样的处理：

# %%
education = tf.feature_column.categorical_column_with_vocabulary_list(
    'education', [
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
        'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
        '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
    'marital_status', [
        'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
        'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

workclass = tf.feature_column.categorical_column_with_vocabulary_list(
    'workclass', [
        'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
        'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])


my_categorical_columns = [relationship, occupation, education, marital_status, workclass]

# %% [markdown]
# 最后，很容易就能使用所有的特征列来对模型进行配置：

# %%
classifier = tf.estimator.LinearClassifier(feature_columns=my_numeric_columns+my_categorical_columns)
classifier.train(train_inpf)
result = classifier.evaluate(test_inpf)

clear_output()

for key,value in sorted(result.items()):
  print('%s: %s' % (key, value))

# %% [markdown]
# ### 派生特征列
# %% [markdown]
# #### 通过 Bucketization 将连续特征转变为类别
# 
# 有时候连续特征与标签的关系是非线性的。比如，*age* 和 *income*（年龄与收入），一个人的收入可能在职业生涯的早期快速增长，在退休后则开始下降。在这种情况下，用 `age` 的原始数据作为真值特征列可能并不好，因为模型没法学习以下三种情况：
# 
# 1.  收入总是随着年龄的增长按照一定比例增长（正相关）；
# 2.  收入总是随着年龄的增长按照一定比例减少（负相关）；
# 3.  收入在任何年龄都保持不变（不相关）。
# 
# 如果你想得到收入与年龄间良好的关系，我们可以使用 *bucketization* （桶化）。Bucketization 是一种将连续特征按照一定范围截断放在一系列的桶中，然后将原始的数值特征根据数值会落入哪个桶，转换为桶 ID（类似于类别特征）。因此我们在 `age` 上定义了一个桶特征列 `bucketized_column`：

# %%
age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

# %% [markdown]
# `boundaries` 是一系列桶的边界。在本例子中，用了 10 个边界，将年龄分为了 11 个桶（17 岁及以下、18-24 岁、25-29 岁…65岁及以上）。
# 
# 通过桶化，模型可以将各个桶都转为 one-hot 特征：

# %%
fc.input_layer(feature_batch, [age, age_buckets]).numpy()

# %% [markdown]
# #### 通过交叉列来学习复杂的关系
# 
# 只用基本的特征列可能还不足以解释数据。比如，教育程序与标签（收入大于 5 万美元）可能在不同的职业中有不同的情况。因此，如果我们只学习 `education="Bachelors"` 或 `education="Masters"` 这样的单一模型权重，将无法捕获到教育与职业间的联系（比如，`education="Bachelors"` AND `occupation="Exec-managerial"` 与 `education="Bachelors" AND occupation="Craft-repair"` 这两种模式是有区别的）。
# 
# 为了将不同特征结合起来学习，我们可以向模型中加入交叉特征列 *crossed feature columns*：

# %%
education_x_occupation = tf.feature_column.crossed_column(
    ['education', 'occupation'], hash_bucket_size=1000)

# %% [markdown]
# 我们可以在两列以上的列间创建交叉列 `crossed_column`。交叉列的每个成分列都可以是一个基本特征列，比如类别（`SparseColumn`）、或者哈希后的数值特征列，或者是其它的交叉列 `CrossColumn`。比如：

# %%
age_buckets_x_education_x_occupation = tf.feature_column.crossed_column(
    [age_buckets, 'education', 'occupation'], hash_bucket_size=1000)

# %% [markdown]
# 这些交叉列通常会用哈希桶以避免类别数量过多，并且这样可以将模型权重的数量交由用户控制。
# 
# 哈希桶对交叉列的可视化样例请参阅 [此 notebook](https://colab.research.google.com/github/tensorflow/models/blob/master/samples/outreach/blogs/housing_prices.ipynb)。
# 
# %% [markdown]
# ## 定义逻辑回归模型
# 
# 在处理好输入数据与特征列后，我们可以将它们放在一起并构建一个**逻辑回归**模型。前面章节展示了几种类型的基本特征列和派生特征列，包括：
# 
# *   `CategoricalColumn`
# *   `NumericColumn`
# *   `BucketizedColumn`
# *   `CrossedColumn`
# 
# 以上所有列都是 `FeatureColumn` 类的子类，并可以通过 `feature_columns` 参数传入模型：

# %%
import tempfile

base_columns = [
    education, marital_status, relationship, workclass, occupation,
    age_buckets,
]

crossed_columns = [
    tf.feature_column.crossed_column(
        ['education', 'occupation'], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        [age_buckets, 'education', 'occupation'], hash_bucket_size=1000),
]

model = tf.estimator.LinearClassifier(
    model_dir=tempfile.mkdtemp(), 
    feature_columns=base_columns + crossed_columns,
    optimizer=tf.train.FtrlOptimizer(learning_rate=0.1))

# %% [markdown]
# 模型开始训练时，会自动学习一种偏差模式，在没有获得任何特征的情况下控制预测值。学习好的模型文件将保存在 `model_dir` 下。
# 
# ## 训练与评价模型
# 
# 在将所有特征都传入模型后，我们开始训练模型。训练模型只需要使用 `tf.estimator` API 中的一个简单命令：

# %%
train_inpf = functools.partial(census_dataset.input_fn, train_file, 
                               num_epochs=40, shuffle=True, batch_size=64)

model.train(train_inpf)

clear_output()  # used for notebook display

# %% [markdown]
# 在模型训练完成后，通过对验证数据预测的准确率来评价模型的效果：

# %%
results = model.evaluate(test_inpf)

clear_output()

for key,value in sorted(result.items()):
  print('%s: %0.2f' % (key, value))

# %% [markdown]
# 输出的第一行应该会显示 `accuracy: 0.83`，表示准确率为 83%。你可以试试更多的特征和转换方法，来试试可否得到更好的效果！
# 
# 模型在评价完成后，我们就能用它通过输入某人的信息，来预测这个人的年收入是否大于 5 万美元了。
# 
# 让我们看看模型的细节：

# %%
import numpy as np

predict_df = test_df[:20].copy()

pred_iter = model.predict(
    lambda:easy_input_function(predict_df, label_key='income_bracket',
                               num_epochs=1, shuffle=False, batch_size=10))

classes = np.array(['<=50K', '>50K'])
pred_class_id = []

for pred_dict in pred_iter:
  pred_class_id.append(pred_dict['class_ids'])

predict_df['predicted_class'] = classes[np.array(pred_class_id)]
predict_df['correct'] = predict_df['predicted_class'] == predict_df['income_bracket']

clear_output()

predict_df[['income_bracket','predicted_class', 'correct']]

# %% [markdown]
# 如需运行一个端到端的实例，请下载 [样例代码](https://github.com/tensorflow/models/tree/master/official/wide_deep/census_main.py)，并将 `model_type` 标记设为 `wide`。
# %% [markdown]
# ## 增加正则化防止过拟合
# 
# 正则化是一种用于防止过拟合的技术。当一个模型在训练数据上表现优秀，在测试数据上表现却不好时，就是发生了过拟合现象。之所以发生过拟合，是因为模型过于复杂，参数相对与训练数据来说过多。正则化可以让你控制模型的复杂度，让模型对没有训练的数据更具泛化性能。
# 
# 你可以根据以下代码对模型进行 L1 与 L2 正则：

# %%
model_l1 = tf.estimator.LinearClassifier(
    feature_columns=base_columns + crossed_columns,
    optimizer=tf.train.FtrlOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=10.0,
        l2_regularization_strength=0.0))

model_l1.train(train_inpf)

results = model_l1.evaluate(test_inpf)
clear_output()
for key in sorted(results):
  print('%s: %0.2f' % (key, results[key]))


# %%
model_l2 = tf.estimator.LinearClassifier(
    feature_columns=base_columns + crossed_columns,
    optimizer=tf.train.FtrlOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.0,
        l2_regularization_strength=10.0))

model_l2.train(train_inpf)

results = model_l2.evaluate(test_inpf)
clear_output()
for key in sorted(results):
  print('%s: %0.2f' % (key, results[key]))

# %% [markdown]
# 这种正则化后的模型并不会比基本模型效果好很多。观察模型的权重分布，来具体看看正则化的效果：

# %%
def get_flat_weights(model):
  weight_names = [
      name for name in model.get_variable_names()
      if "linear_model" in name and "Ftrl" not in name]

  weight_values = [model.get_variable_value(name) for name in weight_names]

  weights_flat = np.concatenate([item.flatten() for item in weight_values], axis=0)

  return weights_flat

weights_flat = get_flat_weights(model)
weights_flat_l1 = get_flat_weights(model_l1)
weights_flat_l2 = get_flat_weights(model_l2)

# %% [markdown]
# 模型包含了许多 0 权重，这是因为一些 hash bin 没有用到造成的（在一些列中，hash bin 的数量比分类的数量要更多）。我们可以隐去这些权重，来看看权重的分布：

# %%
weight_mask = weights_flat != 0

weights_base = weights_flat[weight_mask]
weights_l1 = weights_flat_l1[weight_mask]
weights_l2 = weights_flat_l2[weight_mask]

# %% [markdown]
# 将分布图打出来：

# %%
plt.figure()
_ = plt.hist(weights_base, bins=np.linspace(-3,3,30))
plt.title('Base Model')
plt.ylim([0,500])

plt.figure()
_ = plt.hist(weights_l1, bins=np.linspace(-3,3,30))
plt.title('L1 - Regularization')
plt.ylim([0,500])

plt.figure()
_ = plt.hist(weights_l2, bins=np.linspace(-3,3,30))
plt.title('L2 - Regularization')
_=plt.ylim([0,500])

# %% [markdown]
# 在正则限制后，一些权重更接近与 0。L2 正则可以排除一些极端的群众，效果拔群。L1 正则产生了更多的 0 值，大约将 200 列设为了 0。
