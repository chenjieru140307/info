# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # TensorFlow2教程-自定义回调
# 
# 自定义回调是一个很好用的工具，可以在训练，评估和推理期间自定义模型的行为，包括读取/更改keras模型等。

# %%
from __future__ import absolute_import, division,print_function, unicode_literals
import tensorflow as tf

# %% [markdown]
# ## 1 Keras回调简介
# 在Kreas中，Callback是一个python类，旨在被子类化以提供特定功能，并在训练的各阶段（包括每个batch/epoch的开始和结束），以及测试中调用一组方法。
# 
# 我们可以通过回调列表，传递回调方法，在训练/评估/推断的不同阶段调用回调方法。
# 
# 构建一个模型

# %%
def get_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, activation = 'linear', input_dim = 784))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.1), loss='mean_squared_error', metrics=['mae'])
    return model

# %% [markdown]
# 导入数据

# %%
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

# %% [markdown]
# 定义一个简单的自定义回调，以跟踪每批数据的开始和结束。

# %%
import datetime

class MyCustomCallback(tf.keras.callbacks.Callback):

    def on_train_batch_begin(self, batch, logs=None):
        print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

    def on_train_batch_end(self, batch, logs=None):
        print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

    def on_test_batch_begin(self, batch, logs=None):
        print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

    def on_test_batch_end(self, batch, logs=None):
        print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

# %% [markdown]
# 在训练时传入回调函数

# %%
model = get_model()
_ = model.fit(x_train, y_train,
          batch_size=64,
          epochs=1,
          steps_per_epoch=5,
          verbose=0,
          callbacks=[MyCustomCallback()])

# %% [markdown]
# ### 1.1 以下方法会调用回调函数
# fit(), fit_generator()
# 训练或使用迭代数据进行训练。
# 
# evaluate(), evaluate_generator()
# 评估或使用迭代数据进行评估。
# 
# predict(), predict_generator()
# 预测或使用迭代数据进行预测。

# %%
_ = model.evaluate(x_test, y_test, batch_size=128, verbose=0, steps=5,
          callbacks=[MyCustomCallback()])

# %% [markdown]
# ## 2 回调方法概述
# ### 2.1 训练/测试/预测的常用方法
# 为了进行训练，测试和预测，提供了以下方法来替代。
# 
# on_(train|test|predict)_begin(self, logs=None)
# 在fit/ evaluate/ predict开始时调用。
# 
# on_(train|test|predict)_end(self, logs=None)
# 在fit/ evaluate/ predict结束时调用。
# 
# on_(train|test|predict)_batch_begin(self, batch, logs=None)
# 在培训/测试/预测期间处理批次之前立即调用。在此方法中，logs是带有batch和size可用键的字典，代表当前批次号和批次大小。
# 
# on_(train|test|predict)_batch_end(self, batch, logs=None)
# 在培训/测试/预测批次结束时调用。在此方法中，logs是一个包含状态指标结果的字典。
# %% [markdown]
# ### 2.2 训练时特定方法
# 另外，为了进行培训，提供以下内容。
# 
# on_epoch_begin（self，epoch，logs = None）
# 在训练期间的开始时调用。
# 
# on_epoch_end（self，epoch，logs = None）
# 在训练期间的末尾调用。
# %% [markdown]
# ### 2.3 logsdict的用法
# 该logs字典包含损loss，已经每个batch和epoch的结束时的所有指标。示例包括loss和平均绝对误差。

# %%
class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):

    def on_train_batch_end(self, batch, logs=None):
        print('For batch {}, loss is {:7.2f}.'.format(batch, logs['loss']))

    def on_test_batch_end(self, batch, logs=None):
        print('For batch {}, loss is {:7.2f}.'.format(batch, logs['loss']))

    def on_epoch_end(self, epoch, logs=None):
        print('The average loss for epoch {} is {:7.2f} and mean absolute error is {:7.2f}.'.format(epoch, logs['loss'], logs['mae']))

model = get_model()
_ = model.fit(x_train, y_train,
          batch_size=64,
          steps_per_epoch=5,
          epochs=3,
          verbose=0,
          callbacks=[LossAndErrorPrintingCallback()])

# %% [markdown]
# 同样，可以在evaluate时调用回调。

# %%
_ = model.evaluate(x_test, y_test, batch_size=128, verbose=0, steps=20,
          callbacks=[LossAndErrorPrintingCallback()])

# %% [markdown]
# ## 3 keras回调示例
# ## 3.1 以最小的损失尽早停止
# 第一个示例展示了Callback通过达到最小损失时更改属性model.stop_training（布尔值），停止Keras训练。用户可以提供一个参数patience来指定训练最终停止之前应该等待多少个时期。
# 
# 注：tf.keras.callbacks.EarlyStopping 提供了更完整，更通用的实现。

# %%
import numpy as np
class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    def __init__(self, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        self.best_weights = None  # loss最低时的权重
    def on_train_begin(self, logs=None):
        # loss不再下降时等待的轮数
        self.wait = 0
        # 停止时的轮数
        self.stopped_epoch = 0
        # 开始时的最优loss
        self.best = np.Inf
    
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('loss')
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # 最佳权重
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print('导入当前最佳模型')
                self.model.set_weights(self.best_weights)
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('在%05d: 提前停止训练'% (self.stopped_epoch+1))
        


# %%
model = get_model()
_ = model.fit(x_train, y_train,
          batch_size=64,
          steps_per_epoch=5,
          epochs=30,
          verbose=0,
          callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()])

# %% [markdown]
# ### 自定义学习率
# 在模型训练中通常要做的一件事是随着训练轮次改变学习率。Keras后端公开了可用于设置变量的get_value API。在此示例中，我们展示了如何使用自定义的回调来动态更改学习率。
# 
# 注：这只是示例实现，请参见callbacks.LearningRateScheduler和keras.optimizers.schedules有关更一般的实现。

# %%
class LearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, schedule):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule
        
    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer没有lr参数。')
        # 获取当前lr
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        # 调整lr
        scheduled_lr = self.schedule(epoch, lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print('Epoch %05d: 学习率为%6.4f.'%(epoch, scheduled_lr))

# %% [markdown]
# 按轮次调整学习率

# %%
LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (3, 0.05), (6, 0.01), (9, 0.005), (12, 0.001)
]

def lr_schedule(epoch, lr):
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr

model = get_model()
_ = model.fit(x_train, y_train,
          batch_size=64,
          steps_per_epoch=5,
          epochs=15,
          verbose=0,
          callbacks=[LossAndErrorPrintingCallback(), LearningRateScheduler(lr_schedule)])


# %%


