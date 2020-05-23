# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# %%
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.random(*train_X.shape) * 0.3

## 显示模拟数据
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.legend()
plt.show()

# %% [markdown]
# ## 创建模型

# %%
## 占位符
X = tf.placeholder('float')
Y = tf.placeholder('float')

## 模型参数初始化
## tf.random_normal()函数用于从服从指定正太分布的数值中取出指定个数的值。
## 初始化成[-1，1]的随机数
## Variable：定义变量
W =tf.Variable(tf.random_normal([1]), name="weight")
b =tf.Variable(tf.zeros([1]), name="bias")

## 前向结构
## tf.multiply（）两个矩阵中对应元素各自相乘
z = tf.multiply(X, W) + b

# %% [markdown]
# ### 反向结构

# %%
## reduce_mean计算张量的各个维度上的元素的平均值.
cost = tf.reduce_mean(tf.square(Y - z))
learning_rate = 0.01

## 优化
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# %% [markdown]
# ### 迭代训练模型
# 1.训练模型
# 
# 建立好模型后，可以通过迭代来训练模型了。TensorFlow中的任务是通过session来进行的。
# 
# 下面的代码中，先进行全局初始化，然后设置训练迭代的次数，启动session开始运行任务。

# %%
#初始化所有变量
init = tf.global_variables_initializer()
#定义参数
training_epochs = 20
display_step = 2

# 统计loss平均值
def moving_average(a, w=10):
    if len(a) < w: 
        return a[:]    
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in 
    enumerate(a)]

#启动session
with tf.Session() as sess:
    sess.run(init)
    plotdata={"batchsize":[],"loss":[]} #存放批次值和损失值
    #向模型输入数据
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        #显示训练中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost,feed_dict={X:train_X,Y:train_Y})
            print ("Epoch:", epoch+1,"cost=", loss,"W=",sess.run(W), 
            "b=", sess.run(b))
            if not (loss == "NA" ):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)

    print (" Finished!")
    print ("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}), 
    "W=", sess.run(W), "b=", sess.run(b))
    
    ## 显示拟合曲线
    plt.plot(train_X, train_Y, 'ro', label = 'Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fittedling')
    plt.legend()
    plt.show()

    # 显示loss曲线
    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"], plotdata["avgloss"],'b--')
    plt.xlabel('minibatch number')
    plt.ylabel('loss')
    plt.title('')
    plt.show()

