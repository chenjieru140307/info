# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # 构建单层LSTM网络对MNIST数据集分类
# %% [markdown]
# 这里的输入x当成28个时间段，每段内容为28个值，使用unstack将原始的输入28×28调整成具有28个元素的list
# 
# 每个元素为1×28的数组。这28个时序一次送入RNN中，如图下图所示：
# ![](https://gitee.com/kkweishe/images/raw/master/ML/2019-8-17_17-25-47.png)
# 
# 由于是批次操作，所以每次都取该批次中所有图片的一行作为一个时间序列输入。
# 
# 理解了这个转换之后，构建网络就变得很容易了，先建立一个包含128个cell的类lstm_cell，然后将变形后的x1放进去生成节点outputs，最后通过全连接生成pred，最后使用softmax进行分类。

# %%
import tensorflow as tf
old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# 导入 MINST 数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)
tf.compat.v1.logging.set_verbosity(old_v)


# %%
n_input = 28    #MNIST data 输入(img shape: 28*28)
n_steps = 28    #序列个数
n_hidden = 128  #隐藏层个数
n_classes = 10  #MNIST 分类个数 (0～9 digits)

# 定义占位符
x = tf.placeholder('float', [None, n_steps, n_input])
y = tf.placeholder('float', [None, n_classes])

# 对矩阵进行分解
x1 = tf.unstack(x, n_steps, 1)
lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x1, dtype=tf.float32)

pred = tf.contrib.layers.fully_connected(outputs[-1], n_classes, activation_fn=None)

learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# 平均交叉熵损失
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

## 评估模型
# tf.argmax(input,axis)根据axis取值的不同返回每行或者每列最大值的索引。
# axis = 1: 行
# equal，相等的意思。顾名思义，就是判断，x, y 是不是相等
# tf.cast  数据类型转换
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 1
    
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # 计算批次数据的准确率
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " +                   "{:.6f}".format(loss) + ", Training Accuracy= " +                   "{:.5f}".format(acc))
            
        step += 1
    print (" Finished!")
    
    # 计算准确率 for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print ("Testing Accuracy:",         sess.run(accuracy, feed_dict={x: test_data, y: test_label}))


# %%


