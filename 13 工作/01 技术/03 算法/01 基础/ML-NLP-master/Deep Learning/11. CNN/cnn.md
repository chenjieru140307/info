# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import cifar10_input
import tensorflow as tf
import numpy as np
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import pylab

# %% [markdown]
# ## cifar10数据集下载

# %%
batch_size = 128
print('begin')

# 下载cifar10数据集
images_train, labels_train = cifar10_input.inputs(eval_data=False, batch_size=batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data=True, batch_size=batch_size)

print('begin data')

# %% [markdown]
# ## 定义网络结构
# %% [markdown]
# TensorFlow里使用tf.nn.conv2d函数来实现卷积，其格式如下。
# 
# tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
# 
# 除去参数name参数用以指定该操作的name，与方法有关的共有5个参数。
# 
# - input：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch，in_height，in_width，in_channels]这样的形状（shape），具体含义是“训练时一个batch的图片数量，图片高度，图片宽度，图像通道数”，注意这是一个四维的Tensor，要求类型为float32和float64其中之一。
# - filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height，filter_width，in_channels，out_channels]这样的shape，具体含义是“卷积核的高度，滤波器的宽度，图像通道数，滤波器个数”，要求类型与参数input相同。有一个地方需要注意，第三维in_channels，就是参数input的第四维。
# - strides：卷积时在图像每一维的步长，这是一个一维的向量，长度为4。
# - padding：定义元素边框与元素内容之间的空间。string类型的量，只能是SAME和VALID其中之一，这个值决定了不同的卷积方式，padding的值为'VALID'时，表示边缘不填充，当其为'SAME'时，表示填充到滤波器可以到达图像边缘。
# - use_cudnn_on_gpu：bool类型，是否使用cudnn加速，默认为true。
# - 返回值：tf.nn.conr2d函数结果返回一个Tensor，这个输出就是常说的feature map。
# 
# 注意： 在卷积函数中，padding参数是最容易引起歧义的，该参数仅仅决定是否要补0，因此一定要清楚padding设为SAME的真正含义。在设为SAME的情况下，只有在步长为1时生成的feature map才会与输入值相等。

# %%
def weight_variable(shape):
    # 对于权重w的定义，统一使用函数truncated_normal来生成标准差为0.1的随机数为其初始化。
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # 对于权重b的定义，统一初始化为0.1。
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def avg_pool_6x6(x):
    return tf.nn.avg_pool(x, ksize=[1, 6, 6, 1], strides=[1, 6, 6, 1], padding='SAME')

## 定义占位符
# cifar data的shape 24*24*3
x = tf.placeholder(tf.float32, [None, 24, 24, 3])
# 0～9 数字分类=> 10 classes
y = tf.placeholder(tf.float32, [None, 10])

# 第一层卷积
W_conv1 = weight_variable([5, 5, 3, 64])
b_conv1 = bias_variable([64])

x_image = tf.reshape(x, [-1, 24, 24, 3])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积
W_conv2 = weight_variable([5, 5, 64, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 第三层卷积
W_conv3 = weight_variable([5, 5, 64, 10])
b_conv3 = bias_variable([10])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
nt_hpool3 = avg_pool_6x6(h_conv3)
nt_hpool3_flat = tf.reshape(nt_hpool3, [-1, 10])

# 输出层
y_conv = tf.nn.softmax(nt_hpool3_flat)

cross_entropy = -tf.reduce_sum(y * tf.log(y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# %% [markdown]
# ## 运行session进行训练

# %%
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# QueueRunner类用来启动tensor的入队线程，可以用来启动多个工作线程同时将多个tensor（训练数据）推送入文件名称队列中
tf.train.start_queue_runners(sess=sess)
for i in range(15000):
    image_batch, label_batch = sess.run([images_train, labels_train])
    
    # one hot编码
    # eye生成对角矩阵
    label_b = np.eye(10, dtype=float)[label_batch]
    
    train_step.run(feed_dict={x: image_batch, y: label_b}, session=sess)
    
    if i % 200 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: image_batch, y: label_b}, session=sess)
        print( "step %d, training accuracy %g"%(i, train_accuracy))
    

image_batch, label_batch = sess.run([images_test, labels_test])
label_b = np.eye(10,dtype=float)[label_batch] #one hot编码
print ("finished！ test accuracy %g"%accuracy.eval(feed_dict={x:image_batch, y: label_b},session=sess))


# %%


