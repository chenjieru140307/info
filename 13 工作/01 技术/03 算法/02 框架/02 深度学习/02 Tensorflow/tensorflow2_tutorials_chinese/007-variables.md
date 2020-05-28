# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # TensorFlow2教程-Variables
# 
# TensorFlow 变量是表示程序所控制的共享持久状态的最佳方法。
# 
# 变量通过tf.Variable类进行操作。tf.Variable 表示张量，其值可以通过在其上运行ops进行更改。特定操作允许您读取和修改此张量的值。
# %% [markdown]
# ## 1 创建一个变量

# %%
import tensorflow as tf
# 要创建变量，只需提供初始值
my_var = tf.Variable(tf.ones([2,3]))
print(my_var)
#如果tf.device作用域处于活动状态，则变量将被放置在该设备上；否则，变量将被放置在与其dtype兼容的“最快”设备上
try:
    with tf.device("/device:GPU:0"):
        v = tf.Variable(tf.zeros([10, 10]))
        print(v)
except:
    print('no gpu')

# %% [markdown]
# ## 2 使用变量
# 

# %%
# 变量相加
a = tf.Variable(1.0)
b = (a+2) *3
print(a)
print(b)


# %%
# 更改某个向量
a = tf.Variable(1.0)
b = (a.assign_add(2)) *3
print(a)
print(b)

# %% [markdown]
# # 3 变量追踪
# TensorFlow中的变量是一个Python对象。在构建图层，模型，优化器和其他相关工具时，您可能希望获取（假设）模型中所有变量的列表。

# %%
class MyModuleOne(tf.Module):
    def __init__(self):
        self.v0 = tf.Variable(1.0)
        self.vs = [tf.Variable(x) for x in range(10)]
    
class MyOtherModule(tf.Module):
    def __init__(self):
        self.m = MyModuleOne()
        self.v = tf.Variable(10.0)
    
m = MyOtherModule()
print(m.variables)  # 模型中的所有变量
len(m.variables) 


# %%


