


举例：

```python
import tensorflow as tf

# 注：本书全部代码在tensorflow 2.1版本测试通过
tf.print("tensorflow version:", tf.__version__)

a = tf.constant("hello")
b = tf.constant("tensorflow2")
c = tf.strings.join([a, b], " ")
tf.print(c)
```

输出：

```
tensorflow version: 2.1.0
hello tensorflow2
```

