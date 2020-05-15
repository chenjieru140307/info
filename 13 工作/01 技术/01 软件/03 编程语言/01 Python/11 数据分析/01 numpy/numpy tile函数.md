
# ORIGIN
  * 在机器学习实战中看到的，



numpy 中的 tile 函数：
numpy.tile(A,B)：重复 A，B次，这里的 B 可以时 int 类型也可以是元组类型。

比如：
```Python
import numpy as np
print(np.tile([0,0],5))
print(np.tile([0,0],[2,5]))
```


输出：

```text
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
```


也即，上面的 [2,5] 的意思是：类似把 [0,0] 看成一个整体，然后生成类似 2 行 5 列的。
