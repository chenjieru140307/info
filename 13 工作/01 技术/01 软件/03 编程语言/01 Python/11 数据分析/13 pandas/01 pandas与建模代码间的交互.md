

# CHAPTER 13 Introduction to Modeling Libraries in Python（Python中建模库的介绍）

这一章回顾一下之间 pandas 的一些特性，希望能在我们处理数据的时候有所帮助。然后会简要介绍两个很有用的建模工具：statsmodels和 scikit-learn。


# pandas

作用：

- 加载数据并分析清理。


## dataframe


举例：

```py
import numpy as np
import pandas as pd

df1 = pd.DataFrame({'x0': [1, 2, 3, 4, 5],
                     'x1': [0.01, -0.01, 0.25, -4.1, 0.],
                     'y': [-1.5, 0., 3.6, 1.3, -2.]})
print(df1)
print(df1.columns)
print(df1.values)
print()

df2 = pd.DataFrame(df1.values, columns=['one', 'two', 'three'])
print(df2)
print()

df3 = df1.copy()
df3['strings'] = ['a', 'b', 'c', 'd', 'e']
print(df3)
print(df3.values)
print()
```

输出：

```txt
   x0    x1    y
0   1  0.01 -1.5
1   2 -0.01  0.0
2   3  0.25  3.6
3   4 -4.10  1.3
4   5  0.00 -2.0
Index(['x0', 'x1', 'y'], dtype='object')
[[ 1.    0.01 -1.5 ]
 [ 2.   -0.01  0.  ]
 [ 3.    0.25  3.6 ]
 [ 4.   -4.1   1.3 ]
 [ 5.    0.   -2.  ]]

   one   two  three
0  1.0  0.01   -1.5
1  2.0 -0.01    0.0
2  3.0  0.25    3.6
3  4.0 -4.10    1.3
4  5.0  0.00   -2.0

   x0    x1    y strings
0   1  0.01 -1.5       a
1   2 -0.01  0.0       b
2   3  0.25  3.6       c
3   4 -4.10  1.3       d
4   5  0.00 -2.0       e
[[1 0.01 -1.5 'a']
 [2 -0.01 0.0 'b']
 [3 0.25 3.6 'c']
 [4 -4.1 1.3 'd']
 [5 0.0 -2.0 'e']]
```


说明：

- `.values` 可以把一个 DataFrame 变为 Numpy 数组。
- `pd.DataFrame(data.values, columns=['one', 'two', 'three'])` 可以将 numpy 数组传入，来创建 dataframe
- `.values` 属性最好用于同质的数据，即数据类型都是数值型。如果有异质的数据，结果会变为 Python 对象。



## loc

```py
import numpy as np
import pandas as pd

df1 = pd.DataFrame({'x0': [1, 2, 3, 4, 5],
                     'x1': [0.01, -0.01, 0.25, -4.1, 0.],
                     'y': [-1.5, 0., 3.6, 1.3, -2.]})
model_cols = ['x0', 'x1']
print(df1.loc[:, model_cols].values)
```

输出：

```txt
[[ 1.    0.01]
 [ 2.   -0.01]
 [ 3.    0.25]
 [ 4.   -4.1 ]
 [ 5.    0.  ]]
```

说明：

- loc 可以使用列中的一部分数据。




## 使用哑变量代替 category


举例：

```py
import numpy as np
import pandas as pd

df1 = pd.DataFrame({'x0': [1, 2, 3, 4, 5],
                    'x1': [0.01, -0.01, 0.25, -4.1, 0.],
                    'y': [-1.5, 0., 3.6, 1.3, -2.]})

df1['category'] = pd.Categorical(['a', 'b', 'a', 'a', 'b'],
                                 categories=['a', 'b'])
print(df1)
print()

dummies = pd.get_dummies(df1.category, prefix='category')
print(dummies)
print()

df1_summies = df1.drop('category', axis=1).join(dummies)
print(df1_summies)
```

输出：

```txt
   x0    x1    y category
0   1  0.01 -1.5        a
1   2 -0.01  0.0        b
2   3  0.25  3.6        a
3   4 -4.10  1.3        a
4   5  0.00 -2.0        b

   category_a  category_b
0           1           0
1           0           1
2           1           0
3           1           0
4           0           1

   x0    x1    y  category_a  category_b
0   1  0.01 -1.5           1           0
1   2 -0.01  0.0           0           1
2   3  0.25  3.6           1           0
3   4 -4.10  1.3           1           0
4   5  0.00 -2.0           0           1
```




说明：

- `pd.get_dummies`  可以创建哑变量
- `df1.drop('category', axis=1).join(dummies)` 去除 category 列，将哑变量添加进来。



补充：

- 在不同的统计模型上使用哑变量有一些细微的不同。当我们有更很多非数值型列的时候，使用 Patsy 的话会更简单易用一些。关于 Patsy 的内容会在下一节进行介绍。

