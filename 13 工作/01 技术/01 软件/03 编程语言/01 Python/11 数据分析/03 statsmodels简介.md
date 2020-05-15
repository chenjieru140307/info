# statsmodels

作用：

- 有很多统计模型，能完成很多统计测试，数据探索以及可视化。它也包含一些经典的统计方法，比如贝叶斯方法和一个机器学习的模型。


文档：

- [文档](http://www.statsmodels.org/stable/index.html)



statsmodels 中模型：

- 线性模型（linear models），广义线性模型（generalized linear models），鲁棒线性模型（robust linear models）
- 线性混合效应模型（Linear mixed effects models）
- 方差分析(ANOVA)方法（Analysis of variance (ANOVA) methods）
- 时间序列处理（Time series processes）和 状态空间模型（state space models）
- 广义矩估计方法（Generalized method of moments）

接下来我们用一些 statsmodels 中的工具，并了解如何使用 Patsy 公式和 pandas DataFrame进行建模。

## 估计线性模型

statsmodels中的线性模型大致分为两种：基于数组的（array-based），和基于公式的（formula-based）。调用的模块为：


举例：

```py
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd


def dnorm(mean, variance, size=1):
    if isinstance(size, int):
        size = size
    return mean + np.sqrt(variance) * np.random.randn(size)


# For reproducibility
np.random.seed(12345)

N = 100
X = np.c_[dnorm(0, 0.4, size=N),
          dnorm(0, 0.6, size=N),
          dnorm(0, 0.2, size=N)]

eps = dnorm(0, 0.1, size=N)
beta = [0.1, 0.3, 0.5]

y = np.dot(X, beta) + eps

print(X.shape)
print(eps.shape)
print(X[:5])
print(y[:5])
X_model = sm.add_constant(X)
print(X_model[:5])

model = sm.OLS(y, X)
results = model.fit()
print(results.params)
print(results.summary())


data = pd.DataFrame(X, columns=['col0', 'col1', 'col2'])
data['y'] = y
print(data.head())

results = smf.ols('y ~ col0 + col1 + col2', data=data).fit()
print(results.params)
print(results.tvalues)

results.predict(data[:5])
```

输出：

```txt
(100, 3)
(100,)
[[-0.12946849 -1.21275292  0.50422488]
 [ 0.30291036 -0.43574176 -0.25417986]
 [-0.32852189 -0.02530153  0.13835097]
 [-0.35147471 -0.71960511 -0.25821463]
 [ 1.2432688  -0.37379916 -0.52262905]]
[ 0.42786349 -0.67348041 -0.09087764 -0.48949442 -0.12894109]
[[ 1.         -0.12946849 -1.21275292  0.50422488]
 [ 1.          0.30291036 -0.43574176 -0.25417986]
 [ 1.         -0.32852189 -0.02530153  0.13835097]
 [ 1.         -0.35147471 -0.71960511 -0.25821463]
 [ 1.          1.2432688  -0.37379916 -0.52262905]]
[0.17826108 0.22303962 0.50095093]
                                 OLS Regression Results                                
=======================================================================================
Dep. Variable:                      y   R-squared (uncentered):                   0.430
Model:                            OLS   Adj. R-squared (uncentered):              0.413
Method:                 Least Squares   F-statistic:                              24.42
Date:                Sat, 16 May 2020   Prob (F-statistic):                    7.44e-12
Time:                        00:33:36   Log-Likelihood:                         -34.305
No. Observations:                 100   AIC:                                      74.61
Df Residuals:                      97   BIC:                                      82.42
Df Model:                           3                                                  
Covariance Type:            nonrobust                                                  
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1             0.1783      0.053      3.364      0.001       0.073       0.283
x2             0.2230      0.046      4.818      0.000       0.131       0.315
x3             0.5010      0.080      6.237      0.000       0.342       0.660
==============================================================================
Omnibus:                        4.662   Durbin-Watson:                   2.201
Prob(Omnibus):                  0.097   Jarque-Bera (JB):                4.098
Skew:                           0.481   Prob(JB):                        0.129
Kurtosis:                       3.243   Cond. No.                         1.74
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
       col0      col1      col2         y
0 -0.129468 -1.212753  0.504225  0.427863
1  0.302910 -0.435742 -0.254180 -0.673480
2 -0.328522 -0.025302  0.138351 -0.090878
3 -0.351475 -0.719605 -0.258215 -0.489494
4  1.243269 -0.373799 -0.522629 -0.128941
Intercept    0.033559
col0         0.176149
col1         0.224826
col2         0.514808
dtype: float64
Intercept    0.952188
col0         3.319754
col1         4.850730
col2         6.303971
dtype: float64
```



- 我们对一些随机数据生成一个线性模型：
- 真正的模型用的参数是 beta，dnorm的功能是产生指定平均值和方差的随机离散数据
- 一个线性模型通常会有一个截距，这里我们用 sm.add_constant 函数添加一个截距列给 X
- sm.OLS可以拟合（fit）普通最小二乘线性回归
- fit方法返回的是一个回顾结果对象，包含预测模型的参数和其他一些诊断数据
- 在 results 上调用 summary 方法，可能得到一些详细的诊断数据：
- 参数的名字通常为 x1, x2，以此类推。假设所有的模型参数都在一个 DataFrame 里：

- 现在我们可以使用 statsmodels formula API（公式 API）和 Patsy 的公式字符串 `y ~ col0 + col1 + col2`
- 可以看到 statsmodel 返回的结果是 Series，而 Series 的索引部分是 DataFrame 的列名。当我们使用公式和 pandas 对象的时候，不需要使用 add_constant。
- `results.predict(data[:5])` 如果得到新的数据，我们可以用预测模型的参数来进行预测：





## 预测时序过程

statsmodels 中的另一个类是用于时间序列分析的，其中有：

- 自动回归处理（autoregressive processes）
- 卡尔曼滤波（Kalman filtering）
- 状态空间模型（state space models）
- 多元回归模型（multivariate autoregressive models）。

让我们用自动回归结果和噪音模拟一个时间序列数据：


举例：



```Python
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd

def dnorm(mean, variance, size=1):
    if isinstance(size, int):
        size = size
    return mean + np.sqrt(variance) * np.random.randn(size)


init_x = 4

import random
values = [init_x, init_x]
N = 1000

b0 = 0.8
b1 = -0.4
noise = dnorm(0, 0.1, N)
for i in range(N):
    new_x = values[-1] * b0 + values[-2] * b1 + noise[i]
    values.append(new_x)

print(values[:6])


MAXLAGS = 5
model = sm.tsa.AR(values)
results = model.fit(MAXLAGS)

print(results.params)
```

说明：

- 这种数据有 AR(2)结构（two lags，延迟两期），延迟参数是 0.8和-0.4。当我们拟合一个 AR 模型，我们可能不知道延迟的期间是多少，所以可以在拟合时设一个比较大的延迟数字
- 结果里的预测参数，第一个是解决，之后是两个延迟（lags）
