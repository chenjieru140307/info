# 大纲

原理要清楚些。

XGBoost 里面的目标函数：


$$
\operatorname{obj}^{(t)} = \sum_{i=1}^{n}\left[l\left(y_{i}, \hat{y}_{i}^{(t-1)}\right)+g_{i} f_{t}\left(x_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(x_{i}\right)\right]+\Omega\left(f_{t}\right)+\text { constant }
$$


式子中的 $g_{i}$ 和 $h_{i}$ 为：

$$
g_{i}=\partial_{\hat{y}_{i}^{(j-1)}} l\left(y_{i}, \hat{y}_{i}^{(t-1)}\right)
$$


$$
h_{i}=\partial_{\hat{y}_{i}^{(j-1)}}^{2} l\left(y_{i}, \hat{y}_{i}^{(t-1)}\right)
$$