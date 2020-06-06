

## KNN 在二分类问题上的性能

在二分类问题上，若 $k=1$


给定测试样本 $\boldsymbol{x}$ ，若其最近邻样本为 $\boldsymbol{z}$ ，则最近邻分类器出错的概率就是 $\boldsymbol{x}$ 与 $\boldsymbol{z}$ 类别标记不同的概率，即

$$
P(e r r)=1-\sum_{c \in \mathcal{Y}} P(c | \boldsymbol{x}) P(c | \boldsymbol{z})\tag{10.1}
$$


假设样本独立同分布，且对任意 $\boldsymbol{x}$ 和任意小正数 $\delta$ ，在 $\boldsymbol{x}$ 附近 $\delta$ 距离范围内总能找到一个训练样本；换言之，对任意测试样本，总能在任意近的范围内找到训练样本 $\boldsymbol{z}$ 。

令 $c^{*}=\arg \max _{c \in \mathcal{Y}} P(c | \boldsymbol{x})$ 表示贝叶斯最优分类器的结果，有

$$
\begin{aligned} P(e r r) &=1-\sum_{c \in \mathcal{Y}} P(c | \boldsymbol{x}) P(c | \boldsymbol{z}) \\ & \simeq 1-\sum_{c \in \mathcal{Y}} P^{2}(c | \boldsymbol{x}) \\ & \leqslant 1-P^{2}\left(c^{*} | \boldsymbol{x}\right) \\ &=\left(1+P\left(c^{*} | \boldsymbol{x}\right)\right)\left(1-P\left(c^{*} | \boldsymbol{x}\right)\right) \\ & \leqslant 2 \times\left(1-P\left(c^{*} | \boldsymbol{x}\right)\right) \end{aligned}
$$

于是我们得到了有点令人惊讶的结论：最近邻分类器虽简单，但它的泛化错误率不超过贝叶斯最优分类器的错误率的两倍！<span style="color:red;">确认下贝叶斯最优分类器的错误率的计算</span>



K 近邻的最大误差：

- 例如，假设我们有一个用 0-1误差度量性能的多分类任务。那么，当训练样本数目趋向于无穷大时，1-最近邻的误差将收敛到两倍贝叶斯误差。
  - 超出贝叶斯误差的原因是它会随机从等距离的临近点中随机挑一个。而存在无限的训练数据时，所有测试点 $\boldsymbol{x}$ 周围距离为零的邻近点有无限多个。
  - 如果我们使用所有这些临近点投票的决策方式，而不是随机挑选一个，那么该过程将会收敛到贝叶斯错误率。

