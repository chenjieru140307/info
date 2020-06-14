

## 证明 K 均值算法的收敛性

首先，我们需要知道 K 均值聚类的迭代算法实际上是一种最大期望算法（Expectation-Maximization algorithm），简称 EM 算法。

EM 算法解决的是在概率模型中含有无法观测的隐含变量情况下的参数估计问题。

假设有 $m$ 个观察样本，模型的参数为$θ$，最大化对数似然函数可以写成如下形式：

$$
\theta=\arg \max _{\theta} \sum_{i=1}^{m} \log P\left(x^{(i)} | \theta\right)\tag{5.5}
$$

当概率模型中含有无法被观测的隐含变量时，参数的最大似然估计变为：

$$
\theta=\underset{\theta}{\operatorname{argmax}} \sum_{i=1}^{m} \log \sum_{z^{(i)}} P\left(x^{(i)}, z^{(i)} | \theta\right)\tag{5.6}
$$

由于 $z^{(i)}$ 是未知的，无法直接通过最大似然估计求解参数，这时就需要利用 EM 算法来求解。假设 $z^{(i)}$ 对应的分布为 $Q_i(z^{i)})$，并满足 $\sum_{z(i)}Q_i(z^{(i)})=1$。利用 Jensen 不等式，可以得到：

$$
\begin{aligned}
\sum_{i=1}^{m} \log \sum_{z^{(i)}} P\left(x^{(i)}, z^{(i)} | \theta\right)=&\sum_{i=1}^{m} \log \sum_{z^{(i)}} Q_{i}\left(z^{(i)}\right) \frac{P\left(x^{(i)}, z^{(i)} | \theta\right)}{Q_{i}\left(z^{(i)}\right)}
\\\geqslant & \sum_{i=1}^{m} \sum_{z^{(i)}} Q_{i}\left(z^{(i)}\right) \log \frac{P\left(x^{(i)}, z^{(i)} | \theta\right)}{Q_{i}\left(z^{(i)}\right)}
\end{aligned}\tag{5.7}
$$


要使上式中的等号成立，需要满足 $\frac{P\left(x^{(i)}, Z^{(i)} | \theta\right)}{Q_{i}\left(Z^{(i)}\right)}=c$，其中 $c$ 为常数，且满足 $\sum_{z^{(i)}} Q_{i}\left(z^{(i)}\right)=1$；

因此，

$$Q_{i}\left(z^{(i)}\right)=\frac{P\left(x^{(i)}, z^{(i)} | \theta\right)}{\sum_{z^{(i)}} P\left(x^{(i)}, z^{(i)} | \theta\right)}=P\left(z^{(i)} | x^{(i)}, \theta\right)$$

不等式右侧函数记为 $r(x|\theta)$。当等式成立时，我们相当于为待优化的函数找到了一个逼近的下界，然后通过最大化这个下界可以使得待优化函数向更好的方向改进。


图 5.5是一个θ为一维的例子，其中棕色的曲线代表我们待优化的函数，记为 $f(θ)$，优化过程即为找到使得 $f(θ)$ 取值最大的θ。在当前θ的取值下（即图中绿色的位置），可以计算 $Q_{i}\left(z^{(i)}\right)=P\left(z^{(i)} | x^{(i)}, \theta\right)$ ，此时不等式右侧的函数（记为 $r(x|\theta)$）给出了优化函数的一个下界，如图中蓝色曲线所示，其中在θ处两条曲线的取值时相等的。接下来找到使得  $r(x|\theta)$ 最大化的参数 $\theta′$，即图中红色的位置，此时 $f(\theta')$ 的取值比 $f(\theta)$ （绿色的位置处）有所提升。可以证明， $f(\theta')\geq r(x|\theta)=f(\theta)$ ，因此函数是单调的，而且 $P(x^{(i)},z^{(i)}|\theta)\in(0,1)$ 从而函数是有界的。根据函数单调有界必收敛的性质，EM算法的收敛性得证。但是 EM 算法只保证收敛到局部最优解。当函数为非凸时，以图 5.5为例，如果初始化在左边的区域时，则无法找到右侧的高点。

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20190401/yGM76Ccn6q4s.png?imageslim">
</p>

由上面的推导，EM算法框架可以总结如下，由以下两个步骤交替进行直到收敛。


（1）E步骤：计算隐变量的期望

$$Q_{i}\left(z^{(i)}\right)=P\left(z^{(i)} | x^{(i)}, \theta\right)$$

（2）M步骤：最大化．

$$\theta=\underset{\theta}{\operatorname{argmax}} \sum_{i=1}^{m} \sum_{z^{(i)}} Q_{i}\left(z^{(i)}\right) \log \frac{P\left(x^{(i)}, z^{(i)} | \theta\right)}{Q_{i}\left(z^{(i)}\right)}\tag{5.9}
$$

剩下的事情就是说明 K 均值算法与 EM 算法的关系了。K均值算法等价于用 EM 算法求解以下含隐变量的最大似然问题：

$$P\left(x, z | \mu_{1}, \mu_{2}, \ldots, \mu_{k}\right) \propto\left\{
\begin{aligned}
\exp \left(-\left\|x-\mu_{z}\right\|_{2}^{2}\right),&\left\|x-\mu_{z}\right\|_{2}=\min _{k}\left\|x-\mu_{k}\right\|_{2} 
\\0,&\left\|x-\mu_{z}\right\|_{2}>\min _{k}\left\|x-\mu_{k}\right\|_{2}
\end{aligned}
\right.\tag{5.10}
$$


其中 $z \in\{1,2, \ldots, k\}$ 是模型的隐变量。直观地理解，就是当样本 x 离第 k 个簇的中心点 $\mu_k$ 距离最近时，概率正比于 $exp(-||x-\mu_z||_2^2)$ ，否则为 0。


在 E 步骤，计算

$$
Q_{i}\left(z^{(i)}\right)=P\left(z^{(i)} | x^{(i)}, \mu_{1}, \mu_{2}, \ldots, \mu_{k}\right) \propto\left\{\begin{array}{l}
1,\left\|x^{(i)}-\mu_{z^{(i)}}\right\|_{2}=\min _{k}\left\|x-\mu_{k}\right\|_{2} \\
0,\left\|x^{(i)}-\mu_{z^{(i)}}\right\|_{2}>\min _{k}\left\|x-\mu_{k}\right\|_{2}
\end{array}\right.\tag{5.11}
$$


这等同于在 K 均值算法中对于每一个点 $x(i)$ 找到当前最近的簇 $z(i)$。

在 M 步骤，找到最优的参数 $\theta=\left\{\mu_{1}, \mu_{2}, \ldots, \mu_{k}\right\}$ ，使得似然函数最大：

$$\theta=\underset{\theta}{\operatorname{argmax}} \sum_{i=1}^{m} \sum_{z^{(i)}} Q_{i}\left(z^{(i)}\right) \log \frac{P\left(x^{(i)}, z^{(i)} | \theta\right)}{Q_{i}\left(z^{(i)}\right)}\tag{5.12}
$$

经过推导可得：

$$\sum_{i=1}^{m} \sum_{z^{(i)}} Q_{i}\left(z^{(i)}\right) \log \frac{P\left(x^{(i)}, z^{(i)} | \theta\right)}{Q_{i}\left(z^{(i)}\right)}=\text { const }-\sum_{i=1}^{m}\left\|x^{(i)}-\mu_{z^{(i)}}\right\|^{2}\tag{5.13}
$$


因此，这一步骤等同于找到最优的中心点 $\mu_{1}, \mu_{2}, \ldots, \mu_{k}$ ，使得损失函数 $\sum_{i=1}^{m}\left\|x^{(i)}-\mu_{z^{(i)}}\right\|^{2}$ 达到最小，此时每个样本 $x^{(i)}$ 对应的簇 $z^{(i)}$ 已确定，因此每个簇 k 对应的最优中心点 $\mu_k$ 可以由该簇中所有点的平均计算得到，这与 K 均值算法中根据当前簇的分配更新聚类中心的步骤是等同的。
