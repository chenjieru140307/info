

# 半朴素贝叶斯分类器 独依赖估计 ODE


缘由：

- 朴素贝叶斯分类器采用了属性条件独立性假设，但在现实任务中这个假设往往很难成立。
- 于是，人们尝试对属性条件独立性假设进行一定程度的放松，由此产生了一类称为 “半朴素贝叶斯分类器” (semi-naive Bayes classifiers)的学习方法。


半朴素贝叶斯分类器：

- 适当考虑一部分属性间的相互依赖信息。


独依赖估计 One-Dependence Estimator ODE

- 独依赖估计是半朴素贝叶斯分类器最常用的一种策略。
- 独依赖是假设每个属性在类别之外最多依赖一个其他属性。
- 即：$P(\boldsymbol{x}|c_i)=\prod_{j=1}^d P(x_j|c_i,{\rm pa}_j)$
  - 其中 $pa_j$ 为属性 $x_i$ 所依赖的属性，成为 $x_i$ 的父属性。
  - 假设父属性 $pa_j$ 已知，那么可以使用下面的公式估计 $P(x_j|c_i,{\rm pa}_j)$：
  - $P(x_j|c_i,{\rm pa}_j)=\frac{P(x_j,c_i,{\rm pa}_j)}{P(c_i,{\rm pa}_j)}$



于是，问题的关键就转化为如何确定每个属性的父属性，

此时，不同的做法产生不同的独依赖分类器.


- SPODE (Super-Parent ODE)
  - 假设所有属性都依赖于同一个属性：
    - 这个共同父属性称为 “超父” (superparent)
  - 然后通过交叉验证等模型选择方法来确定超父属性
- TAN (Tree Augmented naive Bayes) 
  - 是在最大带权生成树(maximum weighted spanning tree)算法的基础上，通过以下步骤将属性间依赖关系约简为下图 TAN 中所示的树形结构：
    1. 计算任意两个属性之间的条件互信息(conditional mutual information)

    $$
    I\left(x_{i}, x_{j} | y\right)=\sum_{x_{i}, x_{j} ; c \in \mathcal{Y}} P\left(x_{i}, x_{j} | c\right) \log \frac{P\left(x_{i}, x_{j} | c\right)}{P\left(x_{i} | c\right) P\left(x_{j} | c\right)}
    $$

    2. 以属性为结点构建完全图，任意两个结点之间边的权重设为  $I\left(x_{i}, x_{j} | y\right)$ 。
    3. 构建此完全图的最大带权生成树，挑选根变量，将边置为有向；
    3. 加入类别结点 $y$ ，增加从 $y$ 到每个属性的有向边.
  - 容易看出，条件互信息  $I\left(x_{i}, x_{j} | y\right)$  刻画了属性 $x_i$ 和 $x_j$ 在已知类别情况下的相关性，因此，通过最大生成树算法，TAN 实际上仅保留了强相关属性之间 的依赖性.
- AODE (Averaged One-Dependent Estimator)
  - 是一种基于集成学习机制、更为强大的独依赖分类器。
  - 与 SPODE 通过模型选择确定超父属性不同，AODE尝试将每个属性作为超父来构建 SPODE，然后将那些具有足够训练数据支撑的 SPODE 集成起来作为最终结果，即：

  $$
  P(c | \boldsymbol{x}) \propto \sum_{i=1 \atop D_{x_{i}} | \geqslant m^{\prime}}^{d} P\left(c, x_{i}\right) \prod_{j=1}^{d} P\left(x_{j} | c, x_{i}\right)\tag{7.23}
  $$

  - 推导：


  $$\begin{aligned}
  P(c|\boldsymbol x)&=\cfrac{P(\boldsymbol x,c)}{P(\boldsymbol x)}\\
  &=\cfrac{P\left(x_{1}, x_{2}, \ldots, x_{d}, c\right)}{P(\boldsymbol x)}\\
  &=\cfrac{P\left(x_{1}, x_{2}, \ldots, x_{d} | c\right) P(c)}{P(\boldsymbol x)} \\
  &=\cfrac{P\left(x_{1}, \ldots, x_{i-1}, x_{i+1}, \ldots, x_{d} | c, x_{i}\right) P\left(c, x_{i}\right)}{P(\boldsymbol x)} \\
  \end{aligned}$$

  $$\begin{aligned}
  P(c|\boldsymbol x)&\propto P(c,x_{i})P(x_{1},…,x_{i-1},x_{i+1},…,x_{d}|c,x_{i}) \\
  &=P(c,x_{i})\prod _{j=1}^{d}P(x_j|c,x_i)
  \end{aligned}$$

  $$P(c|\boldsymbol x)\propto\sum\limits_{i=1 \atop |D_{x_{i}}|\geq m'}^{d}P(c_{i}|\boldsymbol x_{i})\prod_{j=1}^{d}P(c_{i}|\boldsymbol x_{i})$$

  - 当使用拉普拉斯修正时，有：

  $$
  \hat{P}\left(c, x_{i}\right)=\frac{\left|D_{c, x_{i}}\right|+1}{|D|+N_{i}}\tag{7.24}
  $$
  $$
  \hat{P}\left(x_{j} | c, x_{i}\right)=\frac{\left|D_{c, x_{i}, x_{j}}\right|+1}{\left|D_{c, x_{i}}\right|+N_{j}}\tag{7.25}
  $$

  - 说明：
    - 其中 $D_{x_i}$ 是在第 $i$ 个属性上取值为 $x_i$ 的样本的集合， $m'$ 为阈值常数。$N_i$ 是第 $i$ 个属性可能的取值数, $D_{c,x_i}$ 是类别为 $c$ 且在第 $i$ 个属性上取值为 $x_i$ 的样本集合，$D_{c,x_i,x_j}$ 是类别为 $c$ 且在第 $i$ 和第 $j$ 个属性上取值分别为 $x_i$ 和 $x_j$ 的样本集合。
    - 由于上面两式使用到了 $|D_{c,x_{i}}|$ 与 $|D_{c,x_{i},x_{j}}|$，若 $|D_{x_{i}}|$ 集合中样本数量过少，则 $|D_{c,x_{i}}|$ 与 $|D_{c,x_{i},x_{j}}|$ 将会更小，因此在式(7.23)中要求 $|D_{x_{i}}|$ 集合中样本数量不少于 $m'$。
  - 因此，与朴素贝叶斯分类器类似，AODE的训练过程也是“计数”，即在训练数据集上对符合条件的样本进行计数的过程。与朴素贝叶斯分类器相似，AODE无需模型选择，既能通过预计算节省预测时间，也能采取懒惰学习方式在预测时再进行计数，并且易于实现增量学习.

- kDE：

    - 既然将属性条件独立性假设放松为独依赖假设可能获得泛化性能的提升，那么，能否通过考虑属性间的高阶依赖来进一步提升泛化性能呢？
    - 也就是说，将 $P(\boldsymbol{x}|c_i)=\prod_{j=1}^d P(x_j|c_i,{\rm pa}_j)$ 中的属性 $pa_i$ 替换为包含 $k$ 个属性的集合 $\mathbf{p a}_{i}$ ，从而将 ODE 拓展为 kDE。
    - 需注意的是，隨着 $k$ 的增加，准确估计概率 $P\left(x_{i} | y, \mathbf{p a}_{i}\right)$ 所需的训练样本数量将以指数级增加。因此，若训练数据非常充分，泛化性能有可能提升；但在有限样本条件下，则又陷入估计高阶联合概率的泥沼.


朴素贝叶斯与两种半朴素贝叶斯分类器所考虑的属性依赖关系：

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200611/Jh2mAkzYmlnd.png?imageslim">
</p>

说明：

- SPODE 中，$x_1$ 是超父属性.