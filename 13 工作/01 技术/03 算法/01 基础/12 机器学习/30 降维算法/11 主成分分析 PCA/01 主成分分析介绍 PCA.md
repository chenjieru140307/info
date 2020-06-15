
# 主成分分析 PCA 

主成分分析 Principal Components Analysis PCA



主成分分析：

- 主成分分析是降维中最经典的方法。



特点：

- 线性
- 非监督
- 全局：也就是说，不是针对某种类别的，而是所有的数据



优点：

- 仅仅需要以方差衡量信息量，不受数据集以外的因素影响。
- 各主成分之间正交，可消除原始数据成分间的相互影响的因素。
- 计算方法简单，主要运算是特征值分解，易于实现。


缺点：

- 主成分各个特征维度的含义具有一定的模糊性，不如原始样本特征的解释性强。
- 方差小的非主成分也可能含有对样本差异的重要信息，因降维丢弃可能对后续数据处理有影响。



介绍：

- PCA 旨在找到数据中的主成分，并利用这些主成分表征原始数据，从而达到降维的目的。
- 也相当于将高维的数据通过线性变换投影到低维空间上去。
- 因此，要找出最能够代表原始数据的投影方法。使用 PCA 降掉的那些维度最好只包含噪声或冗余。

作用：

- 去冗余：
  - 去除可以被其他向量代表的线性相关向量，这部分信息量是多余的。
- 去噪声：
  - 去除较小特征值对应的特征向量，特征值的大小反映了变换后在特征向量方向上变换的幅度，幅度越大，说明这个方向上的元素差异也越大，要保留。

应用：

- 主要情形：
  - 当数据维度很大的时候，如果相信大部分变量之间存在线性关系，那么我们就希望降低维数，用较少的变量来抓住大部分的信息。


解决的问题：

- 可解决训练数据中存在数据特征过多或特征累赘的问题。

核心思想：

- 将 $m$ 维特征映射到 $n$ 维（$n < m$），这 $n$ 维形成主元，是重构出来最能代表原始数据的正交特征。

介绍：

- 假设数据集是 $m$ 个 $n$ 维，$(\boldsymbol x^{(1)}, \boldsymbol x^{(2)}, \cdots, \boldsymbol x^{(m)})$。如果 $n=2$，需要降维到 $n'=1$，现在想找到某一维度方向代表这两个维度的数据。
- 下图有 $u_1, u_2$ 两个向量方向，但是哪个向量才是我们所想要的，可以更好代表原始数据集的呢？

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20190722/4x9JJbBcxXPG.png?imageslim">
</p>

- 从图可看出，$u_1$ 比 $u_2$ 好，为什么呢？有以下两个主要评价指标：
  - 样本点到这个直线的距离足够近。
  - 样本点在这个直线上的投影能尽可能的分开。<span style="color:red;">是的。</span>
- OK，那么如果我们需要降维的目标维数是其他任意维，则我们关注的是：
  - 最近重构性：样本点到这个超平面的距离都足够近；
  - 最大可分性：样本点在这个超平面上的投影能尽可能分开。
- 基于最近重构性和最大可分性，能分别得到主成分分析的两种等价推导。

准备：


- 在做 PCA 之前先标准化
  - 在使用 PCA 之前，一般来讲要做 normalization 使得变量中心为 0，而且方差为 1.
  - 原因：（没写）
    - 主成分分析通常会得到协方差矩阵和相关矩阵。这些矩阵可以通过原始数据计算出来。
    - 协方差矩阵包含平方和与向量积的和。相关矩阵与协方差矩阵类似，但是第一个变量，也就是第一列，是标准化后的数据。如果变量之间的方差很大，或者变量的量纲不统一，我们必须先标准化再进行主成分分析。




最近重构性出发：

过程：

- 假定数据样本进行了中心化, 即 $\sum_{i} \boldsymbol{x}_{i}=\mathbf{0} ;$ 
- 再假定投影变换后得到的新坐标系为 $\left\{\boldsymbol{w}_{1}, \boldsymbol{w}_{2}, \ldots, \boldsymbol{w}_{d}\right\},$ 其中 $\boldsymbol{w}_{i}$ 是标准正交基向量, $\left\|\boldsymbol{w}_{i}\right\|_{2}=1, \boldsymbol{w}_{i}^{\mathrm{T}} \boldsymbol{w}_{j}=0$ $(i \neq j)$ 
- 若丢弃新坐标系中的部分坐标, 即将维度降低到 $d^{\prime}<d,$ 则样本点 $\boldsymbol{x}_{i}$ 在低维坐标系中的投影是 $\boldsymbol{z}_{i}=\left(z_{i 1} ; z_{i 2} ; \ldots ; z_{i d^{\prime}}\right),$ 其中 $z_{i j}=\boldsymbol{w}_{j}^{\mathrm{T}} \boldsymbol{x}_{i}$ 是 $\boldsymbol{x}_{i}$ 在低维坐标系下第 j 维的坐标. 
- 若基于 $\boldsymbol{z}_{i}$ 来重构 $\boldsymbol{x}_{i},$ 则会得到 $\hat{\boldsymbol{x}}_{i}=\sum_{j=1}^{d^{\prime}} z_{i j} \boldsymbol{w}_{j}$
- 考虑整个训练集, 原样本点 $\boldsymbol{x}_{i}$ 与基于投影重构的样本点 $\hat{\boldsymbol{x}}_{i}$ 之间的距离为

$$\begin{aligned}
\sum_{i=1}^{m}\left\|\sum_{j=1}^{d^{\prime}} z_{i j} \boldsymbol{w}_{j}-\boldsymbol{x}_{i}\right\|_2^2 &=\sum_{i=1}^{m} \boldsymbol{z}_{i}^{\mathrm{T}} \boldsymbol{z}_{i}-2 \sum_{i=1}^{m} \boldsymbol{z}_{i}^{\mathrm{T}} \mathbf{W}^{\mathrm{T}} \boldsymbol{x}_{i}+\mathrm{const} \\
& \propto-\operatorname{tr}\left(\mathbf{W}^{\mathrm{T}}\left(\sum_{i=1}^{m} \boldsymbol{x}_{i} \boldsymbol{x}_{i}^{\mathrm{T}}\right) \mathbf{W}\right)
\end{aligned}$$

- 根据最近重构性，上式应被最小化, 考虑到 $\boldsymbol{w}_{j}$ 是标准正交基, $\sum_{i} \boldsymbol{x}_{i} \boldsymbol{x}_{i}^{\mathrm{T}}$
是协方差矩阵, 有

$$\begin{array}{c}
\min _{\mathbf{W}}-\operatorname{tr}\left(\mathbf{W}^{\mathrm{T}} \mathbf{X} \mathbf{X}^{\mathrm{T}} \mathbf{W}\right) \\
\text { s.t. } \mathbf{W}^{\mathrm{T}} \mathbf{W}=\mathbf{I}\tag{10.15}
\end{array}$$

- 这就是主成分分析的优化目标。

最大可分性出发：

- 我们知道, 样本点
$\boldsymbol{x}_{i}$ 在新空间中超平面上的投影是 $\mathbf{W}^{\mathrm{T}} \boldsymbol{x}_{i},$ 若所有样本点的投影能尽可能分开, 则应该使投影后样本点的方差最大化, 如图所示。
<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200615/CkPUhssbtwRH.png?imageslim">
</p>

- 投影后样本点的方差是 $\sum_{i} \mathbf{W}^{\mathrm{T}} \boldsymbol{x}_{i} \boldsymbol{x}_{i}^{\mathrm{T}} \mathbf{W},$ 于是优化目标可写为

$$\begin{array}{c}
\max _{\mathbf{W}} \operatorname{tr}\left(\mathbf{W}^{\mathrm{T}} \mathbf{X} \mathbf{X}^{\mathrm{T}} \mathbf{W}\right) \\
\text { s.t. } \mathbf{W}^{\mathrm{T}} \mathbf{W}=\mathbf{I}
\end{array}\tag{10.16}
$$

显然, 式(10.16)与(10.15)等价.

求解：

- 已知

    $$\begin{array}{c}
    \min _{\mathbf{W}}-\operatorname{tr}\left(\mathbf{W}^{\mathrm{T}} \mathbf{X} \mathbf{X}^{\mathrm{T}} \mathbf{W}\right) \\
    \text { s.t. } \mathbf{W}^{\mathrm{T}} \mathbf{W}=\mathbf{I}
    \end{array}$$

- 运用拉格朗日乘子法可得：

    $$\begin{aligned}
    J(\boldsymbol W)&=-tr(\boldsymbol W^T\boldsymbol X\boldsymbol X^T\boldsymbol W+\boldsymbol\lambda'(\boldsymbol W^T\boldsymbol W-\boldsymbol I))\\
    \cfrac{\partial J(\boldsymbol W)}{\partial \boldsymbol W} &=-(2\boldsymbol X\boldsymbol X^T\boldsymbol W+2\boldsymbol\lambda'\boldsymbol W)
    \end{aligned}$$

- 令 $\cfrac{\partial J(\boldsymbol W)}{\partial \boldsymbol W}=\boldsymbol 0$，故

$$\begin{aligned}
\boldsymbol X\boldsymbol X^T\boldsymbol W&=-\boldsymbol\lambda'\boldsymbol W\\
\boldsymbol X\boldsymbol X^T\boldsymbol W&=\boldsymbol\lambda\boldsymbol W\\
\end{aligned}$$

- 其中，
  - $\boldsymbol W=\{\boldsymbol w_1,\boldsymbol w_2,\cdot\cdot\cdot,\boldsymbol w_d\}$
  - $\boldsymbol \lambda=\boldsymbol{diag}(\lambda_1,\lambda_2,\cdot\cdot\cdot,\lambda_d)$。

- 于是，只需对协方差矩阵 $\mathbf{X X}^{\mathrm{T}}$ 进行特征值分解，将求得的特征值排序: $\lambda_{1} \geqslant \lambda_{2} \geqslant \ldots \geqslant \lambda_{d},$ 再取前 $d^{\prime}$ 个特征值对应的特征向量构成 $\mathbf{W}=\left(\boldsymbol{w}_{1},\boldsymbol{w}_{2}, \ldots, \boldsymbol{w}_{d^{\prime}}\right) .$ 这就是主成分分析的解。


PCA 算法描述如图 10.5 所示.

- 输入:
  - 样本集 $D=\left\{\boldsymbol{x}_{1}, \boldsymbol{x}_{2}, \ldots, \boldsymbol{x}_{m}\right\}$
  - 低维空间维数 $d^{\prime}$ 
- 过程:
  - 01：对所有样本进行中心化: $\boldsymbol{x}_{i} \leftarrow \boldsymbol{x}_{i}-\frac{1}{m} \sum_{i=1}^{m} \boldsymbol{x}_{i}$
  - 02：计算样本的协方差矩阵 $\mathbf{X X}^{\mathrm{T}}$
  - 03：对协方差矩阵 $\mathbf{X X}^{\mathrm{T}}$ 做特征值分解;
  - 04：取最大的 $d^{\prime}$ 个特征值所对应的特征向量 $\boldsymbol{w}_{1}, \boldsymbol{w}_{2}, \ldots, \boldsymbol{w}_{d^{\prime}}$
- 输出：
  - 投影矩阵 $\mathbf{W}=\left(\boldsymbol{w}_{1}, \boldsymbol{w}_{2}, \dots, \boldsymbol{w}_{d^{\prime}}\right)$

说明：

- 降维后低维空间的维数 $d^{\prime}$ 通常是由用户事先指定, 或通过在 $d^{\prime}$ 值不同的低维空间中对 $k$ 近邻分类器(或其他开销较小的学习器) 进行交叉验证来选取 较好的 $d^{\prime}$ 值. 
- 对 PCA, 还可从重构的角度设置一个重构阈值, 例如 $t=95 \%,$ 然 后选取使下式成立的最小 $d^{\prime}$ 值:
$$
\frac{\sum_{i=1}^{d^{\prime}} \lambda_{i}}{\sum_{i=1}^{d} \lambda_{i}} \geqslant t
$$

- PCA 仅需保留 $W$ 与样本的均值向量即可通过简单的向量減法和矩阵-向量乘法将新样本投影至低维空间中。
- 显然，低维空间与原始高维空间必有不同，因为对应于最小的 $d-d^{\prime}$ 个特征值的特征向量被舍弃了，这是降维导致的结果。
- 但舍弃这部分信息往往是必要的: 
  - 一方面，舍弃这部分信息之后能使样本的采样密度增大, 这正是降维的重要动机; 
  - 另一方面, 当数据受到噪声影响时，最小的特征值所对应的特征向量往往与噪声有关, 将它们舍弃能在一定程度上起到去噪的效果。




疑问：

- 使用 PCA 后很难进行分类了吗？
