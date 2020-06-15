
# 核化线性降维 KPCA


缘由：

- 线性降维方法假设从高维空间到低维空间的函数映射是线性的。
- 然而，在不少现实任务中，可能需要非线性映射才能找到恰当的低维嵌入。

举例：

- 下图中，样本点从二维空间中的矩形区域采样后以 S 形曲面嵌入到三维空间，若直接使用线性降维方法对三维空间观察到的样本点进行降维，则将丢失原本的低维结构。
  - 为了对 “原本采样的” 低维空间与降维后的低维空间加以区别，我们称前者为“本真”(intrinsic)低维空间。


<p align="center">
    <img width="80%" height="70%" src="http://images.iterate.site/blog/image/180629/LmGdkc27L7.png?imageslim">
</p>


非线性降维：

- 常用方法，是基于核技巧对线性降维方法进行“核化” (kernelized)。
  - 核主成分分析(Kernelized PCA，简称 KPCA)

核主成分分析 KPCA

- 假定我们将在高维特征空间中把数据投影到由 $\mathbf{W}$ 确定的超平面上，即 PCA 欲求解

    $$
    \left(\sum_{i=1}^{m} z_{i} z_{i}^{\mathrm{T}}\right) \mathbf{W}=\lambda \mathbf{W}\tag{10.19}
    $$

- 其中 $\boldsymbol{z}_{i}$ 是样本点 $\boldsymbol{x}_{i}$ 在高维特征空间中的像。易知

$$
\begin{aligned} \mathbf{W} &=\frac{1}{\lambda}\left(\sum_{i=1}^{m} z_{i} z_{i}^{\mathrm{T}}\right) \mathbf{W}=\sum_{i=1}^{m} z_{i} \frac{z_{i}^{\mathrm{T}} \mathbf{W}}{\lambda} \\ &=\sum_{i=1}^{m} \boldsymbol{z}_{i} \boldsymbol{\alpha}_{i} \end{aligned}\tag{10.20}
$$

其中 $\boldsymbol{\alpha}_{i}=\frac{1}{\lambda} \boldsymbol{z}_{i}^{\mathrm{T}} \mathbf{W}$ 。假定 $\boldsymbol{z}_{i}$ 是由原始属性空间中的样本点 $\boldsymbol{x}_{i}$ 通过映射 $\phi$ 诊产生， 即 $\boldsymbol{z}_{i}=\phi\left(\boldsymbol{x}_{i}\right), i=1,2, \dots, m$ 。若 $\phi$ 能被显式表达出来，则通过它将样本映射至高维特征空间，再在特征空间中实施 PCA 即可。式(10.19)变换为：

$$
\left(\sum_{i=1}^{m} \phi\left(\boldsymbol{x}_{i}\right) \phi\left(\boldsymbol{x}_{i}\right)^{\mathrm{T}}\right) \mathbf{W}=\lambda \mathbf{W}\tag{10.21}
$$

式(10.20)变换为


$$
\mathbf{W}=\sum_{i=1}^{m} \phi\left(\boldsymbol{x}_{i}\right) \boldsymbol{\alpha}_{i}\tag{10.22}
$$


一般情形下，我们不清楚 $\phi$ 的具体形式，于是引入核函数

$$
\kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)=\phi\left(\boldsymbol{x}_{i}\right)^{\mathrm{T}} \phi\left(\boldsymbol{x}_{j}\right)\tag{10.23}
$$

将式(10.22)和(10.23)代入式(10.21)后化简可得

$$
\mathbf{K A}=\lambda \mathbf{A}\tag{10.24}
$$


其中 $\mathbf{K}$ 为 $\kappa$ 对应的核矩阵，$(\mathbf{K})_{i j}=\kappa\left(\boldsymbol{x}_{\boldsymbol{i}}, \boldsymbol{x}_{\boldsymbol{j}}\right)$，$\mathbf{A}=\left(\boldsymbol{\alpha}_{1} ; \boldsymbol{\alpha}_{2} ; \ldots ; \boldsymbol{\alpha}_{m}\right)$ 。显然，式(10.24)是特征值分解问题，取 $\mathbf{K}$ 最大的 $d'$ 个特征值对应的特征向量即可。

对新样本 $\boldsymbol{x}$ ，其投影后的第 $j\left(j=1,2, \dots, d^{\prime}\right)$ 维坐标为


$$
\begin{aligned} z_{j} &=\boldsymbol{w}_{j}^{\mathrm{T}} \phi(\boldsymbol{x})=\sum_{i=1}^{m} \alpha_{i}^{j} \phi\left(\boldsymbol{x}_{i}\right)^{\mathrm{T}} \phi(\boldsymbol{x}) \\ &=\sum_{i=1}^{m} \alpha_{i}^{j} \kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}\right) \end{aligned}\tag{10.25}
$$


其中 $\boldsymbol{\alpha}_{i}$ 已经过规范化， $\alpha_i^j$ 是 $\boldsymbol{\alpha}_{i}$ 的第 $j$ 个分量。


式(10.25)显示出，为获得投影后的坐标，KPCA 需对所有样本求和，因此它的计算开销较大。











缘由：

- PCA 算法前提是假设存在一个线性超平面，进而投影。
- 那如果数据不是线性的呢？该怎么办？

KPCA：

- 数据集从 $n$ 维映射到线性可分的高维 $N >n$，然后再从 $N$ 维降维到一个低维度 $n'(n'<n<N)$ 。

过程：

- 假设高维空间数据由 $n​$ 维空间的数据通过映射 $\phi​$ 产生。
- $n$ 维空间的特征分解为：

$$
\sum^m_{i=1} x^{(i)} \left( x^{(i)} \right)^T W = \lambda W
$$

- 其映射为：

$$
\sum^m_{i=1} \phi \left( x^{(i)} \right) \phi \left( x^{(i)} \right)^T W = \lambda W
$$

- ​通过在高维空间进行协方差矩阵的特征值分解，然后用和 PCA 一样的方法进行降维。


注意：

- 由于 KPCA 需要核函数的运算，因此它的计算量要比 PCA 大很多。


