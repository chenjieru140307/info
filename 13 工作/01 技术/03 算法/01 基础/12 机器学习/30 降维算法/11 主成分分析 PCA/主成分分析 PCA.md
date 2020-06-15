# 主成分分析

主成分分析 principal components analysis PCA 

主成分分析想解决的问题：

- 假设在 $\mathbb{R}^{n}$ 空间中有 $m$ 个点 $\left\{\boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{(m)}\right\}$
- 我们希望对这些点进行有损压缩。即想使用更少的内存来存储这些点，虽然损失一些精度，但是希望损失的精度尽可能少。

解决方法：

- 编码这些点的一种方式是用低维表示。
  - 对于每个点 $\boldsymbol{x}^{(i)} \in \mathbb{R}^{n}$ ，会有一个对应的编码向量 $\boldsymbol{c}^{(i)} \in \mathbb{R}^{l}$ 。如果 $l$ 比 $n$ 小，那么我们便使用了更少的内存来存储原来的数据。
  - 我们希望找到一个编码函数，根据输入返回编码，$f(\boldsymbol{x})=\boldsymbol{c}$
  - 我们也希望找到一个解码函数，给定编码重构输入，$\boldsymbol{x} \approx g(f(\boldsymbol{x}))$，这个解码函数如果使用矩阵来表示，就是：$g(\boldsymbol{c})=\boldsymbol{D} \boldsymbol{c}$，其中 $\boldsymbol{D} \in \mathbb{R}^{n \times l}$ 是定义解码的矩阵。


关注这个解码矩阵：

- 这个解码矩阵可能会有很多个解。为了使问题有唯一解，我们限制 $\boldsymbol{D}$ 中所有列向量都有单位范数。
  - 因为如果我们按比例地缩小所有点对应的编码向量 $c_{i}$，那么只需按比例放大 $\boldsymbol{D}_{ :, i}$，即可保持结果不变。

计算这个解码矩阵的最优解：

- 为了使编码问题简单一些，PCA 限制 $\boldsymbol{D}$ 的列向量彼此正交。
  - 注意，除非 $l=n$ ，否则严格意义上 $\boldsymbol{D}$ 不是一个正交矩阵)。
- 为了将这个基本想法变为我们能够实现的算法，首先我们需要明确如何根据每一个输入 $\boldsymbol{x}$ 得到一个最优编码 $\boldsymbol{c}*$ 。
  - 一种方法是最小化原始输入向量 $\boldsymbol{x}$ 和重构向量 $g(\boldsymbol{c}*)$ 之间的距离。我们使用范数来衡量它们之间的距离。在 PCA 算法中，我们使用 $L^{2}$ 范数，这样来求解得到 满足距离最小化的 c。
  - $\boldsymbol{c}^{*}=\underset{c}{\arg \min }\|\boldsymbol{x}-g(\boldsymbol{c})\|_{2}$
  - 我们可以用平方 $L^{2}$ 范数替代 $L^{2}$ 范数，因为两者在相同的值 $\boldsymbol{c}$ 上取得最小值。这是因为 $L^{2}$ 范数是非负的，并且平方运算在非负值上是单调递增的。这样就转化成：
  - $\boldsymbol{c}^{*}=\arg \min _{\boldsymbol{c}}\|\boldsymbol{x}-g(\boldsymbol{c})\|_{2}^{2}$

求解：

$$
\begin{aligned}
\boldsymbol{c}^{*}=& \arg \min _{\boldsymbol{c}}\|\boldsymbol{x}-g(\boldsymbol{c})\|_{2}^{2}
\\= &\arg \min _{\boldsymbol{c}}\left((\boldsymbol{x}-g(\boldsymbol{c}))^{\top}(\boldsymbol{x}-g(\boldsymbol{c}))\right)
\\= &\arg \min _{\boldsymbol{c}}\left(\boldsymbol{x}^{\top} \boldsymbol{x}-\boldsymbol{x}^{\top} g(\boldsymbol{c})-g(\boldsymbol{c})^{\top} \boldsymbol{x}+g(\boldsymbol{c})^{\top} g(\boldsymbol{c})\right)
\\= &\arg \min _{\boldsymbol{c}}\left(\boldsymbol{x}^{\top} \boldsymbol{x}-2 \boldsymbol{x}^{\top} g(\boldsymbol{c})+g(\boldsymbol{c})^{\top} g(\boldsymbol{c})\right)
\\= &\arg \min _{\boldsymbol{c}}\left(-2 \boldsymbol{x}^{\top} g(\boldsymbol{c})+g(\boldsymbol{c})^{\top} g(\boldsymbol{c})\right)
\\= &\arg \min _{\boldsymbol{c}}\left(-2 \boldsymbol{x}^{\top} \boldsymbol{D} \boldsymbol{c}+\boldsymbol{c}^{\top} \boldsymbol{D}^{\top} \boldsymbol{D} \boldsymbol{c}\right)
\\= &\arg \min _{\boldsymbol{c}}\left(-2 \boldsymbol{x}^{\top} \boldsymbol{D} \boldsymbol{c}+\boldsymbol{c}^{\top} \boldsymbol{I}_{l} \boldsymbol{c}\right)
\\= &\arg \min _{\boldsymbol{c}}\left(-2 \boldsymbol{x}^{\top} \boldsymbol{D} \boldsymbol{c}+\boldsymbol{c}^{\top} \boldsymbol{c}\right)
\end{aligned}
$$

说明：

- 第三行到第四行：因为标量 $g(\boldsymbol{c})^{\top} \boldsymbol{x}$ 的转置等于自己。
- 第四行到第五行：因为第一项 $\boldsymbol{x}^{\top} \boldsymbol{x}$ 不依赖于 $\boldsymbol{c}$，所以我们可以忽略它。
- 第五行到第六行：代入 $g(\boldsymbol{c})$ 的定义。
- 第六行到第七行：矩阵 $\boldsymbol{D}$ 的正交性和单位范数约束

此时，求解：

- 令 $-2 \boldsymbol{x}^{\top} \boldsymbol{D} \boldsymbol{c}+\boldsymbol{c}^{\top} \boldsymbol{c}$ 最小化的 $\boldsymbol{c}$

可以通过向量微积分来求解这个最优化问题：

$$
\nabla_{c}\left(-2 \boldsymbol{x}^{\top} \boldsymbol{D} \boldsymbol{c}+\boldsymbol{c}^{\top} \boldsymbol{c}\right)=0
$$

$$
-2 \boldsymbol{D}^{\top} \boldsymbol{x}+2 \boldsymbol{c}=0
$$

$$
\boldsymbol{c}=\boldsymbol{D}^{\top} \boldsymbol{x}
$$

此时，得到：

- 使用编码函数 $f(\boldsymbol{x})=\boldsymbol{D}^{\top} \boldsymbol{x}$ 即可进行编码。

则：

$$
r(\boldsymbol{x})=g(f(\boldsymbol{x}))=\boldsymbol{D} \boldsymbol{D}^{\top} \boldsymbol{x}
$$

下面，我们需要挑选编码矩阵 $\boldsymbol{D}$ 。


- 要做到这一点，先来回顾最小化输入和重构之间 $L^2$ 距离的这个想法。因为用相同的矩阵 $\boldsymbol{D}$ 对所有点进行解码，我们不能再孤立地看待每个点。反之，我们必须最小化所有维数和所有点上的误差矩阵的 Frobenius 范数。这样来保证编码解码后的整体误差最小。

即：

$$
\boldsymbol{D}^{*}=\underset{D}{\arg \min } \sqrt{\sum_{i, j}\left(\boldsymbol{x}_{j}^{(i)}-r\left(\boldsymbol{x}^{(i)}\right)_{j}\right)^{2}} \text { subject to } \boldsymbol{D}^{\top} \boldsymbol{D}=\boldsymbol{I}_{l}
$$

为了推导用于寻求 $\boldsymbol{D}^{*}$ 的算法，我们首先考虑 $l=1$ 的情况：

- 在这种情况下，$\boldsymbol{D}$ 是一个单一向量 $\boldsymbol{d}$ 。将 $r(\boldsymbol{x})=g(f(\boldsymbol{x}))=\boldsymbol{D} \boldsymbol{D}^{\top} \boldsymbol{x}$ 代入上面式子，简化 $\boldsymbol{D}$ 为 $\boldsymbol{d}$ ，问题简化为：


$$
\begin{aligned}
\boldsymbol{d}^{*}=&\underset{d}{\arg \min } \sum_{i}\left\|\boldsymbol{x}^{(i)}-\boldsymbol{d} \boldsymbol{d}^{\top} \boldsymbol{x}^{(i)}\right\|_{2}^{2} \text { subject to }\|\boldsymbol{d}\|_{2}=1
\\=&\underset{d}{\arg \min } \sum_{i}\left\|\boldsymbol{x}^{(i)}-\boldsymbol{d}^{\top} \boldsymbol{x}^{(i)} \boldsymbol{d}\right\|_{2}^{2} \text { subject to }\|\boldsymbol{d}\|_{2}=1
\\=&\underset{d}{\arg \min } \sum_{i}\left\|\boldsymbol{x}^{(i)}-\boldsymbol{x}^{(i) \top} \boldsymbol{d} \boldsymbol{d}\right\|_{2}^{2} \text { subject to }\|\boldsymbol{d}\|_{2}=1
\\=&\underset{d}{\arg \min }\left\|\boldsymbol{X}-\boldsymbol{X} \boldsymbol{d} \boldsymbol{d}^{\top}\right\|_{F}^{2}  \text { subject to } \boldsymbol{d}^{\top} \boldsymbol{d}=1
\end{aligned}
$$

说明：

- 从第一行到第二行：我们将标量 $\boldsymbol{d}^{\top} \boldsymbol{x}^{(i)}$ 放在向量 $\boldsymbol{d}$ 的右边。将该标量放在左边的写法更为传统。
- 第二行到第三行：标量的转置和自身相等。
- 第三行到第四行：我们将表示各点的向量堆叠成一个矩阵，记为 $\boldsymbol{X} \in \mathbb{R}^{m \times n}$ ，其中 $\boldsymbol{X}_{i, :}=\boldsymbol{x}^{(i)^{\top}}$。此时，使用单一矩阵来重述问题，比将问题写成求和形式更有帮助，这有助于我们使用更紧凑的符号。

此时，我们先不考虑约束，我们可以将 Frobenius 范数简化成下面的形式：

$$
\begin{aligned}
\underset{d}{\arg \min }\left\|\boldsymbol{X}-\boldsymbol{X} \boldsymbol{d} \boldsymbol{d}^{\top}\right\|_{F}^{2}=&\underset{d}{\arg \min } \operatorname{Tr}\left(\left(X-X d d^{\top}\right)^{\top}\left(X-X d d^{\top}\right)\right)
\\=& \underset{d}{\arg \min } \operatorname{Tr}\left(\boldsymbol{X}^{\top} \boldsymbol{X}-\boldsymbol{X}^{\top} \boldsymbol{X} d \boldsymbol{d}^{\top}-\boldsymbol{d} \boldsymbol{d}^{\top} \boldsymbol{X}^{\top} \boldsymbol{X}+\boldsymbol{d} \boldsymbol{d}^{\top} \boldsymbol{X}^{\top} \boldsymbol{X} \boldsymbol{d} \boldsymbol{d}^{\top}\right)
\\=&\underset{d}{\arg \min } \operatorname{Tr}\left(\boldsymbol{X}^{\top} \boldsymbol{X}\right)-\operatorname{Tr}\left(\boldsymbol{X}^{\top} \boldsymbol{X} \boldsymbol{d} \boldsymbol{d}^{\top}\right)-\operatorname{Tr}\left(\boldsymbol{d} \boldsymbol{d}^{\top} \boldsymbol{X}^{\top} \boldsymbol{X}\right)+\operatorname{Tr}\left(\boldsymbol{d} \boldsymbol{d}^{\top} \boldsymbol{X}^{\top} \boldsymbol{X} \boldsymbol{d} \boldsymbol{d}^{\top}\right)
\\=&\underset{d}{\arg \min }-\operatorname{Tr}\left(\boldsymbol{X}^{\top} \boldsymbol{X} d \boldsymbol{d}^{\top}\right)-\operatorname{Tr}\left(\boldsymbol{d} \boldsymbol{d}^{\top} \boldsymbol{X}^{\top} \boldsymbol{X}\right)+\operatorname{Tr}\left(d \boldsymbol{d}^{\top} \boldsymbol{X}^{\top} \boldsymbol{X} d d^{\top}\right)
\\=&\underset{d}{\arg \min }-2 \operatorname{Tr}\left(\boldsymbol{X}^{\top} \boldsymbol{X} \boldsymbol{d} \boldsymbol{d}^{\top}\right)+\operatorname{Tr}\left(\boldsymbol{d} \boldsymbol{d}^{\top} \boldsymbol{X}^{\top} \boldsymbol{X} d \boldsymbol{d}^{\top}\right)
\\=&\underset{d}{\arg \min }-2 \operatorname{Tr}\left(\boldsymbol{X}^{\top} \boldsymbol{X} \boldsymbol{d} \boldsymbol{d}^{\top}\right)+\operatorname{Tr}\left(\boldsymbol{X}^{\top} \boldsymbol{X} d \boldsymbol{d}^{\top} \boldsymbol{d} \boldsymbol{d}^{\top}\right)
\end{aligned}
$$

说明：

- 第一个式子：$\|A\|_{F}=\sqrt{\operatorname{Tr}\left(\boldsymbol{A} \boldsymbol{A}^{\top}\right)}$
- 第三行到第四行：与 $\boldsymbol{d}$ 无关的项不影响 $\arg \min$，可以直接消去
- 第四行到第五行：因为循环改变迹运算中相乘矩阵的顺序不影响结果，即：$\operatorname{Tr}\left(\prod_{i=1}^{n} F^{(i)}\right)=\operatorname{Tr}\left(\boldsymbol{F}^{(n)} \prod_{i=1}^{n-1} \boldsymbol{F}^{(i)}\right)$
- 第五行到第六行：再次使用 循环改变迹运算中相乘矩阵的顺序不影响结果 的性质。


此时，我们再来考虑约束条件：




$$
\begin{aligned}
\boldsymbol{d}^{*}=&\underset{d}{\arg \min }\left\|\boldsymbol{X}-\boldsymbol{X} \boldsymbol{d} \boldsymbol{d}^{\top}\right\|_{F}^{2}  \text { subject to } \boldsymbol{d}^{\top} \boldsymbol{d}=1
\\=&\underset{d}{\arg \min }-2 \operatorname{Tr}\left(\boldsymbol{X}^{\top} \boldsymbol{X} \boldsymbol{d} \boldsymbol{d}^{\top}\right)+\operatorname{Tr}\left(\boldsymbol{X}^{\top} \boldsymbol{X} \boldsymbol{d} \boldsymbol{d}^{\top} \boldsymbol{d} \boldsymbol{d}^{\top}\right)\text { subject to }\boldsymbol{d}^{\top} \boldsymbol{d}=1
\\=&\underset{d}{\arg \min }-2 \operatorname{Tr}\left(\boldsymbol{X}^{\top} \boldsymbol{X} d \boldsymbol{d}^{\top}\right)+\operatorname{Tr}\left(\boldsymbol{X}^{\top} \boldsymbol{X} d \boldsymbol{d}^{\top}\right)\text { subject to }\boldsymbol{d}^{\top} \boldsymbol{d}=1
\\=&\underset{d}{\arg \min }-\operatorname{Tr}\left(\boldsymbol{X}^{\top} \boldsymbol{X} \boldsymbol{d} \boldsymbol{d}^{\top}\right)\text { subject to }\boldsymbol{d}^{\top} \boldsymbol{d}=1
\\=&\underset{d}{\arg \max } \operatorname{Tr}\left(\boldsymbol{X}^{\top} \boldsymbol{X} \boldsymbol{d} \boldsymbol{d}^{\top}\right)\text { subject to }\boldsymbol{d}^{\top} \boldsymbol{d}=1
\\=&\underset{d}{\arg \max } \operatorname{Tr}\left(\boldsymbol{d}^{\top} \boldsymbol{X}^{\top} \boldsymbol{X} \boldsymbol{d}\right)\text { subject to }\boldsymbol{d}^{\top} \boldsymbol{d}=1
\end{aligned}
$$

说明：

- 第二行到第三行：因为 $\boldsymbol{d}^{\top} \boldsymbol{d}=1$
- 第五行到第六行：使用 循环改变迹运算中相乘矩阵的顺序不影响结果 的性质。


此时：

- 这个优化问题可以通过特征分解来求解。具体来讲，最优的 $\boldsymbol{d}$ 是 $\boldsymbol{X}^{\top} \boldsymbol{X}$ 最大特征值对应的特征向量。（怎么求解）

以上推导特定于 $l=1$ 的情况，仅得到了第一个主成分。

更一般地，当我们希望得到主成分的基时，矩阵 $\boldsymbol{D}$ 由前 $l$ 个最大的特征值对应的特征向量组成。这个结论可以通过归纳法证明，我们建议将此证明作为练习。（来自《深度学习》对里面的习题进行补充。）


