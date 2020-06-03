# GNN

## 一、GNN

1. 很多领域的数据的底层关系可以表示为图结构，如计算机视觉、分子化学、分子生物学、模式识别、数据挖掘等领域。

   最简单的图结构为单节点图，以及作为节点序列的图，更复杂的图结构包括树、无环图、带环图等。

2. 关于图的任务可以分为两类：

   - 基于图的任务`graph-focused`：以图为单位，直接在图结构的数据上实现分类或者回归。

     如：图表示化学化合物，每个顶点表示一个原子或者化学基团、边表示化学键。模型可以用于评估被检测的化合物的类别。

     <p align="center">
        <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/s7NHdj8iN7zg.png?imageslim">
     </p>
     

   - 基于节点的任务 `node-focused`：以节点为单位，在每个结点上实现分类或者回归。如：

     - 目标检测任务中需要检测图像是否包含特定的目标并进行目标定位。该问题可以通过一个映射函数来解决，该映射函数根据相应区域是否属于目标对象从而对邻接的顶点进行分类。

       如下图所示，对应于城堡的黑色顶点的输出为`1`，其它顶点输出为 `0` 。

      <p align="center">
         <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/IBNogtPor5GY.png?imageslim">
      </p>
      

     - 网页分类任务中需要判断每个网页的类别。我们将所有的网页视作一个大图，每个网页代表一个顶点、网页之间的超链接代表边。可以利用网页的内容和图结构来进行建模。

      <p align="center">
         <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/eVjzzSKEWV9E.png?imageslim">
      </p>
      

3. 传统的机器学习算法通过使用预处理阶段来处理图结构的数据。在这一阶段，算法将图结构数据映射到一个更简单的表达`representation`，如一个实值向量。即：预处理阶段首先将图结构的数据“压缩” 为实值向量，然后使用 `list-based`数据处理技术来处理。

   在数据压缩过程中，一些重要的信息（如：每个顶点的拓扑依赖性）可能会被丢失。最终的效果严重依赖于数据预处理算法。

   最近有很多算法尝试在预处理阶段保留数据的图结构性质，其基本思路是：利用图的结点之间的拓扑关系对图结构的数据进行编码，从而在数据预处理过程中融合图的结构信息。递归神经网络`RNN`、马尔科夫链都是这类技术。

   论文 `《The graph neural network model》` 提出了 `GNN` 模型，该模型扩展了`RNN` 和马尔科夫链技术，适合处理图结构的数据。

   - `GNN` 既适用于 `graph-focused` 任务，又适用于 `node-focused` 任务。
   - `GNN` 扩展了`RNN`，它可以处理更广泛的图任务，包括无环图、带环图、有向图、无向图等。
   - `GNN` 扩展了马尔科夫链，它通过引入学习算法并扩大了可建模的随机过程的类型从而扩展了随机游走理论。

4. `GNN` 是基于消息扩散机制 `information diffusion mechanism` 。图由一组处理单元来处理，每个处理单元对应于图的一个顶点。

   - 处理单元之间根据图的连接性来链接。
   - 处理单元之间交换信息并更新单元状态，直到达到稳定的平衡为止。
   - 通过每个处理单元的稳定状态可以得到对应顶点的输出。
   - 为确保始终存在唯一的稳定状态，消息扩散机制存在约束。

### 1.1 模型

1. 定义图$G=( V, E)$，其中$V$为顶点集合、$E$为边集合。边可以为有向边，也可以为无向边。

   对于顶点$v$，定义$\mathcal N_v$为其邻接顶点集合，定义$\mathcal E_v$为包含顶点$v$的边的集合。

2. 顶点和边可能含有额外的信息，这些信息统称为标签信息（它和监督学习中的标记`label` 不是一个概念）。

   - 标签信息以实值向量来表示，定义顶点$v$的标签为$\vec l _v \in \mathbb R^{n_v}$，定义边$(v_1,v_2)$的标签为$\vec l_{v_1,v_2} \in \mathbb R^{n_E}$，其中$n_v$为顶点标签的维度、$n_E$为边标签的维度。

   - 顶点标签通常包含顶点的特征，边标签通常包含顶点之间关系的特征。如下图中：

     - 顶点标签可能代表区块的属性，如：面积、周长、颜色的平均强度。
     - 边标签可能代表区块之间的相对位置，如：重心之间的距离、轴线之间的角度。

     <p align="center">
        <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/fh0UD2gdVMXq.png?imageslim">
     </p>
     

3. 一个直觉的想法是：将图中的顶点视作对象或者概念 `concept`，边表示对象之间的关系。每个对象通过各自的特征以及关联对象的特征来定义。因此，我们可以通过顶点$v$包含的信息，以及顶点$v$邻居顶点包含的信息，从而给顶点$v$定义一个状态向量$\mathbf{\vec x}_v \in \mathbb R^s$，其中$s$为状态向量的维度。

   -$\mathbf{\vec x}_v$是顶点$v$对应对象的 `representation`，可用于产生顶点$v$的输出$\mathbf{\vec o}_v \in \mathbb R^m$。

   - 定义局部转移函数 `local transition function`$f_w$：它是一个参数可学习的函数，刻画了顶点$v$和邻居顶点的依赖性。

    $\mathbf{\vec x}_v = f_w( {\vec l}_v, {\vec l}_{\mathcal E_v}, \mathbf{\vec x}_{\mathcal N_v}, {\vec l}_{\mathcal N_v})$

     其中：

     -$\vec l_v$为顶点$v$的标签信息向量。
     -$\vec l_{\mathcal E_v}$为包含顶点$v$的所有边的标签信息向量拼接的向量。
     -$\mathbf{\vec x}_{\mathcal N_v}$为顶点$v$的所有邻居的状态向量拼接的向量。
     -${\vec l}_{\mathcal N_v}$为顶点$v$的所有邻居的标签信息向量拼接的向量。

   - 定义局部输出函数 `local output function`$g_w$：它也是一个参数可学习的函数，刻画了顶点$v$的输出。

    $\mathbf{\vec o}_v = g_w(\mathbf{\vec x}_v, {\vec l}_v)$

4. 可以采取不同的邻域定义。

   - 可以移除${\vec l}_{\mathcal N_v}$，因为$\mathbf{\vec x}_{\mathcal N_v}$中已经包含了邻居顶点的信息。
   - 邻域不仅可以包含顶点$v$的直接链接的顶点，还可以包含顶点$v$的二跳、三跳的顶点。

5. 上述定义仅仅针对无向图，也可以通过改造$f_w$来支持有向图。我们引入一个变量$d_l, l \in \mathcal E_v$：如果边$l$的终点为$v$则$d_l = 1$；如果边$l$的起点为$v$则$d_l = 0$。则有：

  $\mathbf{\vec x}_v = f_w( {\vec l}_v, {\vec l}_{\mathcal E_v}, \mathbf {\vec d}_{\mathcal E_v}, \mathbf{\vec x}_{\mathcal N_v}, {\vec l}_{\mathcal N_v})$

6. 如果图的顶点可以划分为不同的类别，则对不同类别的顶点建立不同的模型是合理的。假设顶点$v$的类别为$k_v$，转移函数为$f_w^{k_v}$，输出函数为$g_w^{k_v}$，对应参数为$w_{k_v}$，则有：

   但是为了表述方便，我们假设所有的顶点都共享相同的参数。

7. 令$\mathbf{\vec x}, \mathbf{\vec o},\vec l,\vec l_V$分别代表所有顶点状态向量的拼接、所有顶点输出向量的拼接、所有标签（包含顶点标签以及边的标签）向量的拼接、所有顶点标签向量的拼接：

   则有：

   其中：

   -$F_w$称作全局转移函数`global transition fucntion`，它由$|V|$个$f_w$组成。
   -$G_w$称作全局输出函数 `global output function` ，它由$|V|$个$g_w$组成。

   令图和顶点的 `pair` 对的集合为$\mathcal D = \mathcal G\times \mathcal V$，其中$\mathcal G$为所有图的集合，$\mathcal V$为这些图中所有顶点的集合。全局转移函数和全局输出函数定义了一个映射$\varphi_w: \mathcal D \rightarrow \mathbb R^m$，它以一个图作为输入，然后对每个顶点$v$返回一个输出$\mathbf{\vec o}_v$。

8. 图结构的数据可以是位置相关的 `positional` 或者是位置无关的 `non-positional` 。前面描述的都是位置无关的，位置相关的有所不同：需要为顶点$v$的每个邻居分配唯一的整数标识符来指示其位置。即：对于顶点$v$的邻域$\mathcal N_v$，存在一个映射函数$\nu_v: \mathcal N_v \rightarrow \{1,2,\cdots,|\mathcal N_v|\}$，使得顶点$v$的每个邻居$u \in \mathcal N_v$都有一个位置$\nu_v(u)$。

   - 邻居位置可以隐式的表示一些有意义的信息。 如下图所示，邻居位置可以表示区块的相对空间位置。在这个例子中，我们可以按照顺时针来枚举顶点的邻居。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/l20HR4OXcgT6.png?imageslim">
   </p>
   

   - 对于位置相关的图，$f_w$必须接收邻居顶点的位置信息作为额外的输入。

     实际中我们可以根据邻居的位置进行排序，然后对$\vec l_{\mathcal E_v}, \mathbf{\vec x}_{\mathcal N_v}, \vec l_{\mathcal N_v}$按照排序之后的顺序进行拼接。如果在某些位置处的邻居不存在，则需要填充 `null` 值。

     如：

     其中：

     -$M = \max_{v,u}\nu_v(u)$为所有顶点的最大邻居数。

     -$\mathbf{\vec y}_i$为第$i$个位置邻居的状态向量：

       即：如果$u$为$v$的第$i$个邻居节点，则$\mathbf{\vec y}_i = \mathbf{\vec x}_u$；如果$v$没有第$i$个邻居节点，则$\mathbf{\vec y}_i$为`null` 值$\mathbf{\vec x}_0$。

   - 对于位置无关的图，我们可以将$f_w$替换为：

    $\mathbf{\vec x}_v = \sum_{u\in \mathcal N_v}h_{w} ({\vec l}_v,{\vec l}_{(u,v)}, \mathbf{\vec x}_{u},{\vec l}_{u})$

     其中$h_w$为待学习的函数，它和邻居的位置无关，也和邻居的邻居无关。这种形式被称作 `nonpositional form`，而原始形式被称作 `positional form`。

9. 为实现 `GNN` 模型，我们必须解决以下问题：

   - 求解以下方程的算法：
   - 从训练集种学习$f_w$和$g_w$参数的学习算法。
   -$f_w$和$g_w$的实现方式，即：解空间。

#### 1.1.1 方程求解算法

1. `Banach` 不动点理论为上述方程解的存在性和唯一性提供了理论依据。根据 `Banach` 不动点理论，当$F_w$满足以下条件时，方程$\mathbf{\vec x} = F_w(\mathbf{\vec x}, {\vec l})$存在唯一解：$F_w$是一个收缩映射 `contraction map` 。即存在$\mu, 0\le\mu\lt 1$，使得对任意$\mathbf{\vec x}, \mathbf{\vec y}$都有：

  $||F_w(\mathbf{\vec x}, {\vec l}) - F_w(\mathbf{\vec y}, {\vec l})|| \le \mu ||\mathbf{\vec x} - \mathbf{\vec y}||$

   其中$||\cdot||$表示向量范数。

   本文中我们假设$F_w$是一个收缩映射。实际上在 `GNN` 模型中，这个条件是通过适当的选择转移函数来实现的。

2. `Banach` 不动点理论不仅保证了解的存在性和唯一性，还给出了求解的方式：采用经典的迭代式求解：

  $\mathbf{\vec x}(t+1) = F_w(\mathbf{\vec x}(t), {\vec l})$

   其中$\mathbf{\vec x}(t)$表示状态$\mathbf{\vec x}$的第$t$次迭代值。

   - 对于任意初始值$\mathbf{\vec x}(0)$，上式指数级收敛到方程$\mathbf{\vec x} = F_w(\mathbf{\vec x}, {\vec l})$的解。

   - 我们将$\mathbf{\vec x}(t)$视为状态向量，它由转移函数$F_w$来更新。因此输出向量$\mathbf{\vec o}_v(t)$和状态向量$\mathbf{\vec x}(t)$的更新方程为：

     这可以解释为由很多处理单元`unit`组成的神经网络，每个处理单元通过$f_w$计算其状态，并通过$g_w$计算其输出。这个神经网络被称作编码网络 `encoding network`，它类似于 `RNN` 的编码网络。

     在编码网络中，每个单元根据邻居单元的状态、当前顶点的信息、邻居顶点的信息、边的信息，通过$f_w$计算下一个时刻的状态$\mathbf{\vec x}_v(t+1)$。

     > 注意：这里没有根据$\mathbf{\vec x}_v(t+1)$没有考虑$\mathbf{\vec x}_v(t)$，也就是没有自环。

   - 当$f_w$和$g_w$通过前馈神经网络实现时，编码网络就成为 `RNN` ，其中神经元之间的连接可以分为内部连接和外部连接。内部连接由实现神经元的神经网络架构决定，外部连接由图的链接决定。

     如下图所示：上半图对应一个图`Graph`，中间图对应于编码网络，下半图对应于编码网络的展开图。在展开图中，每一层`layer` 代表一个时间步，`layer` 之间的链接（外部连接）由图的连接性来决定，`layer` 内神经元的链接（内部连接）由神经网络架构决定。

     <p align="center">
        <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/eiJirjiT73ST.png?imageslim">
     </p>
     

#### 1.1.2 参数学习算法

1. 假设训练集为：

   其中$G_i$表示第$i$个图，$V_i$表示第$i$个图的顶点集合，$E_i$表示第$i$个图的边集合，$v_{i,j}$表示第$i$个图的第$j$个顶点，$\mathbf{\vec t}_{i,j}$表示顶点$v_{i,j}$的监督信息`target` ，$q_i$为图$G_i$中的标记样本。

   - 训练集的所有图$G_i$也可以合并为一张大图，其中大图内部存在孤立的子区域（各个子图）。因此上述框架可以简化为：

    $\mathcal L = (G,\mathcal T)$

     其中$G=(V,E)$表示大图，$\mathcal T = \{(v_i,\mathbf{\vec t}_i)\mid v_i\in V, \mathbf{\vec t}_i\in \mathbb R^m, 1\le i\le q\}$表示 “顶点-目标” `pair` 对的集合。这种表述方式不仅简单而且实用，它直接捕获了某些仅包含单个图的问题的本质。

   - 对于 `graph-focused` 任务可以引入一个特殊的顶点，该顶点和任务目标相关。只有该顶点包含监督信息，即$q_i = 1$。

   - 对于`node-focused` 任务，每个顶点都可以包含监督信息。

   假设采用平方误差，则训练集的损失函数为：

  $\mathcal L_w = \sum_{i=1}^p\sum_{j=1}^{q_i}||\mathbf{\vec t}_{i,j} - \varphi_w(G_i,v_{i,j})||_2^2$

   也可以在损失函数中增加罚项从而对模型施加约束。

2. 我们可以基于梯度下降算法来求解该最优化问题，求解方法由以下几步组成：

   - 通过下面的迭代公式求解求解$\mathbf{\vec x}_v(t)$，直到时间$T$：

     其解接近$\mathbf{\vec x} = F_w(\mathbf{\vec x}, {\vec l})$的不动点：$\mathbf{\vec x}(T) \simeq \mathbf{\vec x}$。

     注意：这一步要求$F_w$是一个压缩映射，从而保证方程能够收敛到一个不动点。

   - 求解梯度$\nabla_{\mathbf{\vec w}} \mathcal L_w$。

     这一步可以利用 `GNN` 中发生的扩散过程以非常高效的方式进行。这种扩散过程与 `RNN` 中发生的扩散过程非常相似，而后者是基于`backpropagation-through-time: BPTT` 算法计算梯度的。

     `BPTT` 是在展开图上执行传统的反向传播算法。 首先计算时间步$T$的损失函数，以及损失函数在每个时间步$t$相对于$f_w$和$g_w$的梯度。最终$\nabla_{\mathbf{\vec w}} \mathcal L_w(T)$由所有时间步的梯度之和得到。

     `BPTT` 要求存储每个单元在每个时间步$t$的状态$\mathbf {\vec x}(t)$，当$T-t_0$非常大时内存需求太大。为解决该问题，论文基于 `Almeida-Pineda` 算法提出了一个非常高效的处理方式：由于我们假设状态向量最终收敛到不动点$\mathbf{\vec x}$，因此我们假设对于任意$t\ge t_0$都有$\mathbf{\vec x}(t) = \mathbf{\vec x}$。因此 `BPTT` 算法仅需要存储$\mathbf{\vec x}$即可。

     下面两个定理表明：这种简单直观方法的合理性。

     - 定理：如果全局转移函数$F_w(\mathbf{\vec x}, {\vec l})$和全局输出函数$G_w(\mathbf{\vec x}, {\vec l}_V)$对于状态$\mathbf{\vec x}$和参数$\mathbf{\vec w}$都是连续可微的，则$\varphi_w$对于参数$\mathbf{\vec w}$也是连续可微的。

       其证明见原始论文。值得注意的是，对于一般动力学系统而言该结论不成立。对于这些动力学系统而言，参数的微小变化会迫使其从一个固定点转移到另一个固定点。而 `GNN` 中的$\varphi_w$可微的原因是由于$F_w$是收缩映射。

     - 定理：如果全局转移函数$F_w(\mathbf{\vec x}, {\vec l})$和全局输出函数$G_w(\mathbf{\vec x}, {\vec l}_V)$对于状态$\mathbf{\vec x}$和参数$\mathbf{\vec w}$都是连续可微的，定义$\mathbf{\vec z}(t) \in \mathbb R^s$为：

      $\mathbf{\vec z}(t) = \left(\frac{\partial F_w(\mathbf{\vec x},\vec l) }{\partial \mathbf{\vec x}}\right)^T\mathbf{\vec z}(t+1)+ \left(\frac{\partial G_w(\mathbf{\vec x},\vec l_V)}{\partial \mathbf{\vec x}}\right)^T\nabla_{\mathbf {\vec o}}\mathcal L_w(t)$

       则序列$\mathbf{\vec z}(T),\mathbf{\vec z}(T-1),\cdots$以指数级收敛到一个向量$\mathbf{\vec z} = \lim_{t\rightarrow -\infty} \mathbf{\vec z}(t)$，且收敛结果和初始状态$\mathbf{\vec z}(T)$无关。

       更进一步有：

      $\nabla_{\mathbf{\vec w}} \mathcal L_w = \left(\frac{\partial G_w(\mathbf{\vec x},\vec l_V)}{\partial \mathbf{\vec w}}\right)^T\nabla_{\mathbf {\vec o}}\mathcal L_w + \left(\frac{\partial F_w(\mathbf{\vec x},\vec l)}{\partial \mathbf{\vec w}}\right)^T \mathbf{\vec z}$

       其中$\mathbf{\vec x}$为`GNN` 的不动点，$\mathbf{\vec z}$为上述收敛的向量。

       证明见论文原文。其中：

       - 第一项表示输出函数$G_w$对于梯度的贡献。反向传播的梯度在通过$g_w$的 `layer` 时计算这一项。

       - 第二项表示转移函数$F_w$对于梯度的贡献。反向传播的梯度在通过$f_w$的 `layer` 时计算这一项。

         可以证明：

        $\mathbf{\vec z}(t) = \sum_{i=T}^t\frac{\partial \mathcal L_w(T)}{\partial \mathbf{\vec x}(i)}$

         即：$\mathbf{\vec z}(t)$等价于$\frac{\partial \mathcal L_w(T)}{\partial \mathbf{\vec x}(i)}$的累加。

         证明过程：

         其中最外层的$()^T$表示矩阵的转置，内层的$()^i$表示矩阵的幂。

         考虑到收敛结果和初始状态$\mathbf{\vec z}(T)$无关，因此假设：

        $\mathbf{\vec z}(T) = \left(\frac{\partial G_w(\mathbf{\vec x}(T),\vec l_V)}{\partial \mathbf{\vec x}(T)}\right)^T\nabla_{\mathbf {\vec o}}\mathcal L_w(T)$

         并且当$t_0 \le t \le T$时有$\mathbf{\vec x}(t) = \mathbf{\vec x}$，以及$\mathcal L_w(t) = \mathcal L_w(T)$，则有：

         考虑到：

         因此有：

        $\mathbf{\vec z}(t) = \sum_{i=0}^{T-t} \frac{\partial \mathcal L_w(T)}{\partial \mathbf{\vec x}(T-i)} = \sum_{i=T}^t\frac{\partial \mathcal L_w(T)}{\partial \mathbf{\vec x}(i)}$

   - 通过梯度来更新参数$\mathbf{\vec w}$。

3. `GNN` 参数学习算法包含三个部分：

   - `FORWARD`前向计算部分：前向计算部分用于计算状态向量$\mathbf{\vec x}$，即寻找不动点。
   - `BACKWARD` 反向计算部分：反向计算部分用于计算梯度$\nabla_{\mathbf{\vec w}} \mathcal L_w$。
   - `MAIN` 部分：该部分用户求解参数。该部分更新权重$\mathbf{\vec w}$直到满足迭代的停止标准。

4. `FORWARD` 部分：

   - 输入：
     - 图$G = (V,E)$
     - 当前参数$\mathbf{\vec w}$
     - 迭代停止条件$\epsilon_f$
   - 输出：不动点$\mathbf{\vec x}$
   - 算法步骤：
     - 随机初始化$\mathbf{\vec x}(0)$，令$t=0$
     - 循环迭代，直到满足$||\mathbf{\vec x}(t) - \mathbf{\vec x}(t-1)||\le \epsilon_f$。迭代步骤为：
       - 计算$\mathbf{\vec x}(t+1)$：$\mathbf{\vec x}(t+1) = F_w(\mathbf{\vec x}(t),l)$
       - 令$t = t+1$
     - 返回$\mathbf{\vec x}(t)$

5. `BACKWARD` 部分：

   - 输入：

     - 图$G=(V,E)$
     - 不动点$\mathbf{\vec x}$
     - 当前参数$\mathbf{\vec w}$
     - 迭代停止条件$\epsilon_b$

   - 输出：

   - 算法步骤：

     - 定义：

     - 随机初始化$\mathbf{\vec z}(T)$，令$t = T$

     - 循环迭代，直到满足$||\mathbf{\vec z}(t-1) - \mathbf{\vec z}(t)||\le \epsilon_b$。迭代步骤为：

       - 更新$\mathbf{\vec z}(t)$：$\mathbf{\vec z}(t) = \mathbf A\mathbf{\vec z}(t+1) + \mathbf{\vec b}$
       - 令$t = t - 1$

     - 计算梯度：

      $\nabla_{\mathbf{\vec w}} \mathcal L_w = \left(\frac{\partial G_w(\mathbf{\vec x},\vec l_V)}{\partial \mathbf{\vec w}}\right)^T\nabla_{\mathbf {\vec o}}\mathcal L_w + \left(\frac{\partial F_w(\mathbf{\vec x},\vec l)}{\partial \mathbf{\vec w}}\right)^T \mathbf{\vec z}(t)$

     - 返回梯度$\nabla_{\mathbf{\vec w}}\mathcal L_w$

6. `Main` 部分：

   - 输入：
     - 图$G=(V,E )$
     - 学习率$\lambda$
   - 输出：模型参数$\mathbf{\vec w}$
   - 算法步骤：
     - 随机初始化参数$\mathbf{\vec w}$
     - 通过前向计算过程计算状态：$\mathbf{\vec x} = \text{Forward}(\mathbf{\vec w})$
     - 循环迭代，直到满足停止条件。循环步骤为：
       - 通过反向计算过程计算梯度：$\nabla_{\mathbf{\vec w}} \mathcal L_w = \text{Backward}(\mathbf{\vec x},\mathbf{\vec w})$
       - 更新参数：$\mathbf{\vec w} = \mathbf{\vec w} - \lambda \nabla_{\mathbf{\vec w}} \mathcal L_w$
       - 通过新的参数计算状态：$\mathbf{\vec x} = \text{Forward}(\mathbf{\vec w})$
     - 返回参数$\mathbf{\vec w}$

7. 目前 `GNN` 只能通过梯度下降算法求解，非梯度下降算法目前还未解决，这是未来研究的方向。

8. 实际上编码网络仅仅类似于静态的前馈神经网络，但是编码网络的`layer` 层数是动态确定的，并且网络权重根据输入图的拓扑结构来共享。因此为静态网络设计的二阶学习算法、剪枝算法、以及逐层学习算法无法直接应用于 `GNN` 。

#### 1.1.3 转移函数和输出函数

1. 局部输出函数$g_w$的实现没有任何约束。通常在 `GNN` 中，$g_w$采用一个多层前馈神经网络来实现。

2. 局部转移函数$f_w$在 `GNN` 中起着关键作用，它决定了不动点的存在性和唯一性。`GNN` 的基本假设是：全局转移函数$F_w$是收缩映射。

   论文给出了两种满足该约束的$f_w$的实现，它们都是基于`nonpositional form`，`positional form` 也可以类似地实现。

3. `nonpositional linear GNN` 线性 `GNN`：

  $h_{w} ({\vec l}_v,{\vec l}_{(u,v)}, \mathbf{\vec x}_{u},{\vec l}_{u}) = \mathbf A_{v,u} \mathbf{\vec x}_u + \mathbf{\vec b}_v$

   其中$\mathbf{\vec b}_v\in \mathbb R^s$和矩阵$\mathbf A_{v,u}\in \mathbb R^{s\times s}$分别由两个前馈神经网络的输出来定义，这两个前馈神经网络的参数对应于 `GNN` 的参数。更准确的说：

   - 转移神经网络 `transition network` 是一个前馈神经网络，它用于生成$\mathbf A_{v,u}$。

     设该神经网络为一个映射$\phi_w:\mathbb R^{2n_V+n_E} \rightarrow \mathbb R^{s^2}$，则定义：

    $\mathbf A_{v,u} = \frac{\mu}{s\times |\mathcal N_u|}\mathbf B$

     其中：

     -$\mathbf B$是由$\phi_w(\vec l_v,\vec l_{v,u},\vec l_u)$的$s^2$个元素进行重新排列得到的矩阵。
     -$\mu\in (0,1)$为缩放系数，$\frac{\mu}{s\times |\mathcal N_u|}$用于对矩阵$\mathbf B$进行缩放。

   - 约束神经网络`forcing network` 是另一个前馈神经网络，它用于生成$\mathbf{\vec b}_v$。

     设该神经网络为一个映射$\rho_w: \mathbb R^{n_V} \rightarrow \mathbb R^s$，则定义：

    $\mathbf{\vec b}_v = \rho_w(\vec l_v)$

   假设有：$||\phi_w(\vec l_v,\vec l_{v,u},\vec l_u)||_1 \le s$，即$|\mathbf B|_1 \le s$。事实上如果转移神经网络的输出神经元采用有界的激活函数（如双曲正切），则很容易满足该假设。

   根据$h_{w} ({\vec l}_v,{\vec l}_{(u,v)}, \mathbf{\vec x}_{u},{\vec l}_{u}) = \mathbf A_{v,u} \mathbf{\vec x}_u + \mathbf{\vec b}_v$有：

  $F_w(\mathbf{\vec x},\vec l) = \mathbf A \mathbf{\vec x} + \mathbf{\vec b}$

   其中：

   -$\mathbf{\vec b}$是由所有的$\mathbf{\vec b}_v$拼接而来，$\mathbf{\vec x}$是由所有的$\mathbf{\vec x}_v$拼接而来：

   -$\mathbf A$是一个分块矩阵，每一块为$\bar{\mathbf A}_{v,u}$：

     其中：

     - 如果$u$是$v$的邻居顶点，则有$\bar{\mathbf A}_{v,u} = \mathbf A_{v,u}$
     - 如果$u$不是$v$的邻居顶点，则由$\bar{\mathbf A}_{v,u} = \mathbf 0$

   由于$\mathbf{\vec b}_v$和$\mathbf A_{v,u}$不依赖于状态$\mathbf{\vec x}$（它们仅仅依赖于图的结构和顶点标签信息、边标签信息），因此有：

  $\frac{\partial F_w(\mathbf{\vec x},\vec l)}{\partial \mathbf{\vec x}} = \mathbf A$

   则有：

   因此对于任意的参数$\mathbf{\vec w}$，$F_w(\cdot)$都是约束映射。

4. `nonpositional nonlinear GNN` 非线性 `GNN`：$h_{w} ({\vec l}_v,{\vec l}_{(u,v)}, \mathbf{\vec x}_{u},{\vec l}_{u})$通过一个多层前馈神经网络来实现。考虑到$F_w$必须是一个约束映射，因此我们在损失函数中增加罚项：

  $\mathcal L_w = \sum_{i=1}^p\sum_{j=1}^{q_i}||\mathbf{\vec t}_{i,j} - \varphi_w(G_i,v_{i,j})||_2^2 + \beta L\left(\left\|\frac{\partial F_w}{\partial \mathbf{\vec x}}\right\|_1\right)$

   其中罚项$L(\cdot)$定义为：

   超参数$\mu \in (0,1)$定义了针对$F_w$的约束。

### 1.2 模型分析

#### 1.2.1 RNN

1. 事实上，`GNN` 是其它已知模型的扩展，特别地，`RNN` 是 `GNN` 的特例。当满足以下条件时，`GNN` 退化为 `RNN`：
   - 输入为有向无环图。
   -$f_w$的输入为$\vec l_v, \mathbf{\vec x}_{ch[v]}$，其中$ch[v]$为顶点$v$的子结点的集合。
   - 一个超级源点$v_s$，从该源点可以到达其它所有顶点。该源点通常对应于 `graph-focused`任务的输出$\mathbf{\vec o}_{v_s}$。
2. 实现$f_w,g_w$的神经网络形式包括：多层前馈神经网络、`cascade correlation`、自组织映射 `self-orgnizing map`。在 `RNN` 中，编码网络采用多层前馈神经网络。这个简化了状态向量的计算。

#### 1.2.2 随机游走

1. 当选择$f_w$为线性函数时，`GNN` 模型还捕获了图上的随机游走过程。

   定义顶点的状态$\mathbf{\vec x}_v$为一个实数，其定义为：

  $x_v = \sum_{i\in pa[v]} a_{v,i}x_i$

   其中$pa[v]$表示顶点$v$的父节点集合；$a_{v,i}$为归一化系数，满足：

2. 事实上$x_v = \sum_{i\in pa[v]} a_{v,i}x_i$定义了一个随机游走生成器：

   -$a_{v,i}$表示当前位于顶点$v$时，随机转移到下一个顶点$i$的概率。
   -$x_v$表示当系统稳定时，随机游走生成器位于顶点$v$的概率。

3. 当所有的$x_v$拼接成向量$\mathbf{\vec x}$，则有：

   其中：

   可以很容易的验证$||\mathbf A||_1 = 1$。

   马尔可夫理论认为：如果存在$t$使得矩阵$\mathbf A^t$的所有元素都是非零的，则$x_v = \sum_{i\in pa[v]} a_{v,i}x_i$就是一个收缩映射。

   因此假设存在$t$使得矩阵$\mathbf A^t$的所有元素都是非零的，则图上的随机游走是 `GNN` 的一个特例，其中$F_w$的参数$\mathbf A$是一个常量随机矩阵，而不是由神经网络产生的矩阵。

#### 1.2.3 计算复杂度

1. 我们关心三种类型的 `GNN` 模型：`positional GNN` ，其中$f_w$和$g_w$通过前馈神经网络来实现；`nonpositional linear GNN`；`nonpositional nonlinear GNN` 。

   训练过程中一些复杂运算的计算复杂度见下表。为方便表述，我们假设训练集仅包含一张图。这种简化不影响结论，因为训练集所有的图总是可以合并为一张大图。另外，复杂度通过浮点预算量来衡量。

   具体推导见论文。其中：

   - `instruction` 表示具体的运算指令；`positional/non-linear/linear` 分别给出了三类 `GNN` 模型在对应运算指令的计算复杂度；`execs` 给出了迭代的次数。

   - 下图中的$e_w$就是损失函数$\mathcal L_w$，$\mathbf l$就是$\vec l$，$\mathbf l_N$就是$\vec l_V$，$\mathbf N$就是$V$。这是因为原始论文采用了不同的符号。

   -$p_w$为罚项。设${\mathbf A}= \frac{\partial F_w(\mathbf{\vec x},\vec l)}{\partial \mathbf{\vec x}}$，则有：

    $p_w = \sum_{j=1}^s L(||\mathbf{ A}^j||_1)=\sum_{u\in V}\sum_{j=1}^sL\left(\sum_{(v,u)\in E}\sum_{i=1}^s|\mathbf A^{v,u}_{i,j}|-\mu\right)=\sum_{u\in V}\sum_{j=1}^s\alpha_{u,j}$

     其中：

     -$\mathbf A^{v,u}_{i,j}$表示矩阵$\mathbf A$的分块$\mathbf A^{v,u}$的第$i$行第$j$列
     -$\mathbf{ A}^j$表示矩阵$\mathbf A$的第$j$列
     -$\alpha_{u,j}=L\left(\sum_{(v,u)\in E}\sum_{i=1}^s |\mathbf A_{i,j}^{v,u}|-\mu\right )$

   - 定义$\mathbf R^{v,u}$为一个矩阵，其元素为$\mathbf R^{v,u}_{i,j} = \alpha_{u,j}\times \text{sgn}(\mathbf A^{v,u}_{i,j})$，则$t_R$为：对所有的顶点$v$，满足$\mathbf R^{v,u} \ne \mathbf 0$的顶点$u$的数量的均值。通常它是一个很小的数值。

   -$\overrightarrow C_f$和$\overleftarrow C_f$分别表示前向计算$f$和反向计算$f$梯度的计算复杂度。

   -$\text{hi}$表示隐层神经元的数量，即隐层维度。如$\text{hi}_f$表示函数$f$的实现网络的隐层神经元数量。

   -$\text{it}_l$表示迭代的 `epoch` 数量，$\text{it}_b$表示平均每个`epoch` 的反向迭代次数(`BACKWARD` 过程中的循环迭代次数)，$\text{it}_f$表示平均每个`epoch` 的前向迭代次数（`FORWARD` 过程中的循环迭代次数）。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/AtM0rNW7odfg.png?imageslim">
   </p>
   

2. 当 `GNN` 模型训练完成之后，其推断速度也很快。

   - 对于`positional GNN`，其推断的计算复杂度为：$O(|V|\overrightarrow C_g + \text{it}_f|V|\overrightarrow C_f)$
   - 对于 `nonpositional nonliear GNN`，其推断的计算复杂度为：$O(|V|\overrightarrow C_g +\text{it}_f|E|\overrightarrow C_h)$
   - 对于 `nonpositional linear GNN`，其推断的计算复杂度为：$O(|V|\overrightarrow C_g+\text{it}_f|E|s^2+|V|\overrightarrow C_\rho+|E|\overrightarrow C_\phi)$

3. 推断阶段的主要时间消耗在计算状态$\mathbf{\vec x}$的重复计算中，每次迭代的计算代价和输入图的维度（如边的数量）成线性关系，和前馈神经网络的隐层维度成线性关系，和状态向量的维度成线性关系。

   - 线性 `GNN` 是一个例外。线性 `GNN` 的单次迭代成本是状态维度的二次关系。
   - 状态向量的收敛速度取决于具体的问题。但是 `Banach` 定理可以确保它是以指数级速度收敛。实验表明：通常`5` 到 `15` 次迭代足以逼近不动点。

4. 在 `positional GNN` 中转移函数需要执行$\text{it}_f|V|$次，在 `nonpositional nonliear GNN` 中转移函数需要执行$\text{it}_f|E|$次。虽然边的数量$|E|$通常远大于顶点数量$|V|$，但是`positional GNN` 和 `nonpositional nonlinear GNN`的推断计算复杂度是相近的，这是因为$f_w$的网络通常要比$h_w$的网络更复杂。

   - 在 `positional GNN` 中，实现$f_w$的神经网络有$M\times (s + n_E + 2n_V)$个神经元，其中$M$为所有顶点的最大邻居数量。因为$f_w$的输入必须确保能够容纳最多的邻居。
   - 在 `nonpositonal nonliear GNN`中，实现$h_w$的神经网络有$(s+n_E + 2n_V)$个神经元。

5. 在 `nonpositonal linear GNN` 中，转移函数就是一个简单的矩阵乘法，因此这部分的计算复杂度为$O(|E|s^2)$而不是$O(|E|\overrightarrow C_h)$。

   通常线性 `GNN` 比非线性 `GNN` 速度推断速度更快，但是前者的效果更差。

6. `GNN` 的训练阶段要比推断阶段消耗更多时间，主要在于需要在多个`epoch` 中重复执行 `forward` 和 `backward` 过程。

   实验表明：`forward` 阶段和 `backward` 阶段的时间代价都差不多。类似 `forward`阶段的时间主要消耗在重复计算$\mathbf{\vec x}(t)$，`backward` 阶段的时间主要消耗在重复计算$\mathbf{\vec z}(t)$。前述定理可以确保$\mathbf{\vec z}(t)$是指数级收敛的，并且实验表明$\text{it}_b$通常很小。

7. 训练过程中，每个 `epoch` 的计算代价可以由上表中所有指令的计算复杂度的加权和得到，权重为指令对应的迭代次数。

   - 所有指令的计算复杂度基本上都是输入图的维度（如：边的数量）的线性函数，也是前馈神经网络隐单元维度的线性函数，也是状态维度$s$的线性函数。

     有几个例外，如计算$\mathbf{\vec z}(t) = \mathbf A^T \mathbf{\vec z}(t+1) + \mathbf{\vec b}, \mathbf A = \frac{F_w(\mathbf{\vec x,\vec l})}{\partial \mathbf{\vec x}}, \nabla_{\mathbf{\vec w}}p_w$的计算复杂度都是$s$的平方关系。

   - 最耗时的指令是 `nonpositional nonlinear GNN` 中计算$\nabla_{\mathbf{\vec w}}p_w$，其计算复杂度为$t_R\times \max(s^2\text{hi}_h,\overleftarrow C_h)$。

     - 实验表明，通常$t_R$是一个很小的数字。在大多数 `epoch` 中$t_R=0$，因为雅可比矩阵$\mathbf A$并未违反施加的约束；另一些情况中，$t_R$通常在 `1~5` 之间。

       因此对于较小的状态维度$s$，计算$\nabla_{\mathbf{\vec w}}p_w$的复杂度较低。

     - 理论上，如果$s$非常大则可能导致$s^2\times \text{hi}_h \gg \overleftarrow C_h$。如果同时还有$t_R\gg 0$，则这将导致计算$\nabla_{\mathbf{\vec w}}p_w$非常慢。但是论文表示未在实验中观察到何种情况。

#### 1.2.4 不动点

1. `GNN` 的核心是不动点理论，通过顶点的消息传播使得整张图的每个顶点的状态收敛，然后在收敛的状态基础上预测。

   这里存在两个局限：

   - `GNN` 将顶点之间的边仅仅视为一种消息传播手段，并未区分边的功能。

   - 基于不动点的收敛会导致顶点之间的状态存在较多的消息共享，从而导致顶点状态之间过于光滑 `over smooth` ，这将使得顶点之间缺少区分度。

     如下图所示，每个像素点和它的上下左右、以及斜上下左右八个像素点相邻。初始时刻蓝色没有信息量，绿色、黄色、红色各有一部分信息。

     开始时刻，不同像素点的区分非常明显；在不动点的收敛过程中，所有像素点都趋向于一致，最终整个系统的信息分布比较均匀。最终，虽然每个像素点都感知到了全局信息，但是我们已经无法根据每个像素点的最终状态来区分它们。

     <p align="center">
        <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/gAV56PJtcka4.gif">
     </p>
     

### 1.3 实验

1. 根据 `RNN` 的已有经验，`nonpositional`转移函数效果要优于 `positional` 转移函数，因此这里测试了 `nonpositional linear GNN` 和 `nonpositional nonlinear GNN` 。

   所有`GNN` 中涉及到的函数，如 `nonpositional linear GNN` 中的$g_w,\phi_w,\rho_w$，以及 `nonpositional nonlinear GNN` 中的$g_w,h_w$都采用三层的前馈神经网络来实现，并使用 `sigmoid` 激活函数。

2. 数据集划分为训练集、验证集和测试集。

   - 如果原始数据仅包含一张大图$G$，则训练集、验证集、测试集分别包含$G$的不同顶点。
   - 如果原始数据包含多个图$G_i$，则每张图整个被划分到训练集、验证集、测试集之一。

   所有模型都随机执行五次并报告在测试集上指标的均值。每次执行时都随机生成训练集：以一定的概率$\delta$随机连接每一对顶点，直到图的连接性满足指定条件。

   - 对于分类问题，$\mathbf{\vec t}_{i,j}$为一个标量，取值范围为$\{+1,-1\}$。模型的评估指标为预测准确率：如果$t_{i,j} \varphi_w(G_i,v_{i,j}) \gt 0$则分类正确；否则分类不正确。

   - 对于回归问题，$\mathbf{\vec t}_{i,j}$为一个标量，取值范围为$\mathbb R$。模型的评估指标为相对误差：

    $\left|\frac{t_{i,j} - \varphi_w(G_i,v_{i,j})}{t_{i,j}}\right|$

   在每轮实验中，训练最多 `5000` 个 `epoch`，每隔 `20` 个 `epoch` 就在验证集上执行一次评估。然后选择验证集损失最小的模型作为最佳模型，从而在测试集上评估。

#### 1.3.1 子图匹配问题

1. 子图匹配问题：在更大的图$G$中寻找给定的子图$S$。即，需要学习一个函数$\tau$：如果顶点$v_{i,j}$属于图$G_i$中的一个和$S$同构的子图，则$\tau(G_i,v_{i,j}) = 1$；否则$\tau(G_i,v_{i,j}) = -1$。

   如下图所示，图$G_1,G_2$都包含子图$S$。顶点内的数字表示顶点的标签信息向量$\vec l_v$（这里是一个标量）。最终学到的函数$\tau$为：如果为黑色顶点则$\tau(G_i,v_{i,j}) = 1$，否则$\tau(G_i,v_{i,j}) = -1$。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/nKQK4BhdcAI3.png?imageslim">
   </p>
   

2. 子图匹配问题有很多实际应用，如：目标定位、化合物检测。子图匹配问题是评估图算法的基准测试。实验表明 `GNN` 模型可以处理该任务。

   - 一方面 `GNN` 模型解决子图匹配问题的结果可能无法与该领域的专用方法相比，后者的速度更快、准确率更高。
   - 另一方面 `GNN` 模型是一种通用算法，可以在不经修改的情况下处理子图匹配问题的各自扩展。如：同时检测多个子图、子图的结构和标签信息向量带有噪音、待检测的目标图$G_i$是未知的且仅已知它的几个顶点。

3. 数据集：由 `600` 个随机图组成（边的连接概率为$\delta = 0.2$），平均划分为训练集、验证集、测试集。在每轮实验中，随机生成一个子图$S$，将子图$S$插入到数据集的每个图中。因此每隔图$G_i$至少包含了$S$的一份拷贝。

   - 每个顶点包含整数标签，取值范围从 `[0,10]`。
   - 我们使用一个一个均值为`0`、标准差为 `0.25` 的高斯噪声添加到标签上，结果导致数据集中每个图对应的$S$的拷贝都不同。
   - 为了生成正确的监督目标$t_{i,j}$，我们使用一个暴力搜索算法从每个图$G_i$中搜索$S$。

4. `GNN` 配置：

   - 所有实验中，状态向量的维度$s=5$
   - 所有实验中，`GNN` 的所有神经网络的隐层为三层，隐层维度为 `5` 。

   为评估子图匹配任务中，标签信息和图结构的相对重要性，我们还应用了前馈神经网络`FNN` 。`FNN` 的隐层为三层、隐层维度为 `20` 、输入层维度为`1`、输出层维度为 `1`。因此 `FNN` 仅仅使用标签信息$l_{v_{i,j}}$来预测监督目标$t_{i,j}$，它并没有利用图的结构。

5. 实验结果如下图所示，其中 `NL` 表示 `nonpositional nonlinear GNN`，`L` 表示 `nonpositional linear GNN` ，`FNN`表示前馈神经网络。

   结论：

   - 正负顶点的比例影响了所有方法的效果。

     - 当$|S|$接近$|G|$时，几乎所有顶点都是正样本，所有方法预测的准确率都较高。
     - 当$|S|$只有$|G|$的一半时，正负顶点比较均匀，次数所有方法预测的准确率都较低。

   - 子图规模$|S|$影响了所有方法的结果。

     因为标签只能有 `11` 种不同取值，当$|S|$很小时，子图的大多数顶点都可以仅仅凭借其标签来识别。因此$|S|$越小预测准确率越高，即使是在$|G| = 2|S|$时。

   - `GNN` 总是优于 `FNN`，这表明 `GNN` 可以同时利用标签内容和图的拓扑结构。

   - 非线性 `GNN` 略优于线性 `GNN`，这可能是因为非线性 `GNN` 实现了更为通用的模型，它的假设空间更大。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/3r0IsRCuLJ7z.png?imageslim">
   </p>
   

6. 为评估`GNN` 的计算复杂度和准确性，我们评估了不同顶点数、不同边数、不同隐层维度、不同状态向量维度的效果。

   在基准情况下：训练集包含`10` 个随机图，每个图包含`20` 个顶点和 `40` 条边；`GNN` 隐层维度为`5`，状态向量维度为 `2` 。

   `GNN` 训练 `1000` 个 `epoch` 并报告十次实验的平均结果。如预期的一样，梯度计算中需要的 `CPU` 时间随着顶点数量、边的数量、隐层维度呈线性增长，随着状态向量维度呈二次增长。

   - 下图为顶点数量增加时，梯度计算花费的`CPU` 时间。实线表示非线性`GNN`，虚线表示线性 `GNN` 。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/IpdMSFNPE14i.png?imageslim">
   </p>
   

   - 下图为状态向量维度增加时，梯度计算花费的 `CPU` 时间。实线表示非线性`GNN`，虚线表示线性 `GNN` 。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/3D9sJPPvNuaY.png?imageslim">
   </p>
   

7. 非线性 `GNN` 中，梯度和状态向量维度的二次关系取决于计算雅可比矩阵$\frac{\partial F_w(\mathbf{\vec x},\vec l)}{\partial \mathbf{\vec x}}$以及梯度$\nabla_{\mathbf{\vec w}} p_w$的时间代价。下图给出了计算梯度过程中的总时间代价。

   - 线条 `-o-` 给出了计算$\mathcal L_w$和$\nabla_{\mathbf{\vec w}}\mathcal L_w$的时间代价
   - 线条 `-*-` 给出了计算雅可比矩阵$\frac{\partial F_w(\mathbf{\vec x},\vec l)}{\partial \mathbf{\vec x}}$的时间代价
   - 线条 `-x-` 给出了计算$\nabla_{\mathbf{\vec w}} p_w$的时间代价
   - 点线 `...`和给出了剩下的前向计算的时间代价
   - 虚线 `---`给出了剩下的反向计算的时间代价
   - 实线表示剩下的计算梯度的时间代价

   可以看到：$\nabla_{\mathbf{\vec w}} p_w$的计算复杂度虽然是状态向量维度的二次关系，但是实际上影响较小。实际上该项的计算复杂度依赖于参数$t_R$：对所有的顶点$v$，满足$\mathbf R^{v,u} \ne \mathbf 0$的顶点$u$的数量的均值。通常它是一个很小的数值。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/ooNsycdgfidQ.png?imageslim">
   </p>
   

   下图给出每个`epoch` 中$\mathbf R^{v,u} \ne \mathbf 0$的顶点$u$的数量的直方图。可以看到$\mathbf R^{v,u}$的顶点$u$的数量通常为零，且从未超过`4` 。

   另外下图也给出计算稳定状态$\mathbf{\vec x}$和计算梯度（如计算$\mathbf{\vec z}$）所需要的平均迭代次数的直方图，可以看到这些值通常也很小。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/t1bSMQT5JKWI.png?imageslim">
   </p>
   

#### 1.3.2 Mutagenesis问题

1. `Mutagenesis` 数据集：一个小型数据集，经常作为关系学习和 `inductive logic programming` 中的基准。它包含 `230` 种硝基芳香族化合物的数据，这些化合物是很多工业化学反应中的常见中间副产品。

   任务目标是学习识别 `mutagenic` 诱变化合物。我们将对数诱变系数 `log mutagenicity` 的阈值设为`0`，因此这个任务是一个二类分类问题。

   数据集中的每个分子都被转换为一张图：

   - 顶点表示原子、边表示原子键 `atom-bond：AB` 。平均的顶点数量大约为 `26` 。

   - 边和顶点的标签信息包括原子键 `AB`、原子类型、原子能量状态，以及其它全局特征。全局特征包括：

     - `chemical measurement`化学度量 `C`： 包括 `lowest unoccupied molecule orbital`， `the water/octanol partition coefficient` 。
     - `precoded structural` 预编码结构属性 `PS` 。

     另外原子键可以用于定义官能团 `functional groups:FG` 。

   - 在每个图中存在一个监督顶点：分子描述中的第一个原子。如果分子为诱变的，则该顶点的期望输出为`1`；否则该顶点的期望输出为 `-1` 。

   - 在这 `230` 个分子中，有 `188` 个适合线性回归分析，这部分分子被称作回归友好 `regression friendly`；剩下的 `42` 个分子称作回归不友好 `regression unfriendly` 。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/7Xd4hFKwmdsC.png?imageslim">
   </p>
   

2. `GNN` 在诱变化合物问题上的结果如下表所示。我们采用十折交叉验证进行评估：将数据集随机拆分为十份，重复实验十次，每次使用不同的部分作为测试集，剩余部分作为训练集。我们运行`5` 次十折交叉，并取其均值。

   在回归友好分子上的效果：

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/XJGA5C4w2r5E.png?imageslim">
   </p>
   

   在回归不友好分子上的效果：

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/TnMCuI7fkOYF.png?imageslim">
   </p>
   

   在所有分子上的效果：

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/yN8GF6p1uTT1.png?imageslim">
   </p>
   

   结论：

   - `GNN` 在回归不友好分子和所有分子上的效果都达到最佳，在回归友好分子上的效果达到了最新的技术水平。
   - 大多数方法在应用于整个数据集时，相对于回归不友好分子部分显示出更高的准确率。但是`GNN` 与此相反。这表明 `GNN` 可以捕获有利于解决问题，但是在回归友好分子、回归不友好分子这两部分中分布不均的模式特征。

#### 1.3.3 Web PageRank

1. 受到谷歌的 `PageRank` 启发，这里我们的目标是学习一个网页排名。网页$v$的排名得分$p_v$定义为：

  $p_v = d\times \frac{\sum_{u\in pa[v]} p_u}{o_v} + (1-d)$

   其中：

   -$o_v$为顶点$v$的出度 `out-degree`
   -$d\in [0,1]$为阻尼因子 `damping factor`
   -$pa[v]$为顶点$v$的父顶点集合

   图$G$以$\delta = 0.2$随机生成，包含 `5000` 个顶点。训练集、验证集、测试集由图的不同顶点组成，其中 `50` 个顶点作为训练集、`50` 个顶点作为验证集、剩下顶点作为测试集。

   每个顶点$v$对应于一个二维标签$\vec l_v = [a_v,b_v]$，其中$a_v\in \{0,1\},b_v \in \{0,1\}$表示顶点$v$是否属于两个给定的主题：$[a_v,b_v]=[1,1]$表示顶点$v$同时属于这两个主题；$[a_v,b_v] = [1,0]$表示顶点$v$仅仅属于第一个主题；$[a_v,b_v]=[0,1]$表示顶点$v$仅仅属于第二个主题；$[a_v,b_v]=[0,0]$表示顶点$v$不属于任何主题。

   需要拟合的目标`target` 为：

   其中$\mathbf{\vec p}$表示每个顶点的`PageRank` 得分组成的向量。

2. 这里我们使用线性 `GNN` 模型，因为线性 `GNN` 模型很自然的类似于 `PageRank` 线性模型。

   - 转移网络和 `forcing` 网络都使用三层前馈神经网络，隐层维度为`5`
   - 状态向量维度为$s=1$
   - 输出函数为：$g_w(x_v,\vec l_v) = x_v^\prime \times \pi_w(x_v,\vec l_v)$。其中$\pi_w$为三层前馈神经网络，隐层维度为 `5` 。

3. 下图给出了 `GNN` 模型的结果。其中图 `(a)`给出了仅属于一个主题的结果，图 `(b)` 给出了其它网页的结果。

   实线表示目标$t_v$，点线表示 `GNN` 模型的输出。横轴表示测试集的顶点数量，纵轴表示目标得分$t_v$。顶点按照$t_v$得分进行升序排列。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/l9qKRIar8JC6.png?imageslim">
   </p>
   

   下图给出学习过程中的误差。实线为训练集的误差，虚线是验证集的误差。注意：两条曲线总是非常接近，并且验证集的误差在 `2400` 个 `epoch` 之后仍在减少。这表明尽管训练集由 `5000` 个顶点中的 `50` 个组成，`GNN` 仍然未经历过拟合。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/bzHaGbFEywdo.png?imageslim">
   </p>
   

## 二、GCN

1. 卷积神经网络 `CNN` 要求输入数据为网格结构，并且要求数据在网格中具有平移不变性，如一维语音、二维图像、三维视频都是这类数据的典型代表。`CNN` 充分利用了以下几个特点来大幅度降低模型的参数数量：

   - 权重共享：同一个卷积核可以应用于不同的位置。
   - 空间局部性：卷积核的大小通常都远远小于输入信号的尺寸。
   - 多尺度：通过步长大于一的卷积或者池化操作来减少参数，并获得更大的感受野 `receptive field` 。

   在网格数据结构中，顶点的邻居数量都是固定的。但是在很多任务中，数据并不是网格结构，如社交网络数据。在这些图数据结构中，顶点的邻居数量是不固定的。

   图结构是一种比网格结构更通用的结构，论文 `《Spectral Networks and Deep Locally Connected Networks on Graphs》` 在图结构上扩展了卷积的概念，并提出了两种构建方式：

   - 基于空域的卷积构建`Spatial Construction` ：直接在原始图结构上执行卷积。
   - 基于谱域的卷积构建`Spectral Construction` ：对图结构进行傅里叶变换之后，在谱域进行卷积。

2. 在网格数据中的卷积可以采用固定大小的卷积核来抽取特征；但是在图数据中，传统的卷积核无能为力。图卷积的本质是找到适合于图结构的可学习的卷积核。

### 2.1 空域构建

1. 空域构建主要考虑的是 `CNN` 的空间局部性、多尺度特点。

   给定图$G = (\mathbf\Omega, \mathbf W)$，其中$\mathbf\Omega$为大小为$m$的顶点集合，$\mathbf W\in \mathbb R^{m\times m}$为对称、非负的邻接矩阵。即这里为无向图。

   - 可以很容易的在图结构中推广空间局部性。对于顶点$j$，定义其邻域为：

    $\mathcal N_\delta(j) = \{i\mid i\in \mathbf\Omega, W_{i,j} \gt \delta \}$

     其中$\delta \gt 0$为阈值。

     在对顶点$j$卷积时我们仅考虑其邻域：

    $\mathbf{\vec o}_j = \sum_{i\in \mathcal N_\delta(j)} F_{i,j} \mathbf{\vec x}_i$

     其中$\mathbf{\vec x}_i$为顶点$i$的 `representation` 向量，$F_{i,j}$为卷积核。

   - `CNN` 通过池化层和下采样层来减小`feature map` 的尺寸，在图结构上我们同样可以使用多尺度聚类的方式来获得多尺度结构。

     在图上如何进行多尺度聚类仍然是个开发的研究领域，论文这里根据顶点的邻域进行简单的聚类。

     下图给出了多尺度层次聚类的示意图（两层聚类)：

     - 原始的`12` 个顶点为灰色，每个顶点可以同时被划分到多个聚类。
     - 第一层有`6` 个聚类，聚类中心为彩色顶点，聚类以彩色块给出。
     - 第二层有`3` 个聚类，聚类以彩色椭圆给出。

     <p align="center">
        <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/oXkIJNnDgBH3.png?imageslim">
     </p>
     

2. 现在考虑$K$个尺度。定义第 `0`个尺度表示原始图，即$\mathbf\Omega_0 = \mathbf\Omega$；对于之后每个尺度的`feature map`，定义为$\mathbf\Omega_k,k=1,2,\cdots,K$。我们采用聚类算法将`feature map`$\mathbf\Omega_{k-1}$划分到$d_k$个聚类，其中$d_0$为原始图的顶点数量$m$。

   定义$\mathbf\Omega_{k-1}$每个顶点的邻域集合的集合为：

  $\mathbb N_k = \{\mathcal N_{k,1},\cdots, \mathcal N_{k,d_{k-1}}\}$

   在$\mathbf\Omega_0$中的顶点就是原始图的顶点，在$\mathbf\Omega_{k},k=1,2,\cdots,K$中的顶点就是第$k-1$层聚类的中心点。

   不失一般性，假设$\mathbf\Omega_0$中原始图每个顶点的特征为标量。我们假设第$k$层卷积核的数量为$f_k$，则网络的第$k$层将$f_{k-1}$维特征（因为第$k-1$层卷积核数量为$f_{k-1}$）转化为$f_k$维特征。

   假设第$k$层网络的输入为：

   其中$\mathbf X ^k$的第$i$行定义为$\mathbf {\vec x}^k_j = (x^k_{j,1},\cdots ,x^k_{j,f_{k-1}})^T\in \mathbb R^{f_{k-1}}$为第$k-1$层聚类的第$j$个中心点的 `feature` ，$\mathbf X^k$为第$k$层网络的输入 `feature map`。

   网络第$k$层的输出经过三步：

   - 卷积操作：

     其中$i$表示输入通道，$j$表示输出通道，$v$表示当前顶点，$u$表示邻域顶点。

     写成矩阵的形式为：

    $o^k_{v,j} = \mathbf F^k_{j,v} * \mathbf X^k$

     其中$*$表示卷积操作，卷积核$\mathbf F^k_{j,v} \in \mathbb R^{d_{k-1}\times f_{k-1}}$，它和位置$v$有关，其定义为：

     则矩阵$\mathbf F^k_{j,v}$仅仅在顶点$v$的邻域所在的列非零，其它列均为零。

   - 非线性层：$h(o^k_{v,j})$。其中$h(\cdot)$为非线性激活函数。

   - 聚类：通过$\epsilon\text{-covering}$聚类算法（或者其它聚类算法）将$d_{k-1}$个顶点聚成$d_k$个簇。$\epsilon\text{-covering}$算法将所有的顶点聚合为$d_k$个簇，使得簇内距离小于$\epsilon$、簇间距离大于$\epsilon$。

   - 池化层：

     构造池化矩阵$\mathbf L^k$，行表示聚类 `cluster id`，列表示顶点`id` ，矩阵中的原始表示每个顶点对应于聚类中心的权重：如果是均值池化，则就是 `1` 除以聚类中的顶点数；如果是最大池化，则是每个聚类的最大值所在的顶点。

   最终第$k$层网络的输出为：

  $\mathbf H^k =\begin{bmatrix}h\left(\mathbf F^k_{1,1} * \mathbf X^k \right) & h\left(\mathbf F^k_{1,2} * \mathbf X^k \right) &\cdots & h\left(\mathbf F^k_{1,f_k} * \mathbf X^k \right) \\h\left(\mathbf F^k_{2,1} * \mathbf X^k \right) & h\left(\mathbf F^k_{2,2} * \mathbf X^k \right) &\cdots & h\left(\mathbf F^k_{2,f_k} * \mathbf X^k \right) \\\vdots&\vdots&\ddots &\vdots\\h\left(\mathbf F^k_{d_{k-1},1} * \mathbf X^k \right) & h\left(\mathbf F^k_{d_{k-1},2} * \mathbf X^k \right) &\cdots & h\left(\mathbf F^k_{d_{k-1},f_k} * \mathbf X^k \right) \end{bmatrix}\in \mathbb R^{d_{k-1}\times f_{k}} \\\mathbf X^k = \mathbf L^k \mathbf H ^k\in \mathbb R^{d_k \times f_k}$

   其中：矩阵$\mathbf H^k$的第$v$行第$j$列表示顶点$v$的第$j$个卷积核的输出。

   如下图所示$K=2$。

   -$\mathbf\Omega_0$表示第零层，它有 `12` 个顶点（灰色），每个顶点只有一个通道（标量）
   -$\mathbf\Omega_1$表示第一层，其输入为$\mathbf\Omega_0$，输出 `6`个顶点，每个顶点有四个通道（四个卷积核）
   -$\mathbf\Omega_2$表示第二层，其输入为$\mathbf \Omega_1$，输出 `3` 个顶点，每个顶点有六个通道（六个卷积核）

   每一层卷积都降低了空间分辨率`spatial resolution`，但是增加了空间通道数。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/dpH5EepLykfE.png?imageslim">
   </p>
   

3.$\mathbf\Omega_k$和$\mathbb N_k$的构建过程：

   - 初始化：$\mathbf W_0 = \mathbf W$

   - 构建簇中心连接权重：对于$\mathbf\Omega_k$，其簇中心$i,j$之间的连接权重为两个簇之间的所有连接的权重之和：

    $A_k(i,j) = \sum_{s\in \mathbf\Omega_k(i)}\sum_{t\in \mathbf\Omega_k(j)} W_{k-1}(s,t)$

     然后按行进行归一化：

    $\mathbf W_k = \text{row-normalize}(\mathbf A_k)$

   - 最后进行层次聚类，得到$\mathbf\Omega_k$和$\mathbb N_k$。

     这里聚类按照对$\mathbf W_k$的$\epsilon\text{-covering}$得到，理论上也可以采取其它聚类算法。

4. 假设$S_k$为$\mathbb N_k$中平均邻居数量，则第$k$层卷积的平均参数数量为：

  $O(S_k\times d_{k} \times f_k\times f_{k-1}) = O(m)$

   实际应用中我们可以使得$S_k\times d_k\simeq \alpha d_{k-1}$，$\alpha$为一个升采样系数，通常为$\alpha \in (1,4)$。

   其中$d_1,d_2,\cdots,d_K$依次递减，$f_1,f_2,\cdots,f_K$依次递增，总体有$d_{k-1}\times f_{k-1}\simeq m$，而$S_k\times d_k\times f_k\times f_{k-1}\simeq \alpha \times d_{k-1}\times f_{k-1}\times f_k\simeq \alpha\times m\times f_k = O(m)$。

5. 空域构建的实现非常朴素，其优点是不需要对图结构有很高的规整性假设 `regularity assumption`。缺点是无法在顶点之间实现权重共享。

### 2.2 图卷积

#### 2.2.1 拉普拉斯算子

1. 给定向量场$\mathbf{\vec F}(\mathbf{\vec x})$，设$\Sigma$为围绕某一点$\mathbf{\vec x}$的一个封闭曲面，$dS$为曲面上的微元，$\mathbf{\vec n}$为该微元的法向量，则该曲面的通量为：

  $\mathbf\Phi_{\mathbf{\vec F}}(\Sigma) = \oint_{\Sigma} \mathbf{\vec F} \cdot \mathbf{\vec n} d S$

   当$\Sigma$趋近于零时，即可得到$\mathbf{\vec x}$点的散度：

  $\text{div}\mathbf{\vec F}(\mathbf{\vec x}) = \nabla\cdot \mathbf{\vec F} = \nabla\cdot \mathbf{\vec F} = \sum_{i=1}^n \frac{\partial F_i}{\partial x_i}$

   其中$\mathbf{\vec x} = (x_1,\cdots,x_n)^T, \mathbf{\vec F} = (F_1,\cdots,F_n)^T$。

   散度的物理意义为：在向量场中从周围汇聚到该点或者从该点流出的流量。

2. 给定向量场$\mathbf{\vec F}(\mathbf{\vec x})$，设$\Gamma$为围绕某一点$\mathbf{\vec x}$的一个封闭曲线，$dl$为曲线上的微元，${\vec \tau }$为该微元的切向量，则该曲线的环量为：

  $\mathbf \Theta_{\mathbf {\vec F}} (\Gamma) = \oint_{\Gamma} \mathbf{\vec F} \cdot \vec\tau dl$

   当$\Gamma$趋近于零时，即可得到$\mathbf{\vec x}$点的旋度：

  $\text{curl} \mathbf{\vec F}(\mathbf{\vec x}) = \nabla \times \mathbf{\vec F}$

   - 在三维空间中，上式等于：
   - 旋度的物理意义为：向量场对于某点附近的微元造成的旋转程度，其中:
     - 旋转的方向表示旋转轴，它与旋转方向满足右手定则
     - 旋转的大小是环量与环面积之比

3. 给定函数$f(\mathbf{\vec x})$，其中$\mathbf{\vec x} = (x_1,\cdots,x_n)^T$，则有：

   - 梯度：

    $\nabla f = \left(\frac{\partial f}{\partial x_1},\cdots\frac{\partial f}{\partial x_n}\right)^T$

     梯度的物理意义为：函数值增长最快的方向。

   - 梯度的散度为拉普拉斯算子，记作：

    $\nabla^2f = \nabla\cdot \nabla f = \sum_{i=1}^n\frac{\partial^2 f}{\partial x_i^2}$

     - 由于所有的梯度都朝着$f$极大值点汇聚、从$f$极小值点流出，因此拉普拉斯算子衡量了空间中每一点，该函数的梯度是倾向于流出还是流入。
     - 拉普拉斯算子也能够衡量函数的平滑度`smoothness`：函数值没有变化或者线性变化时，二阶导数为零；当函数值突变时，二阶导数非零。

4. 假设$f(x)$为离散的一维函数，则一阶导数为一阶差分：

  $f^\prime(x) = \frac{\partial f(x)}{\partial x} \simeq f(x+1) - f(x)$

   二阶导数为二阶差分：

  $\nabla^2f = f^{\prime\prime}(x) = \frac{\partial ^2 f(x)}{\partial x^2} = f^\prime(x) - f^\prime(x-1) = [f(x+1) - f(x)] - [f(x) - f(x-1)]\\= f(x+1) + f(x-1) - 2f(x)$

   一维函数其自由度可以理解为`2`，分别是 `+1` 和 `-1` 两个方向。因此二阶导数等于函数在所有自由度上微扰之后获得的增益。

   推广到图$G=(V,E)$，其中顶点数量为$|V|$。假设邻接矩阵为$\mathbf W$，任意两个顶点之间存在边（如果不存在边则$w_{i,j} = 0$）。对于其中任意顶点$v$，对其施加微扰之后该顶点可以到达其它任意顶点，因此图的自由度为$|V|$。

   令$f_i$为函数$f$在顶点$i$的值，定义$\mathbf{\vec f} = (f_1,f_2,\cdots,f_{|V|})^T\in \mathbb R^{|V|}$，它表示函数$f$在图$G=(V,E)$上的取值。对于顶点$i$，其扰动到顶点$j$的增益时$(f_j-f_i)$，不过这里通常写成负的形式，即$(f_i-f_j)$。考虑到边的权重，则增益为：$w_{i,j}(f_i-f_j)$。

   对于顶点$i$，总的增益为拉普拉斯算子在顶点$i$的值。即：

  $(\nabla^2 f)_i = \sum_j \frac{\partial^2 f_i}{\partial j^2} \simeq \sum_{j\in \mathcal N_i}w_{i,j}(f_i-f_j)=\sum_{j}w_{i,j}(f_i-f_j)\\= (\sum_j w_{i,j}) f_i - \sum_j w_{i,j}f_j\\= (\mathbf D \mathbf{\vec f})_i - (\mathbf W \mathbf{\vec f})_i= ((\mathbf D - \mathbf W)\mathbf{\vec f})_i$

   其中$\mathcal N_i$为顶点$i$的邻域，$\mathbf D$为图的度矩阵。

   考虑所有的顶点，则有：

  $\nabla^2 \mathbf{\vec f} = (\mathbf D - \mathbf W) \mathbf{\vec f}$

   定义拉普拉斯矩阵$\mathbf L = \mathbf D - \mathbf W$，因此在图的拉普拉斯算子就是拉普拉斯矩阵。

   上述结果都是基于$f_i$为标量推导，实际上当$f_i$为向量时也成立。

5. 图的拉普拉斯矩阵$\mathbf L$是一个半正定对称矩阵，它具有以下性质：

   - 对称矩阵一定有$m$个线性无关的特征向量，其中$m$为顶点数。
   - 半正定矩阵的特征值一定是非负的。
   - 杜晨矩阵的特征向量相互正交，即：所有特征向量构成的矩阵为正交矩阵。

   因此有拉普拉斯矩阵的谱分解：

  $\mathbf L\mathbf{\vec u}_k = \lambda_k \mathbf{\vec u}_k$

   其中$\mathbf{\vec u}_k$为第$k$个特征向量，$\lambda_k$为第$k$个特征值。

   解得：$\mathbf L = \mathbf U \mathbf\Lambda \mathbf U^T$，其中 ：

  $\mathbf U$每一列为特征向量构成的正交矩阵，$\mathbf\Lambda$为对应特征值构成的对角矩阵。

#### 2.2.2 卷积

1. 给定函数$f(x)$， 其傅里叶变换为：

  $f(x) = \int_{-\infty}^{\infty} F(k)e^{ikx} dk$

   其中$F(k) = \frac{1}{2\pi}\int_{-\infty}^{\infty} f(x) e^{-ikx} dx$为傅里叶系数，即频率为$k$的振幅，$e^{-iwx}$为傅里叶基 `fouries basis` 。

   可以证明：$e^{-ikx}$为拉普拉斯算子的特征函数。证明：

  $\nabla^2 e^{_-ikx} = \frac{\partial^2 e^{-ikx}}{\partial x^2} = -k^2 e^{-ikx}$

2. 如果将傅里叶变换推广到图上，则有类比：

   - 拉普拉斯算子对应于拉普拉斯矩阵$\mathbf L$。

   - 频率$k$对应于拉普拉斯矩阵的特征值$\lambda_k$。

   - 傅里叶基$e^{-ikx}$对应于特征向量$\mathbf{\vec u}_k$。

   - 傅里叶系数$F(k)$对应于$F(\lambda_k)$，其中

    $F(\lambda_k) = \hat f_k = \mathbf{\vec f}\cdot \mathbf{\vec u}_k\\$

     写成矩阵形式为：

    $\hat{\mathbf{\vec f}} = \mathbf U^T \mathbf{\vec f}$

     其中$\hat{\mathbf{\vec f}}$为图的傅里叶变换，它是不同特征值下对应的振幅构成的向量；$\mathbf{\vec f} \in \mathbb R^{m}$为顶点特征构成的$m$维向量。

   - 传统的傅里叶逆变换：

    $\mathcal F^{-1}(F(k)) = f(x) = \int_{-\infty}^\infty F(k)e^{ikx} dk$

     对应于：

    $f_i = \sum_{k=1}^m \hat f_k u_{k,i}$

     其中$u_{k,i}$对应于特征向量$\mathbf{\vec u}_k$的第$i$个分量。

     写成矩阵的形式为：

    $\mathbf{\vec f} = \mathbf U\hat{\mathbf{\vec f}}$

   因此图的傅里叶变换是将图从`Spatial Domain` 转换到 `Spectural Domain` 。

3. 卷积定理：两个函数在时域的卷积等价于在频域的相乘。

   对应于图上有：

  $\mathbf{\vec f}*\mathbf{\vec h} = \mathcal F^{-1}(\hat{\mathbf{\vec f}}\circ \hat{\mathbf{\vec h}})=\mathbf U\left(\mathbf K (\mathbf U^T\mathbf{\vec f})\right)= \mathbf U\mathbf K\mathbf U^T\mathbf{\vec f}$

   其中$\circ$为逐元素乘积，$\mathbf U$为拉普拉斯矩阵$\mathbf L$特征向量组成的矩阵，$\mathbf K$为对角矩阵：

   这里将逐元素乘积转换为矩阵乘法。

4. 图卷积神经网络的核心就是设计卷积核，从上式可知卷积核就是$\mathbf K$。我们可以直接令$\mathbf{\vec h}\cdot \mathbf{\vec u}_k = \theta_k$，然后学习卷积核：

   我们并不关心$\mathbf{\vec h}$，仅仅关心傅里叶变换之后的$\hat{\mathbf{\vec h}}$。

### 2.3 频域构建

1. 假设构建一个$K$层的卷积神经网络，在第$k$层将输入的 `feature map`$\mathbf X^k\in \mathbb R^{|\mathbf\Omega| \times f_{k-1}}$映射到$\mathbf X^{k+1}\in \mathbb R^{|\mathbf\Omega| \times f_{k}}$，则第$k$层网络为：

  $\mathbf{\vec x}^{k+1}_{\cdot,j} = h\left(\mathbf U\sum_{i=1}^{f_{k-1}} \mathbf K^k_{j,i}\mathbf U^T\mathbf{\vec x}^k_{\cdot,i}\right)$

   其中$\mathbf{\vec x}^k_{\cdot,i}$表示$\mathbf X^k$的第$k$列，$h(\cdot)$为非线性激活函数，$\mathbf U$为拉普拉斯矩阵特征向量组成的矩阵，$\mathbf K^k_{j,i}\in \mathbb R^{m\times m}$为第$k$层针对第$i$个 `feature map` 的第$j$个卷积核。

   > 将$\mathbf X^k$按列进行计算，则根据空域构建的傅里叶变换有：

   - 实际应用中，通常仅仅使用拉普拉斯矩阵的最大$d$个特征向量，截断阈值$d$取决于图的固有规整性 `regularity` 以及图的顶点数量。此时上式中的$\mathbf U$替换为$\mathbf U_d$。这可以减少参数和计算量，同时去除高频噪声。

   - 第$k$层卷积核的参数数量为：$f_{k-1}\times f_k\times m$。

   - 非线性必须作用在空域而不是频域，这使得我们必须使用代价很高的矩阵乘法将频域的卷积结果映射回空域。

     如何在频域进行非线性处理目前还未解决。

2. 我们还可以使用三次样条插值来增加平滑约束。具有空间局部性的函数（如脉冲函数）在谱域具有连续的频率响应。

   此时有：

  $\vec\theta^k_{j,i}=(\theta^k_{j,i,1},\cdots,\theta^k_{j,i,m})^T = \mathcal K^k \vec \alpha^k_{i,j}$

   其中$\mathcal K^k\in \mathbb R^{m\times q_k}$为三次样条核函数，$\alpha^k_{i,j}\in \mathbb R^{q_k}$为$q_k$个样条参数，$k$代表第$k$层网络，$j$代表第$j$个输出`feature map`，$i$代表第$i$个输入 `feature map` 。

   假设采样步长正比于顶点数量，即步长$\alpha \sim m$，则$q_k\sim m\times \frac{1}{\alpha} = O(1)$， 则频域卷积的参数数量降低为：$f_{k-1}\times f_k$。

3. 三次样条曲线：给定以下顶点：$\{(x_0,y_0),\cdots,(x_n,y_n)\}$，其中$a = x_0\lt x_1\lt\cdots\lt x_n = b$，定义样条曲线$S(x)$，它满足以下条件：

   - 在每个分段区间$[x_i,x_{i+1}]$，$S(x) = S_i(x)$为一个三次多项式，且满足$S(x_i) = y_i$
   - 函数$S(x)$、一阶导数$S^\prime(x)$、二阶导数$S^{\prime\prime}(x)$在 `[a,b]` 上是连续的。

   令$S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i), i =0,1,\cdots,n-1$，则有：

   令$h_i=x_{i+1} - x_i$，则有：

   - 根据$S_i(x_i) = y_i$得到：$a_i = y_i$

   - 根据$S_{i}(x_{i+1}) = y_{i+1}$得到：

    $a_i + h_ib_i+h_i^2c_i + h_i^3d_i = y_{i+1}$

   - 根据$S^\prime_i(x_{i+1}) = S_{i+1}^\prime(x_{i+1})$得到：

    $b_i + 2h_ic_i + 3h_i^2d_i- b_{i+1} = 0$

   - 根据$S^{\prime\prime}_i(x_{i+1}) = S_{i+1}^{\prime\prime}({i+1})$得到：

    $2c_i + 6h_id_i -2c_{i+1} = 0$

   令$m_i= S_{i}^{\prime\prime}(x_i) = 2c_i$，则可以得到：

   代入$b_i + 2h_ic_i + 3h_i^2d_i- b_{i+1} = 0$有：

  $h_im_i + 2(h_i+h_{i+1})m_{i+1} + h_{i+1}m_{i+2} = 6\left(\frac{y_{i+2}-y_{i+1}}{h_{i+1}} - \frac{y_{i+1} - y_i}{h_i}\right)$

   由$i$的取值范围可知，一共有$n-1$个公式，但是却有$n+1$个未知量。如果需要求解方程，则需要两个端点的限制。有三种限制条件：

   - 自由边界`Natural`：首尾两端的曲率为零，即$S^{\prime\prime}(x_0) = 0, S^{\prime\prime}(x_n) = 0$，即$m_0=0,m_n=0$。则有：

   - 固定边界`Clamped`：首位两端的一阶导数被指定，即$S^\prime(x_0) = A,S^\prime(x_n) = B$，即$b_0 = A, b_{n-1} =B$。则有：

   - 非结点边界`Not-A-Knot` ：指定样条曲线的三次微分匹配，即$S_0^{\prime\prime\prime}(x_1) = S_1^{\prime\prime\prime}(x_1), S_{n-2}^{\prime\prime\prime}(x_{n-1}) = S_{n-1}^{\prime\prime\prime}(x_{n-1})$。则有：

     不同边界的效果如下：

     <p align="center">
        <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/y9b0KfQgWmcc.png?imageslim">
     </p>
     

4. 频域卷积的问题：

   - 计算复杂度太高。我们需要对拉普拉斯矩阵进行谱分解求解$\mathbf U$，当图很大时谱分解的计算代价太大。另外每次前向传播都需要计算矩阵乘积，其计算复杂度为$O(m^2)$。
   - 每个卷积核的参数数量为$O(m)$，当图很大时参数数量太大。
   - 卷积的空间局部性不好。

### 2.4 实验

1. 论文对 `MNIST` 数据集进行实验，其中`MNIST` 有两个变种。所有实验均使用 `ReLU` 激活函数以及最大池化。

   模型的损失函数为交叉熵，初始学习率为`0.1` ，动量为 `0.9` 。

#### 2.4.1 降采样 MNIST

1. 我们将`MNIST` 原始的 `28x28` 的网格数据降采样到 `400` 个像素，这些像素仍然保留二维结构。由于采样的位置是随机的，因此采样后的图片无法使用标准的卷积操作。

   采样后的图片的示例，空洞表示随机移除的像素点。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/OxxPiPpV4GtL.png?imageslim">
   </p>
   

   采用空域卷积进行层次聚类的结果，不同的颜色表示不同的簇，颜色种类表示簇的数量。图 `a` 表示$k=1$，图 `b` 表示$k=3$。可以看到：层次越高，簇的数量越少。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/l0kbSsopV3nB.png?imageslim">
   </p>
   

   采用频域卷积的拉普拉斯特征向量（特征值降序排列）的结果，结果经过了傅里叶逆变换。图`a` 表示$\mathbf{\vec v}_2$，图`b` 表示$\mathbf{\vec v_{20}}$。可以看到：特征值越大的特征向量对应于低频部分，特征值越小的部分对应于高频部分。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/kRCP5YDcBA9t.png?imageslim">
   </p>
   

2. 不同模型在 `MNIST` 上分类的结果如下。基准模型为最近邻模型 `kNN` ，`FCN` 表示带有 `N` 个输出的全连接层，`LRFN` 表示带有 `N`个输出的空域卷积层，`MPN` 表示带有 `N` 个输出的最大池化层，`SPN` 是带有 `N` 个输出的谱域卷积层。

   - 基准模型 `kNN` 的分类性能比完整的（没有采样的）`MNIST` 数据集的 `2.8%` 分类误差率稍差。
   - 两层全连接神经网络可以将测试误差降低到 `1.8%` 。
   - 两层空域图卷积神经网络效果最好，这表明空域卷积层核池化层可以有效的将信息汇聚到最终分类器中。
   - 谱域卷积神经网络表现稍差，但是它的参数数量最少。
   - 采用平滑约束的谱域卷积神经网络的效果优于常规的谱域卷积神经网络。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/Dcd4kGB6Fsea.png?imageslim">
   </p>
   

3. 由于 `MNIST` 中的数字由笔画组成，因此具有局部性。空域卷积很明确的满足这一约束，而频域卷积没有强制空间局部性。

   - 图 `(a),(b)` 表示同一块感受野在空域卷积的不同层次聚类中的结果。
   - 图 `(c),(d)` 表示频域卷积的两个拉普拉斯特征向量，可以看到结果并没有空间局部性。
   - 图 `(e),(f)` 表示采用平滑约束的频域卷积的两个拉普拉斯特征向量，可以看到结果有一定的空间局部性。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/SRrOC0GB4Hci.png?imageslim">
   </p>
   

#### 2.4.2 球面 MNIST

1. 我们将`MNIST` 图片映射到一个球面上，构建方式为：

   - 首先从单位球面上随机采样 `4096`个点$\mathbb S =\{s_1,\cdots,s_{4096}\}$。
   - 然后考虑三维空间的一组正交基$\mathbf E = (\mathbf{\vec e}_1,\mathbf{\vec e}_2,\mathbf{\vec e}_3)$，其中$||\mathbf{\vec e}_1|| = 1,||\mathbf{\vec e}_2||=2,||\mathbf{\vec e}_3||=3$，以及一个随机方差算子$\mathbf\Sigma = (\mathbf E + \mathbf W)^T(\mathbf E + \mathbf W)$，其中$\mathbf W$是一个方差为$\sigma^2\lt 1$的独立同部分的高斯分布的分布矩阵。
   - 对原始 `MNIST` 数据集的每张图片，我们采样一个随机方差$\Sigma_i$并考虑其`PCA` 的一组基$\{\mathbf{\vec u}_1,\mathbf{\vec u}_2,\mathbf{\vec u}_3 \}$。这组基定义了一个平面内的旋转。我们将图片按照这组基进行旋转并使用双三次插值将图片投影到$\mathbb S$上。

   由于数字 `6` 和 `9` 对于旋转是等价的，所以我们从数据集中移除了所有的 `9` 。

   下面给出了两个球面 `MNIST` 示例：

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/IYRBA5n7CFNI.png?imageslim">
   </p>
   

   下面给出了频域构建的图拉普拉斯矩阵的两个特征向量。结果经过了傅里叶逆变换。图`a` 表示$\mathbf{\vec v}_{20}$，图`b` 表示$\mathbf{\vec v_{100}}$。可以看到：特征值越大的特征向量对应于低频部分，特征值越小的部分对应于高频部分。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/jrcnEthWnyCA.png?imageslim">
   </p>
   

2. 首先考虑“温和”的旋转：$\sigma^2=0.2$，结果如下表所示。

   - 基准的 `kNN` 模型的准确率比上一个实验（随机采样 `MNIST` ）差得多。
   - 所有神经网络模型都比基准 `KNN` 有着显著改进。
   - 空域构建的卷积神经网络、频域构建的卷积神经网络在比全连接神经网络的参数少得多的情况下，取得了相差无几的性能。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/aBv5a6AjF3GW.png?imageslim">
   </p>
   

3. 不同卷积神经网络学到的卷积核如下图所示。

   - 图 `(a),(b)` 表示同一块感受野在空域卷积的不同层次聚类中的结果。
   - 图 `(c),(d)` 表示频域卷积的两个拉普拉斯特征向量，可以看到结果并没有空间局部性。
   - 图 `(e),(f)` 表示采用平滑约束的频域卷积的两个拉普拉斯特征向量，可以看到结果有一定的空间局部性。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/09uUBjsmL2Be.png?imageslim">
   </p>
   

4. 最后我们考虑均匀旋转，此时$\{\mathbf{\vec u}_1,\mathbf{\vec u}_2,\mathbf{\vec u}_3 \}$代表$\mathbb R^3$中的随机的一组基，此时所有的模型的效果都较差。这时需要模型有一个完全的旋转不变性，而不仅仅是平移不变性。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/S0jGils8qi8k.png?imageslim">
   </p>
   

## 三、Fast GCN

1. 基于空域卷积可以通过有限大小的卷积核来实现空间局部性，但是由于在图上无法定义平移，因此它无法实现平移不变性。

   基于频域卷积可以通过平滑约束实现近似的空间局部性，但是频域定义的卷积不是原生的空间局部性，并且频域卷积涉及到计算复杂度为$O(m^2)$的矩阵乘法，其中$m$为图中顶点数量。

   论文 `《Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering》` 提出了一种快速的、满足空间局部性的卷积核。这种卷积核有以下优点：

   - 新卷积核与经典`CNN` 卷积核的计算复杂度相同，适用于所有的图结构。

     在评估阶段，其计算复杂度正比于卷积核的尺寸$K$核边的数量$|E|$。由于真实世界的图几乎都是高度稀疏的，因此有$|E| \ll m^2$以及$|E|= \bar k \times m$，其中$\bar k$为所有顶点的平均邻居数。

   - 新卷积核具有严格的空间局部性，对于每个顶点$i$，它仅考虑$i$附近半径为$K$的球体，其中$K$表示距离当前顶点的最短路径，即$K$跳`hop`。`1` 跳表示直接邻居。

### 3.1 模型

1. 模型的整体架构如下图所示。其中：

   - `1,2,3,4` 为图卷积的四个步骤。
   - 特征抽取层获取粗化图的多通道特征。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/znCwFJvOYy3W.png?imageslim">
   </p>
   

2. 将卷积推广到图上需要考虑三个问题：

   - 如何在图上涉及满足空间局部性的卷积核
   - 如何执行图的粗化`graph coarsening`，即：将相似顶点聚合在一起
   - 如何执行池化操作

#### 3.1.1 卷积核

1. 给定输入$\mathbf{\vec x}$，其频域卷积为：$\mathbf{\vec y} = \mathbf U \mathbf K \mathbf U^T\mathbf{\vec x}$，其中$\mathbf K$为卷积核：

   其中$\vec\theta=(\theta_1,\cdots,\theta_m)^T\in \mathbb R^{m}$为卷积核参数。

   这种卷积有两个不足：

   - 频域卷积在空域不满足空间局部性
   - 卷积核参数数量为$O(m)$

   我们可以通过多项式卷积核来克服这两个缺陷。令：

  $\theta_i = \sum_{k=0}^{K-1} \alpha_k\times \lambda_i^k$

   其中$\lambda_i$为拉普拉斯矩阵$\mathbf L$的第$i$个特征值。即用特征值的$K-1$次多项式来拟合卷积核参数。

   则有：

   由于$\mathbf U\mathbf\Lambda\mathbf U^T = \mathbf L, \mathbf U^T\mathbf U = \mathbf I$，因此有$\mathbf U\mathbf \Lambda^k\mathbf U^T = \mathbf L^k$。

   则卷积结果为：

  $\mathbf{\vec y} = \mathbf U \left(\sum_{k=0}^{K-1}\alpha_k\mathbf\Lambda^k\right)\mathbf U^T\mathbf{\vec x}= \sum_{k=0}^{K-1}\alpha_k \mathbf U\mathbf\Lambda^k\mathbf U^T\mathbf{\vec x} = \sum_{k=0}^{K-1}\alpha_k \mathbf L^k\mathbf{\vec x}$

   现在的参数为$\vec\alpha = (\alpha_0,\cdots,\alpha_{K-1})\in \mathbb R^K$，参数数量降低到$O(K)$。

2. 给定 `Kronecker delta` 函数$\vec\delta_i=(\delta_{i,1},\cdots,\delta_{i,m})^T\in \mathbb R^m$，其中：

   即$\vec\delta_i$的第$i$个分量为`1`，其余分量为`0` 。

   则有：

  $\mathbf{\vec y}_{\delta_i} = \sum_{k=0}^{K-1}\alpha_k \mathbf L^k\vec\delta_i$

   输出的第$j$个分量记作$y_{i,j}$，令$\mathbf L^k$的第$j$行第$i$列为$L^k(j,i)$，则有：

  $y_{i,j} = \sum_{k=0}^{K-1}\alpha_k\times L^k(j,i)$

   可以证明：当$d_{\mathcal G}(i,j) \gt k$时，有$L^k(j,i) = 0$，其中$d_{\mathcal G}$为顶点$i$和$j$之间的最短路径。

   因此当$d_{\mathcal G}(i,j) \gt K\gt K-1\gt\cdots\gt 1$时，有$L^{K-1}(j,i)=\cdots = L^0(j,i)=0$，则有$y_{i,j} = 0$。即顶点$i$经过卷积之后，只能影响距离为$K$范围内的输出。

   因此$K-1$阶多项式实现的频域卷积满足$K$阶局部性，其卷积核大小和经典 `CNN` 相同。

3. 切比雪夫多项式：

   - 第一类切比雪夫多项式：

   - 第二类切比雪夫多项式：

   - 第一类切比雪夫多项式性质：

     -$T_n(\cos(\theta)) = \cos (n\theta)$
     - 切比雪夫多项式$T_n(x)$是$n$次代数多项式，其最高次幂$x^n$的系数为$2^{n-1}$
     - 当$|x|\le 1$时，有$|T_n(x)| \le 1$
     -$T_n(x)$在$[-1,1]$之间有$n+1$个点$x^*_k = \cos\left(\frac{k\pi}{n}\right),k=0,\cdots,n$，轮流取得最大值 `1` 和最小值 `-1` 。
     - 当$n$为奇数时$T_n(x)$为奇函数；当$n$为偶数时，$T_n(x)$为偶函数

   - 切比雪夫逼近定理：在$x \in [-1,1]$之间的所有首项系数为`1`的一切$n$次多项式中，$w_n(x) = 2^{1-n} T_n(x)$对 `0` 的偏差最小。即：

    $\max_{-1\le x\le 1}|w_n(x) - 0| \le \max_{-1\le x \le 1} |P_n(x) - 0|$

     其中$P_n(x)$为任意首项系数为 `1` 的$n$次多项式。

   - 定理：

     - 令$\mathcal C[a,b]$为定义域在$[a,b]$之间的连续函数的集合，$\mathcal P_n$为所有的$n$次多项式的集合。如果$f\in \mathcal C[a,b]$，则$\mathcal P_n$中存在一个$f$的最佳切比雪夫逼近多项式。
     - 如果$f\in \mathcal C[a,b],P\in \mathcal P_n$，则$P$是$f$的最佳切比雪夫逼近的充要条件是：$f-P$在$[a,b]$之间有一个至少有$n+2$个点的交错点组（即包含$n+2$个点$x_0\lt x_1\lt\cdots\lt x_{n+1}$）。

   - 推论：

     - 如果$f\in \mathcal C[a,b]$，则$\mathcal P_n$中存在唯一的、函数$f$的最佳切比雪夫逼近。
     - 如果$f\in \mathcal C[a,b]$，则其最佳一致逼近$n$次多项式就是$f$在$[a,b]$上的某个$n$次拉格朗日插值多项式。

4. 注意到这里的卷积$\mathbf{\vec y} = \mathbf U \mathbf K \mathbf U^T\mathbf{\vec x}$计算复杂度仍然较高，因为这涉及到计算复杂度为$O(m^2)$的 `矩阵-向量` 乘法运算。

   我们可以利用切比雪夫多项式来逼近卷积。

   定义：

  $\tilde{\mathbf\Lambda} = \frac{2}{\lambda_\max}\mathbf\Lambda - \mathbf I$

   为归一化的对角矩阵，其对角线元素都位于$[-1,1]$之间。

   令$\sum_{k=0}^{K-1}\alpha_k \mathbf\Lambda^k = \sum_{k=0}^{K-1}\beta_k T_k(\tilde{\mathbf\Lambda})$，其中$T_k$为$k$阶切比雪夫多项式，$\beta_k$为$k$阶切比雪夫多项式的系数。

   根据切比雪夫多项式的性质我们有：

   则卷积结果为：

  $\mathbf{\vec y} = \sum_{k=0}^{K-1}\alpha_k \mathbf U\mathbf\Lambda^k\mathbf U^T\mathbf{\vec x} =\sum_{k=0}^{K-1}\beta_k\mathbf UT_k(\tilde{\mathbf\Lambda}) \mathbf U^T\mathbf{\vec x}$

   由于有$\mathbf U\mathbf \Lambda^k\mathbf U^T = \mathbf L^k$以及$\mathbf U$为正交矩阵，因此有$\mathbf U\tilde{\mathbf\Lambda}^k\mathbf U^T = \tilde{\mathbf L}^k$，其中$\tilde{\mathbf L} = \frac{2}{\lambda_{\max}} \mathbf L - \mathbf I$。 考虑到$T_k(\tilde{\mathbf\Lambda})$为$\tilde{\mathbf\Lambda}$的多项式，则有：

  $\mathbf UT_k(\tilde{\mathbf\Lambda}) \mathbf U^T = T_k(\tilde{\mathbf L})$

   则有：

  $\mathbf{\vec y} = \sum_{k=0}^{K-1}\beta_kT_k(\tilde{\mathbf L})\mathbf{\vec x}$

   定义$\bar{\mathbf{\vec x}}_k = T_k(\tilde{\mathbf L})\mathbf{\vec x} \in \mathbb R^m$，则有：

   最终有：$\mathbf{\vec y} = \sum_{k=0}^{K-1} \beta_k\bar{\mathbf{\vec x}}_k$。其计算代价为$O(K\times |E|)$。考虑到$\mathbf L$为一个稀疏矩阵，则有$O(K\times |E|) \ll O(m^2)$。

5. 给定输入$\mathbf X\in \mathbb R^{m\times F_{in}}$，其中$F_{in}$为输入通道数。设$F_{out}$为输出通道数，则第$s$个顶点的第$j$个输出通道为：

  $y_{s,j} = \sum_{i=1}^{F_{in}}\sum_{k=0}^{K-1} \beta_{j,i,k} \bar{x}_{s,i,k}$

   其中$i$表示第$i$个输入通道，$\bar x_{s,i,k} = (T_k(\tilde{\mathbf L})_{s,\cdot}\cdot \mathbf{\vec x}_{\cdot,i})$为第$s$个顶点在输入通道$i$上的、对应于$T_k$的取值，$\mathbf{\vec x}_{\cdot,i}$为$\mathbf X$的第$i$列，$T_k()_{s,\cdot}$为$T_k$矩阵的第$s$行。$\beta_{j,i,k}$为可训练的参数，参数数量为$O(K\times F_{in}\times F_{out})$。

   定义$\vec\beta_{j,i} = (\beta_{j,i,1},\cdots,\beta_{j,i,K})^T$为第$j$个输出通道对应于第$i$个输入通道的参数，则有：

   以及：

  $\frac{\partial \mathcal L }{\partial x_{s,i}} = \sum_{j=1}^{F_{out}} \frac{\partial \mathcal L}{\partial y_{s,j}} \sum_{k=0}^{K-1}\beta_{j,i,k}T_k(\tilde{\mathbf L})_{s,s}$

   其中$T_k(\tilde{\mathbf L})_{s,s}$为矩阵的第$s$行第$s$列，$\mathcal L$为训练集损失函数，通常选择交叉熵损失函数。

   上述三个计算的复杂度为$O(K\times F_{out}\times F_{in}\times m\times |E|)$，但是可以通过并行架构来提高计算效率。

   另外，$\{\bar x_{s,i,k}\},s=1,2,\cdots,m,i=1,2,\cdots,F_{in},k=0,2,\cdots,K-1$只需要被计算一次。

#### 3.1.2 图粗化

1. 池化操作需要在图上有意义的邻域上进行，从而将相似的顶点聚合在一起。因此，在多层网络中执行池化等价于保留图的局部结构的多尺度聚类。

   但是图聚类是 `NP` 难的，必须使用近似算法。这里我们仅考虑图的多层次聚类算法，在该算法中，每个层次都产生一个粗化的图，这个粗化图对应于图的不同分辨率。

2. 论文选择使用 `Graclus` 层次聚类算法，该算法已被证明在图聚类上非常有效。每经过一层，我们将图的分辨率降低一倍。

   `Graclus` 层次聚类算法：

   - 选择一个未标记的顶点$i$，并从其邻域内选择一个未标记的顶点$j$，使得最大化局部归一化的割 `cut`$W_{i,j}(1/d_i + 1/d_j)$。

     然后标记顶点$(i,j)$为配对的粗化顶点，并且新的权重等于所有其它顶点和$i,j$权重之和。

   - 持续配对，直到所有顶点都被探索。这其中可能存在部分独立顶点，它不和任何其它顶点配对。

   这种粗化算法非常块，并且高层粗化的顶点大约是低一层顶点的一半。

#### 3.1.3 池化

1. 池化操作将被执行很多次，因此该操作必须高效。

   图被粗化之后，输入图的顶点及其粗化后的顶点的排列损失是随机的，取决于粗化时顶点的遍历顺序。因此池化操作需要一个表格来存储所有的配对顶点，这将导致内存消耗，并且运行缓慢、难以并行实现。

   可以将图的顶点及其粗化后的顶点按照固定的顺序排列，从而使得图的池化操作和经典的一维池化操作一样高效。这包含两步：

   - 创建一颗平衡二叉树
   - 重排列图的顶点或者粗化后的顶点

2. 在图的粗化之后，每个粗化后的顶点要么有两个子结点（低层的两个配对顶点），要么只有一个子结点（低层该子结点未能配对）。

   对于多层粗化图，从最高层到最底层，我们添加一批 “伪顶点” 添加到单个子结点的顶点，使得每个顶点都有两个子结点。同时每个“伪顶点”都有两个低层级的“伪顶点” 作为子结点。

   - 这批“伪顶点”和其它任何顶点都没有连接，因此是“断开”的顶点。
   - 添加假顶点人工增加了图的大小，从而增加计算成本。但是我们发现 `Graclus` 算法得到的单子结点的顶点数量很少。
   - 当使用 `ReLU` 激活函数以及最大池化时，这批“伪顶点” 的输入为 `0`。由于这些“伪顶点”是断开的，因此卷积操作不会影响这些初始的`0`值。

   最终得到一颗平衡二叉树。

3. 对于多层粗化图，我们在最粗粒度的图上任意排列粗化顶点，然后将其顺序扩散到低层图。假设粗化顶点编号为$k$，则其子结点的编号为$2k$和$2k+1$。因此这将导致低层顶点有序。

   池化这样一个重排列的图类似于一维池化，这使得池化操作非常高效。

   由于内存访问是局部的，因此可以满足并行架构的需求。

4. 一个池化的例子如下图。

   - 图粗化过程：$\mathcal G_0$为原始图，包含`8` 个顶点。深色链接表示配对，红色圆圈表示未能配对顶点，蓝色圆圈表示“伪顶点”

     - 一级粗化$\mathcal G_1$：

       1

       ```
       G0顶点    G1顶点    备注
       ```

       2

       ```
       0,1        0
       ```

       3

       ```
       4,5        2
       ```

       4

       ```
       8,9        4
       ```

       5

       ```
       6          3       未能配对
       ```

       6

       ```
       10         5       未能配对
       ```

     - 二级粗化$\mathcal G_2$：

       ```
       xxxxxxxxxx
       ```

       4

       1

       ```
       G1顶点     G2顶点    备注
       ```

       2

       ```
       2,3         1
       ```

       3

       ```
       4,5         2
       ```

       4

       ```
       0           0       未能配对
       ```

   - 构建二叉树过程：

     -$\mathcal G_2$的 `0`顶点只有一个子结点，因此在$\mathcal G_1$中添加一个“伪顶点”作为$\mathcal G_2$的 `0`顶点的另一个子结点。
     -$\mathcal G_1$的`3,5` 顶点只有一个子结点，因此在$\mathcal G_0$中添加两个“伪顶点”作为$\mathcal G_1$的 `3,5` 顶点的另一个子结点。
     -$\mathcal G_1$的`1`顶点是一个“伪顶点”，因此在$\mathcal G_0$中添加两个“伪顶点”作为$\mathcal G_1$的 `1`顶点的两个子结点。

   - 重新编号：

     - 将$\mathcal G_2$的三个顶点依次编号为 `0,1,2`
     - 根据$k \rightarrow (2k,2k+1)$的规则为$\mathcal G_1$的六个顶点依次编号为 `0,1,2,3,4,5`
     - 根据$k \rightarrow (2k,2k+1)$的规则为$\mathcal G_0$的十二个顶点依次编号为 `0,1,...,11`

   - 重排，重排结果见右图所示。

   - 按照重排后的一维顶点顺序，依次从低层向高层执行“卷积+池化”。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/zGuTCCRPvyEO.png?imageslim">
   </p>
   

### 3.2 实验

1. 我们将常规的`Graph CNN` 的卷积核称作 `Non-Param` 、增加平滑约束的样条插值卷积核称作 `Spline`、`Fast Graph CNN` 的卷积核称作 `Chebyshev` 。

   在这些图卷积神经网络中：

   - `FCk` 表示一个带$k$个神经元的全连接层；`Pk` 表示一个尺寸和步长为$k$的池化层（经典池化层或者图池化层）；`GCK` 表示一个输出$k$个`feature map` 的图卷积层；`Ck` 表示一个输出$k$个 `feature map` 的经典卷积层。
   - 所有的`FCk,Ck,GCk` 都使用`ReLU` 激活函数。
   - 所有图卷积神经网络模型的粗化算法统一为`Graclus` 算法而不是`Graph CNN` 中的朴素聚合算法，因为我们的目标是比较卷积核而不是粗化算法。
   - 所有神经网络模型的最后一层统一为 `softmax` 输出层
   - 所有神经网络模型采用交叉熵作为损失函数，并对全连接层的参数使用$L_2$正则化
   - 随机梯度下降的`mini-batch` 大小为 `100`

2. `MNIST` 实验：

   论文构建了`8` 层图神经网络，每张`MNIST`图片对应一个图`Graph`，图的顶点数量为`978` （$28\times 28 = 784$，再加上 `192` 个“伪顶点”）；边的数量为 `3198` 。

   顶点$i,j$之间的边的权重为：

  $W_{i,j} = \exp\left(- \frac{||\mathbf{\vec z}_i - \mathbf{\vec z}_j||_2^2}{\sigma^2}\right)$

   其中$\mathbf{\vec z}_i$表示顶点$i$的二维坐标。

   各模型的参数为：

   - 标准卷积核的尺度为 `5x5`，图卷积核的$K=25$，二者尺寸相同

     标准 `CNN` 网络从 `TensorFlow MNIST` 教程而来，`dropout = 0.5`，正则化系数为$5\times 10^{-4}$，初始学习率为`0.03`，学习率衰减系数 `0.95`，动量 `0.9` 。

   - 所有模型训练 `20` 个 `epoch` 。

   模型效果如下表所示，结果表明我们的`Graph CNN` 模型和经典`CNN` 模型的性能非常接近。

   - 性能上的差异可以用频域卷积的各向同性来解释：一般图形中的边不具有方向性，但是`MNIST` 图片作为二维网格具有方向性，如像素的上下左右。这是优势还是劣势取决于具体的问题。
   - 另外`Graph CNN` 模型也缺乏架构涉及经验，因此需要研究更适合的优化策略和初始化策略。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/zqoEPkDJQbTK.png?imageslim">
   </p>
   

3. `20NEWS` 实验：

   为验证`Graph CNN` 可以应用于非结构化数据，论文对 `20NEWS` 数据集应用图卷积来解决文本分类问题。

   `20NEWS` 数据集包含 `18846` 篇文档，分为`20` 个类别。我们将其中的 `11314` 篇文档用于训练、`7532` 篇文档用于测试。

   我们首先抽取文档的特征：

   - 从所有文档的 `93953` 个单词中保留词频最高的一万个单词。
   - 每篇文档使用词袋模型提取特征，并根据文档内单词的词频进行归一化。

   论文构建了`16` 层图神经网络，每篇文档对应一个图`Graph`，图的顶点数量为一万、边的数量为 `132834` 。顶点$i,j$之间的权重为：

  $W_{i,j} = \exp\left(- \frac{||\mathbf{\vec z}_i - \mathbf{\vec z}_j||_2^2}{\sigma^2}\right)$

   其中$\mathbf{\vec z}_i$为对应单词的 `word2vec embedding` 向量。

   所有神经网络模型都使用`Adam` 优化器，初始学习率为 `0.001` ；`GC32` 模型的$K=5$。结果如下图所示，虽然我们的模型未能超越`Multinomial Naive Bayes` 模型，但是它超越了所有全连接神经网络模型，而这些全连接神经网络模型具有更多的参数。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/gC3tmeTjVlsi.png?imageslim">
   </p>
   

4. 我们在`MNIST` 数据集上比较了不同的图卷积神经网络架构的效果，其中包括：常规图卷积的`Non-Param` 架构、增加平滑约束的样条插值`Spline` 架构、`Fast Graph CNN` 的`Chebyshev` 架构，其中$K=25$。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/2oPsq856U2wu.png?imageslim">
   </p>
   

   训练过程中，这几种架构的验证集准确率、训练集损失如下图所示，横轴表示迭代次数。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/b7kMO0Px17cf.png?imageslim">
   </p>
   

5. 我们在 `20NEWS` 数据集上比较了不同的图卷积神经网络架构的计算效率，其中$K=25$。

   我们评估了每个`mini-batch` 的处理时间，其中`batch-size = 10` 。横轴为图的顶点数量$m$。可以看到 `Fast Graph CNN` 的计算复杂度为$O(m)$，而传统图卷积（包括样条插值）的计算复杂度为$O(m^2)$。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/xS9HN5bgJzQ5.png?imageslim">
   </p>
   

6. 我们在 `MNIST` 数据集上验证了 `Fast Graph CNN` 的并行性。下面比较了经典卷积神经网络和`Fast Graph CNN` 采用`GPU`时的加速比，其中 `batch-size = 100` 。

   可以看到我们的 `Fast Graph CNN` 获得了与传统 `CNN` 相近的加速比，这证明我们提出的`Fast Graph CNN` 有良好的并行性。我们的模型仅依赖于矩阵乘法，而矩阵乘法可以通过`NVIDA` 的 `cuBLAS` 库高效的支持。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/BExPPwKQsWhK.png?imageslim">
   </p>
   

7. 要想图卷积取得良好的效果，数据集必须满足一定条件：图数据必须满足局部性`locality`、平稳性`stationarity`、组成性`compositionality` 的统计假设。

   因此训练得到的卷积核的质量以及图卷积模型的分类能力很大程度上取决于数据集的质量。

   从`MNIST` 实验我们可以看到：从欧式空间的网格数据中采用 `kNN` 构建的图，这些图数据质量很高。我们基于这些图数据采用图卷积几乎获得标准`CNN` 的性能。并且我们发现，`kNN` 中 `k` 的值对于图数据的质量影响不大。

   作为对比，我们从`MNIST` 中构建随机图，其中顶点之间的边是随机的。可以看到在随机图上，图卷积神经网络的准确率下降。在随机图中，数据结构发生丢失，因此卷积层提取的特征不再有意义。

   这表明图数据满足数据的统计假设的重要性。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/OWeHkkXe1Kwe.png?imageslim">
   </p>
   

8. 在`20NEWS` 数据集构建图数据时，对于顶点$i$对应的单词，我们有三种表示单词的$\mathbf{\vec z}_i$的方法：

   - 将每个单词表示为一个 `one-hot` 向量
   - 通过 `word2vec` 从数据集中学习每个单词的 `embedding` 向量
   - 使用预训练的单词`word2vec embedding` 向量

   不同的方式采用 `GC32` 的分类准确率如下所示。其中：

   - `bag-of-words` 表示 `one-hot` 方法

   - `pre-learned` 表示预训练的 `embedding` 向量

   - `learned` 表示从数据集训练 `embedding` 向量

   - `approximate` 表示对 `learned` 得到的 `embedding` 向量进行最近邻搜索时，使用`LSHForest` 近似算法。

     这是因为当图的顶点数量较大时，找出每个顶点的`kNN` 顶点的计算复杂度太大，需要一个近似算法。

   - `random` 表示对 `learned` 得到的 `embedding` 向量采用随机生成边，而不是基于 `kNN` 生成边。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/UEBjnTg7FNnm.png?imageslim">
   </p>
   

## 四、Semi-Supervised GCN

1. 图的半监督学习方法大致分为两大类：

   - 基于图的拉普拉斯矩阵正则化的方法， 包括标签传播算法`label propagation`、流行正则化算法`manifold regularization` 。

   - 基于图嵌入的方法，包括 `DeepWalk、LINE` 等。

     但是基于图嵌入的方法需要一个`pipeline`，其中包括生成随机游走序列、执行半监督训练等多个步骤，而每个步骤都需要分别进行优化。

   论文`《 SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS》` 提出了一种可扩展的、基于`Graph` 的半监督学习方法，该方法基于一个有效的、能直接对`Graph` 进行操作的卷积神经网络变种。

   该模型基于频域卷积的局部一阶近似来选择卷积网络结构，其计算复杂度为$O(|E|)$，其中$|E|$为边的数量。

   该模型学到的隐层`representation` 既能够编码图的局部结构，又能够编码顶点的特征。

2. 考虑图中顶点的分类问题，其中仅有一小部分顶点具有标签信息。如：对引文网络的文章进行分类，只有一小部分文中具有类别标签。

   该问题是一个图的半监督学习问题，其中标签信息通过显式的正则化而被平滑。例如，考虑在损失函数中使用图的拉普拉斯正则项：

   其中：

   -$\mathcal L_0$表示图中有标签部分的监督损失；$\mathcal L_{reg}$为正则化项；$\lambda$为罚项系数。其中：

    $\mathcal L_0 = \sum_{i\in \mathcal Y_L} ||f(\mathbf{\vec x}_i) - y_i||^2$

    $\mathbf{\vec x}_i$为顶点$i$的输入特征，$y_i$为顶点$i$的标签；$f(\cdot)$是一个类似神经网络的可微函数，它将输入$\mathbf{\vec x}_i$映射到类别空间；$\mathcal Y_L$为带标签顶点的集合。

   -$\mathbf W\in \mathbb R^{m\times m}$为邻接矩阵，$m$为顶点数量；$\mathbf D$为度矩阵，其中$D_{i,i} = \sum_j W_{i,j}$；$\Delta = \mathbf D- \mathbf W$为无向图$G=(V,E)$的未归一化的拉普拉斯算子 ；$\mathbf X$为顶点的特征向量拼接的矩阵

   正则化项的物理意义为：

   - 如果两个顶点距离较近（即$W_{i,j}$较大），则它们的 `representation` 应该比较相似（即$f(\mathbf{\vec x}_i)$和$f(\mathbf{\vec x}_j)$距离相近）。
   - 如果两个顶点距离较远(即$W_{i,j}$较小)，则它们的 `representation` 可以相似也可以不相似。

   因此该模型假设：`Graph` 中直接相连的顶点很可能共享相同的标签。这种假设会限制模型的表达能力，因为图中的边不一定代表相似性，边也可能代表其它信息。

   在论文`Semi-Supervised GCN` 中，作者直接使用神经网络模型$f(\mathbf X,\mathbf W)$对图结构进行编码，并对所有带标签的顶点进行监督目标$\mathcal L_0$的训练，从而避免在损失函数中进行显式的、基于图的正则化。$f(\cdot)$依赖于图的邻接矩阵$\mathbf W$，这允许模型从监督损失$\mathcal L_0$中分配梯度信息，并使得模型能够学习带标签顶点的`representation` 和不带标签顶点的 `representation`。

### 4.1 模型

1. 考虑一个多层`GCN` 网络，其中每层的传播规则为：

  $\mathbf H^{(l+1)} = \sigma\left(\tilde{\mathbf D}^{-1/2} \tilde{\mathbf W}\tilde{\mathbf D}^{-1/2} \mathbf H^{(l)} \mathbf \Theta^{(l)}\right)$

   其中：

   -$\tilde{\mathbf W} = \mathbf W + \mathbf I_m$是带自环的无向图的邻接矩阵，$\mathbf I_m$为单位矩阵，$m$为顶点数量；$\tilde D_{i,i} = \sum_j \tilde W_{i,j}$为带自环的度矩阵。
   -$\mathbf H^{(l)}\in \mathbb R^{m\times d}$为第$l$层的激活矩阵，$\mathbf H^{0} = \mathbf X$，$d$为特征向量维度；$\mathbf \Theta^{(l)}$为第$l$层的权重矩阵；$\sigma(\cdot)$为激活函数。

   其基本物理意义：

   - 我们认为在第$l+1$层，每个顶点的`representation` 由其邻域内顶点（包含它自身）在第$l$层的`representation`的加权和得到，加权的权重为边的权重：

    $\mathbf H^{(l+1)}= \tilde{\mathbf W}\mathbf H^{(l)}$

   - 考虑到不同顶点的度不同，这将使得顶点的度越大（即邻域内顶点越多），则`representation` 越大。因此这里对顶点的度进行归一化：

    $\mathbf H^{(l+1)} = \tilde{\mathbf D}^{-1/2} \tilde{\mathbf W}\tilde{\mathbf D}^{-1/2} \mathbf H^{(l)}$

   - 考虑到非线性映射，则得到最终的结果 ：

    $\mathbf H^{(l+1)} = \sigma\left(\tilde{\mathbf D}^{-1/2} \tilde{\mathbf W}\tilde{\mathbf D}^{-1/2} \mathbf H^{(l)} \mathbf \Theta^{(l)}\right)$

2. 可以证明，上述形式的传播规则等价于`localized` 局部化的频域图卷积的一阶近似。

   给定输入$\mathbf{\vec x}$，其频域卷积为$\mathbf{\vec y} = \mathbf U \mathbf K \mathbf U^T\mathbf{\vec x}$，其中：

  $\mathbf K$为卷积核，$\mathbf U$各列为拉普拉斯矩阵$\mathbf L$的特征向量。这里我们采用归一化的拉普拉斯矩阵：

  $\mathbf L = \mathbf I - \mathbf D^{-1/2}\mathbf W \mathbf D^{-1/2}$

   以及$\mathbf L = \mathbf U\mathbf\Lambda \mathbf U^T$。

   这种卷积的参数数量为$O(m)$，计算复杂度为$O(m^2)$。因此 `Fast Graph CNN` 采用切比雪夫多项式来构造多项式卷积核，因此有：

  $\mathbf{\vec y} = \sum_{k=0}^{K-1}\beta_k\mathbf UT_k(\tilde{\mathbf\Lambda}) \mathbf U^T\mathbf{\vec x}$

   其中$T_k$为$k$阶切比雪夫多项式，$\beta_k$为其系数，$\tilde{\mathbf\Lambda} = \frac{2}{\lambda_\max}\mathbf\Lambda - \mathbf I$。

   令$\tilde{\mathbf L} = \frac{2}{\lambda_{\max}} \mathbf L - \mathbf I$，则有：

  $\mathbf{\vec y} = \sum_{k=0}^{K-1}\beta_kT_k(\tilde{\mathbf L})\mathbf{\vec x}$

   该卷积满足$K$阶局部性，其计算复杂度为$O(|E|)$。

   现在我们选择$K=1$，则有$T_0 (\tilde{\mathbf L}) =\mathbf I,\; T_1 (\tilde{\mathbf L})= \tilde{\mathbf L}$。因此有：

  $\mathbf{\vec y} = \beta_0\mathbf{\vec x} + \beta_1\tilde{\mathbf L}\mathbf{\vec x} = \left(\beta_0\mathbf I+\frac{2\beta_1}{\lambda_{\max} }\mathbf L- \beta_1\mathbf I\right)\mathbf{\vec x}$

   该卷积为$\mathbf L$的线性函数。

   - 我们仍然可以堆叠多个这样的层来实现图卷积神经网络，但是现在我们不再局限于切比雪夫多项式这类的参数化方式。
   - 由于$\mathbf L$为归一化的拉普拉斯矩阵，因此我们预期该模型能够缓解顶点的`degree` 分布范围很大的图的局部邻域结构的过拟合问题，这些图包括社交网络、引文网络、知识图谱等等。
   - 当计算资源有限时，这种线性卷积核使得我们可以构建更深的模型，从而提高模型的能力。

   进一步的，我们令$\lambda_{max} \simeq 2$，因为我们预期模型可以在训练过程中适应这种变化。另外我们减少参数数量，令$\beta= \beta_0= -\beta_1$，则有：

  $\mathbf{\vec y} = \beta \left(2\mathbf I- \mathbf L\right)\mathbf{\vec x} = \beta\left(\mathbf I +\mathbf D^{-1/2}\mathbf W \mathbf D^{-1/2} \right)\mathbf{\vec x}$

   注意到$\mathbf I + \mathbf D^{-1/2}\mathbf W \mathbf D^{-1/2}$的特征值在 `[0,2]`之间，在深层网络种应用这种卷积操作会导致数值不稳定核梯度爆炸、消失的现象。为缓解该问题，我们应用了正则化技巧：

  $\mathbf I + \mathbf D^{-1/2}\mathbf W \mathbf D^{-1/2} \rightarrow \tilde{\mathbf D}^{-1/2}\tilde{\mathbf W }\tilde{\mathbf D}^{-1/2}$

   其中$\tilde{\mathbf W } = \mathbf I + \mathbf W$，$\tilde{\mathbf D}$为由$\tilde{\mathbf W}$定义的度矩阵：$\tilde D_{i,i} = \sum_j\tilde W_{i,j}$。

   我们可以将这个定义推广到$F_{in}$个输入通道、$F_{out}$个输出通道的卷积上，其中输入$\mathbf X\in \mathbb R^{m\times F_{in}}$，则卷积输出为：

  $\mathbf Y = \tilde{\mathbf D}^{-1/2} \tilde{\mathbf W} \tilde{\mathbf D}^{-1/2}\mathbf X\mathbf\Theta$

   其中$\mathbf Y\in \mathbb R^{m\times F_{out}}$为输出 `featue map`，$\mathbf\Theta \in \mathbb R^{F_{in}\times F_{out}}$为卷积核的参数矩阵。

   卷积操作的计算复杂度为$O(|E|\times F_{in}\times F_{out})$，因为$\tilde{\mathbf W} \mathbf X$可以实现为一个稀疏矩阵和一个稠密矩阵的乘积。

3. 介绍完这个简单灵活的、可以在图上传播信息的模型$f(\mathbf X,\mathbf W)$之后，我们回到图的半监督顶点分类问题上。由于$\mathbf W$包含了一些$\mathbf X$中为包含的图的结构信息，如引文网络中的文档引用关系、知识图谱的实体之间的关系等等，我们预期模型比传统的半监督学习模型表现更好。

   整个半监督学习模型是一个多层`GCN`模型，结构如下图所示。

   - 左图：一个多层卷积神经网络模型，其中包含$F_{in}$个输入通道、$F_{out}$个输出通道，图中$F_{in}=3,F_{out}= 5$。

     黑色的线条为图的边，彩色线条为信息流动的方向（注意：颜色有重叠）；$Y_i$为监督的标签信息。图结构在不同卷积层之间共享。

   - 右图：在`Cora` 数据集上的一个双层`GCN` 网络（两个隐层 + 一个输出层）的最后一个隐层的隐向量（经过激活函数），在使用 `t-SNE` 可视化的结果，颜色代表不同类别。其中数据集仅仅使用 `5%` 的顶点标签。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/LHMyLbq17j2H.png?imageslim">
   </p>
   

4. 考虑一个双层 `GCN` 模型，其邻接矩阵$\mathbf W$是对称的。

   - 我们首先在预处理中计算$\hat {\mathbf W} = \tilde{\mathbf D}^{-1/2} \tilde{\mathbf W} \tilde{\mathbf D}^{-1/2}$

   - 然后计算前向传播：

    $\mathbf Y = f(\mathbf X,\mathbf W) = \text{softmax}(\hat{\mathbf W}\text{ReLU}(\hat{\mathbf W}\mathbf X \mathbf \Theta^{(0)})\mathbf \Theta^{(1)})$

     其中：

     -$\mathbf \Theta^{(0)}\in \mathbb R^{F_{in}\times H}$为输入到隐层的权重矩阵，有$H$个输出 `feature map`

     -$\mathbf \Theta^{(1)}\in \mathbb R^{H\times F_{out}}$为隐层到输出的权重矩阵

     -$\text{softmax}$激活函数定义为：

       按行进行归一化。

     前向传播的计算复杂度为$O(|E|F_{in}HF_{out})$。

   - 对于半监督多标签分类，我们使用交叉熵来衡量所有标记样本的误差：

    $\mathcal L = -\sum_{l\in \mathcal Y_L}\sum_{f=1}^{F_{out}}Y_{l,f}\ln(Z_{l,f})$

     其中$\mathcal Y_L$为有标签的顶点的下标集合。顶点标签$y_l$经过 `one-hot` 为：$(Y_{l,1},\cdots,Y_{l,C})$，其中$C$为类别数量。

   - 神经网络权重$\mathbf\Theta^{(0)},\mathbf\Theta^{(1)}$使用梯度下降来训练。

     - 只要数据集能够全部放到内存中，训练过程中我们就使用全部数据集来执行梯度下降。由于邻接矩阵$\mathbf W$是稀疏的，因此内存占用为$O(|E|)$。

       在未来工作中，我们考虑使用 `mini-batch` 随机梯度下降。

     - 训练过程中使用 `dropout` 增加随机性。

### 4.2 WL-1 算法

1. 理想情况下图神经网络模型应该能够学到图中顶点的`representation`，该`representation` 必须能够同时考虑图的结构和顶点的特征。

   一维 `Weisfeiler-Lehman:WL-1` 算法提供了一个研究框架。给定图以及初始顶点标签，该框架可以对顶点标签进行分配。

2. `WL-1` 算法：令$h_i^{(t)}$为顶点$v_i$在第$t$轮分配的标签，$\mathcal N_i$为顶点$v_i$的邻居顶点集合。

   - 输入：初始顶点标签$\{h_1^{(0)},h_2^{(0)},\cdots,h_N^{(0)}\}$

   - 输出：最终顶点标签$\{h_1^{(T)},h_2^{(T)},\cdots,h_N^{(T)}\}$

   - 算法步骤：

     - 初始化$t=0$

     - 迭代直到$t=T$或者顶点的标签到达稳定状态。迭代步骤为：

       - 循环遍历$v_i\in V$，执行：

        $h_i^{(t+1)} = \text{hash}\left(\sum_{j\in \mathcal N_i} h_j^{(t)}\right)$

       -$t = t+1$

     - 返回每个顶点的标签

3. 如果我们采用一个神经网络来代替 `hash`函数，同时假设$h_i$为向量，则有：

  $\mathbf{\vec h}_i^{(l+1)} = \sigma\left(\sum_{j\in \mathcal N_i}\frac {1}{c_{i,j}} \mathbf W^{(l)}\mathbf{\vec h}_j^{(l)} \right)$

   其中：

   -$\mathbf{\vec h}_i^{(l)}$为第$l$层神经网络顶点$i$的激活向量 `vector of activations` 。
   -$\mathbf W^{(l)}$为第$l$层的权重矩阵。
   -$\sigma(\cdot)$为非线性激活函数。
   -$c_{i,j}$为针对边$(v_i,v_j)$的正则化常数。

   我们定义$c_{i,j} = \sqrt{d_id_j}$，其中$d_i = |\mathcal N_i|$为顶点$v_i$的度`degree`，则上式等价于我们的 `GCN` 模型。因此我们可以将 `GCN` 模型解释为图上 `WL-1` 算法的微分化和参数化的扩展。

4. 通过与 `WL-1` 算法的类比，我们可以认为：即使是未经训练的、具有随机权重的 `GCN` 模型也可以充当图中顶点的一个强大的特征提取器。如：考虑下面的一个三层`GCN` 模型：

  $\mathbf Y = \tanh\left(\hat{\mathbf W}\tanh\left(\hat{\mathbf W}\tanh\left(\hat{\mathbf W} \mathbf X \mathbf \Theta^{(0)}\right)\mathbf \Theta^{(1)}\right)\mathbf \Theta^{(2)}\right)$

   其中权重矩阵是通过 `Glorot & Bengio (2010)` 初始化的：$\mathbf\Theta^{(k)}\sim \text{Uniform}\left[-\sqrt{\frac{6}{h_k+h_{k+1}}},\sqrt{\frac{6}{h_k+h_{k+1}}}\right]$。

5. 我们将这个三层 `GCN` 模型应用于 `Zachary`的 `karate club network` ，该网络包含`34`个顶点、`154` 条边。每个顶点都属于一个类别，一共四种类别。顶点的类别是通过 `modularity-based` 聚类算法进行标注的。如下图所示，颜色表示顶点类别。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/5o0lwdyWLeft.png?imageslim">
   </p>
   

   我们令$\mathbf X = \mathbf I$，即每个顶点除了顶点`ID` 之外不包含任何其它特征。另外顶点的`ID` 是随机分配的，也不包含任何信息。我们选择隐层的维度为`4`、输出层的维度为`2` ，因此输出层的输出$\mathbf Y$能够直接视为二维数据点来可视化。

   下图给出了未经训练的 `GCN` 模型获得的顶点`embedding`，这些结果与从`DeepWalk`获得的顶点`embedding` 效果相当，而`DeepWalk` 使用了代价更高的无监督训练过程。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/7Sm4fnYIpQCa.png?imageslim">
   </p>
   

6. 在`karate club network`数据集上，我们观察半监督分类任务期间，`embedding`如何变化。这种可视化效果提供了关于 `GCN` 模型如何利用图结构从而学到对于分类任务有益的顶点`embedding` 。

   训练配置：

   - 在上述三层`GCN` 之后添加一个 `softmax` 输出层，输出顶点属于各类别的概率
   - 每个类别仅使用一个带标签的顶点进行训练，一共有四个带标签的顶点
   - 使用`Adam` 优化器来训练，初始化学习率为 `0.01`
   - 采用交叉熵损失函数
   - 迭代 `300` 个 `step`

   下图给出多轮迭代中，顶点`embedding` 的演变。图中的灰色直线表示图的边，高亮顶点（灰色轮廓）表示标记顶点。可以看到：模型最终基于图结构以及最少的监督信息，成功线性分离出了簇团。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/J2C1LsAliO5H.png?imageslim">
   </p>
   

### 4.3 局限性

1. 我们的 `Semi-GCN` 模型存在一些局限，我们计划在将来克服这些局限性。

2. 内存需求局限性：在`full-batch` 梯度下降算法中，内存需求随着数据集的大小线性增长。

   - 一种解决方式是：采用 `CPU` 训练来代替 `GPU` 训练。这种方式我们在实验中得到验证。

   - 另一种解决方式是：采用 `mini-batch` 梯度下降算法。

     但是`mini-batch` 梯度下降算法必须考虑 `GCN` 模型的层数。因为对于一个$K$层的 `GCN` 模型，其$K$阶邻居必须全部存储在内存中。对于顶点数量庞大、顶点链接很密集的图，这可能需要进一步的优化。

3. 边类型的局限性：目前我们的模型不支持边的特征，也不支持有向图。

   通过`NELL` 数据集的实验结果表明：可以通过将原始的有向图转化为无向二部图来处理有向图以及边的特征。这通过额外的顶点来实现，这些顶点代表原始`Graph` 中的边。

4. 假设局限性：我们的模型有两个基本假设：

   - 假设$K$层`GCN` 依赖于$K$阶邻居，即模型的局部性。

   - 假设自链接和邻居链接同样重要。

     在某些数据集中，我们可以引入一个折衷参数：$\tilde{\mathbf W} = \mathbf W + \lambda \mathbf I$。其中参数$\lambda$

     平衡了自链接和邻居链接的重要性，它可以通过梯度下降来学习。

### 4.4 实验

1. 我们在多个任务中验证模型性能：在引文网络中进行半监督文档分类、在从知识图谱抽取的二部图中进行半监督实体分类。然后我们评估图的各种传播模型，并对随机图的运行时进行分析。

2. 数据集：

   - 引文网络数据集：我们考虑 `Citeseer,Cora,Pubmed` 三个引文网络数据集，每个数据集包含以文档的稀疏 `BOW` 特征向量作为顶点，文档之间的引文链接作为边。我们将引文链接视为无向边，并构造一个二元的对称邻接矩阵$\mathbf W$。

     每个文档都有一个类别标签，每个类别仅包含 `20`个标记顶点作为训练样本。

   - `NELL` 数据集：该数据集是从`Carlson et al.2010` 引入的知识图谱中抽取的数据集。知识图谱是一组采用有向的、带标记的边链接的实体。我们为每个实体对$(e_1,r,e_2)$分配顶点$\{e_1,e_2,r_1,r_2\}$，以及边$(e_1,r_1)$和$(e_2,r_2)$。其中，$r_1,r_2$是知识图谱链接$r$得到的两个“拷贝”的关系顶点`relation node`，它们之间不存在边。最终我们得到 `55864` 个关系顶点和 `9891` 个实体顶点。

     实体顶点`entity node` 通过稀疏的特征向量来描述。我们为每个关系顶点分配唯一的 `one-hot` 向量从而扩展 `NELL` 的实体特征向量，从而使得每个顶点的特征向量为 `61278`维稀疏向量。

     对于顶点$i$和$j$，如果它们之间存在一条边或者多条边，则设置$W_{i,j} = 1$从而构建一个二元对称邻接矩阵。

     在顶点的半监督分类任务中，我们为每个类别标记一个顶点作为训练集，因此属于非常极端的情况。

   - 随机图：我们生成各种规模的随机`Graph` 数据集，从而评估每个`epoch` 的训练时间。

     对于具有$N$个顶点的图，我们创建一个随机图：

     - 随机均匀分配$2N$条边
     - 令$\mathbf X = \mathbf I$，即每个顶点除了其`id` 之外没有任何特征，且顶点`id` 是随机分配的
     - 每个顶点标签为$y_i=1$

   各数据集的整体统计如下表所示。标记率：表示监督的标记顶点数量占总的顶点数量的比例。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/OUoYjp0usBu5.png?imageslim">
   </p>
   

3. 模型设置：除非另有说明，否则我们的`GCN` 模型就是前面描述的两层`GCN` 模型。

   - 我们将数据集拆分为`labled` 数据、`unlabled` 数据、测试数据。其中我们在`labled` 数据和 `unlabled` 数据上学习，在测试数据上测试。我们选择测试数据包含 `1000` 个顶点。

     另外我们还使用额外的 `500` 个带标签的顶点作为验证集，用于超参数优化。这些超参数包括：所有层的 `dropout` 比例、第一层的$L_2$正则化系数、隐层的维度。

     注意：验证集不用于训练。

   - 对于引文网络数据集，我们仅在`Cora` 数据集上优化超参数，并对`Citeseer` 和 `Pubmed` 数据集采用相同的超参数。

   - 所有模型都使用 `Adam` 优化器，初始化学习率为 `0.01` 。

   - 所有模型都使用早停策略，早停的 `epoch` 窗口为 `10`。即：如果连续 `10` 个 `epoch` 的验证损失没有下降，则停止继续训练。

   - 所有模型最多训练 `200` 个 `epoch` 。

   - 我们使用 `Glorot & Bengio (2010)` 初始化策略：$\mathbf\Theta^{(k)}\sim \text{Uniform}\left[-\sqrt{\frac{6}{h_k+h_{k+1}}},\sqrt{\frac{6}{h_k+h_{k+1}}}\right]$。

   - 我们对输入的特征向量进行按行的归一化 `row-normalize`，这类似于 `Batch-Normalization`，使得每个维度的取值都在相近的取值范围。

   - 在随机图数据集上，我们选择隐层维度为 `32`，并省略正则化：即不进行`dropout`，也不进行$L_2$正则化。

4. `Baseline` 模型：我们比较了 `Yang et al.(2016)` 相同的 `baseline` 方法，即：即：标签传播算法`label propagation:LP`、半监督`embedding` 算法 `semi-supervised embedding:SemiEmb` 、流形正则化算法`manifold regularization:MainReg` 、基于`skip-gram` 的图嵌入算法`DeepWalk`。

   - 我们忽略了 `TSVM` 算法，因为它无法扩展到类别数很大的数据集。
   - 我们还还比较了`Planetoid(2016)`算法， 以及 `Lu&Getoor(2003)` 提出的迭代式分类算法`iterative classification algorithm:ICA`。

5. 模型比较结果如下表所示。

   - 对于`ICA` ，我们随机运行 `100` 次、每次以随机的顶点顺序训练得到的平均准确率； 所有其它基准模型的结果均来自于 `Planetoid` 论文，`Planetoid*` 表示论文中提出的针对每个数据集的最佳变体。

   - 我们的`GCN` 执行`100`次、每次都是随机权重初始化的平均分类准确率，括号中为平均训练时间。

   - 我们为 `Citeseer,Cora,Pubmed`使用的超参数为：`dropout = 0.5`、$L_2$正则化系数为$5\times 10^{-4}$、隐层的维度为`64` 。

   - 最后我们报告了`10` 次随机拆分数据集，每次拆分的`labled` 数据、`unlabled` 数据、测试数据比例与之前相同，然后给出`GCN` 的平均准确率和标准差（以百分比表示），记作 `GCN(rand. splits)` 。

     > 前面七行是针对同一个数据集拆分，最后一行是不同的数据集拆分。

   实验表明：我们的半监督顶点分类方法明显优于其它方法。

   - 基于图的拉普拉斯正则化方法（`LP, maniReg` 方法）最有可能受到限制，因为它们假设图的边仅仅编码了顶点的相似性。
   - 基于 `SkipGram` 的方法也受到限制，因为它们难以优化一个多步骤的 `pipeline` 。

   我们的模型可以克服这些局限性，同时在计算效率上和有关方法相比仍然是有利的。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/t4HQFB5auVXg.png?imageslim">
   </p>
   

6. 我们在引文网络数据集上比较了我们提出的逐层传播模型的不同变体，实验配置和之前相同，结果如下表所示。

   我们原始的 `GCN` 模型应用了 `renormalization` 技巧（粗体），即：

  $\mathbf I + \mathbf D^{-1/2}\mathbf W \mathbf D^{-1/2} \rightarrow \tilde{\mathbf D}^{-1/2}\tilde{\mathbf W }\tilde{\mathbf D}^{-1/2}$

   其它的`GCN` 变体采用`Propagation model`对应的传播模型。

   - 对于每一种变体模型，我们给出执行`100`次、每次都是随机权重初始化的平均分类准确率。
   - 对于每层有多个权重$\mathbf{\Theta}_k$的情况下（如 `Chebyshev filter, 1st-order model`），我们对第一层的所有权重执行$L_2$正则化。

   实验表明：和单纯的一阶切比雪夫多项式卷积模型，以及更高阶的切比雪夫多项式卷积模型相比，`renormalization` 模型在很多数据集上都能够效率更高（更少的参数和更少的计算量）、预测能力更强。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/UhhfbL4xFsXT.png?imageslim">
   </p>
   

7. 我们在随机图上报告了 `100` 个 `epoch` 的每个 `epoch` 平均训练时间。我们在 `Tensorflow` 上比较了 `CPU` 和 `GPU` 实现的结果，其中 `*` 表示内存溢出错误`Out Of Memory Error` 。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/wo2BuglaP6j2.png?imageslim">
   </p>
  

8. 最后我们考虑模型的深度对于性能的影响。这里我们报告对 `Cora,Citeseer,Pubmed` 数据集进行`5` 折交叉验证的结果。

   除了标准的 `GCN` 模型之外，我们还报告了模型的一种变体：隐层之间使用了残差连接：

  $\mathbf H^{(l+1)} = \sigma\left(\tilde{\mathbf D}^{-1/2} \tilde{\mathbf W}\tilde{\mathbf D}^{-1/2} \mathbf H^{(l)}\mathbf \Theta^{(l)}\right) + \mathbf H^{(l)}$

   在`5` 折交叉验证的每个拆分中，我们训练`400` 个 `epoch` 并且不使用早停策略。我们使用`Adam` 优化器，初始学习率为 `0.01`。我们对第一层和最后一层使用`dropout = 0.5` ，第一层权重执行正则化系数为$5\times 10^{-4}$的$L_2$正则化。`GCN` 的隐层维度选择为 `16`。

   结果如下图所示，其中标记点表示`5` 折交叉验证的平均准确率，阴影部分表示方差。

   可以看到：

   - 当使用两层或三层模型时，`GCN` 可以获得最佳效果。
   - 当模型的深度超过七层时，如果不使用残差连接则训练会变得非常困难，表现为训练准确率骤降。
   - 当模型深度增加时，模型的参数数量也会增加，此时模型的过拟合可能会成为问题。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/d76ebxU1fVNx.png?imageslim">
   </p>
   

## 五、分子指纹GCN

1. 在材料设计领域的最新工作已经将神经网络用于材料筛选，其任务是通过学习样本来预测新型分子的特性。预测分子特性通常需要将分子图作为输入，然后构建模型来预测。在分子图中顶点表示原子，边表示化学键。

   这个任务的一个难点在于：输入的分子图可以具有任意大小和任意形状，而大多数机器学习模型只能够处理固定大小、固定形状的输入。

   目前主流的方法是通过`hash` 函数对分子图进行预处理从而生成固定大小的指纹向量`fingerprint vector`，该指纹向量作为分子的特征灌入后续的模型中。

   论文`《Convolutional Networks on Graphs for Learning Molecular Fingerprints》` 提出了分子指纹`GCN` 模型，该模型用一个可微的神经网络代替了分子指纹部分。

   神经网络以原始的分子图作为输入，采用卷积层来抽取特征，然后通过全局池化来结合所有原子的特征。这种方式使得我们可以端到端的进行分子预测。

   相比较传统的指纹向量的方式，我们的方法具有以下优势：

   - 预测能力强：通过实验比较可以发现，我们的模型比传统的指纹向量能够提供更好的预测能力。

   - 模型简洁：为了对所有可能的子结构进行编码，传统的指纹向量必须维度非常高。而我们的模型只需要对相关特征进行编码，模型的维度相对而言低得多，这降低了下游的计算量。

   - 可解释性：传统的指纹向量对每个片段`fragment` 进行不同的编码，片段之间没有相似的概念。即：相似的片段不一定有相似的编码；相似的编码也不一定代表了相似的片段。

     我们的模型中，每个特征都可以由相似但是不同的分子片段激活，因此相似的片段具有相似的特征，相似的特征也代表了相似的片段。这使得特征的`representation` 更具有意义。

### 5.1 模型

#### 5.1.1 圆形指纹算法

1. 分子指纹的最新技术是扩展连接性圆形指纹 `extended-connectivity circular fingerprints:ECFP` 。`ECFP` 是对`Morgan` 算法的改进，旨在以无关于原子标记顺序`atom-relabling`的方式来识别分子中存在哪些亚结构。

   `ECFP` 通过对前一层邻域的特征进行拼接，然后采用一个固定的哈希函数来抽取当前层的特征。哈希函数的结果视为整数索引，然后对顶点 `feature vector` 在索引对应位置处填写 `1` 。

   - 不考虑`hash` 冲突，则指纹向量的每个索引都代表一个特定的亚结构。索引表示的亚结构的范围取决于网络深度，因此网络的层数也被称为指纹的“半径”。
   - `ECFP` 类似于卷积网络，因为它们都在局部采用了相同的操作，并且在全局池化中聚合信息。

2. `ECFP` 的计算框架如下图所示：

   - 首先通过分子结构构建分子图，其中顶点表示原子、边表示化学键。

   - 在每一层，信息在邻域之间流动

   - 图的每个顶点在一个固定的指纹向量中占据一个`bit`。

     其中这只是一个简单的示意图，实际上每一层都可以写入指纹向量。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/WV4uAHuckGtR.png?imageslim">
   </p>
   

3. 圆形指纹算法：

   - 输入：

     - 分子结构
     - 半径参数$R$
     - 指纹向量长度$S$

   - 输出：指纹向量$\mathbf{\vec f}$

   - 算法步骤：

     - 初始化指纹向量：

      $\mathbf{\vec f} = (\underbrace{0,0,\cdots,0}_{S})^T$

     - 遍历每个原子$a$，获取每个原子的特征$\mathbf{\vec r}_a^{(0)} = g(a)$

     - 遍历每一层。对于第$l, 1\le l\le R$层，迭代步骤为：

       - 遍历分子中的每个原子$a$，对原子$a$计算：
         - 获取顶点$a$邻域原子的特征：$\mathbf{\vec r}_1^{(l-1)},\cdots,\mathbf{\vec r}_N^{(l-1)}$
         - 拼接顶点$a$及其邻域原子特征：$\mathbf{\vec v}^{(l)} = [\mathbf{\vec r}_a^{(l-1)},\mathbf{\vec r}_1^{(l-1)},\cdots,\mathbf{\vec r}_N^{(l-1)}]$
         - 执行哈希函数得到顶点$a$的当前特征：$\mathbf{\vec r}_a^{(l)} = \text{hash}(\mathbf{\vec v}^{(l)})$
         - 执行索引函数：$i = \text{mod}(\mathbf{\vec r}_a^{(l)},S)$
         - 登记索引：$f_i = 1$

     - 最终返回$\mathbf{\vec f}$

#### 5.1.2 分子指纹GCN算法

1. 我们选择类似于现有`ECFP` 的神经网络架构：

   - 哈希操作`Hashing`：在`ECFP` 算法中，每一层采用哈希操作的目的是为了组合关于每个原子及其邻域子结构的信息。

     我们利用一层神经网络代替哈希运算。当分子的局部结构发生微小的变化时（神经网络是可微的，因此也是平滑的），这种平滑函数可以得到相似的激活值。

   - 索引操作`Indexing`：在 `ECFP` 算法中，每一层采用索引操作的目的是将每个原子的特征向量组合成整个分子指纹。每个原子在其特征向量的哈希值确定的索引处，将指纹向量的单个比特位设置为`1`，每个原子对应一个`1` 。这种操作类似于池化，它可以将任意大小的`Graph` 转换为固定大小的向量。

     这种索引操作的一个缺点是：当分子图比较小而指纹长度很大时，最终得到的指纹向量非常稀疏。

     我们使用`softmax` 操作视作索引操作的一个可导的近似。本质上这是要求将每个原子划分到一组类别的某个类别中。所有原子的这些类别向量的总和得到最终的指纹向量。其操作也类似于卷积神经网络中的池化操作。

   - 规范化`Canonicalization`：无论原子的邻域原子的顺序如何变化，圆形指纹是不变的。实现这种不变性的一种方式是：在算法过程中，根据相邻原子的特征和键特征对相邻原子进行排序。

     我们尝试了这种排序方案，还对局部邻域的所有可能排列应用了局部特征变换。

     另外，一种替代方案是应用排序不变函数`permutation-invariant`， 如求和。为了简单和可扩展性，我们选择直接求和。

2. 神经网络指纹算法：

   - 输入：

     - 分子结构

     - 半径参数$R$

     - 指纹长度$S$

     - 隐层参数$\mathbf H_1^1,\cdots, \mathbf H_R^5$， 输出层参数$\mathbf W_1,\cdots,\mathbf W_R$。

       对不同的键数量，采用不同的隐层参数$1,2,3,4,5$（最多五个键）。

   - 输出：指纹向量$\mathbf{\vec f}$

   - 算法步骤：

     - 初始化指纹向量：

      $\mathbf{\vec f} = (\underbrace{0,0,\cdots,0}_{S})^T$

     - 遍历每个原子$a$，获取每个原子的特征$\mathbf{\vec r}_a = g(a)$

     - 遍历每一层。对于第$l, 1\le l\le R$层，迭代步骤为：

       - 遍历分子中的每个原子$a$，对原子$a$计算：
         - 获取顶点$a$邻域原子的特征：$\mathbf{\vec r}_1^{(l-1)},\cdots,\mathbf{\vec r}_N^{(l-1)}$
         - 池化顶点$a$及其邻域$\mathcal N_a$的原子的特征：$\mathbf{\vec v}^{(l)} = \mathbf{\vec r}_a^{(l-1)}+\sum_{i=1}^N\mathbf{\vec r}_i^{(l-1)}$
         - 执行哈希函数：$\mathbf{\vec r}_a^{(l)} = \sigma(\mathbf H_l^N \mathbf{\vec v}^{(l)})$，$N$为邻域顶点数量。
         - 执行索引函数：$\mathbf{\vec i} = \text{softmax}(\mathbf W_l\mathbf{\vec r}_a^{(l)})$
         - 登记索引：$\mathbf{\vec f} = \mathbf{\vec f} + \mathbf{\vec i}$

     - 最终返回$\mathbf{\vec f}$

3. 设指纹向量的长度为$S$，顶点特征向量的维度为$F$，则$\mathbf W_{l}$的参数数量为$O(F\times S)$，$\mathbf H_l^N$的参数数量为$O(F\times F)$。

4. 上述 `ECFP` 算法和神经网络指纹算法将每一层计算得到的指纹叠加到全局指纹向量中。我们也可以针对每一层计算得到一个层级指纹向量，然后将它们进行拼接，而不是相加。

   以神经网络指纹算法为例：

   - 在第$l$层计算索引为：$\mathbf{\vec i}^{(l)} = \text{softmax}(\mathbf W_l\mathbf{\vec r}_a^{(l)})$。然后登记索引：$\mathbf{\vec f} ^{(l)}= \mathbf{\vec f} ^{(l)}+ \mathbf{\vec i}^{(l)}$
   - 最终将所有层的索引拼接：$\mathbf{\vec f} = [\mathbf{\vec f}^{(1)} ,\mathbf{\vec f}^{(2)} ,\cdots,\mathbf{\vec f}^{(R)} ]$

5. `ECFP` 圆形指纹可以解释为具有较大随机权重的神经网络指纹算法的特殊情况。

   在较大的输入权重情况下，当$\sigma(\cdot)$为$\tanh(\cdot)$时，该激活函数接近阶跃函数。而级联的阶跃函数类似于哈希函数。

   在较大的输入权重情况下，`softmax` 函数接近一个`one-hot` 的 `argmax` 操作，这类似于索引操作。

#### 5.1.3 限制

1. 计算代价：神经网络指纹在原子数、网络深度方面与圆形指纹具有相同的渐进复杂度，但是由于在每一步都需要通过矩阵乘法来执行特征变换，因此还有附加的计算复杂度。

   假设分子的特征向量维度为$F$，指纹向量长度为$S$，网络深度为$R$，原子数量为$N$，则神经网络指纹的计算复杂度为$O(RNFS + RNF^2)$。

   实际上在圆形指纹上训练一个单隐层的神经网络只需要几分钟，而对神经网络指纹以及指纹顶部的单隐层神经网络需要一个小时左右。

2. 每层的计算限制：从网络的一层到下一层之间应该采取什么结构？本文采用最简单的单层神经网络，实际上也可以采用多层网络或者 `LSTM` 结构，这些复杂的结构可能效果更好。即：用复杂的神经网络"block"结构代替简单的神经网络单层结构。

3. 图上信息传播的限制：图上信息传播的能力受到网络深度的限制。对于一些规模较小的图如小分子的图，这可能没有问题；对于一些大分子图， 这可能受到限制。

   最坏情况下，可能需要深度为$\frac N2$的网络来处理规模（以原子数来衡量）为$N$的图。

   为了缓解该问题，`Spectral networks and locally connected networks on graphs` 提出了层次聚类，它只需要$\log N$层就可以在图上传播信息。这种方式需要解析分子为树结构，可以参考`NLP` 领域的相关技术。

4. 无法区分立体异构体`stereoisomers` ：神经网络指纹需要特殊处理来区分立体异构体，包括`enantomers` 对映异构体（分子的镜像）、`cis/trans isomers` 顺/反异构体（绕双键旋转）。大多数圆形指纹的实现方案都可以区分这些异构体。

### 5.2 实验

#### 5.2.1 随机权重

1. 分子指纹的一个用途是计算分子之间的距离。这里我们检查基于 `ECFP` 的分子距离是否类似于基于随机的神经网络指纹的分子距离。

   我们选择指纹向量的长度为 `2048`，并使用`Jaccard` 相似度来计算两个分子的指纹向量之间的距离：

  $\text{distance}(\mathbf{\vec x},\mathbf{\vec y}) = 1-\frac{\sum_{i}\min(x_i,y_i)}{\sum_{i}\max(x_i,y_i)}$

   我们的数据集为溶解度数据集，下图为使用圆形指纹和神经网络指纹的成对距离散点图，其相关系数为$r=0.823$。

   图中每个点代表：相同的一对分子，采用圆形指纹计算到的分子距离、采用神经网络指纹计算得到的分子距离，其中神经网络指纹模型采用大的随机权重。

   距离为`1.0` 代表两个分子的圆形指纹没有任何重叠；距离为`0.0` 代表两个分子的圆形指纹完全重叠。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/sP1MlBaE6guu.png?imageslim">
   </p>
   

2. 我们将圆形指纹、随机神经网络指纹接入一个线性回归层，从而比较二者的预测性能。

   - 圆形指纹、大的随机权重的随机神经网络指纹，二者的曲线都有类似的轨迹。这表明：通过大的随机权重初始化的随机神经网络指纹和圆形指纹类似。
   - 较小随机权重初始化的随机神经网络指纹，其曲线与前两者不同，并且性能更好。
   - 即使是未经训练的神经网络，神经网络激活值的平滑性也能够有助于模型的泛化。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/vbVL5KxTj4jF.png?imageslim">
   </p>
   

#### 5.2.2 可解释性

1. 圆形指纹向量的特征（即某一组`bit` 的组合）只能够通过单层的单个片段激活（偶然发生的哈希碰撞除外），神经网络指纹向量的特征可以通过相同结构的不同变种来激活，从而更加简洁和可解释。

2. 为证明神经网络指纹是可接受的，我们展示了激活指纹向量中每个特征对应的亚结构类别。

   - 溶解性特征：我们将神经网络指纹模型作为预训溶解度的线性模型的输入来一起训练。下图展示了对应的片段（蓝色），这些片段可以最大程度的激活神经网络指纹向量中最有预测能力的特征。

     - 上半图：激活的指纹向量的特征与溶解性具有正向的预测关系，这些特征大多数被包含亲水性`R-OH` 基团（溶解度的标准指标）的片段激活。
     - 下半图：激活的指纹向量的特征与溶解性具有负向的预测关系（即：不溶解性），这些特征大多数被非极性的重复环结构激活。

     <p align="center">
        <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/9XJUGNTcxvoV.png?imageslim">
     </p>
     

   - 毒性特征：我们用相同的架构来预测分子毒性。下图展示了对应的片段（红色），这些片段可以最大程度的激活神经网络指纹向量中最有预测能力的特征。

     - 上半图：激活的指纹向量的特征与毒性具有正向的预测关系，这些特征大多数被包含芳环相连的硫原子基团的片段激活。
     - 下半图：激活的指纹向量的特征与毒性具有正向的预测关系，这些特征大多数被稠合的芳环（也被称作多环芳烃，一种著名的致癌物）激活。

    <p align="center">
       <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/A6bKM8eU4ee6.png?imageslim">
    </p>
    

#### 5.2.3 模型比较

1. 数据集：我们在多个数据集上比较圆形指纹和神经网络指纹的性能：

   - 溶解度数据集：包含 `1144` 个分子，及其溶解度标记。
   - 药物功效数据集：包含 `10000` 个分子，及其对恶行疟原虫（一种引发疟疾的寄生虫）的功效。
   - 有机光伏效率数据集：哈佛清洁能源项目使用昂贵的 `DFT` 模拟来估算有机分子的光伏效率，我们从该数据集中使用 `20000` 个分子作为数据集。

   我们的 `pipeline` 将每个分子编码的 `SMILES` 字符串作为输入，然后使用 `RDKit` 将其转换为`Graph` 。我们也使用 `RDKit` 生成的扩展圆形指纹作为 `baseline` 。这个过程中，氢原子被隐式处理。

   我们的 `ECFP` 和神经网络中用到的特征包括：

   - 原子特征：原子元素类型的 `one-hot`、原子的度`degree`、连接氢原子的数量、隐含价`implicit valence`、极性指示`aromaticity indicator`。
   - 键特征：是否单键、是否双键、是否三键、是否芳族键、键是否共轭、键是否为环的一部分。

2. 我们采用 `Adam` 优化算法，训练步数为 `10000`，`batch size = 100` 。我们还使用了 `batch normalization` 技术。

   - 我们还对神经网络进行了 `tanh` 和 `relu` 激活函数的对比实验，我们发现`relu` 在验证集上一直保持优势并且优势不大。
   - 我们还对神经网络进行了 `drop-connect` 实验，它是 `dropout` 的一个变种，其中权重被随机设置为零（而不是隐单元被随机设置为零）。我们发现这会导致更差的验证误差。

3. 我们使用 `Random-Search` 来优化以下超参数：学习率的对数 、初始权重的对数、$L_2$正则化系数的对数、指纹向量长度$S$、指纹深度$R$（最深六层）、全连接网络层的维度、神经网络指纹的隐层维度。

   所有超参数使用$k$折交叉验证来优化，其中每一折随机执行`50` 次。

4. 我们比较了两种情况下圆形指纹和神经网络指纹的性能：

   - 第一种情况：一个线性层使用指纹向量作为输入来执行预测，即 `linear layer`。
   - 第二种情况：一个单隐层的神经网络使用指纹向量作为输入来执行预测，即 `neural net` 。

   结果如下图所示。可以看到在所有实验中，神经网络指纹均达到或者超过圆形指纹的性能，并且使用神经网络层的方式（`neural net` ）超过了线性层的方式（`linear layer`）。


## 六、GGS-NN

1. 目前关于图神经网络模型的工作主要集中在单输出模型上，如`graph-level` 的分类。实际上有一些图任务需要输出序列，如：输出一条包含特定属性的顶点组成的路径。

   论文`《GATED GRAPH SEQUENCE NEURAL NETWORKS》` 基于 `GNN` 模型进行修改，它使用门控循环单元 `gated recurrent units` 以及现代的优化技术，并扩展到序列输出的形式。这种模型被称作门控图序列神经网络`Gated Graph Sequence Neural Networks:GGS-NNs` 。

   在`Graph` 数据结构上，`GGS-NNs` 相比于单纯的基于序列的模型（如`LSTM`）具有有利的归纳偏置`inductive biases` 。

   > 归纳偏置：算法对于学习的问题做的一些假设，这些假设就称作归纳偏置。它可以理解为贝叶斯学习中的“先验`prior`”，即对模型的偏好。
   >
   > - `CNN` 的归纳偏置为：局部性`locality`和平移不变性`spatial invariance`。即：空间相近的元素之间联系较紧、空间较远的元素之间没有联系；卷积核权重共享。
   > - `RNN` 的归纳偏置为：序列性`sequentiality` 和时间不变性 `time invariance`。即：序列顺序上的 `timestep` 之间有联系；`RNN`权重共享。

2. 在图上学习`representation` 有两种方式：

   - 学习输入图的 `representation` 。这也是目前大多数模型的处理方式。
   - 在生成一系列输出的过程中学习内部状态的 `representation`，其中的挑战在于：如何学习 `representation`，它既能对已经生成的部分输出序列进行编码（如路径输出任务中，已经产生的路径），又能对后续待生成的部分输出序列进行编码（如路径输出任务中，剩余路径）。

### 6.1 模型

#### 6.1.1 GNN回顾

1. 定义图$G=( V, E)$，其中$V$为顶点集合、$E$为边集合。顶点$v\in V$， 边$(v_1,v_2)\in E$。我们关注于有向图，因此$(v_1,v_2)$代表有向边$v_1\rightarrow v_2$，但是我们很容易用这套框架处理无向边。

   - 定义顶点$v$的标签为$\vec l _v \in \mathbb R^{n_v}$，定义边$(v_1,v_2)$的标签为$\vec l_{v_1,v_2} \in \mathbb R^{n_E}$，其中$n_v$为顶点标签的维度、$n_E$为边标签的维度。

   - 对给定顶点$v$，定义其状态向量为$\mathbf{\vec h}_v \in \mathbb R^D$。

     > 在原始 `GNN` 论文中状态向量记作$\mathbf{\vec x}_v$，为了和 `RNN` 保持一致，这里记作$\mathbf{\vec h}_v$。

   - 定义函数$\text{IN}(v) = \{v^\prime \mid (v^\prime,v) \in E\}$为顶点$v$的前序顶点集合，定义函数$\text{OUT}(v) = \{v^\prime\mid (v,v^\prime) \in E\}$为顶点$v$的后序顶点集合，定义函数$\text{NBR}(v) = \text{IN}(v)\bigcup \text{OUT}(v)$为顶点$v$的所有邻居顶点集合。

     > 在原始 `GNN` 论文中，邻居顶点仅仅考虑前序顶点集合，即指向顶点$v$的顶点集合。

   - 定义函数$\text{CO}(v) = \{ (v^\prime,v^{\prime\prime})\mid v= v^\prime\;\text{or} \; v= v^{\prime\prime}\}$表示所有包含顶点$v$的边（出边 + 入边）。

     > 在原始 `GNN` 论文中，仅考虑入边。即信息从邻居顶点流向顶点$v$。

   `GNN` 通过两个步骤来得到输出：

   - 首先通过转移函数`transition function` 得到每个顶点的`representation`$\mathbf{\vec h}_v$。其中转移函数也被称作传播模型`propagation model` 。
   - 然后通过输出函数 `output function` 得到每个顶点的输出$\mathbf{\vec o}_v$。 其中输出函数也被称作输出模型 `output model` 。

   该系统是端到端可微的，因此可以利用基于梯度的优化算法来学习参数。

2. 传播模型：我们通过一个迭代过程来传播顶点的状态。

   顶点的初始状态$\mathbf h_v^{(1)}$可以为任意值，然后每个顶点的状态可以根据以下方程来更新直到收敛，其中$t$表示时间步：

  $\mathbf{\vec h}_v^{(t)} = f_w(\vec l_v, \vec l_{\text{CO}(v)},\vec l_{\text{NER}(v)},\mathbf{\vec h}_{\text{NBR}(v)}^{(t-1)})$

   其中$f_w(\cdot)$为转移函数，它有若干个变种，包括：`nonpositional form` 和`posistional form`、线性和非线性。 论文建议按照 `nonpositional form` 进行分解：

  $f_w(\vec l_v, \vec l_{\text{CO}(v)},\vec l_{\text{NER}(v)},\mathbf{\vec h}_{\text{NBR}(v)}^{(t-1)}) = \sum_{v^\prime \in \text{IN}(v)} h_w(\vec l_v,\vec l_{v^\prime, v},\vec l_{v^\prime},\mathbf{\vec h}_{v^\prime}^{(t-1)}) + \sum_{v^\prime \in \text{OUT}(v)} h_w(\vec l_v,\vec l_{v, v^\prime},\vec l_{v^\prime},\mathbf{\vec h}_{v^\prime}^{(t-1)})$

   其中$h_w(\cdot)$可以为线性函数，或者为一个神经网络。当$h_w(\cdot)$为线性函数时，$h_w(\cdot)$为：

  $h_w(\vec l_v,\vec l_{v^\prime,v},\vec l_{v^\prime},\mathbf{\vec h}_{v^\prime}^{(t)}) = \mathbf A^{(v^\prime,v)} \mathbf{\vec h}_{v^\prime}^{(t-1)} + \mathbf{\vec b}^{(v^\prime,v)}$

   其中$\mathbf{\vec b}^{(v^\prime,v)}\in \mathbb R^D$和矩阵$\mathbf A^{(v^\prime,v)}\in \mathbb R^{D\times D}$分别由两个前馈神经网络的输出来定义，这两个前馈神经网络的参数对应于 `GNN` 的参数。

3. 输出模型：模型输出为$\mathbf{\vec o}_v = g_w(\mathbf{\vec h}_v, \vec l_v)$。其中$g_w$可以为线性的，也可以使用神经网络；$\mathbf{\vec h}_v$为传播模型最后一次迭代的结果$\mathbf{\vec h}_v^{(T)}$。

4. 为处理 `graph-level` 任务，`GNN` 建议创建一个虚拟的超级顶点`super node`，该超级顶点通过特殊类型的边连接到所有其它顶点，因此可以使用 `node-level` 相同的方式来处理 `graph-level` 任务。

5. `GNN` 模型是通过 `Almeida-Pineda` 算法来训练的，该算法首先执行传播过程并收敛，然后基于收敛的状态来计算梯度。

   其优点是我们不需要存储传播过程的中间状态（只需要存储传播过程的最终状态）来计算梯度，缺点是必须限制参数从而使得传播过程是收缩映射。

   转移函数是收缩映射是模型收敛的必要条件，这可能会限制模型的表达能力。当$f_w(\cdot)$为神经网络模型时，可以通过对网络参数的雅可比行列式的$L_1$范数施加约束来实现收缩映射的条件：

  $\mathcal L_w = \sum_{i=1}^p\sum_{j=1}^{q_i}||\mathbf{\vec t}_{i,j} - \varphi_w(G_i,v_{i,j})||_2^2 + \beta L\left(\left\|\frac{\partial F_w}{\partial \mathbf{\vec x}}\right\|_1\right)$

   其中$p$表示图的个数，$q_i$表示第$i$个图的顶点数量，$\mathbf{\vec t}_{i,j}$为第$i$个图、第$j$个顶点的监督信息，$\phi_w(G_i,v_{i,j})$为第$i$个图、第$j$个顶点的预测，$L(\cdot)$为罚项：

   超参数$\mu \in (0,1)$定义了针对转移函数的约束。

6. 事实上一个收缩映射很难在图上进行长距离的信息传播。

   考虑一个环形图，图有$N$个顶点，这些顶点首位相连。假设每个顶点的隐状态的维度为`1` ，即隐状态为标量。假设$h_w(\cdot)$为线性函数。为简化讨论，我们忽略了所有的顶点标签信息向量、边标签信息向量，并且只考虑入边而未考虑出边。

   在每个时间步$t$，顶点$v$的隐单元为：

  $h_v^{(t)} = m_v\times h_{v-1}^{(t-1)} + b_v$

   其中$m_v,b_v$为传播模型的参数。

   考虑到环形结构，我们认为：$v\le 0$时有$h_v = h_{N+v}$。

   令$\mathbf{\vec h}^{(t)} = [h_1^{(t)},\cdots,h_N^{(t)}]^T, \mathbf{\vec b} = [b_1,\cdots,b_N]^T$，令：

   则有：$\mathbf{\vec h}^{(t)} = \mathbf M\mathbf{\vec h}^{(t-1) } + \mathbf{\vec b}$。记$T(\mathbf{\vec h}^{(t-1)}) = \mathbf M\mathbf{\vec h}^{(t-1) } + \mathbf{\vec b}$，则$T(\cdot)$必须为收缩映射，则存在$\rho\le 1$使得对于任意的$\mathbf{\vec h},\mathbf{\vec h}^\prime$，满足:

  $||T(\mathbf{\vec h}) - T(\mathbf{\vec h}^\prime)|| \lt \rho ||\mathbf{\vec h}-\mathbf{\vec h}^\prime||$

   即：

  $||\mathbf M(\mathbf{\vec h} - \mathbf{\vec h}^\prime)||\lt \rho ||\mathbf{\vec h} - \mathbf{\vec h}^\prime||$

   如果选择$\mathbf{\vec h}^\prime = \mathbf{\vec 0}$，选择$\mathbf{\vec h} = (\underbrace{0,0,\cdots,0}_{v-2},1,0,\cdots,0)^T$（即除了位置$v-1$为`1`、其它位置为零） ，则有$|m_v|\lt \rho$。

   扩展$h_v^{(t)} = m_v\times h_{v-1}^{(t-1)} + b_v$，则有：

   考虑到$|m_v|\lt \rho$，这意味着从顶点$j\rightarrow j+1 \rightarrow j+2\cdots \rightarrow v$传播的信息以指数型速度$\rho^\delta$衰减，其中$\delta$为顶点$j$到顶点$v$的距离（这里$j$是$v$的上游顶点）。因此 `GNN` 无法在图上进行长距离的信息传播。

7. 事实上，当$h_w(\cdot)$为非线性函数时，收缩映射也很难在图上进行长距离的信息传播。令

  $h_v^{(t)} = \sigma \left( m_v\times h_{v-1}^{(t-1)} + b_v\right)$

   其中$\sigma(\cdot)$为非线性激活函数。则有$T(\mathbf{\vec h}^{(t-1)}) = \sigma\left(\mathbf M\mathbf{\vec h}^{(t-1) } + \mathbf{\vec b}\right)$，$T(\cdot)$为一个收缩映射。则存在$\rho\le 1$使得对于任意的$\mathbf{\vec h},\mathbf{\vec h}^\prime$，满足:

  $||T(\mathbf{\vec h}) - T(\mathbf{\vec h}^\prime)|| \lt \rho ||\mathbf{\vec h}-\mathbf{\vec h}^\prime||$

   这意味着函数$T(\mathbf{\vec h})$的雅可比矩阵的每一项都必须满足：

  $\left|\frac{\partial T_i}{\partial h_j}\right|\lt \rho, \forall i, \forall j$

   证明：考虑两个向量$\mathbf{\vec h},\mathbf{\vec h}^{\prime}$，其中 ：

   则有$||T_i(\mathbf{\vec h}) - T_i(\mathbf{\vec h}^\prime)||\le ||T(\mathbf{\vec h}) - T(\mathbf{\vec h}^\prime)|| \lt \rho |\Delta|$，则有：

  $\left\|\frac{T_i(h_1,\cdots,h_{j-1},h_j,h_{j+1},\cdots,h_N) - T_i(h_1,\cdots,h_{j-1},h_j+\Delta,h_{j+1},\cdots,h_N)}{\Delta}\right\|\lt \rho$

   其中$T_i(\mathbf{\vec h})$为$T(\mathbf{\vec h})$的第$i$个分量。当$\Delta\rightarrow 0$时， 左侧等于$\left|\frac{\partial T_i}{\partial h_j}\right|$，因此得到结论$\left|\frac{\partial T_i}{\partial h_j}\right|\lt \rho, \forall i, \forall j$。

   当$j= i-1$时，有$\left|\frac{\partial T_i}{\partial h_{i-1}}\right| \lt \rho$。考虑到图为环状结构，因此对于$j\ne i-1$的顶点有$\frac{\partial T_i}{\partial h_j} = 0$。

   考虑到时刻$t$的更新，则有：

  $\left|\frac{\partial h_i^{(t)}}{\partial h_{i-1}^{(t-1)}}\right| \lt \rho$

   现在考虑$h_1(1)$如何影响$h_t^{(t)}$。考虑链式法则以及图的环状结构，则有：

   当$\rho \lt 1$时，偏导数$\frac{\partial h_t^{(t)}}{\partial h_{1}^{(1)}}$随着$t$的增加指数级降低到`0` 。这意味着一个顶点对另一个顶点的影响将呈指数级衰减，因此 `GNN` 无法在图上进行长距离的信息传播。

#### 6.1.2 GG-NN模型

1. 门控图神经网络 `Gated Graph Neural Networks:GG-NNs` 对 `GNN` 进行修改，采用了门控循环单元`GRU` ，并对固定的$T$个时间步进行循环展开。

   - `GG-NNs` 使用基于时间的反向传播`BPTT` 算法来计算梯度。

   - 传统 `GNN` 模型只能给出非序列输出，而`GG-NNs` 可以给出序列输出。

     本节给出的 `GG-NNs` 模型只支持非序列输出，但是 `GG-NNs` 的变种 `GGS-NNs` 支持序列输出。

   - 相比`Almeida-Pineda` 算法，`GG-NNs` 需要更多内存，但是后者不需要约束参数以保证收缩映射。

2. 在 `GNN` 中顶点状态的初始化值没有意义，因为不动点理论可以确保不动点独立于初始化值。但是在 `GG-NNs` 模型中不再如此，顶点的初始化状态可以作为额外的输入。因此顶点的初始化状态可以视为顶点的标签信息的一种。

   为了区分顶点的初始化状态和其它类型的顶点标签信息，我们称初始化状态为顶点的注解`node annotation`，以向量$\mathbf{\vec x}$来表示。

3. 注解向量的示例：对于给定的图，我们希望预测是否存在从顶点$s$到顶点$t$的路径。

   该任务存在任务相关的两个顶点$s,t$，因此我们定义注解向量为：

   注解向量使得顶点$s$被标记为任务的第一个输入参数，顶点$t$被标记为任务的第二个输入参数。我们通过$\mathbf{\vec x}_v$来初始化状态向量$\mathbf{\vec h}_v^{(1)}$：

  $\mathbf{\vec h}_v^{(1)} = [x_{v,0},x_{v,1},0,\cdots,0]^T$

   即：$\mathbf{\vec h}_v^{(1)}$的前面两维为$\mathbf{\vec x}_v$、后面的维度填充为零。

   传播模型很容易学得将顶点$s$的注解传播到任何$s$可达的顶点。如，通过设置传播矩阵为：所有存在后向边的位置都为`1` 。这将使得$\mathbf{\vec h}_s^{(1)}$的第一维沿着后向边进行复制，使得从顶点$s$可达的所有顶点的$\mathbf{\vec h}_v^{(T)}$的第一维均为`1` 。

   最终查看顶点$t$的状态向量前两维是否为`[1,1]` 即可判断从$s$是否可达顶点$t$。

4. 传播模型：

   - 初始化状态向量：$\mathbf{\vec h}_v^{(1)} = [\mathbf{\vec x}_v^T,\mathbf{\vec 0}]^T \in \mathbb R^D$，其中$D$为状态向量的维度。这一步将顶点的注解信息拷贝到状态向量的前几个维度。

   - 信息传递：$\mathbf{\vec a}_v^{(t)} = \mathbf A_{v:}^T[\mathbf{\vec h}_1^{(t-1)T},\cdots,\mathbf{\vec h}_{|V|}^{(t-1)T} ]+ \mathbf{\vec b}_v$，它包含所有方向的边的激活值。

     如下图所示 `(a)` 表示一个图，颜色表示不同的边类型（类型 `B` 和类型 `C` ）；`(b)` 表示展开的一个计算步；`(c)` 表示矩阵$\mathbf A$，$B^\prime$表示$B$的反向边，采用不同的参数。

    $\mathbf A \in \mathbb R^{D|V|\times 2D|V|}$对应于图中的边，它由$\mathbf A^{(out)}\in \mathbb R^{D|V|\times D|V|}$和$\mathbf A^{(in)}\in \mathbb R^{D|V|\times D|V|}$组成，其参数由边的方向和类型决定。通常它们都是稀疏矩阵。

    $\mathbf A_{v:}\in \mathbb R^{D|V|\times 2D}$由$\mathbf A^{(out)},\mathbf A^{(in)}$对应于顶点$v$的两列组成；$\mathbf{\vec b}_v \in \mathbb R^{2D}$。

     <p align="center">
        <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/WLxWFXfjl2zp.png?imageslim">
     </p>
     

   - `GRU` 更新状态：

     这里采用类似 `GRU` 的更新机制，基于顶点的历史状态向量和所有边的激活值来更新当前状态。$\mathbf{\vec z}$为更新门，$\mathbf{\vec r}$为复位门，$\sigma(x) = 1/(1+e^{-x})$为 `sigmoid` 函数，$\odot$为逐元素乘积。

     我们最初使用普通的 `RNN` 来进行状态更新，但是初步实验结论表明：`GRU` 形式的状态更新效果更好。

5. 输出模型：我们可以给出最终时间步的输出。

   - `node-level` 输出：$\mathbf{\vec o}_v = g(\mathbf{\vec h}_v^{(T)}, \mathbf{\vec x}_v)$。然后可以对$\mathbf{\vec o}_v$应用一个 `softmax` 函数来得到每个顶点在各类别的得分。

   - `graph-level` 输出：定义`graph-level` 的 `representation` 向量为：

    $\mathbf{\vec h}_G = \tanh\left(\sum_{v\in V}\sigma\left(g_1\left(\mathbf{\vec h}_v^{(T)},\mathbf{\vec x}_v\right)\right)\odot \tanh\left( g_2\left(\mathbf{\vec h}_v^{(T)},\mathbf{\vec x}_v\right)\right)\right)$

     其中：

     -$\sigma\left(g_1\left(\mathbf{\vec h}_v^{(T)},\mathbf{\vec x}_v\right)\right)$起到 `soft attention` 机制的作用，它决定哪些顶点和当前的`graph-level`任务有关。
     -$g_1(\cdot),g_2(\cdot)$都是神经网络，它们拼接$\mathbf{\vec h}_v^{(T)}$和$\mathbf{\vec x}_v$作为输入， 输出一个实值向量。
     -$\tanh (\cdot)$函数也可以替换为恒等映射。

   > 注意：这里的 `GG-NNs` 给出的是非序列输出，实际上 `GG-NNs` 支持序列输出，这就是下面介绍的 `GGS-NNs` 模型

#### 6.1.3 GGS-NNs 模型

1. 门控图序列神经网络 `Gated Graph Sequence Neural Networks :GGS-NNs`使用若干个 `GG-NNs` 网络依次作用从而生成序列输出$\mathbf{\vec o}^{(1)},\cdots, \mathbf{\vec o}^{(K)}$。

   在第$k$个输出：

   - 定义所有顶点的注解向量组成矩阵$\mathbf X^{(k)} = [\mathbf{\vec x}_1^{(k)},\cdots, \mathbf{\vec x}_{|V|}^{(k)}]^T\in \mathbb R^{|V|\times D_a}$，其中$D_a$为注解向量的维度。

     定义所有顶点的输出向量组成矩阵$\mathbf O^{(k)} = [\mathbf{\vec o}_1^{(k)},\cdots,\mathbf{\vec o}_{|V|}^{(k)}]\in \mathbb R^{|V|\times D_o}$，其中$D_o$为输出向量的维度。

   - 我们使用两个 `GG-NNs` 网络$\mathcal F_{\mathcal O}^{(k)}$和$\mathcal F_{\mathcal X}^{(k)}$，其中$\mathcal F_{\mathcal O}^{(k)}$用于从$\mathbf X^{(k)}$中预测$\mathbf O ^{(k)}$、$\mathcal F_{\mathcal X}^{(k)}$用于从$\mathbf X^{(k)}$中预测$\mathbf X^{(k+1)}$。$\mathbf X^{(k+1)}$可以视作一个“状态”，它从输出步$k$转移到输出步$k+1$。

     - 每个$\mathcal F_{\mathcal O}^{(k)}$和$\mathcal F_{\mathcal X}^{(k)}$均包含各自的传播模型和输出模型。我们定义第$k$个输出步的第$t$个时间步的状态矩阵分别为：

其中$D_{\mathcal O},D_{\mathcal X}$为各自传播模型的状态向量的维度。如前所述，$\mathbf H^{(k,1)}$可以通过$\mathbf X^{(k)}$通过填充零得到，因此有：$\mathbf H^{(k,1)}_{\mathcal O} = \mathbf H^{(k,1)}_{\mathcal X}$，记作$\mathbf H^{(k,1)}$。

- 我们也可以选择$\mathcal F_{\mathcal O}^{(k)}$和$\mathcal F_{\mathcal X}^{(k)}$共享同一个传播模型，然后使用不同的输出模型。这种方式的训练速度更快，推断速度更快，并且大多数适合能够获得原始模型相差无几的性能。但是如果$\mathcal F_{\mathcal O}^{(k)}$和$\mathcal F_{\mathcal X}^{(k)}$的传播行为不同，则这种变体难以适应。

  -$\mathcal F_{\mathcal X}^{(k)}$的输出模型称为顶点`annotation output` 模型，它用于从$\mathbf H^{(k,T)}_{\mathcal X}$中预测$\mathbf X^{(k+1)}$。该模型在每个顶点$v$上利用神经网络独立的预测：

   $\mathbf{\vec x}_v^{(k+1)} = \sigma\left(g_a(\mathbf{\vec h}_{\mathcal X,v}^{(k,T)}, \mathbf{\vec x}_v^{(k)})\right)$

    其中$g_a(\cdot)$为神经网络，$\mathbf{\vec h}_{\mathcal X,v}^{(k,T)}$和$\mathbf{\vec x}_v^{(k)}$的拼接作为网络输入，$\sigma(\cdot)$为`sigmoid` 函数。

  

  整个网络的结构如下图所示，如前所述有$\mathbf H^{(k,1)}_{\mathcal O} = \mathbf H^{(k,1)}_{\mathcal X}$，记作$\mathbf H^{(k,1)}$。

  <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/yS5vS1tfaEsh.png?imageslim">
  </p>
  

1. `GGS-NNs` 的训练有两种方式：

   - 仅仅给定$\mathbf X^{(1)}$，然后执行端到端的模型训练。这种方式更为通用。

     我们将$\mathbf X^{(k)},k\gt 1$视为网络的隐变量，然后通过反向传播算法来联合训练。

   - 指定所有的中间注解向量：$\mathbf X^{(1)},\mathbf X^{(2)},\cdots,\mathbf X^{(K)}$。当我们已知关于中间注解向量的信息时，这种方式可以提高性能。

     考虑一个图的序列输出任务，其中每个输出都仅仅是关于图的一个部分的预测。为了确保图的每个部分有且仅被预测一次，我们需要记录哪些顶点已经被预测过。我们为每个顶点指定一个`bit` 作为注解，该比特表明顶点到目前为止是否已经被“解释”过。因此我们可以通过一组注解来捕获输出过程的进度。

     此时，我们可以将尚未“解释”的顶点的注解作为模型的额外输入。因此我们的 `GGS-NNs` 模型中，`GG-NNs` 和给定的注解是条件独立的。

     - 训练期间序列输出任务将被分解为单个输出任务，并作为独立的 `GG-NNs` 来训练。
     - 测试期间，第$k$个输出得到的注解（已“解释”过的顶点）当作为第$k+1$个输出的网络输入。

### 6.2 application

#### 6.2.1 bAbI 任务

1. `bAbI` 任务旨在测试 `AI` 系统应该具备的推理能力。在 `bAbI suite` 中有`20` 个任务来测试基本的推理形式，包括演绎、归纳、计数和路径查找。

   - 我们定义了一个基本的转换过程 `transformation procedure` 从而将 `bAbI` 任务映射成 `GG-NNs` 或者 `GGS-NNs` 任务。

     我们使用已发布的 `bAbI` 代码中的 `--symbolic` 选项从而获取仅涉及`entity` 实体之间一系列关系的`story` 故事，然后我们将每个实体映射为图上的一个顶点、每个关系映射为图上的一条边、每个`story`被映射为一张图。

   - `Question` 问题在数据中以 `eval` 来标记，每个问题由问题类型（如`has_fear`）、问题参数（如一个或者多个顶点）组成。我们将问题参数转换为初始的顶点注解，第$i$个参数顶点注解向量的第$i$位设置为 `1` 。

     如问题`eval E > A true` ，则：问题类型为 `>` ；问题参数为`E, A` ；顶点的注解向量为：

     问题的监督标签为`true` 。

   - `bAbI` 任务`15` （`Basic Deduction`任务）转换的符号数据集`symbolic dataset` 的一个示例：

     ```
     xxxxxxxxxx
     ```

     12

     1

     ```
     D is A
     ```

     2

     ```
     B is E
     ```

     3

     ```
     A has_fear F
     ```

     4

     ```
     G is F
     ```

     5

     ```
     E has_fear H
     ```

     6

     ```
     F has_fear A
     ```

     7

     ```
     H has_fear A
     ```

     8

     ```
     C is H
     ```

     9

     ```
     eval B has_fear H
     ```

     10

     ```
     eval G has_fear A
     ```

     11

     ```
     eval C has_fear A
     ```

     12

     ```
     eval D has_fear F
     ```

     - 前`8` 行描述了事实 `fact`，`GG-NNs` 将基于这些事实来构建`Graph` 。每个大写字母代表顶点，`is` 和 `has_fear` 代表了边的`label` （也可以理解为边的类型）。
     - 最后`4` 行给出了四个问题，`has_fear` 代表了问题类型。
     - 每个问题都有一个输入参数，如 `eval B has_fear H`中，顶点 `B` 为输入参数。顶点 `B` 的初始注解为标量`1`（只有一个元素的向量就是标量）、其它顶点的初始注解标量为 `0` 。

   - 某些任务具有多个问题类型，如`bAbI` 任务 `4` 具有四种问题类型：`e,s,w,n` 。对于这类任务，我们为每个类型的任务独立训练一个 `GG-NNs` 模型。

   - 在任何实验中，我们都不会使用很强的监督标签，也不会给`GGS-NNs`任何中间注解信息。

2. 我们的转换方式虽然简单，但是这种转换并不能保留有关`story` 的所有信息，如转换过程丢失了输入的时间顺序；这种转换也难以处理三阶或者更高阶的关系，如 `昨天 John 去了花园` 则难以映射为一条简单的边。

   注意：将一般化的自然语言映射到符号是一项艰巨的任务，因此我们无法采取这种简单的映射方式来处理任意的自然语言。

   即使是采取这种简单的转化，我们仍然可以格式化描述各种`bAbI` 任务，包括任务`19`（路径查找任务）。我们提供的 `baseline` 表明：这种符号化方式无助于 `RNN/LSTM` 解决问题，但是`GGS-NNs` 可以基于这种方式以少量的训练样本来解决问题。

   `bAbI` 任务`19` 为路径查找 `path-finding`任务，该任务几乎是最难的任务。其符号化的数据集中的一个示例：

   ```
   xxxxxxxxxx
   ```

   5

   1

   ```
   E s A
   ```

   2

   ```
   B n C
   ```

   3

   ```
   E w F
   ```

   4

   ```
   B w E
   ```

   5

   ```
   eval path B A w,s
   ```

   - 开始的`4` 行描述了四种类型的边，`s,n,w,e` 分别表示`东，南，西，北`。在这个例子中，`e` 没有出现。
   - 最后一行表示一个路径查找问题：`path` 表示问题类型为路径查找；`B, A` 为问题参数；`w,s` 为答案序列，该序列是一个方向序列。该答案表示：从`B` 先向西（到达顶点`E`）、再向南可以达到顶点 `A` 。

3. 我们还设计了两个新的、类似于 `bAbI` 的任务，这些任务涉及到图上输出一个序列。这两个任务包括：最短路径问题和欧拉回路问题。

   - 最短路径问题需要找出图中两个点之间的最短路径，路径以顶点的序列来表示。

     我们首先生成一个随机图并产生一个 `story`，然后我们随机选择两个顶点 `A` 和 `B` ，任务是找出顶点 `A` 和 `B` 之间的最短路径。

     为了简化任务，我们限制了数据集生成过程：顶点`A` 和 `B` 之间存在唯一的最短路径，并且该路径长度至少为 `2` (即 `A` 和 `B` 的最短路径至少存在一个中间结点)。

   - 如果图中的一个路径恰好包括每条边一次，则该路径称作欧拉路径。如果一个回路是欧拉路径，则该回路称作欧拉回路。

     对于欧拉回路问题，我们首先生成一个随机的、`2-regular` 连接图，以及一个独立的随机干扰图。然后我们随机选择两个顶点`A` 和 `B` 启动回路，任务是找出从 `A` 到 `B` 的回路。

     为了增加任务难度，这里添加了干扰图，这也使得输出的回路不是严格的“欧拉回路”。

     > 正则图是每个顶点的`degree`都相同的无向简单图，`2-regular` 正则图表示每个顶点都有两条边。

4. 对于`RNN` 和 `LSTM` 这两个 `baseline`，我们将符号数据集转换为 `token` 序列：

   ```
   xxxxxxxxxx
   ```

   2

   1

   ```
   n6 e1 n1 eol n6 e1 n5 eol n1 e1 n2 eol n4 e1 n5 eol n3 e1 n4
   ```

   2

   ```
   eol n3 e1 n5 eol n6 e1 n4 eol q1 n6 n2 ans 1
   ```

   其中 `n` 表示顶点、`e` 表示边、`q` 表示问题类型。额外的 `token` 中，`eol` 表示一行的结束`end-of-line`、`ans`代表答案`answer` 、最后一个数字`1` 代表监督的类别标签。

   我们添加`ans` 从而使得 `RNN/LSTM` 能够访问数据集的完整信息。

5. 训练配置：

   - 本节中的所有任务，我们生成 `1000`个训练样本（其中有 `50` 个用于验证，只有 `950` 个用于训练）、`1000`个测试样本。

   - 在评估模型时，对于单个样本包含多个问题的情况，我们单独评估每个问题。

   - 由于数据集生成过程的随机性，我们为每个任务随机生成`10` 份数据集，然后报告了这`10` 份数据集上评估结果的均值和标准差。

   - 我们首先以 `50` 个训练样本来训练各个模型，然后逐渐增加训练样本数量为`100、250、500、950` （最多`950` 个训练样本）。

     由于 `bAbI` 任务成功的标准是测试准确率在 `95%` 及其以上，我们对于每一个模型报告了测试准确率达到 `95%` 所需要的最少训练样本，以及该数量的训练样本能够达到的测试准确率。

   - 在所有任务中，我们展开传播过程为 `5` 个时间步。

   - 对于 `bAbI` 任务`4、15、16、18、19` ，我们的 `GG-NNs` 模型的顶点状态向量$\mathbf{\vec h}_v^{(t)}$的维度分别为 `4、5、6、3、6` 。

     对于最短路径和欧拉回路任务，我们的`GG-NNs` 模型的顶点状态向量$\mathbf{\vec h}_v^{(t)}$维度为 `20` 。

   - 对于所有的 `GGS-NNs` ，我们简单的令$\mathcal F_{\mathcal O}^{(k)},\mathcal F_{\mathcal X}^{(k)}$共享同一个传播模型。

   - 所有模型都基于 `Adam` 优化器训练足够长的时间，并使用验证集来选择最佳模型。

6. 单输出任务：`bAbI`的任务`4`（`Tow Argument Relations`）、任务`15`（`Basic Deduction`）、任务`16`（`Basic Induction`）、任务`18`（`Size Reasoning`） 这四个任务都是单输出任务。

   - 对于任务`4、15、16`，我们使用 `node-level GG-NNs`；对于任务 `18` 我们使用 `graph-level GG-NNs` 。

   - 所有 `GG-NNs` 模型包含少于 `600` 个参数。

   - 我们在符号化数据集上训练 `RNN` 和 `LSTM` 模型作为 `baseline`。 `RNN` 和 `LSTM` 使用 `50` 维的`embedding` 层和 `50` 维的隐层，它们在序列末尾给出单个预测输出，并将输出视为分类问题。

     这两个模型的损失函数为交叉熵，它们分别包含大约`5k` 个参数（`RNN`）和`30k` 个参数 （`LSTM` ）。

   预测结果如下表所示。对于所有任务，`GG-NNs` 仅需要`50` 个训练样本即可完美的预测（测试准确率 `100%`）；而 `RNN/LSTM`要么需要更多训练样本（任务`4`）、要么无法解决问题（任务`15、16、18`）。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/IYdM0D74TlnM.png?imageslim">
   </p>
   

   对于任务`4`，我们进一步考察训练数据量变化时，`RNN/LSTM` 模型的性能。可以看到，尽管 `RNN/LSTM` 也能够几乎完美的解决任务，但是 `GG-NNs` 可以使用更少的数据达到 `100%` 的测试准确率。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/TzrWFclNnWpx.png?imageslim">
   </p>
   

7. 序列输出任务：所有 `bAbI` 任务中，任务`19`（路径查找任务）可以任务是最难的任务。我们以符号数据集的形式应用 `GGS-NNs` 模型，每个输出序列的末尾添加一个额外的 `end` 标签。在测试时，网络会一直预测直到预测到 `end` 标签为止。

   另外，我们还对比了最短路径任务和欧拉回路任务。

   下表给出了任务的预测结果。可以看到 `RNN/LSTM` 都无法完成任务， `GGS-NNs` 可以顺利完成任务。另外 `GGS-NNs` 仅仅利用 `50` 个训练样本就可以达到比 `RNN/LSTM`更好的测试准确率。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/dAUujneeXVdp.png?imageslim">
   </p>
   

8. 为什么`RNN/LSTM` 相对于单输出任务，在序列输出任务上表现很差？

   欧拉回路任务是 `RNN/LSTM` 最失败的任务，该任务的典型训练样本如下：

   ```
   xxxxxxxxxx
   ```

   19

   1

   ```
   3 connected-to 7
   ```

   2

   ```
   7 connected-to 3
   ```

   3

   ```
   1 connected-to 2
   ```

   4

   ```
   2 connected-to 1
   ```

   5

   ```
   5 connected-to 7
   ```

   6

   ```
   7 connected-to 5
   ```

   7

   ```
   0 connected-to 4
   ```

   8

   ```
   4 connected-to 0
   ```

   9

   ```
   1 connected-to 0
   ```

   10

   ```
   0 connected-to 1
   ```

   11

   ```
   8 connected-to 6
   ```

   12

   ```
   6 connected-to 8
   ```

   13

   ```
   3 connected-to 6
   ```

   14

   ```
   6 connected-to 3
   ```

   15

   ```
   5 connected-to 8
   ```

   16

   ```
   8 connected-to 5
   ```

   17

   ```
   4 connected-to 2
   ```

   18

   ```
   2 connected-to 4
   ```

   19

   ```
   eval eulerian-circuit 5 7       5,7,3,6,8
   ```

   这个图中有两个回路 `3-7-5-8-6`和 `1-2-4-0`，其中 `3-7-5-8-6` 是目标回路，而 `1-2-4-0` 是一个更小的干扰图。为了对称性，所有边都出现两次，两个方向各一次。

   对于 `RNN/LSTM`，上述符号转换为 `token`序列：

   ```
   xxxxxxxxxx
   ```

   4

   1

   ```
   n4 e1 n8 eol n8 e1 n4 eol n2 e1 n3 eol n3 e1 n2 eol n6 e1 n8 eol
   ```

   2

   ```
   n8 e1 n6 eol n1 e1 n5 eol n5 e1 n1 eol n2 e1 n1 eol n1 e1 n2 eol
   ```

   3

   ```
   n9 e1 n7 eol n7 e1 n9 eol n4 e1 n7 eol n7 e1 n4 eol n6 e1 n9 eol
   ```

   4

   ```
   n9 e1 n6 eol n5 e1 n3 eol n3 e1 n5 eol q1 n6 n8 ans 6 8 4 7 9
   ```

   注意：这里的顶点`ID` 和原始符号数据集中的顶点 `ID` 不同。

   - `RNN/LSTM` 读取整个序列，并在读取到 `ans` 这个`token` 的时候开始预测第一个输出。然后在每一个预测步，使用`ans` 作为输入，目标顶点`ID` (视为类别标签) 作为输出。这里每个预测步的输出并不会作为下一个预测步的输入。

     我们的 `GGS-NNs` 模型使用相同的配置，其中每个预测步的输出也不会作为下一个预测步的输入，仅有当前预测步的注解$\mathbf X^{(k)}$延续到下一个预测步，因此和 `RNN/LSTM` 的比较仍然是公平的。这使得我们的 `GGS-NNs` 有能力得到前一个预测步的信息。

     一种改进方式是：在`RNN/LSTM/GGS-NNs` 中，每个预测步可以利用前一个预测步的结果。

   - 这个典型的样本有 `80` 个 `token`，因此我们看到 `RNN/LSTM` 必须处理很长的输入序列。如第三个预测步需要用到序列头部的第一条边`3-7`，这需要 `RNN/LSTM` 能够保持长程记忆。`RNN` 中保持长程记忆具有挑战性，`LSTM` 在这方面比 `RNN` 更好但是仍然无法完全解决问题。

   - 该任务的另一个挑战是：输出序列出现的顺序和输入序列不同。实际上输入数据并没有顺序结构，即使边是随机排列的，目标顶点的输出顺序也不应该改变。`bAbI` 任务`19`路径查找、最短路径任务也是如此。

     `GGS-NNs` 擅长处理此类“静态”数据，而`RNN/LSTM` 则不然。实际上 `RNN/LSTM` 更擅长处理动态的时间序列。如何将 `GGS-NNs` 应用于动态时间序列，则是将来的工作。

#### 6.2.2 讨论

1. 思考`GG-NNs` 正在学习什么是有启发性的。为此我们观察如何通过逻辑公式解决`bAbI` 任务`15` 。为此考虑回答下面的问题：

   ```
   xxxxxxxxxx
   ```

   3

   1

   ```
   B is E
   ```

   2

   ```
   E has_fear H
   ```

   3

   ```
   eval B has_fear
   ```

   要进行逻辑推理，我们不仅需要对 `story 里存在的事实进行逻辑编码，还需要将背景知识编码作为推理规则。如：$\text{is}(x,y)\land \text{has-fear}(y,z) \rightarrow \text{has-fear}(x,z)$。

   我们对任务的编码简化了将 `story` 解析为`Graph` 的过程，但是它并不提供任何背景知识。因此可以将 `GG-NNs` 模型视为学习背景知识的方法，并将结果存储在神经网络权重中。

2. 当前的 `GG-NNs` 必须在读取所有 `fact` 事实之后才能回答问题，这意味着网络必须尝试得出所见事实的所有后果，并将所有相关信息存储到其顶点的状态中。这可能并不是一个理想的形式，最好将问题作为初始输入，然后动态的得到回答问题所需要的事实。

## 七、PATCHY-SAN

1. 很多重要问题都可以视为图数据得学习问题。考虑以下两个问题：

   - 给定一组图作为训练数据，学习一个用于对未见过的图（即测试图）进行分类或者回归的函数。其中训练集中任意两个图的结构不一定是相同的。例如：给定一组化合物以及它们对于癌细胞活性抑制效果，用于预测新的化合物对于癌细胞活性抑制的结果。
   - 给定一张大图，学习该图的`representation` 从而可以推断未见过的图属性，如顶点类型（即顶点分类）、缺失边（即链接预测）。

2. 卷积神经网络`CNN` 在图像领域中大获成功。图像`image` 也是一种图`graph` ，但是它是一种特殊的正方形的网格图，其中顶点表示像素。如下图所示，黑色/白色顶点表示像素的不同取值（黑色取值为`1`、白色取值为`0` ），红色顶点表示当前卷积核的中心位置。`(a)` 图给出了一个 `3x3` 卷积核在一个 `4x4` 图像上的卷积过程，其中步幅为`1`、采用非零填充。

   我们可以将 `CNN` 视为遍历一个顶点序列（即图`(a)` 中的红色顶点 `1,2,3,4` ）、然后为该序列中的每个顶点生成固定大小的邻域子图（即图`(b)` 中的 `3x3` 网格） 的过程。其中邻域子图作为感受野`receptive field`，用于特征抽取（抽取感受野中顶点的特征）。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/4fRmexuAAsY1.png?imageslim">
   </p>
   

   由于像素点的隐式空间顺序，从左到右以及从上到下唯一确定了这个顶点序列。在`NLP` 问题中，句子也隐式的从左到右确定了单词的顺序。但是对于图`graph` 问题，难以针对不同的问题给予一个合适的顶点顺序。同时数据集中不同图的结构不同，不同图的顶点之间都无法对应。

   因此如果希望将卷积神经网络应用到图结构上，必须解决两个问题：

   - 为图确定一个顶点序列，其中我们为序列中的每个顶点创建邻域子图。
   - 邻域图的归一化，即将邻域子图映射到向量空间，从而方便后续的卷积运算（因为卷积核无法作用于子图上）。

   论文`《Learning Convolutional Neural Networks for Graphs》` 提出了一种学习任意图的卷积神经网络框架 `PATCHY-SAN` ，该框架解决了这两个问题。`PATCHY-SAN` 适用于无向图/有向图，可以处理连续/离散的顶点和边的属性，也支持多种类型的边。

   对于每个输入的`graph`，`PATCHY-SAN` 方法首先确定了一个顶点序列；然后，对序列中每个顶点提取一个正好由$k$个顶点组成的邻域并归一化邻域，归一化的邻域将作为当前顶点的感受野；最后，就像 `CNN` 的感受野一样，可以将一些特征学习组件（如卷积层、`dense` 层）作用到归一化邻域上。其整体架构如下图所示，其中红色顶点表示顶点序列中的顶点，$k=5$。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/fE5AEhEBRLUS.png?imageslim">
   </p>
   

3. `PATCHY-SAN` 方法具有以下优势：

   - 计算高效，原生支持并行计算，并且可用于大型图。
   - 支持特征可视化，可以深入了解图的结构属性。
   - 相比较与 `graph kernel` 方法，`PATCHY-SAN` 无需依赖任何特征工程就可以学到具体任务的特征。

### 7.1 模型

#### 7.1.1 Graph Kernel

1. 目前现有的大多数 `Graph Kernel` 算法都是基于 `R-Convolution` 理论构建而来，其理论思想是：设计一种图的分解算法，两个图的核函数和图分解后的子结构相似程度有关。

   给定两个图$G_1(V_1,E_1),G_2(V_2,E_2)$以及一种图的分解方式$\mathcal F(\cdot)$，则分解后的子结构为：

   基于该子结构，则$G_1$和$G_2$的核函数可以表示为：

  $k_R(G_1,G_2) = \sum_{i=1}^{n_1}\sum_{j=1}^{n_2}\delta(S_{1,i},S_{2,j})$

   其中：

   因此，任意一种图的分解方式$\mathcal F(\cdot)$以及任意一种子结构同构判断方式$\delta(\cdot)$的组合都可以定义一种新的 `Graph Kernel` ，常见的主要分为三类：

   - 基于游走的`Graph Kernel`，如 `Random Walk Kernel`。
   - 基于路径的 `Graph Kernel`，如 `Shortest-Path Kernel` 。
   - 基于子树`subtree` 或者子图 `subgraph` 的 `Graph Kernel` ，如 `Weisfeiler-Lehman Subtree Kernel`。

   另外，除了 `R-Convolution` 系列之外，还有其它的 `Graph Kernel` 。

2. `Random Walk Kernel`：随机游走`Kernel`的基本思想是：统计两个输入图中相同的随机游走序列的数量。

   给定输入图$G_1(V_1,E_1),G_2(V_2,E_2)$，设顶点$v$的`label` 为$l_v$。定义`direct product graph`$G_{\times}$为：$G_{\times} = (V_{\times},E_{\times})$，其中：

   其中$l_v$表示顶点$v$的`label`，$l_{(v_1,v_2)}$表示边$(v_1,v_2)$的 `label` 。

   定义图$G_{\times}$的邻接矩阵为$\mathbf A_{\times}$，则随机游走 `kernel` 定义为：

  $k_{\times}(G_1,G_2) = \sum_{i,j=1}^{|V_{\times}|}\left[\sum_{n=0}^\infty \lambda_n \mathbf A^n_{\times}\right]_{i,j}$

   其中$\lambda_n$必须仔细挑选从而保证$k_\times$的收敛性。

   -$V_{\times}$给出了$(G_1,G_2)$中两个顶点`label`相同的顶点`pair` 对

   -$E_{\times}$给出了$(G_1,G_2)$中两条边`label` 相同、边的对应顶点分别相同的边 `pair` 对。

     ```
     xxxxxxxxxx
     ```

     2

     1

     ```
     G1:        v1(label=A) -------- v2(label=B), label(v1,v2) = s
     ```

     2

     ```
     G2:        w1(label=A) -------- w2(label=B), label(w1,w2) = s
     ```

   -$\sum_{i,j=1}^{|V_{\times}|}\left[ \mathbf A^n_{\times}\right]_{i,j}$：给出了图$G_1$和$G_2$中，长度为$n$的路径的数量，该路径满足以下条件：路径的顶点`label` 序列完全相同、顶点的边`label` 序列完全相同。

     ```
     xxxxxxxxxx
     ```

     2

     1

     ```
     G1: (label = A1) ---s1--- (label = A2) ---s2--- (label = A3)
     ```

     2

     ```
     G2: (label = A1) ---s1--- (label = A2) ---s2--- (label = A3)
     ```

3. `Shortest-Path Kernel`：随机游走`Kernel` 的基本思想是：统计两个输入图中相同标签之间的最短路径。

   给定输入图$G_1(V_1,E_1),G_2(V_2,E_2)$：

   - 首先通过`Floyd` 成对最短路径生成算法，构建每个图的顶点之间的最短路径，得到新的图$G_1^F(V_1,E_1^F),G_2^F(V_2,E_2^F)$，其中$E_1^F$给出了$V_1$的两两顶点之间最短路径、$E_2^F$给出了$V_2$的两两顶点之间最短路径。

   - 计算：

    $k_{shortest-path}(G_1,G_2) = \sum_{e_1\in E_1^F}\sum_{e_2\in E_2^F} k_{walk}^{(1)}(e_1,e_2)$

     其中$k_{walk}^{(1)}$为一个定义在长度为`1` 的 `edge walk` 上的正定核（？需要参考代码）。

4. `Weisfeiler-Lehman Subtree Kernel`：它基于 `Weisfeiler-Lehman` 算法。

   - 顶点 `label` 更新：对于图$G$的顶点$v$，获取顶点$v$的邻居顶点$\mathcal N_v$，通过一个 `hash` 函数得到顶点$v$的新`label`：

    $l_v\leftarrow \text{hash}(l_v,l_{\mathcal N_v})$

     更新后的新`label` 包含了其直接邻域的顶点信息。因此如果两个顶点更新后的 `label` 相同，我们可以认为其邻域结构是同构的。

   - 更新图的所有顶点 、重复更新最多$K$轮直到收敛，最终得到图$G^\prime$。

     每一轮更新后，顶点$v$的 `label` 就包含了更大规模的邻域的顶点信息，最终每个顶点的 `label` 编码了图的全局结构信息。

   - 对于输入图$G_1,G_2$我们分别对其执行 `Weisfeiler-Lehman` 算法，最终根据$G^\prime_1,G_2^\prime$的 顶点`label` 集合的相似性（如 `Jaccard` 相似性）来得到核函数：

    $k_{WL}(G_1,G_2) = \frac{|l_{V_1}\bigcap l_{V_2}|}{|l_{V_1}\bigcup l_{V_2}|}$

     其中$l_{V}$为顶点的`label` 集合。

5. 一旦定义了 `Graph Kernel`，则我们可以使用基于核计巧的方法，如 `SVM` 来直接应用在图上。

#### 7.1.2 PATCHY-SAN

1. 给定图$G=(V,E)$，假设顶点数量为$|V|=n$，边的数量$|E|=m$。

   - 定义图的邻接矩阵$\mathbf A\in \mathbb R^{n\times n}$为：
   - 每个顶点以及每条边可以包含一组属性，这些属性可以为离散的，也可以为连续的。
   - 定义一个游走序列 `walk` 是由连续的边组成的一个顶点序列；定义一条路径 `path` 是由不重复顶点构成的`walk` 。
   - 定义$d(u,v)$为顶点$u$和$v$之间的距离，它是$u$和$v$之间的最短路径距离。
   - 定义$\mathcal N_1(v)$为顶点$v$的一阶邻居顶点集合，它由与$v$直连的所有顶点构成。

2. `PATCHY-SAN` 利用了`graph labeling` 对顶点进行排序。

   - 如果图的顶点自带`label`， 则我们可以直接用该`label` ；如果顶点没有`label` ，则我们可以通过一个`graph labeling` 函数$F_l: V\rightarrow \mathbb S$为图注入`label` ，其中$\mathbb S$为一个有序集（如，实数集$\mathbb R$或整数集$\mathbb Z$）。

     - `graph labeling` 的例子包括：通过顶点的度`degree` 计算`label`、通过顶点的中介中心性`between centrality` 计算 `label` 。

       一个顶点$v$的中介中心性为：网络中经过顶点$v$的最短路径占所有最短路径的比例。

     - `graph labeling` 引入顶点集合$V$的一个划分：$\{V_1,\cdots,V_K\}$，其中$K$为`label` 的取值类别数。顶点$u,v\in V_i$当且仅当$F_l(u) = F_l(v)$。

   - 一个排序 `ranking` 是一个函数$F_r: V\rightarrow \{1,2,\cdots,|V|\}$。每个`graph labeling` 引入一个排序函数，使得当且仅当$F_l(u)\gt F_l(v)$时有$F_r(u)\lt F_r(v)$，即：`label` 越大则排序越靠前。

   - 给定一个 `graph labeling`，则它决定了图$G$中顶点的顺序，根据该顺序我们定义一个邻接矩阵$\mathbf A^l(G)$，它定义为：

     其中顶点$v$在$\mathbf A^l(G)$中的位置由它的排名$F_r(v)$决定。

3. 如前所述，如果希望将卷积神经网络应用到图结构上，必须解决两个问题：

   - 为图确定一个顶点序列，其中我们为序列中的每个顶点创建邻域子图
   - 邻域图的归一化，即将邻域子图映射到向量空间，使得相似的邻域子图具有相似的`vector` 。

   `PATCHY-SAN` 通过`graph labeling` 过程来解决这些问题。给定一组图，`PATCHY-SAN` 对每个图执行以下操作：

   - 采用 `Node Sequence Selection`算法从图中选择一个固定长度的顶点序列。
   - 采用 `Neighborhood Assembly` 算法为顶点序列中的每个顶点拼装一个固定大小的邻域。
   - 通过 `Graph Normalization` 对每个邻域子图进行归一化处理，从而将无序的图转换为有序的、长度固定的顶点序列。
   - 利用卷积神经网络学习邻域的 `representation`。

##### a. Node Sequence Selection

1. 顶点序列选择是为每个输入的图鉴别需要创建感受野的顶点的过程。

   顶点序列选择过程首先将输入图的顶点根据给定的 `graph labeling` 进行排序；然后，使用给定的步幅$s$遍历所有的顶点，并对每个访问到的顶点采用创建感受野的算法来构造一个感受野，直到恰好创建了$w$个感受野为止。

   - 步幅$s$决定了顶点序列中，需要创建感受野的两个顶点之间的距离。

   -$w$决定了卷积运算输出 `feature map` 的尺寸，它对应于`CNN` 中的图像宽度。

     如果顶点数量小于$w$，则算法创建全零的感受野，用于填充。

   - 也可以采用其它顶点序列选择方法，如根据 `graph labeling` 进行深度优先遍历。

2. `Select Node Sequence` 算法：

   - 算法输入：

     - `graph labeling` 函数$F_l(\cdot)$
     - 输入图$G=(V,E)$
     - 步幅$s$、宽度$w$、感受野尺寸$k$

   - 算法输出：被选择的顶点序列，以及对应的感受野

   - 算法步骤：

     - 根据$F_l(\cdot)$选择顶点集合$V$中的 `top`$w$个顶点，记作$V_{sort}$。

     - 初始化：$i=1,j=1$

     - 迭代，直到$j\ge w$停止迭代。迭代步骤为：

       - 如果$i\le |V_{sort}|$，则对排序后的顶点$i$创建感受野$f=\text{CreateReceptiveField}(V_{sort}[i],k)$；否则创建全零感受野$f=Zero(k)$。

       - 将$f$应用到每个输入通道。

         > 因为顶点的特征可能是一个向量，表示多维度属性。

       - 更新：$i = i+s,\quad j=j+1$。

     - 返回访问到的顶点序列，以及创建的感受野序列。

##### b. Neighborhood Assembly

1. 对于被选择的顶点序列，必须为其中的每个顶点构建一个感受野。创建感受野的算法首先调用邻域拼接算法来构建一个局部邻域，邻域内的顶点是感受野的候选顶点。

   给定顶点$v$和感受野的尺寸$k$，邻域拼接算法首先执行广度优先搜索 `BFS` 来探索与顶点$v$距离依次增加的顶点，并将这些被探索到的顶点添加到集合$\mathcal N_v$。如果收集到的顶点数量小于$k$，则使用最近添加到$\mathcal N_v$顶点的一阶邻居（即 `BFS` 过程），直到$\mathcal N_v$中至少有$k$个顶点；或者没有更多的邻居顶点添加为止。注意，最终$\mathcal N_v$的大小可能超过$k$。

2. `Neighborhood Assembly` 算法：

   - 算法输入：
     - 当前顶点$v$
     - 感受野尺寸$k$
   - 算法输出：顶点$v$的局部邻域$\mathcal N_v$
   - 算法步骤：
     - 初始化：$\mathcal N_v = [v], \mathcal B=[v]$，其中$\mathcal B$存放`BFS` 遍历到的顶点。
     - 迭代，直到$|\mathcal N_v|\ge k$或者$|\mathcal B|=0$停止迭代。迭代步骤：
       - 获取当前`BFS` 遍历顶点的一阶邻居：$\mathcal B = \bigcup_{w\in \mathcal B}\mathcal N_w^{(1)}$，其中$\mathcal N_w^{(1)}$表示顶点$w$的一阶邻居集合。
       - 合并`BFS` 遍历到的顶点：$\mathcal N_v = \mathcal N_v\bigcup \mathcal B$。
     - 返回$\mathcal N_v$。

##### c. Graph Normalization

1. 子图归一化是对邻域子图的顶点施加一个顺序，使得顶点从无序的图空间映射到有序的、长度固定的顶点序列，从而为下一步的感受野创建做准备。

   子图归一化的基本思想是利用 `graph labeling`，对于不同子图中的顶点，当且仅当顶点在各自子图中的结构角色相似时，才给它们分配相似的位置。

   为了形式化该思想，我们定义了一个 `Optimal Graph Normalization` 问题，该问题的目标是找到给定的一组图的最佳 `labeling` 。

2. `Optimal Graph Normalization` 问题：令$\mathcal G$为一组图，每个图包含$k$个顶点。令$F_l(\cdot)$为一个 `graph labeling`过程。令$d_G(\cdot,\cdot)$为$k$顶点图之间的距离函数，令$d_A(\cdot,\cdot)$为$k\times k$矩阵之间的距离函数。则我们的目标是寻找$\hat F_l(\cdot)$，使得：

  $\hat F_l = \arg\min_{F_l}\mathbb E_\mathcal G\left[|d_A(\mathbf A^{l}(G),\mathbf A^l(G^\prime)) - d_G(G,G^\prime)|\right]$

   即：从$\mathcal G$中随机均匀采样的两个子图，子图在排序空间的距离与图空间的距离的差距最小。这使得不同子图中结构相似的顶点映射到各子图中相近的顶点排名，从而实现不同子图的顶点对齐。

   图的最优归一化问题是经典的图规范化问题`graph canonicalization problem`的推广。但是经典的`labeling` 算法仅针对同构图`isomorphic graph` 最佳，对于相似但是不同构的图可能效果不佳。这里相似度由$d_G(G,G^\prime)$来衡量。

   - 图的同构问题是 `NP` 难的。在某些限制下，该问题是 `P` 的，如：图的`degree` 是有界的。图的规范化是对于图$G$，用固定的顶点顺序来代表$G$的整个同构类。

   - 关于图的最优归一化问题，论文给出了两个定理：

     - 定理一：图的最优归一化问题是 `NP` 难的，证明见论文。

       因此`PATCHY-SAN` 无法完全解决图的最优归一化问题，它只是比较了不同的 `graph labeling` 方法，然后选择其中表现最好的那个。

     - 定理二：设$\mathcal G$为一组图，令$(G_1,G_1^\prime),\cdots,(G_N,G_N^\prime)$是从$\mathcal G$中独立随机均匀采样的一对图的序列。令：

       如果$d_A(\mathbf A^l(G),\mathbf A^l(G^\prime)) \ge d_G(G,G^\prime)$，则当且仅当$\theta_{l_1}\lt \theta_{l_2}$时有$\mathbb E_\mathcal G[\hat \theta_{l_1}]\lt \mathbb E_{\mathcal G}[\hat\theta_{l_2}]$。其中$l_1,l_2$表示采用不同的 `graph labeling`。证明见论文。

       该定理使得我们通过比较估计量$\hat\theta_l$来以无监督的方式比较不同的 `graph labeling` 。我们可以简单的选择使得$\hat\theta_l$最小的`graph labeling` 。

       当我们在图上选择编辑距离`edit distance`、在矩阵$\mathbf A^l(G)$上选择汉明距离时，假设条件$d_a\ge d_G$成立。

       另外，上述结论不仅对无向图成立，对于有向图也成立。

3. 图的归一化问题，以及针对该问题的合适的`graph labeling` 方法是`PATCHY-SAN`算法的核心。我们对顶点$v$的邻域子图进行归一化，并约束顶点的 `label`：任意两个其它顶点$u,w$，如果$d(u,v) \lt d(w,v)$则$F_r(u) \lt F_r(w)$，即距离$v$越近，排名越靠前。该约束保证了顶点$v$的排名总是`1`（即排名最靠前）。

   众所周知，对于有界`degree` 的图的同构问题可以在多项式时间求解，由于邻域子图的规模为$k$是一个常数，因此图的归一化算法可以在多项式时间内求解。我们的实验证明：图的邻域计算`graph labeling` 的过程仅产生一个微不足道的开销。

4. `Graph Normalization` 算法：

   - 算法输入：

     - 从原始图$G$得到的顶点子集$\mathbb U$，即邻域子集

     - 顶点$v$

     - `graph labeling` 过程$F_l(\cdot)$

     - 感受野尺寸$k$

     - 输出：归一化的子图

     - 算法步骤：

       - 对$\mathbb U$中的每个顶点，使用$F_l(\cdot)$计算这些顶点对$v$的排名，使得：

        $\forall u,w\in \mathbb U: d(u,v)\lt d(w,v) \rightarrow F_r(u)\lt F_r(w)$

       - 如果$|\mathbb U|\gt k$，则根据 `ranking` 取$\mathbb U$中的 `top k` 个顶点，对所选择的顶点再执行一次`labeling` 以及 `ranking` 的过程。

         > 这里必须使用$F_r(\cdot)$在筛选出的较小的顶点集合上重新计算，因为新的结构导致了新的 `labeling` 分布。

       - 如果$|\mathbb U|\lt k$，则：添加$k-|\mathbb U|$个断开的虚拟顶点。

       - 根据这$k$个顶点的排名来归一化这些顶点，并返回归一化的结果。

5. 下图为对红色根节点$v$的邻域进行归一化过程，颜色表示到根节点的距离，感受野尺寸为$k=9$。首先利用`graph labeling` 对顶点进行排序，然后创建归一化的邻域。

   - 归一化还包括裁剪多余的顶点和填充虚拟顶点。
   - 顶点的不同属性对应于不同的输入通道。
   - 不仅可以针对顶点创建感受野，还可以针对边创建感受野，边的感受野尺寸为$k\times k$，边的不同属性对应不同的输入通道。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/8Oifc0VP2xMl.png?imageslim">
   </p>
   

6. 创建感受野的 `Create Receptive Field`算法：

   - 算法输入：
     - 顶点$v$
     -$F_l(\cdot)$
     - 感受野大小$k$
   - 算法输出：顶点$v$的感受野
   - 算法步骤：
     - 计算顶点$v$的邻域：$\mathcal N_v=\text{NeighborhoodAssembly}(v,k)$
     - 归一化邻域子图：$G_{norm} = \text{GraphNormalization}(\mathcal N_v,v,F_l,k)$
     - 返回$G_{norm}$

##### d. CNN 等价性

1. 定理：在图像中得到的一个像素序列上应用 `PATCHY-SAN` ，其中感受野尺寸为$(2m-1)^2$、步幅为$s$、非零填充以及采用 `1-WL` 归一化，则这等效于 `CNN` 的一个感受野大小为$2m-1$、步幅为$s$、非零填充的卷积层。

   证明：如果输入图为一个正方形网格，则为顶点构造的 `1-WL` 归一化的感受野始终是具有唯一顶点顺序的正方形网格。

##### e. PATCHY-SAN 架构

1. `PATCHY-SAN` 既能处理顶点，也能处理边；它既能处理离散属性，也能处理连续属性。
2. `PATCHY-SAN` 对每个输入图$G$产生$w$个尺寸为$k$的归一化的感受野。假设$a_v$为顶点属性的数量、$a_e$为边属性的数量，则这将产生一个$w\times k\times a_v$的张量（顶点的感受野）以及一个$w\times k\times k \times a_e$的张量（边的感受野）。这里$a_v$和$a_e$都是输入通道的数量。
   - 我们可以将这两个张量`reshape` 为一个$wk\times a_v$的张量和一个$wk^2\times a_e$的张量，然后对每个输入通道采用一个一维卷积层进行卷积。顶点产生的张量采用步幅$k$尺寸为$k$的卷积核，边产生的张量采用步幅$k^2$尺寸为$k^2$的卷积核 。
   - 剩下的结构可以任意结合 `CNN` 的组件。另外我们可以利用融合层来融合来自顶点的卷积输出`feature map` 核来自边的卷积输出 `feature map` 。

##### f. 算法复杂度

1. `PATCHY-SAN` 的创建感受野算法非常高效。另外，由于这些感受野的生成是相互独立的，因此感受野生成过程原生支持并行化。

2. 定理：令$N$为数据集中图的数量，$k$为感受野尺寸，$w$为宽度（每个图包含的感受野数量）。假设$O(f_l(n,m))$为包含$n$个顶点、$m$条边的图的 `graph labeling` 过程的计算复杂度。则`PATCHY-SAN` 最坏情况下的计算复杂度为$O(N\times w\times [f_l(n,m)+n\log n +\exp(k)])$。

   证明见论文。

   当采用`Weisfeiler-Lehman` 算法作为`graph labeling` 算法时，它的算法复杂度为$O((n+m)\log n)$。考虑到$w\ll n,k \ll n$，则`PATCHY-SAN`的复杂度为$N$的线性、$m$的准线性、$n$的准线性。

### 7.2 实验

#### 7.2.1 运行时分析

1. 论文通过将`PATCHY-SAN` 应用于实际的`graph` 来评估其计算效率，评估指标为感受野的生成速度。另外论文还给出了目前 `CNN` 模型

2. 数据集：所有输入图都来自 `Python` 模块 `GRAPHTOOL` 。

   - `torus` 图：具有`10k` 个顶点的周期性晶格。
   - `random` 图：具有`10` 个顶点的随机无向图，顶点的度的分布满足：$p(k)\propto 1/k$，以及$k_{\max}=3$。
   - `power` 图：美国电网拓扑网络。
   - `polbooks`：`2004`年美国总统大选期间出版的有关美国政治书籍的 `co-purchasing` 网络。
   - `preferential`：一个 `preferential attachment network`，其中最新添加的顶点的`degree` 为 `3` 。
   - `astro-ph`：天体物理学 `arxiv` 上作者之间的 `co-authorship` 网络。
   - `email-enron`：一个由大约 `50万`封已发送 `email` 生成的通信网络。

3. 我们的`PATCHY-SAN` 采用一维 `Weisfeiler-Lehman:1-WL` 算法来归一化邻域子图。下图给出了每个输入图每秒产生感受野的速度。所有实验都是在单台 `2.8 GHZ GPU`、`64G` 内存的机器上执行。

   - 对于感受野尺寸$k=5/k=10$，除了在 `email-eron` 上的速度为 `600/s` 和 `320/s` 之外，在其它所有图上`PATCHY-SAN` 创建感受野的速度超过 `1000/s` 。
   - 对于最大的感受野尺寸$k=50$，`PATCHY-SAN` 创建感受野的速度至少为 `100/s` 。

   对于一个经典的带两层卷积层、两层 `dense` 层的 `CNN` 网络，我们在相同机器上训练速度大概是 `200-400` 个样本/秒，因此`PATCHY-SAN` 感受野的生成速度使得下游 `CNN` 组件饱和。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/zfPNVbNPlWtI.png?imageslim">
   </p>
   

#### 7.2.2 可视化

1. 我们将 `PATCHY-SAN` 学到的尺寸为`9` 的归一化感受野使用 `restricted boltzman machine:RBM` 进行无监督学习，`RNM` 所学到的特征对应于重复出现的感受野模式。其中：

   - `PATCHY-SAN` 采用 `1-WL` 算法进行邻域子图归一化。
   - 采用单层`RBM` ，隐层包含 `100` 个隐单元。
   - `RBM` 采用对比散度算法`contrastive divergence:CD` 训练 `30` 个 `epoch`，学习率设为 `0.01`。

   下图给出了从四张图中得到的样本和特征。我们将`RBM` 学到的特征权重可视化（像素颜色越深，则对应权重重大）。另外我们还采样了每种模式对应的三个顶点的归一化邻域子图，黄色顶点表示当且顶点（排序为`1`）。

   左上角为 `torus` 周期性晶格图，左下角为 `preferential attachment` 图、右上角为 `co-purchasing` 图、右下角为 随机图。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/sXPilmvQM5Ic.png?imageslim">
   </p>
   

#### 7.2.3 图分类

1. 图分类任务是将每个图划分到若干类别之一。我们采用`6` 个标准 `benchmark` 数据集来比较不同图分类模型的分类准确性和运行时间。

   - `MUTAG` 数据集：由`188` 种硝基化合物组成的数据集，其类别表明该化合物是否对细菌具有诱变 `mutagenic` 作用。
   - `PTC` 数据集：由 `344` 种化合物组成的数据集，其类别表明是否对老鼠具有致癌性。
   - `NCI1` 和 `NCI109` 数据集：筛选出的抑制 `non-small` 肺癌细胞和卵巢癌细胞活性的化合物。
   - `PROTEIN`：一个图的数据集，其中图的顶点表示二级结构元素 `secondary structure element`， 边表示氨基酸序列中的相邻关系，或者三维空间中的氨基酸相邻关系。其类别表示酶或者非酶。
   - `D&D`：由 `1178` 种蛋白质组成的数据集，其类别表明是酶还是非酶。

2. 我们将`PATCHY-SAN` 和一组核方法比较，包括`shortest-path kernel:SP` 、`random walk kernel:RW`、`graphlet count kernel:GK`，以及 `Weisfeiler-Lehman sbutree kernel:WL` 。

   - 对于核方法，我们使用 `LIB-SVM` 模型来训练和评估核方法的效果。我们使用`10` 折交叉验证，其中`9-fold` 用于训练，`1-fold` 用于测试。我们重复`10` 次并报告平均准确率和标准差。

     类似之前的工作，我们设置核方法的超参数为：

     - `WL` 的高度参数设置为`2` 。
     - `GK` 的尺寸参数设置为 `7` 。
     - `RW` 的衰减因子从$\{10^{-6},10^{-5},\cdots,10^{-1}\}$中进行挑选。

   - 对于 `PATCHY-SAN:PSCN` 方法，我们设置$w$为平均顶点数量、感受野尺寸分别为$k=5$和$k=10$。

     - 实验中我们仅使用顶点的属性，但是在$k=10$时我们实验了融合顶点的感受野和边的感受野，记作$k=10^E$。

     - 所有 `PSCN` 都使用了具有两个卷积层、一个`dense` 层、一个 `softmax` 层的网络结构。其中第一个卷积层有 `16`个输出通道；第二个卷积层有 `8` 个输出通道，步长为`1`，核大小为 10；`dense` 层有 `128` 个隐单元，采用`dropout = 0.5` 的 `dropout`。我们采用一个较小的隐单元数量以及 `dropout` 从而避免模型在小数据集上过拟合。

       所有卷积层和 `dense` 层的激活函数都是 `reLU` 。 模型的优化算法为 `RMSPROP` 优化算法，并基于`Keras` 封装的 `Theno` 实现。

     - 所有 `PSCN` 需要优化的超参数为 `epoch` 数量以及 `batch-size` 。

     - 当$k=10$时，我们也对 `PATCHY-SAN` 抽取的感受野应用一个逻辑回归分类器 `PSLR` 。

   这些模型在 `benchmark` 数据集上的结果如下表所示。其中前三行给出了各数据集的属性，包括图的最大顶点数`Max`、图的平均顶点数`Avg`、图的数量`Graphs` 。我们忽略了 `NCI109` 的结果，因为它几乎和 `NCI1` 相同。

   - 尽管使用了非常普通的`CNN` 架构，`PSCN` 的准确率相比现有的`graph kernel` 方法具有很强的竞争力。在大多数情况下，采用$k=10$的 `PSCN` 具有最佳的分类准确性。
   - `PSCN` 这里的预测方差较大，这是因为：`benchmark` 数据集较小，另外 `CNN` 的一些超参数（`epoch` 和 `batch-size` 除外）没有针对具体的数据集进行优化。
   - `PATCHY-SAN` 的运行效率是`graph kernel` 中最高效的 `WL` 方法的2到8倍。
   - `PATCHY-SAN` + 逻辑回归的效果较差，这表明 `PATCHY-SAN` 更适合搭配 `CNN` 。`CNN` 学到了归一化感受野的非线性特征组合，并在不同感受野之间共享权重。
   - 采用中介中心性归一化 `betweeness centrality normalization` 结果也类似（未在表中体现），除了它的运行时间大约增加了 `10%` 。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/fss7V7589xQJ.png?imageslim">
   </p>
   

3. 我们在较大的社交网络图数据集上使用相同的配置进行实验，其中每个数据集最多包含 `12k` 张图，每张图平均 `400` 个顶点。我们将 `PATCHY-SAN` 和之前报告的`graphlet count:GL`、`deep graplet count kernel:DGK` 结果相比。

   我们使用归一化的顶点`degree` 作为顶点的属性，这突出了`PATCHY-SAN` 的优势之一：很容易的包含连续特征。

   可以看到 `PSCN` 在六个数据集的四个中明显优于其它两个核方法，并且在剩下两个数据集也取得了很好的性能。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/YTEmVLnkhEBD.png?imageslim">
   </p>
   

## 八、GraphSage

1. 在大型`Graph` 中，顶点的低维`embedding`在从内容推荐到蛋白质功能识别等各项任务中都非常有效。

   之前的工作都集中在从单个指定的图来学习顶点 `embedding`，这些方法都是 `transductive` 的。但是，很多实际应用需要为从未见过的顶点或者全新的图来快速生成 `embedding` ，即需要 `inductive`能力。这种 `inductive` 能力对于生产中的机器学习系统至关重要，这些机器学习系统需要在不断变化的图上运行，并不断遇到从未见过的顶点（如 `Youtube` 上的新用户、新视频）。 另外这种 `inductive` 能力还可以促进图的泛化，例如我们在已知的分子结构图上训练模型，然后该模型可以为新的分子图产生顶点 `embedding` 。

   - 与 `transductiv` 相比，`inductive` 的顶点 `embedding` 更为困难，因为这需要泛化到从未就按过的顶点。而这需要将新观察到的子图 “对齐” 到模型已经训练过的旧的子图。`inductive` 框架必须学会识别顶点邻域的结构属性，从而识别每个顶点（包括新发现的顶点）在图的局部角色以及全局位置。

   - 大多数现有的顶点`embedding` 方法本质都是 `transductive` 的。这些方法都使用基于矩阵分解来直接优化每个顶点的 `embedding` 。因为它们是在单个固定的图上对顶点进行训练和预测，因此天然地无法推广到未见过的顶点。

     也可以修改这些方法来满足`inductinve` 的要求， 如针对新的未见过的顶点进行若干轮额外的梯度下降。但是这种方式的计算代价较高，也容易导致顶点 `embedding`在重新训练期间发生漂移。

   最近也有通过图卷积（如`Semi-Supervised GCN` ）来学习图结构的方法，但是这些方法也是以 `transductive`的方式在使用。论文`《Inductive Representation Learning on Large Graphs》` 提出了一个通用的、称作 `Graph Sample and Aggregage:GraphSAGE` 的学习框架，该框架将图卷积推广到 `inductinve` 无监督学习。

2. `GraphSage` 是一种`inductive` 的顶点 `embedding` 方法。与基于矩阵分解的`embedding` 方法不同，`GraphSage` 利用顶点特征（如文本属性、顶点画像信息、顶点的`degree` 等）来学习，并泛化到从未见过的顶点。

   - 通过将顶点特征融合到学习算法中，`GraphSage` 可以同时学习每个顶点的邻域拓扑结构，以及顶点特征在邻域中的分布。`GraphSage` 不仅可以应用于顶点特征丰富的图（如引文网络、生物分子网络），还可以应用到没有任何顶点特征的简单的图，此时可以采用顶点的结构属性来作为顶点特征（如顶点`degree` ）。
   - `GraphSage` 并没有为每个顶点训练一个 `embedding`，而是训练了一组聚合函数，这些聚合函数用于从顶点的局部邻域中聚合信息特征。在测试期间，我们使用训练好的模型的聚合函数来为从未见过的顶点生成 `embedding` 。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/viJVcz5QnkJf.png?imageslim">
   </p>
   

3. 和之前的 `embedding` 方法类似，`GraphSage` 设计了一个无监督损失函数。该损失函数允许对`GraphSage` 进行无监督训练，而无需任何特定于具体任务监督信息。论文还展示了以监督方式来训练 `GraphSage` 。

   论文在三个顶点分类 `benchmark` 上评估了`GraphSave` 方法，从而验证了`GraphSage` 在从未见过的数据上具有优秀的顶点 `embedding` 能力。

   最后论文研究了 `GraphSave` 的表达能力，并通过理论分析证明了：虽然`GraphSage` 是基于顶点特征的，但是它能够学得顶点在图中角色的结构信息。

4. `GraphSave` 方法和 `Semi-Supervised GCN` 密切相关。原始的 `Semi-Supervised GCN` 以 `transductive` 的方式进行半监督学习，这要求在训练过程中已知完整的图拉普拉斯算子。`GraphSage`的一个简单变种可以视为 `Semi-Supervised GCN` 框架以 `inductive` 方式的扩展。

### 8.1 模型

1. `GraphSage` 的核心思想是：学习如何从顶点的局部邻域中汇聚特征信息，如邻域内顶点的文本特征或者顶点`degree` 特征。

#### 8.1.1 前向传播

1. 假设我们已经学到了$K$个聚合函数$\text{AGG}_k,k\in \{1,2,\cdots,K\}$，这些聚合函数用于聚合顶点的邻域信息。假设我们也学到了$K$个权重矩阵$\mathbf W^{(k)},k\in \{1,2,\cdots,K\}$，它们用于在不同层之间传递信息。$K$也称作搜索深度。

   `GraphSage` 的`embedding` 生成算法为：

   - 输入：

     - 图$G(V,E)$
     - 输入特征$\{\mathbf{\vec x}_v\mid v\in V\}$
     - 搜索深度$K$
     - 权重矩阵$\mathbf W^{(k)},k\in \{1,\cdots,K\}$
     - 非线性激活函数$\sigma(\cdot)$
     -$K$个聚合函数$\text{AGG}_k,k\in \{1,\cdots,K\}$
     - 邻域函数$\mathcal N(\cdot)$

   - 输出：顶点的`embedding` 向量$\{\mathbf{\vec z}_v\mid v \in V \}$

   - 算法步骤：

     - 初始化：$\mathbf{\vec h}_v^{(0)} = \mathbf{\vec x}_v, v\in V$

     - 对每一层迭代，迭代条件为：$k=1,2,\cdots,K$。迭代步骤：

       - 遍历每个顶点$v\in V$，执行：

       - 对每个顶点$v$的隐向量归一化：

        $\mathbf{\vec h}_v^{(k)} = \frac{\mathbf{\vec h}_v^{(k)}}{\left\|\mathbf{\vec h}_v^{(k)}\right\|_2},\quad v\in V$

     -$\mathbf{\vec z}_v= \mathbf{\vec h}_v^{(T)}$。

2. `GraphSage` 前向传播算法的基本思想是：在网络的每一层，顶点都会聚合来自其局部邻域的信息；并且随着层的逐渐加深，顶点将从图的更远范围逐渐获取越来越多的信息。

   在第$k$层，每个顶点$v$首先聚合其邻域顶点的信息$\{\mathbf{\vec h}_u^{(k-1)}\mid u\in \mathcal N(v)\}$到一个向量$\mathbf{\vec h}_{\mathcal N(v)}^{(k)}$中，这个聚合过程依赖于第$k-1$层的顶点隐向量。然后每个顶点$v$拼接它的第$k-1$层 `representation`$\mathbf{\vec h}_v^{(k-1)}$和邻域信息$\mathbf{\vec h}_{\mathcal N(v)}^{(k)}$，然后通过一个全连接层，并使用一个非线性激活函数$\sigma(\cdot)$。

   最终在第$K$层， 有$\mathbf{\vec z}_v = \mathbf{\vec h}_v^{(K)}$。

3. 大多数顶点 `embedding` 方法将学到的 `embedding` 归一化为单位向量，这里我们也做类似处理。

#### 8.1.2 邻域

1. 在`GraphSage` 中我们并没有使用完整的邻域，而是均匀采样一组固定大小的邻域，从而确保每个 `batch` 的计算代价是固定的。因此我们定义$\mathcal N(v)$为：从集合$\{u\mid u\in V,(u,v)\in E\}$中均匀采样的、固定大小的集合。
   - 对于每个顶点$v$, `GraphSage` 在每一层采样不同的邻域，甚至不同层的邻域大小都不同。
   - 如果对每个顶点使用完整的邻域，则每个 `batch` 的内存需求和运行时间是不确定的，最坏情况为$O(|V|)$。如果使用采样后的邻域，则每个 `batch` 的时间和空间复杂度固定为$O(\prod_{k=1}^KS_k)$，其中$S_k$表示第$k$层邻域大小。$K$以及$S_k$均为用户指定的超参数，实验发现$K=2, S_1\times S_2\le 500$时的效果较好。

#### 8.1.3 聚合函数

1. 和网格型数据（如文本、图像）不同，图的顶点之间没有任何顺序关系，因此算法中的聚合函数必须能够在无序的顶点集合上运行。理想的聚合函数是对称的，同时保持较高的表达能力。对称性是指：对于给定的一组顶点集合，无论它们以何种顺序输入到聚合函数，聚合后的输出结果不变。这种对称性可以确保我们的神经网络模型可以用于任意顺序的顶点邻域集合的训练和测试。

   聚合函数有多种形式，论文给出了三种主要的聚合函数：均值聚合函数`mean aggregator`、`LSTM`聚合函数`LSTM aggregator` 、池化聚合函数 `pooling aggregator`。

2. `mean aggregator`：简单的使用邻域顶点的特征向量的逐元素均值来作为聚合结果。这几乎等价于 `Semi-Supervised GCN` 框架中的卷积传播规则。

   如果我们将前向传播：

   替换为：

  $\mathbf{\vec h}_v^{(k)} = \sigma\left(\mathbf W^{(k)}\text{MEAN}\left(\{\mathbf{\vec h}_v^{(k-1)}\}\bigcup \{\mathbf{\vec h}_u^{(k-1)}\mid u\in \mathcal N(v)\}\right)\right)$

   则这得到 `Semi-supervised GCN` 的一个 `inductive` 变种，我们称之为`mean-based aggregator convolutional` 基于均值聚合的卷积。它是局部频域卷积的一个粗糙的线性近似。

3. `LSTM aggregator` ：和均值聚合相比，`LSTM` 具有更强大的表达能力。但是 `LSTM`原生的是非对称的，它依赖于顶点的输入顺序。因此论文通过将 `LSTM` 应用于邻域顶点的随机排序，从而使得 `LSTM` 可以应用于无序的顶点集合。

4. `pooling aggregator` ：邻域每个顶点的特征向量都通过全连接神经网络独立馈入，然后通过一个逐元素的最大池化来聚合邻域信息：

  $\mathbf{\vec h}_{\mathcal N(v)}^{(k)} = \max\left(\left\{\sigma\left(\mathbf W_{pool}\mathbf{\vec h}^{(k-1)}_{u} + \mathbf{\vec b}_{pool}\right)\mid u\in \mathcal N(v)\right\}\right)$

   其中 `max` 表示逐元素的 `max` 运算符。

   - 理论上可以在最大池化之前使用任意深度的多层感知机，但是论文这里专注于简单的单层网络结构。

     直观上看，可以将多层感知机视为一组函数，这组函数为邻域集合内的每个顶点`representation` 计算特征。通过将最大池化应用到这些计算到的特征上，模型可以有效捕获邻域集合的不同特点`aspect`。

   - 理论上可以使用任何的对称向量函数（如均值池化）来替代 `max` 运算符。但是论文在测试中发现最大池化和均值池化之间没有显著差异，因此论文专注于最大池化。

#### 8.1.4 模型学习

1. 和 `DeepWalk` 等方法相同，为了在无监督条件下学习有效的顶点`representation`，论文定义了一个损失函数：

  $\mathcal J_G(\mathbf{\vec z}_u) = -\log\left(\text{sigmoid}\left(\mathbf{\vec z}_u\cdot \mathbf{\vec z}_v\right)\right) - Q\times \mathbb E_{v_b\sim P_n(v)}\log(\text{sigmoid}(-\mathbf{\vec z}_u\cdot \mathbf{\vec z}_{v_n}))$

   其中$\mathbf{\vec z}_u$为顶点$u\in V$的 `representation`，$v$是和顶点$u$在一个长度为$l$的 `random walk` 上共现的顶点， 为`sigmoid` 函数，$P_n(\cdot)$为负采样用到的分布函数，$Q$为负采样的样本数。

   这个损失函数鼓励距离较近的顶点具有相似的 `representation`、距离较远的顶点具有不相似的 `representation` 。

   - 与之前的 `embedding` 方法不同，`GraphSage` 中的顶点 `representation`$\mathbf{\vec z}_u$不仅包含了顶点的结构信息（通过 `embedding look-up` 得到），还包含了顶点局部邻域的特征信息。
   - 模型参数通过损失函数的随机梯度下降算法来求解。

2. 以无监督方式学到的顶点 `embedding` 可以作为通用 `service` 来服务于下游的机器学习任务。但是如果仅在特定的任务上应用，则可以进行监督学习。此时可以通过具体任务的目标函数（如交叉熵损失）来简单的替换无监督损失，或者将监督损失加上无监督损失来融合二者。

#### 8.1.5 GraphSage VS Weisfeiler-Lehman

1. `GraphSage` 算法在概念上受到图的同构性检验的经典算法的启发。在前向传播过程中，如果令$K=|V|$、$\mathbf W^{(k)} = \mathbf I$，并选择合适的`hash` 函数来作为聚合函数，同时移除非线性函数，则该过程是 `Weisfeiler-Lehman:WL` 同构性检验算法的一个特例，被称作 `naive vertex refinement` 。

   如果算法输出的顶点 `representation`$\{\mathbf{\vec z}_v,v\in V\}$在两个子图是相等的，则 `WL` 检验算法认为这两个子图是同构的。虽然在某些情况下该检验会失败，但是大多数情况下该检验是有效的。

2. `GraphSage` 是 `WL test` 算法的一个`continous` 近似，其中`GraphSage` 使用可训练的神经网络聚合函数代替了不连续的哈希函数。

   虽然 `GraphSage` 的目标是生成顶点的有效`embedding` 而不是检验图的同构性，但是`GraphSage` 和 `WL test` 之间的联系为我们学习顶点邻域拓扑结构算法的设计提供了理论背景。

3. 可以证明：即使我们提供的是顶点特征信息，`GraphSage` 也能够学到图的结构信息。证明见原始论文。

#### 8.1.6 mini-batch 训练

1. 为了使用随机梯度下降算法，我们需要对`GraphSage` 的前向传播算法进行修改，从而允许`mini-batch` 中每个顶点能够执行前向传播、反向传播。即：确保前向传播、反向传播过程中用到的顶点都在同一个 `mini-batch` 中。

2. `GraphSage mini-batch` 前向传播算法：

   - 算法输入：

     - 图$G(V,E)$
     - 输入特征$\{\mathbf{\vec x}_v\mid v\in \mathcal B\}$
     - 搜索深度$K$
     - 权重矩阵$\mathbf W^{(k)},k\in \{1,\cdots,K\}$
     - 非线性激活函数$\sigma(\cdot)$
     -$K$个聚合函数$\text{AGG}_k,k\in \{1,\cdots,K\}$
     - 邻域函数$\mathcal N(\cdot)$

   - 输出：顶点的`embedding` 向量$\{\mathbf{\vec z}_v\mid v \in \mathcal B\}$

   - 算法步骤：

     - 初始化：$\mathcal B^{(K)} = \mathcal B$

     - 迭代$k=K,\cdots,1$，迭代步骤为：

       -$\mathcal B^{(k-1)} = \mathcal B^{(k)}$
       - 遍历$u\in \mathcal B^{(k)}$，计算$\mathcal B^{(k-1)} = \mathcal B^{(k-1)}\bigcup \mathcal N_k(u)$

     - 初始化：$\{\mathbf{\vec h}_v^{(0)} = \mathbf{\vec x}_v， v\in \mathcal B^{(0)}\}$

     - 对每一层迭代，迭代条件为：$k=1,2,\cdots,K$。迭代步骤：

       - 遍历每个顶点$v\in \mathcal B^{(k)}$，执行：

       - 对每个顶点$v$的隐向量归一化：

        $\mathbf{\vec h}_v^{(k)} = \frac{\mathbf{\vec h}_v^{(k)}}{\left\|\mathbf{\vec h}_v^{(k)}\right\|_2},\quad v\in V$

     -$\mathbf{\vec z}_v = \mathbf{\vec h}_v^{(K)},v\in \mathcal B$。

3. 在 `mini-batch` 前向传播算法中：

   - 首先计算`mini-batch` 需要用到哪些顶点。集合$\mathcal B^{(k-1)}$包含了第$k$层计算 `representation` 的顶点所依赖的顶点集合。

     由于$\mathcal B^{(k)} \sube \mathcal B^{(k-1)}$，所以在计算$\mathbf{\vec h}_v^{(k)}$时依赖的$\mathbf{\vec h}_v^{(k-1)}$已经在第$k-1$层被计算。另外第$k$层需要计算 `representation`的顶点更少，这避免计算不必要的顶点。

   - 然后计算目标顶点的 `representation`，这一步和 `batch` 前向传播算法相同。

4. 我们使用$\mathcal N_k(\cdot)$的$k$来表明：不同层之间使用独立的 `random walk` 采样。这里我们使用均匀采样，并且当顶点邻域顶点数量少于指定数量时采用有放回的采样，否则使用无放回的采样。

5. 和 `batch` 算法相比，`mini-batch` 算法的采样过程在概念上是相反的：

   - 在 `batch` 算法中，我们在$k=1$时对顶点邻域内的$S_1$个顶点进行采样，在$k=2$时对顶点邻域内$S_2$个顶点进行采样。
   - 在 `mibi-batch` 算法中，我们在$k= 2$时对顶点领域内的$S_2$个顶点进行采样，然后在$k=1$时对顶点领域内$S_1\times S_2$个顶点进行采样。这样才能保证我们的目标$\mathcal B$中包含 `mibi-batch` 所需要计算的所有顶点。

### 8.2 实验

1. 我们在三个 `benchmark` 任务上检验 `GraphSage` 的效果：`Web of Science Citation` 数据集的论文分类任务、`Reddit` 数据集的帖子分类任务、`PPI` 数据集的蛋白质分类任务。

   前两个数据集是对训练期间未见过的顶点进行预测，最后一个数据集是对训练期间未见过的图进行预测。

2. 数据集：

   - `Web of Science Cor Collection` 数据集：包含 `2000` 年到 `2005` 年六个生物学相关领域的所有论文，每篇论文属于六种主题类别之一。数据集包含 `302424` 个顶点，顶点的平均`degree` 为 `9.15`。其中：`Immunology` 免疫学的标签为`NI`，顶点数量 `77356` ；`Ecology` 生态学的标签为 `GU`，顶点数量 `37935` ；`Biophysics` 生物物理学的标签为`DA`，顶点数量 `36688`；`Endocrinology and Metabolism` 内分泌与代谢的标签为 `IA` ，顶点数量 `52225` ；`Cell Biology` 细胞生物学的标签为 `DR`，顶点数量`84231`；`Biology(other)`生物学其它的标签为 `CU`，顶点数量 `13988` 。

     任务目标是预测论文主题的类别。我们根据 `2000-2004` 年的数据来训练所有算法，并用 `2005` 年的数据进行进行测试（其中 `30%` 用于验证）。

     我们使用顶点`degree` 和文章的摘要作为顶点的特征，其中顶点摘要根据`Arora` 等人的方法使用 `sentence embedding` 方法来处理文章的摘要，并使用`Gensim word2vec` 的实现来训练了`300` 维的词向量。

   - `Reddit` 数据集：包含`2014` 年 `9` 月`Reddit` 上发布帖子的一个大型图数据集，顶点为帖子所属的社区。我们对 `50` 个大型社区进行采样，并构建一个帖子到帖子的图。如果一个用户同时在两个帖子上发表评论，则这两个帖子将链接起来。数据集包含 `232965` 个顶点，顶点的平均`degree` 为 `492` 。

     为了对社区进行采样，我们按照每个社区在 `2014` 年的评论总数对社区进行排名，并选择排名在 `[11,50]`（包含）的社区。我们忽略了最大的那些社区，因为它们是大型的、通用的默认社区，会严重扭曲类别的分布。我们选择这些社区上定义的最大连通图`largest connected component` 。

     任务的目标是预测帖子的社区`community`。我们将该月前`20` 天用于训练，剩下的天数作为测试（其中 `30%` 用于验证）。

     我们使用帖子的以下特征：标题的平均`embedding`、所有评论的平均 `embedding`、帖子评分、帖子评论数。其中`embedding` 直接使用现有的 `300` 维的 `GloVe CommonCral`词向量，而不是在所有帖子中重新训练。

   - `PPI` 数据集：包含`Molecular Signatures Dataset` 中的图，每个图对应于不同的人类组织，顶点标签采用`gene ontology sets` ，一共`121` 种标签。平均每个图包含 `2373` 个顶点，所有顶点的平均 `degree` 为 `28.8` 。

     任务的目的是评估模型的跨图泛化的能力。我们在 `20` 个随机选择的图上进行训练、`2` 个图进行验证、 `2` 个图进行测试。其中训练集中每个图至少有 `15000` 条边，验证集和测试集中每个图都至少包含 `35000`条边。注意：对于所有的实验，验证集和测试集是固定选择的，训练集是随机选择的。我们最后给出测试图上的 `micro-F1` 指标。

     我们使用`positional gene sets`、`motif gene sets` 以及 `immunological signatures` 作为顶点特征。我们选择至少在 `10%` 的蛋白质上出现过的特征，低于该比例的特征不被采纳。最终顶点特征非常稀疏，有 `42%` 的顶点没有非零特征，这使得顶点之间的链接非常重要。

3. `Baseline` 模型：随机分类器、基于顶点特征的逻辑回归分类器（完全忽略图的结构信息）、代表因子分解方法的 `DeepWalk` 算法+逻辑回归分类器（完全忽略顶点的特征）、拼接了 `DeepWalk` 的 `embedding` 以及顶点特征的方法（融合图的顶点特征和结构特征）。

   我们使用了不同聚合函数的 `GraphSage`的四个变体。由于卷积的变体是 `Semi-Supervised GCN` 的 `inductive` 扩展，因此我们称其为 `GraphSage-GCN` 。

   我们使用了 `GraphSage` 的无监督版本，也直接使用分类交叉熵作为损失的有监督版本。

4. 模型配置：

   - 所有`GraphSage` 模型都在 `Tensorflow` 中使用 `Adam` 优化器实现， `DeepWalk` 在普通的随机梯度优化器中表现更好。

   - 为公平比较，所有模型都采样相同的 `mini-batch` 迭代器、损失函数（当然监督损失和无监督损失不同）、邻域采样器。

   - 为防止 `GraphSage` 聚合函数的效果比较时出现意外的超参数`hacking`，我们对所有 `GraphSage`版本进行了相同的超参数配置：根据验证集的性能为每个版本提供最佳配置。

   - 对于所有的 `GraphSage` 版本设置 `K=2` 以及邻域采样大小$S_1=25, S_2= 10$。

   - 对于所有的 `GraphSage` ，我们对每个顶点执行以该顶点开始的 `50` 轮长度为 `5` 的随机游走序列，从而得到`pair` 顶点对。我们的随机游走序列生成完全基于 `Python` 代码实现。

   - 对于原生特征模型，以及基于无监督模型的 `embedding` 进行预测时，我们使用 `scikit-learn` 中的 `SGDClassifier` 逻辑回归分类器，并使用默认配置。

   - 在所有配置中，我们都对学习率和模型的维度以及`batch-size` 等等进行超参数选择：

     - 除了 `DeepWalk` 之外，我们为监督学习模型设置初始学习率的搜索空间为$\{0.01,0.001,0.0001\}$，为无监督学习模型设置初始学习率的搜索空间为$\{2\times 10^{-6},2\times 10^{-7},2\times 10^{-8}\}$。

       最初实验表明 `DeepWalk` 在更大的学习率下表现更好，因此我们选择它的初始学习率搜索空间为$\{0.2,0.4,0.8\}$。

     - 我们测试了每个`GraphSage`模型的`big` 版本和 `small` 版本。 对于池化聚合函数，`big` 模型的池化层维度为 `1024`，`small` 模型的池化层维度为 `512` ；对于 `LSTM` 聚合函数，`big` 模型的隐层维度为 `256`，`small` 模型的隐层维度为 `128` 。

     - 所有实验中，我们将`GraphSage` 每一层的$\mathbf{\vec h}_i^{(k)}$的维度设置为 `256`。

     - 所有的 `GraphSage` 以及 `DeepWalk` 的非线性激活函数为 `ReLU` 。

     - 对于无监督 `GraphSage` 和 `DeepWalk` 模型，我们使用 `20` 个负采样的样本，并且使用 `0.75` 的平滑参数对顶点的`degree` 进行上下文分布平滑。

     - 对于监督 `GraphSage`，我们为每个模型运行 `10` 个 `epoch`。

     - 我们对 `GraphSage` 选择 `batch-size = 512`。对于 `DeepWalk` 我们使用 `batch-size=64`，因为我们发现这个较小的 `batch-size` 收敛速度更快。

5. 硬件配置：`DeepWalk` 在`CPU` 密集型机器上速度更快，它的硬件参数为 `144 core`的 `Intel Xeon CPU(E7-8890 V3 @ 2.50 GHz)` ，`2T` 内存。其它模型在单台机器上实验，该机器具有 `4` 个 `NVIDIA Titan X Pascal GPU`( `12 Gb` 显存, `10Gbps` 访问速度)， `16 core` 的`Intel Xeon CPU(E5-2623 v4 @ 2.60GHz)`，以及 `256 Gb` 内存。

   所有实验在共享资源环境下大约进行了`3`天。论文预期在消费级的单 `GPU` 机器上（如配备了 `Titan X GPU` ）的全部资源专用，可以在 `4` 到 `7` 天完成所有实验。

6. 对于 `Reddit` 和引文数据集，我们按照 `Perozzi` 等人的描述对 `DeepWalk` 执行 `oneline` 训练。对于新的测试顶点，我们进行了新一轮的 `SGD` 优化，从而得到新顶点的 `embedding` 。

   现有的 `DeepWalk` 实现仅仅是 `word2vec`代码的封装，它难以支持 `embedding` 新顶点以及其它变体。这里论文根据 `tensorflow` 中的官方 `word2vec` 教程实现了 `DeepWalk` 。为了得到新顶点的 `embedding`，我们在保持已有顶点的 `embedding` 不变的情况下，对每个新的顶点执行 `50` 个长度为 `5` 的随机游走序列，然后更新新顶点的 `embedding` 。

   - 论文还测试了两种变体：一种是将采样的随机游走“上下文顶点”限制为仅来自已经训练过的旧顶点集合，这可以缓解统计漂移；另一种是没有该限制。我们总数选择性能最强的那个。

   - 尽管 `DeepWalk` 在 `inductive` 任务上的表现很差，但是在 `transductive` 环境下测试时它表现出更强的竞争力。因为在该环境下 `DeepWalk` 可以在单个固定的图上进行持续的训练。

     我们观察到在 `inductive` 环境下 `DeepWalk` 的性能可以通过进一步的训练来提高。并且在某种情况下，如果让它比其它方法运行的时间长 `1000` 倍，则它能够达到与无监督 `GraphSage` （而不是 有监督 `GraphSage` ）差不多的性能。但是我们不认为这种比较对于 `inductive` 是有意义的。

7. 在 `PPI` 数据集中我们无法应用 `DeepWalk`，因为在不同的、不相交的图上运行 `DeepWalk` 算法生成的 `embedding`空间可以相对于彼此任意旋转。参考最后一小节的证明。

8. 由于顶点 `degree` 分布的长尾效应，我们将 `GraphSage` 算法中所有图的边执行降采样预处理。经过降采样之后，使得没有任何顶点的 `degree` 超过 `128` 。由于我们每个顶点最多采样 `25` 个邻居，因此这是一个合理的权衡。

9. `GraphSage` 及 `baseline` 在这三个任务上的表现如下表所示。这里给出的是测试集上的 `micro-F1` 指标，对于 `macro-F1` 结果也有类似的趋势。其中 `Unsup` 表示无监督学习，`Sup` 表示监督学习。

   - `GraphSage` 的性能明显优于所有的 `baseline` 模型。
   - 根据 `GraphSage` 不同版本可以看到：与`GCN` 聚合方式相比，可训练的神经网络聚合函数具有明显的优势。
   - 尽管`LSTM` 这种聚合函数是为有序数据进行设计而不是为无序集合准备的，但是通过随机排列的方式，它仍然表现出出色的性能。
   - 和监督版本的 `GraphSage` 相比，无监督 `GraphSage` 版本的性能具有相当竞争力。这表明我们的框架无需特定于具体任务就可以实现强大的性能。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/Ykqd3ic8fDuq.png?imageslim">
   </p>
   

10. 通过在 `Reddit` 数据集上不同模型的训练和测试的运行时间如下表所示，其中 `batch size = 512`，测试集包含 `79534`个顶点。

    这些方法的训练时间相差无几，其中 `GraphSage-LSTM` 最慢。除了 `DeepWalk`之外，其它方法的测试时间也相差无几。由于 `DeepWalk` 需要采样新的随机游走序列，并运行多轮`SGD` 随机梯度下降来生成未见过顶点的 `embedding`，这使得 `DeepWalk` 在测试期间慢了 `100~500` 倍。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/xVxNe4chkXDk.png?imageslim">
   </p>
   

11. 对于 `GraphSage` 变体，我们发现和$K=1$相比，设置$K=2$使得平均准确率可以一致性的提高大约 `10%~15%` 。但是当$K$增加到 `2`以上时会导致性能的回报较低（`0~5%`） ，但是运行时间增加到夸张的 `10~100`倍，具体取决于采样邻域的大小。

    另外，随着采样邻域大小逐渐增加，模型获得的收益递减。因此，尽管对邻域的采样引起了更高的方差，但是 `GraphSage`仍然能够保持较强的预测准确率，同时显著改善运行时间。下图给出了在引文网络数据集上 `GraphSage-mean` 模型采用不同邻域大小对应的模型性能以及运行时间，其中$K=2$以及$S_1=S_2$。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/hwx8rx4Mohgr.png?imageslim">
   </p>
   

12. 总体而言我们发现就平均性能和超参数而言，基于 `LSTM` 聚合函数和池化聚合函数的表现最好。为了定量的刻画这种比较优势，论文将三个数据集、监督学习和无监督学习两种方式一共六种配置作为实验，然后使用 `Wilcoxon Signed-Rank Test`来量化不同模型的性能。

    结论：

    - 基于 `LSTM` 聚合函数和池化聚合函数的效果确实最好。
    - 基于`LSTM` 聚合函数的效果和基于池化聚合函数的效果相差无几，但是由于 `GraphSage-LSTM` 比 `GraphSage-pool` 慢得多（大约`2`倍），这使得基于池化的聚合函数总体上略有优势。

### 8.3 DeepWalk embedding 旋转不变性

1. `DeepWalk,node2vec` 以及其它类似的顶点 `embedding` 方法的目标函数都有类似的形式：

  $\mathcal L = \alpha \sum_{i,j\in \mathcal A} f(\mathbf{\vec z}_i\cdot \mathbf{\vec z}_j) + \beta \sum_{i,j\in \mathcal B} f(\mathbf{\vec z}_i\cdot \mathbf{\vec z}_j)$

   其中$f(\cdot),g(\cdot)$为平滑、连续的函数，$\mathbf{\vec z}_i$为直接优化的顶点的 `embedding`（通过 `embedding` 的 `look up` 得到），$\mathcal A,\mathcal B$为满足某些条件的顶点 `pair` 对。

   事实上这类方法可以认为是一个隐式的矩阵分解$\mathbf Z^T\mathbf Z \simeq \mathbf M \in \mathbb R^{|V|\times |V|}$，其中$\mathbf Z\in \mathbb R^{d\times |V|}$的每一列代表一个顶点的 `embedding` ，$\mathbf M\in \mathbb R^{|V|\times |V|}$是一个包含某些随机游走统计量的矩阵。

   这类方法的一个重要结果是：`embedding`可以通过任意单位正交矩阵变换，从而不影响矩阵分解：

  $(\mathbf Q^T\mathbf Z)^T(\mathbf Q^T\mathbf Z) = \mathbf Z^T\mathbf Q\mathbf Q^T\mathbf Z = \mathbf Z^T\mathbf Z\simeq \mathbf M$

   其中$\mathbf Q\in \mathbb R^{d\times d}$为任意单位正交矩阵。所以整个`embedding` 空间在训练过程中可以自由旋转。

2. `embedding` 矩阵可以在 `embedding` 空间可以自由旋转带来两个明显后果：

   - 如果我们在两个单独的图 `A` 和 `B` 上基于$\mathcal L$来训练 `embedding` 方法，如果没有一些明确的惩罚来强制两个图的顶点对齐，则两个图学到的 `embedding` 空间将相对于彼此可以任意旋转。因此，对于在图 `A` 的顶点 `embedding`上训练的任何顶点分类模型，如果直接灌入图 `B` 的顶点 `embedding`，这这等效于对该分类模型灌入随机数据。

     如果我们有办法在图之间对齐顶点，从而在图之间共享信息，则可以缓解该问题。研究如何对齐是未来的方向，但是对齐过程不可避免地在新数据集上运行缓慢。

     而 `GraphSage` 完全无需做额外地顶点对齐，它可以简单地为新顶点生成 `embedding` 信息。

   - 如果在时刻$t$对图 `A` 基于$\mathcal L$来训练 `embedding` 方法，然后在学到的 `embedding` 上训练分类器。如果在时刻$t+1$，图 `A` 添加了一批新的顶点，并通过运行新一轮的随机梯度下降来更新所有顶点的 `embedding` ，则这会导致两个问题：

     - 首先，类似于上面提到的第一点，如果新顶点仅连接到少量的旧顶点，则新顶点的 `embedding` 空间实际上可以相对于原始顶点的 `embedding` 空间任意旋转。
     - 其次，如果我们在训练过程中更新所有顶点的 `embedding`，则相比于我们训练分类模型所以来的原始 `embedding` 空间相比，我们新的 `embedding` 空间可以任意旋转。

3. 这类`embedding` 空间旋转问题对于依赖成对顶点距离的任务（如通过 `embedding`的点积来预测链接）没有影响。

4. 缓解这类统计漂移问题（即`embedding` 空间旋转）的一些方法为：

   - 为新顶点训练 `embedding` 时，不要更新已经训练的 `embedding` 。
   - 在采样的随机游走序列中，仅保留旧顶点为上下文顶点，从而确保 `skip-gram` 目标函数中的每个点积操作都是一个旧顶点和一个新顶点。

   在 `GraphSage` 论文中，作者尝试了这两种方式，并始终选择效果最好的 `DeepWalk` 变体。

5. 从经验来讲，`DeepWalk` 在引文网络上的效果要比 `Reddit` 网络更好。因为和引文网络相比，`Reddit` 的这种统计漂移更为严重：`Reddit` 数据集中，从测试集链接到训练集的边更少。在引文网络中，测试集有 `96%` 的新顶点链接到训练集；在 `Reddit` 数据集中，测试集只有 `73%` 的新顶点链接到训练集。

## 九、GAT

1. 卷积神经网络`CNN` 已经成功应用于图像分类、语义分割以及机器翻译之类的问题，其底层数据结构为网格状结构`grid-like structure` 。但很多任务涉及到无法以网状结构表示的数据，如：社交网络、电信网络等，而这些数据通常可以用图的方式来组织。

   - 早期图领域任务通常作为有向无环图，采用循环神经网络`RNN` 来处理。后来发展出图神经网络`GNN` 作为 `RNN` 的泛化来直接处理更通用的图，如循环图、有向图、无环图。

     `GNN` 包含一个迭代过程来传播顶点状态直至达到状态平衡，即不动点；然后再根据顶点状态为每个顶点生成输出。`GG-NN` 通过在顶点状态传播过程中使用门控循环单元来改进这一方法。

   - 另一个方向是将卷积推广到图领域，有两种推广思路：谱方法`spectral approach` 、非谱方法`non-spectral approach` 。

     - 谱方法通常和图的谱表达`spectral representation`配合使用，并已成功应用于顶点分类。

       但是这些谱方法中，学习的`filter` 都依赖于拉普拉斯矩阵分解后的特征向量。这种分解依赖于具体的图结构，因此在一种图结构上训练的模型无法直接应用到具有不同结构的图上。

       另外，该方法无法应用于 `graph-level` 任务，因为不同图的结构通常不同，因此拉普拉斯矩阵也不同。

     - 非谱方法可以直接在图上定义卷积，从而对空间相邻的邻域顶点进行运算。

       但是如何定义一种可以处理不同数量邻居顶点、且能保持 `CNN` 权重共享的卷积操作是一个挑战。`PATCHY-SAN`通过归一化邻域得到固定大小的邻域。

   - `attention` 机制在很多基于序列的任务中已经称为标配。注意力机制的一个好处是可以处理变长输入，并聚焦于输入中最相关的部分来做决策。当注意力机制作用于单个序列的`representation` 时，通常称作`self-attention` 或者 `intra-attention` 。

     受此启发，论文 `《GRAPH ATTENTION NETWORKS》` 提出了一种基于注意力的架构 `Graph attention network:GAT` 来对图结构数据进行顶点分类。其基本思想是：通过`self-attention` 策略来“注意”邻居顶点，从而计算每个顶点的 `representation` 。

     `GAT` 堆叠了一些 `masked self-attention layer` ，这些层中的顶点能够注意到邻居顶点的特征，并给邻域中不同的顶点赋予不同权重。在这个过程中不需要进行任何复杂的矩阵操作（如矩阵求逆或者矩阵分解），也不需要任何依赖于图结构的先验知识。

     `GAT` 模型具有以下优势：

     - 计算高效，因为`GAT` 可以在顶点邻居`pair` 对之间并行执行。
     - 通过对邻居顶点赋予任意权重，它可以应用到任意`degree` 的顶点，对网络结构的普适性更强。
     - 该模型可以直接应用于归纳学习`inductive learning`问题，包括将模型推广到从未见过的图。

2. `inductive learning` 和 `transductive learning`的区别：

   - `inductive learning` 是从具体样本中总结普适性规律，然后泛化到训练中从未见过的样本。

   - `transductive learning` 是从具体样本中总结具体性规律，它用于预测训练集中已经出现过的`unlabelled` 样本，常用于半监督学习。

     > 半监督学习不一定是 `transductive`，它也可能是 `inductive` 。如：训练时仅考虑 `labelled` 样本，不使用任何 `unlabelled` 样本的信息。

### 9.1 模型

1. `graph attentional laye:GAL` 是 `GAT`模型中最重要的层，模型通过大量堆叠这种层来实现。

2. `attention` 的实现方式有很多种，`GAT` 不依赖于具体的 `attention` 实现方式。我们以最常见的实现方式为例，来说明 `GAL`。

   `GAL` 的输入为一组顶点特征：$\mathbb H= \{\mathbf{\vec h}_1,\cdots,\mathbf{\vec h}_N\} ,\mathbf{\vec h}_i\in \mathbb R^F$，其中$N$为图的顶点数量、$F$为顶点的`representation`维度。`GAL` 输出这些顶点的新的`representation`：$\mathbb H^\prime = \{\mathbf{\vec h}_1^\prime,\cdots,\mathbf{\vec h}_N^\prime\},\quad \mathbf{\vec h}_i^\prime\in \mathbb R^{F^\prime}$，其中$F^\prime$为顶点新的 `representation` 维度。

   - 为了在将输入特征映射到高维特征时获得足够的表达能力，我们至少需要一个可学习的线性变换，即这个线性变换的权重不是固定的。

     为实现该目的，我们首先对所有顶点应用一个共享权重的线性变换，权重为$\mathbf W \in \mathbb R^{F^\prime \times F}$。

   - 然后，我们在每个顶点上应用 `self-attention`$a: \mathbb R^{F^\prime \times \mathbb R^{F^\prime}} \rightarrow \mathbb R$， 来计算`attention` 系数：

    $e_{i,j} = a\left(\mathbf W \mathbf{\vec h}_i, \mathbf W\mathbf{\vec h}_j\right)$

    $e_{i,j}$的物理意义为：站在顶点$i$的角度看顶点$j$对它的重要性。

   - 理论上讲，我们允许每个顶点关注图中所有其它的顶点，因此这可以完全忽略所有的图结构信息。实际上，我们采用 `masked attention`机制来结合图的结构信息：对于顶点$i$我们仅仅计算其邻域内顶点的 `attention` 系数$e_{i,j},j\in \mathcal N_i$，其中$\mathcal N_i$为顶点$i$的邻域。注意：这里$\mathcal N_i$包含顶点$i$在内。

     目前在我们的所有实验中，邻域$\mathcal N$仅仅包含顶点的一阶邻居（包括顶点$i$在内）。

   - 为使得系数可以在不同顶点之间比较，我们使用 `softmax` 函数对所有的$j$进行归一化：

    $\alpha_{i,j} = \text{softmax}_j(e_{i,j}) = \frac{\exp(e_{i,j})}{\sum_{k\in \mathcal N_i}\exp(e_{i,k})}$

     在我们的实验中，注意力机制$a$是一个单层的前向传播网络，参数为权重$\mathbf{\vec a}\in \mathbb R^{2F^\prime}$，采用 `LeakyReLU` 激活函数，其中负轴斜率$\beta = 0.2$。因此注意力得分为：

    $\alpha_{i,j} = \text{softmax}_j(e_{i,j}) = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{\vec a}\cdot [\mathbf W\mathbf{\vec h}_i,\mathbf W\mathbf{\vec h}_j] \right)\right)}{\sum_{k\in \mathcal N_i}\exp\left(\text{LeakyReLU}\left(\mathbf{\vec a}\cdot [\mathbf W\mathbf{\vec h}_i,\mathbf W\mathbf{\vec h}_k] \right)\right))}$

     其中$[\cdot,\cdot]$表示两个向量的拼接。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/IzgfgKpLagmc.png?imageslim">
   </p>
   

3. 一旦得到注意力得分，我们就可以用它对相应的邻居顶点进行加权，从而得到每个顶点的最终输出：

  $\mathbf{\vec h}_i^\prime = \sigma\left(\sum_{j\in \mathcal N_i}\alpha_{i,j}\mathbf W \mathbf{\vec h}_j\right)$

   其中$\mathbf W\in \mathbb R^{F^\prime \times F}$，它就是前面计算注意力得分的矩阵。

   我们使用 `multi-head attention` 来稳定 `self-attention` 的学习过程。我们采用$K$个 `head`，然后将它们的输出拼接在一起：

   其中$\alpha{i,j}^{(k)}$为第$k$个`head` 的注意力得分，$\mathbf W^{(k)}$为第$k$个 `head` 的权重矩阵。最终的输出$\mathbf{\vec h}^\prime$的维度为$KF^\prime$（而不是$F^\prime$）。

   但是，如果 `GAL` 是网络最后一层（即输出层），我们对 `multi-head` 的输出不再进行拼接，而是直接取平均，因为拼接没有意义。同时我们延迟使用最后的非线性层，对分类问题通常是 `softmax` 或者 `sigmoid` ：

  $\mathbf{\vec h}*i^\prime = \sigma\left(\frac 1K \sum*{k=1}^K\sum{j\in \mathcal N_i}\alpha{i,j}^{(k)} \mathbf W^{(k)}\mathbf{\vec h}*j\right)$*

   > *理论上，最后的`GAL` 也可以拼接再额外接一个输出层。*

   *如下图所示为 `multi head = 3`，当且顶点$i=1$。不同颜色的箭头表示不同的 `head` 。*

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/U4uUEuA5ABJR.png?imageslim">
   </p>
   

4. *`GAL` 解决了基于神经网络对图结构的数据建模的现有方法的问题：*

   - *`GAL` 计算高效。`GAL` 层的操作可以在所有的边上并行计算，输出特征的计算可以在所有顶点上并行计算。这里没有耗时的特征值分解，也没有类似的矩阵的耗时操作（如求逆）。*

     - *单个 `attention head` 计算$F^\prime$个特征的时间复杂度为$O(|V|\times F\times F^\prime + |E|\times F^\prime)$，其复杂度和一些`baseline` 方法（如 `Graph Convolutional Networks:GCN` ）差不多。*

       *首先计算所有顶点的$\mathbf W \mathbf{\vec h}\*i$，计算复杂度为$O(|V|\times F\times F^\prime)$；再计算所有的$\mathbf{\vec a}\cdot [\mathbf W\mathbf{\vec h}\*i,\mathbf W\mathbf{\vec h}\*j]$，计算复杂度为$O(|E|\times F^\prime)$；再计算所有的$\alpha\*{i,j}$，计算复杂度为$O(|V|\times \bar d)$，其中$\bar d = \frac{|E|}{|V|}$为顶点的平均 `degree`，则$\alpha\*{i,j}$的计算复杂度为$O(|E|)$。**

       *最终计算复杂度为$|V|\times F\times F^\prime + |E| \times F^\prime$。*

     - *尽管 `multi-head attention` 使得参数数量和空间复杂度变成$K$倍，但是每个 `head` 可以并行计算。*

   - *和 `Semi-Supervised GCN` 相比，`GAT` 模型允许为同一个邻域内的顶点分配不同的重要性，从而实现模型容量的飞跃。另外，和机器翻译领域一样，对学得的注意力权重进行分析可以得到更好的解释性。*

   - *注意力机制以共享的方式应用于图的所有边，因此它不需要预先得到整个图结构或者所有顶点。这带来几个影响：*

     - *图可以是有向图，也可以是无向图。如果边$j\rightarrow i$不存在，则我们不需要计算系数$\alpha*{i,j}$。
     - `GAT` 可以直接应用到归纳学习 `inductinve learning`：模型可以预测那些在训练集中从未出现的图。

   - `GraphSage` 归纳学习模型对每个顶点采样固定大小的邻域，从而保持计算过程的一致性。这使得模型无法在测试期间访问整个邻域。注意：由于训练期间训练多个 `epoch`，则可能访问到顶点的整个邻域。

     也可以使用`LSTM` 技术来聚合邻域顶点，但是这需要假设邻域中存在一个一致的顶点顺序。

     `GAT` 没有这两个问题：`GAT` 在作用在完整的邻域上，并且不关心邻域内顶点的顺序。

5. 我们可以使用一种利用稀疏矩阵操作的 `GAL` 层，它可以将空间复杂度下降到顶点和边的线性复杂度，从而使得模型能够在更大的图数据集上运行。但是我们的 `tensor` 计算框架仅支持二阶`tensor` 的稀疏矩阵乘法，这限制了当且版本的 `batch`处理能力，特别是在具有很多图的数据集上。解决该问题是未来一个重要的方向。

   另外，在稀疏矩阵的情况下，`GPU` 的运算速度并不会比 `CPU` 快多少。

6. 和`Semi-Supervised GCN`以及其它模型类似，`GAT` 模型的感受野取决于网络深度。但是网络太深容易引起优化困难，此时采用跳跃连接 `skip connection` 可以解决该问题，从而允许 `GAT` 使用更深的网络。

7. 如果在所有边上并行计算（尤其是分布式并行计算）可能包含大量重复计算，因为图中的邻居往往高度重叠。

### 9.2 实验

#### 9.2.1 Transductinve Learning

1. 数据集：三个标准的引文网络数据集`Cora, Citeseer,Pubmed`。

   每个顶点表示一篇文章、边（无向）表示文章引用关系。每个顶点的特征为文章的 `BOW representation`。每个顶点有一个类别标签。

   - `Cora` 数据集：包含`2708` 个顶点、`5429` 条边、`7` 个类别，每个顶点 `1433` 维特征。
   - `Citeseer` 数据集：包含`3327` 个顶点、`4732` 条边、`6` 个类别，每个顶点 `3703` 维特征。
   - `Pubmed` 数据集：包含`19717` 个顶点、`44338` 条边、`3` 个类别，每个顶点 `500` 维特征。

   对每个数据集的每个类别我们使用`20` 个带标签的顶点来训练，然后在 `1000` 个测试顶点上评估模型效果。我们使用额外的 `500` 个带标签顶点作为验证集。注意：训练算法可以利用所有顶点的结构信息和特征信息，但是只能利用每个类别 `20` 个顶点的标签信息。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/nr9lbmx2mfJ8.png?imageslim">
   </p>
   

2. `Baseline` 模型：我们对比了论文 `《Semi-supervised classification with graph convolutional networks》` 的工作，以及其它的 `baseline` 模型，包括：标签传播模型`label propagation: LP`，半监督嵌入模型 `semi-supervised embedding: SemiEmb`，流型正则化模型 `manifold regularization: ManiReg`，基于`SkipGram` 的`graph embeding` 模型（如 `DeepWalk`），迭代式分类算法模型 `iterative classification algorithm: ICA` ，`Planetoid` 模型。

   我们也直接对比了 `Semi-Supervised GCN`模型、利用高阶切比雪夫的图卷积模型、以及 `MoNet` 模型。

   我们还提供了每个顶点共享 `MLP` 分类器的性能，该模型完全没有利用图的结构信息。

3. 参数配置：

   - 我们使用一个双层的 `GAT` 模型：

     - 第一层包含$K=8$个 `attention head`，每个 `head` 得到$F^\prime = 8$个特征，总计 `64` 个特征。第一层后面接一个`exponential linear unit:ELU` 非线性激活层。
     - 第二层用作分类，采用一个 `attention head` 计算$C$个特征，其中$C$为类别数量，然后使用 `softmax` 激活函数。

   - 模型的所有超参数在 `Cora` 上优化之后，复用到`Citeseer` 数据集。

   - 当处理小数据集时，我们在模型上施加正则化：

     - 我们采用$L_2$正则化，其中正则化系数为$\lambda = 0.0005$
     - 两个层的输入，以及 `normalized attention coefficient` 都使用了$p=0.6$（遗忘比例）的 `dropout` 。即每轮迭代时，每个顶点需要随机采样邻居。

   - 对于`60` 个样本的 `Pubmd` 数据集，这些参数都需要微调：

     - 输出为$K=8$个 `attention head` ，而不是一个。
     -$L_2$正则化系数为$\lambda = 0.001$。

     除此之外都和 `Cora/Citeseer` 的一样。

   - 所有模型都采用 `Glorot` 初始化方式来初始化参数，优化目标为交叉熵，使用 `Adam SGD` 优化器来优化。初始化学习率为：`Pubmed` 数据集为 `0.01`，其它数据集为 `0.005`。

     我们在所有任务上执行早停策略，在验证集上的交叉熵、`accuracy`如果连续 `100` 个 `epoch` 没有改善，则停止训练。

4. 我们报告了 `GAT` 随机执行 `100` 次实验的分类准确率的均值以及标准差，也使用了 `Semi-Supervised GCN` （即表中的 `GCN`）和 `Monet` 报告的结果。

   - 对基于切比雪夫过滤器的方法，我们提供了$K=2,K=3$阶过滤器的最佳结果。
   - 我们进一步评估了 `Semi-Supervised GCN` 模型，其隐层为 `64` 维，同时尝试使用 `ReLU` 和 `ELU`激活函数，并记录执行 `100` 次后效果最好的那个（实验表明 `ReLU` 在所有三个数据集上都最佳），记作 `GCN-64*` 。
   - 结论：`GAT` 在 `Cora` 和 `Citeseer`上超过 `Semi-Supervised GCN` 分别为 `1.5%, 1.6%` ，这表明为邻域内顶点分配不同的权重是有利的。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/lekhNpLNiyGW.png?imageslim">
   </p>
   

#### 9.2.2 Inductinve learning

1. 数据集：`protein-protein interaction: PPI` 数据集，该数据集包含了人体不同组织的蛋白质的`24` 个图。其中`20` 个图为训练集、`2` 个图为验证集、`2`个图为测试集。注意：这里测试的`Graph`在训练期间完全未被观测到。

   我们使用 `GraphSage` 提供的预处理数据来构建图，每个图的平均顶点数量为 `2372` 个，每个顶点`50` 维特征，这些特征由 `positional gene sets, motif gene sets, immunological signatures` 组成。

   从 `Molecular Signatuers Database`收集到的`gene ontology` 有 `121` 种标签，这里每个顶点可能同时属于多个标签。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/bdau5pbtV5vn.png?imageslim">
   </p>
   

2. `Baseline` 模型：我们对比了 四个不同版本的监督 `GraphSAGE` 模型，它们提供了多种方法来聚合采样邻域内的顶点特征：`GraphSAGE-GCN`、`GraphSAGE-mean`、`GraphSAGE-LSTM`、`GraphSAGE-pool` 。

   剩下的 `transductinve` 方法要么完全不适用于`inductive` 的情形，要么无法应用于在训练期间完全看不到测试`Graph`的情形，如 `PPI`数据集。

   我们还提供了每个顶点共享 `MLP` 分类器的性能，该模型完全没有利用图的结构信息。

3. 参数配置：

   - 我们使用一个三层`GAT` 模型：

     - 第一层包含$K=4$个 `attention head`，每个 `head` 得到$F^\prime = 256$个特征，总计 `1024`个特征。第一层后面接一个`exponential linear unit:ELU` 非线性激活层。

     - 第二层和第一层配置相同。

     - 第三层为输出层，包含$K=6$个 `attention head` ，每个 `head` 得到 `121`个特征。

       我们对所有 `head` 取平均，并后接一个 `sigmoid` 激活函数。

   - 由于该任务的训练集足够大，因此我们无需执行$L_2$正则化或者 `dropout` 。

   - 我们在 `attention layer` 之间应用 `skip connection` 。

   - 训练的 `batch size = 2` ，即每批`2` 个 `graph` 。

   - 为评估 `attention` 机制的效果，我们提供了一个注意力得分为常数的模型进行对比（$a(x,y) = 1$），其它结构不变。

   - 所有模型都采用 `Glorot` 初始化方式来初始化参数，优化目标为交叉熵，使用 `Adam SGD` 优化器来优化。初始化学习率为：`Pubmed` 数据集为 `0.01`，其它数据集为 `0.005`。

     我们在所有任务上执行早停策略，在验证集上的交叉熵、`micro-F1`如果连续 `100` 个 `epoch` 没有改善，则停止训练。

4. 我们报告了模型在测试集（两个从未见过的 `Graph` ）上的 `micro-F1` 得分。我们随机执行`10` 轮 “训练--测试”，并报告这十轮的均值。

   - 对于其它基准模型，我们使用 `GraphSage` 报告的结果。
   - 为了评估聚合邻域的好处，我们进一步提供了`GraphSAGE` 架构的最佳结果，记作 `GraphSAGE*` 。这是通过一个三层`GraphSAGE-LSTM` 得到的，三层维度分别为 `[512,512,726]`，最终聚合的特征为 `128` 维。
   - 我们报告常数的注意力系数为 `Const-GAT` 。
   - 结论：
     - `GAT` 在 `PPI` 数据集上相对于 `GraphSAGE` 的最佳效果还要提升 `20.5%` ，这表明我们的模型在`inductive` 任务中通过观察整个邻域可以获得更大的预测能力。
     - 相比于 `Const-GAT`，我们的模型提升了 `3.9%`，这再次证明了为不同邻居分配不同权重的重要性。

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/assdWAlwMxj1.png?imageslim">
   </p>
   

#### 9.2.3 其它

1. 我们采用 `t-SNE` 对学到的特征进行可视化。我们对 `Cora` 数据集训练的 `GAT` 模型的第一层的输出进行可视化，该 `representation` 在投影到的二维空间中表现出明显的聚类。这些簇对应于数据集的七种类别，从而验证了模型的分类能力。

   此外我们还可视化了归一化注意力系数的相对强度（在所有`8` 个 `attention head`上的均值）。如果正确的解读这些系数需要有关该数据集的进一步的领域知识。

   下图中：颜色表示顶点类别，线条粗细代表归一化的注意力系数均值：

  $\text{line}{i,j} = \sum{k=1}^K \alpha_{i,j}^k + \alpha_{j,i}^k$

   <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/MYczsH3N2dxt.png?imageslim">
   </p>
   

2. 一些有待改进的点：

   - 如果处理更大的 `batch size` 。
   - 如果利用注意力机制对模型的可解释性进行彻底的分析。
   - 如何执行`graph-level` 的分类，而不仅仅是`node-level` 的分类。
   - 如果利用边的特征，而不仅仅是顶点的特征。因为边可能蕴含了顶点之间的关系，边的特征可能有助于解决各种各样的问题。