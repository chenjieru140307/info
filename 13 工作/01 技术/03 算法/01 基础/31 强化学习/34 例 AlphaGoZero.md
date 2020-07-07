# AlphaGo Zero


AlphaGo Zero

- 介绍
  - Silver et. al., Mastering the game of Go without human knowledge
  - 纯自我对弈，从随机下棋开始，没有用任何人类棋谱
  - 纯棋盘当前黑白子将征作为输入
  - 用同一个 neural network 作为 policy 和 value network 
  - 采用上述 neural network， 用 Monte Carlo Tree Search 来 sample move
- 算法细节
  - 神经网络本身输 raw move probabilities $f_{\theta}(s)$ 
    - s 为当前棋盘状态。
  - Monte Carlo Tree Search (MCTS)可以在它的基础上连出更好的步骤去下棋
  - 用Monte Carlo Tree Search和自己下棋，然后用贏(1)或输(-1) 当作 value， 可以训练一个很好的 policy evaluator
  - 更新 neural network 使得 policy 更加符合 MCTS 的策略，value 更加符合 self-play 的结果
- Monte Carlo Tree Search
  - 每一个 $edge (s, a)$会储存先验概率 $P(s, a)$，访 问次数 $N(s,a)$ 和 action value $Q(s, a)$
    - 我们使用策略 $f$ 基于 $s$ 算出来的概率分布 $a$ 就是我们的先验概率 $P(s,a)$ 
  - 每一次 simulation 我们都从当前局面开始， 选择 $Q(s, a) + \mathrm{P}(\mathrm{s}, \mathrm{a}) /(1+\mathrm{N}(\mathrm{s}, \mathrm{a}))$ 最大的那一步棋，直到一个 leaf node被访问（还没有访问过的状态）
  - 然后我们会展开这个 leaf node，计算 $\left(P\left(s^{\prime}, \cdot\right), V\left(s^{\prime}\right)\right)=f_{\theta}\left(s^{\prime}\right)$
  - simulation 结束之后，每一条被访问的 $edge (s, a)$ 的 counter $\mathrm{N}(\mathrm{s},a)$ 都会加 1 ，$Q(s, a)=1 / N(s, a) \sum_{s^{\prime} \mid s, a \rightarrow s^{\prime}} V\left(s^{\prime}\right)$
- 训练 neural network
  - 首先，我们随机初始化神经网络的参数
  - 在一个iteration 结束之后，我们使用上一个 iteration 训练出来的神经网络，利用 MTCS 对弈，直到对弈分出胜负。（请查看论文中约纸节， 如何判定胜负）
  - 对弈中的每一步，我们都存下来， 包括 $\left(s_{t}, \pi_{t}, z_{t}\right)$ 其中 $S_{t}$ 是当前状态，$\pi_{t}$ 是MCTS 结合 neural network 选出来的策略，$z_{t}$ 表示以当前下棋者的视角来看，这盘棋是赢了了(1)还是输了 $(-1)$
  - 我们从刚刚存下来的棋谱中随机选择一些 batch，然后训练我们的神经网络使得 $(\boldsymbol{p}, v)=f_{\theta_{i}}(s)$ 与我们自我对弈的结果尽量接近。 loss function 如下：$(\boldsymbol{p}, v)=f_{\theta}(s)$ and $l=(z-v)^{2}-\boldsymbol{\pi}^{\mathrm{T}} \log \boldsymbol{p}+c\|\theta\|^{2}$

自我对弈与训练：

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200703/R37PC3kw4PaQ.png?imageslim">
</p>

MonteCarlo Tree Search：

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200703/wpyHnPCzXXcK.png?imageslim">
</p>

说明：

- 上图是 Monte Carlo Tree Search 的过程，整个 simulation 结束之后，我们把该局面下每一种下去被采用的次数转换 成一个概率分布。