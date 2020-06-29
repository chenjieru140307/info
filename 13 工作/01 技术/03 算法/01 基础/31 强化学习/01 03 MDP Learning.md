# MDP Learning


注意：

- 前提是，state 和 action 是离散的，所以可以使用矩阵来描述。

Unknown Environment MDP

- 缘由：
  - 大多数时候模型未知，并不知道 $\mathcal{R}$ 和 $\mathcal{P}_{\mathrm{ss}^{\prime}}^{a}$。
    - 比如，棋局中，reward 已知是，你看对面出车，你就上马，然后你知道这个会胜还是败。$\mathcal{P}_{\mathrm{ss}^{\prime}}^{a}$ 已知就是，对手面对某种棋局的时候，会怎么做。但是，实际上，我们并不能知道对手怎么做。也不知道自己上马会胜还是败。
  - 那么，一无所知的情况下，怎么做？
    - 我随便下，比如下了 1w 次，然后得到这些下法数据，然后 estimate 模型，然后做优化。
      - 先建立模型(Estimate P and R from observations)
      - 这就是一个 model-based 的方法，这种方法好吗？有点问题的，比如，如果是一个游戏，那么游戏死了，是可以的，但是，如果是现实的世界，你只有一次机会，也没有办法重复，那么这种方法是有问题的。
    - 另外一种方式是：我也是乱下，但是，每走一步，都会重新评估自己的策略，吃一堑长一智，完全不考虑建立这个环境模型，完全不考虑去 estimate 我的 P 和 R。
      - 边玩边学 $v(s)$, $q(s, a)$ and $\pi(s)$
      - 即：我每一步，就直接上来评估这个 $v(s)$,$q(s, a)$ 每走一步都会调整自己。而不是像上面一样，走几万步之后，完全了解这个世界，才开始优化自己。
      - 而且，我边成长边总结，不是总结这个世界是什么样的，而是直接总结定义在这个世界上的几个 target 函数即 $v(s)$, $q(s, a)$ 

即，对于未知环境下的 最优策略求解：

- Model-Based 方法:
  - 即 先建立模型(Estimate P and R from observations) 。
  - 相关问题：
    - 策略评估
    - 寻找最优策略
- Model-Free:
  - 边玩边学 $v(s), q(s, a)$ and $\pi(s)$
  - 相关问题：
    - 策略评估：已知一个 policy ，如何评估这个 policy ，我们称之为 prediction，如果不尝试去求解这个模型，那么称为 model-free prediction。有两种方法：
      - Monte Carlo Method
      - TD Method 时间差分方法。
    - 寻找最优策略



## Model-based 方法：

- 流程：
  - 先产生 $k$ 个数据，即先产生一些样本：
    $$
    \begin{array}{l}
    s_{0}^{(1)} \stackrel{a_{0}^{(1)}}{\longrightarrow} s_{1}^{(1)} \stackrel{a_{1}^{(1)}}{\longrightarrow} s_{2}^{(1)} \stackrel{a_{2}^{(1)}}{\longrightarrow} s_{3}^{(1)} \stackrel{a_{3}^{(1)}}{\longrightarrow} \ldots 
    \\s_{0}^{(2)} \stackrel{a_{0}^{(2)}}{\longrightarrow} s_{1}^{(2)} \stackrel{a_{1}^{(2)}}{\longrightarrow} s_{2}^{(2)} \stackrel{a_{2}^{(2)}}{\longrightarrow} s_{3}^{(2)} \stackrel{a_{3}^{(2)}}{\longrightarrow} \ldots
    \\\ldots
    \end{array}
    $$
    - 其中:
      - $\mathrm{s}_{\mathrm{i}}^{(\mathrm{j})}$ is the state at time $i$ of trial $j$
      - $a_{i}^{(j)}$ is the action at time $i$ of trial $j$
  - 然后基于这些样本建立 MDP 模型：
    - 基于这些样本，对于 state transition probabilities 的 MLE 估计为:
        $$P_{s a}\left(s^{\prime}\right)=\frac{\# \text { of times we took action } a \text { in state } s \text { and } \operatorname{got} \text { to } s^{\prime}}{\# \text { of times we took action } a \text { in state } s}$$
      - 说明:
        - 如果 在 state $s$ 中 action $a$ 没有发生过，那么，上面的 ratio 是 $0 / 0$，所以，在这种情况下 $P_{s a}\left(s^{\prime}\right)=1 /|S|$ (uniform distribution over all states)
        - $P_{sa}$ 比较容易更新的，按照分子分母将对应的次数加进去就行。
    - 同样的，$R(s,a)$ 也可以计算：
      - $R(s,a)=$ average reward in state $s$ across all the trials
  - 得到了 $P_{s a}\left(s^{\prime}\right)$ 和 $R(s, a)$，这样问题就转化成，已知 $\mathcal{R}$ 和 $\mathcal{P}_{\mathrm{ss}^{\prime}}^{a}$ 的情况下的问题了，即 MDP Planning。
- 应用：
  - 基本不使用，可以了解下。
- 缺点：
  - 如果想精确的 得到 $P_{s a}\left(s^{\prime}\right)$，那么样本至少是 $P_{s a}\left(s^{\prime}\right)\text{的量级}*1w$，即 $S^2*A*1w$，一般我们的 state 很多，而且，如果是 连续的 state，那么更多。所以，一般这种方式可以在 state 很少的情况时使用。


## Model-Free 方法：

- 策略评估问题：
  - Monte-Carlo
    - 基本思想：大数定理，期望值 经验平均值
    - 介绍：
      - 回顾 $V_{\pi}(\mathrm{s})$ 定义 :
          $$
          v_{\pi}(s)=\mathbb{E}_{\pi}\left[G_{t} \mid S_{t}=s\right]
          $$
          $$
          G_{t}=R_{t+1}+\gamma R_{t+2}+\ldots+\gamma^{T-1} R_{T}
          $$
      - 说明：
        - 即：$V_{\pi}(s)$ 可以由状态从 $s$ 开始的所有 $G_t$ 的平均值.替代。
        - 即，用经验平均代替期望值。
      - 即：如果我产生了很多轨迹样本，如果某个样本是从 $s$ 开始的，而我就能计算出 从 $s$ 开始的样本的 $G_t$，这就是一个样本从 $s$ 开始的 $G_t$，然后我把所有的 从 $s$ 开始的 $G_t$ 全部加起来，求平均，就是我们要求的 $v_{\pi}(s)$。
        - 根据大数定理：我 random 的 样本的平均值会趋近于我随机值的期望。
      - 疑问：
        - $\gamma$ 不知道的情况下，怎么计算 $G_t$？
        - 而且，如果 $R$ 不是一个数字，而是一个图片呢，那么怎么计算 $G_t$？
    - Monte-Carlo 方法细分：
      - 原始方法：
        - First-Visit MC 方法 计算 $V_{\pi}(s):$
          - 流程：
            - 01 Initialize $N(s) \leftarrow 0, s(s) \leftarrow 0$
            - 02 对每条轨迹，如果状态 $s$ 在时间 $t$ 被首次访问，那么 {
            - 03 $\quad$ $N(s) \leftarrow N(s)+1$
            - 04 $\quad$ $S(s) \leftarrow S(s)+G_{t}$
            - 05 }
            - 06 最终 $V(s)=S(s) / N(s)$
          - 理解：
            - 由大数定理 $V(s) \rightarrow V_{\pi}(s)$ as $N \rightarrow \infty$ ，收剑精度 $\sim N(s)^{-1 / 2}$
        - Every-Visit MC 方法 计算 $V_{\pi}(s):$
          - 流程：
            - 01 Initialize $N(s) \leftarrow 0, s(s) \leftarrow 0$
            - 02 对每条轨迹，如果状态 $s$ 在时间 $t$ 被 ~~首次~~ 访问，那么 {
            - 03 $\quad$ $N(s) \leftarrow N(s)+1$
            - 04 $\quad$ $S(s) \leftarrow S(s)+G_{t}$
            - 05 }
            - 06 最终 $V(s)=S(s) / N(s)$
          - 理解：
            - 由大数定理 $V(s) \rightarrow V_{\pi}(s)$ as $N \rightarrow \infty$ ，收剑精度 $\sim N(s)^{-1 / 2}$
            - Every-Visit MC 方法与 First-Visit 方法唯一不同就是在于 ，如果在一个轨迹里，开始遇到了 $s$，然后后面又遇到了一个 $s$，那么，Every-Visit MC 计算的是两次。
      - 由于，可以对数据增量更新：
        - 由于：
          $$
          \begin{aligned}
          \mu_{k} &=\frac{1}{k} \sum_{j=1}^{k} x_{j} \\
          &=\frac{1}{k}\left(x_{k}+\sum_{j=1}^{k-1} x_{j}\right) \\
          &=\frac{1}{k}\left(x_{k}+(k-1) \mu_{k-1}\right) \\
          &=\mu_{k-1}+\frac{1}{k}\left(x_{k}-\mu_{k-1}\right)
          \end{aligned}
          $$
        - 说明：
          - 第一行：即，可以用 $x_j$ 个样本来估计 $\mu$ 
          - 第四行：即，如果新增了一个样本，那么可以使用这个样本 $x_k$ 与之前的 $k-1$ 个样本估计得到的 $\mu_{k-1}$ 方便的得到新的 $\mu_k$，而不用再从头统计。
        - 即：
          - 可以不用将所有的样本都加起来得到 $S(s)$ 和 $N(s)$ ，因为这样可能加起来的 数据非常大，溢出，而是可以每次拿到一个样本都估算出这次的 $V(s)$，然后，下一个样本过来，可以用这次的 $V(s)$ 与 这个样本的 $N(s)$ 计算新的 $V(s)$。
      - 基于增量更新，将算法更新为：Incremental MC 算法：
        - Initialize $N(s) \leftarrow 0, S(s) \leftarrow 0$
        - 对每条轨迹，如果状态 $s$ 在时间 $t$ 被访问，那么
          - $\mathrm{N}(\mathrm{s}) \leftarrow \mathrm{N}(\mathrm{s})+1$
          - $\mathrm{V}(\mathrm{s}) \leftarrow \mathrm{V}(\mathrm{s})+\left(\mathrm{G}_{\mathrm{t}}-\mathrm{V}(\mathrm{s})\right) / \mathrm{N}(\mathrm{s})$
        - Or more general, 可以完全忘记 history trajectory
          - $V(s) \leftarrow V(s)+\alpha\left(G_{t}-V(s)\right)$
    - 注意：
      - 对于 Monte Carlo 方法，如果 state space 非常大，或者 本身就是连续的，比如围棋的棋盘情况，那么，可以使用 DL 或者 Deep Q-Learning 来求解 v(s) 的函数的形式，而不是用矩阵的形式了。
      - 样本采样的时候，这个序列样本一定要是到达终点的，这样才可以计算 $G_t$。
        - 这个是一个缺点，这样就没法在 轨迹产生的过程中进行学习，必须等到轨迹结束。
  - Temporal-Difference 时间差分
    - 介绍：
      - 根据 Monte Carlo 方法，有：
          $$\mathrm{V}\left(\mathrm{s}_{\mathrm{t}}\right) \leftarrow \mathrm{V}\left(\mathrm{s}_{\mathrm{t}}\right)+\alpha\left(\mathrm{G}_{\mathrm{t}}-\mathrm{V}\left(\mathrm{s}_{\mathrm{t}}\right)\right)$$
      - 根据 Bellman Expectation Eq.， 用 $R_{t+1}+\gamma V\left(s_{t+1}\right)$ 替代 $G _{t}$
          $$\quad V\left(s_{t}\right) \leftarrow V\left(s_{t}\right)+\alpha\left(R_{t+1}+\gamma V\left(s_{t+1}\right)-V\left(s_{t}\right)\right)$$
      - 说明：
        - 即，我并不真正的计算出 $G_t$ ，而是用 $R_{t+1}+\gamma V\left(s_{t+1}\right)$ 来近似的代替 $G_t$ 来代到 Monte Carlo 式子里面去。
        - 这样就有一个好处，每走一步，就可以 update $V(s_t)$，我不需要这个轨迹结束，我任意拿一段轨迹都可以用来学。
        - 与 Bootstrap 类似。（没有理解，为什么？）
        - 上式，相当于：
          - TD target 为：$R_{t+1}+\gamma V\left(s_{t+1}\right)$
          - TD error 为：$\delta_t=R_{t+1}+\gamma V\left(s_{t+1}\right)-V\left(s_{t}\right)$
      - 其实，上面的是用 $R_{t+1}+\gamma V\left(s_{t+1}\right)$ 替代 $G _{t}$，而，$G_t$ 可以写成更多步：
        - n-step TD 算法:
          $$
          \begin{array}{rl}
          n=1 &G_{t}^{(1)}=R_{t+1}+\gamma V\left(S_{t+1}\right)  \quad  (\text{即}T D) 
          \\ n=2 & G_{t}^{(2)}=R_{t+1}+\gamma R_{t+2}+\gamma^{2} V\left(S_{t+2}\right) 
          \\ & \vdots 
          \\ n=\infty & G_{t}^{(\infty)}=R_{t+1}+\gamma R_{t+2}+\ldots+\gamma^{T-1} R_{T} \quad (\text{即}M C) 
          \end{array}
          $$
        - 理解：
          - 往下多走几步的时候，variance 会变大一点点，bias 会变小一点点。
        - 这时，我们定义：
          - $\quad G_{t}^{(n)}=R_{t+1}+\gamma R_{t+2}+\ldots+\gamma^{n-1} R_{t+n}+\gamma^{n} V\left(S_{t+n}\right)$
        - 那么，n-step TD learning 可以写成：
          $$\quad V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha\left(G_{t}^{(n)}-V\left(S_{t}\right)\right)$$
      - 那么，基于 n-step TD ，我们可以得到 TD($\lambda$)算法：
        - 可以将 $G_{t}^{(1)}$，$G_{t}^{(2)}$，等加起来，求平均值，根据中心极限定理，这时候的 variance 会比较小。（疑问：为什么可以加起来？为什么可以使用中心极限定理？不是要独立同分布吗？这个时候的 $G_{t}^{(1)}$，$G_{t}^{(2)}$ 并不是独立同分布吧？）
        - 这就是 TD($\lambda$)算法:
          - 基本想法: 综合 n-step TD的return，使结果更加robust
          - 权重函数
              $$
              G_{t}^{\lambda}=(1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_{t}^{(n)}
              $$
          - $\lambda$ 怎么给定，如图：（没明白）
              <p align="center">
                  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200627/20wIySUftEmR.png?imageslim">
              </p>
          - 此时：
              $$V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha\left(G_{t}^{\lambda}-V\left(S_{t}\right)\right)$$

    - MC/TD 关于偏差和方差的权衡 Bias / Variance Trade-Off ：
      - 对比下 MC 和 TD 对于 $\mathrm{V}_{\pi}\left(\mathrm{s}_{\mathrm{t}}\right)$ 的估计：
        - 如果使用：$G_{t}=R_{t+1}+\gamma R_{t+2}+\ldots+\gamma^{T-1} R_{T}$ 来带入 $\mathrm{V}\left(\mathrm{s}_{\mathrm{t}}\right) \leftarrow \mathrm{V}\left(\mathrm{s}_{\mathrm{t}}\right)+\alpha\left(\mathrm{G}_{\mathrm{t}}-\mathrm{V}\left(\mathrm{s}_{\mathrm{t}}\right)\right)$
          - 那么是对 $\mathrm{V}_{\pi}\left(\mathrm{s}_{\mathrm{t}}\right)$ 的一个无偏估计
        - 如果使用：$G_{t}=R_{t+1}+\gamma v_{\pi}\left(S_{t+1}\right)$ 来带入 $\mathrm{V}\left(\mathrm{s}_{\mathrm{t}}\right) \leftarrow \mathrm{V}\left(\mathrm{s}_{\mathrm{t}}\right)+\alpha\left(\mathrm{G}_{\mathrm{t}}-\mathrm{V}\left(\mathrm{s}_{\mathrm{t}}\right)\right)$，即假设使用真正的 $v_{\pi}\left(S_{t+1}\right)$。
          - 那么也是对 $\mathrm{V}_{\pi}\left(\mathrm{s}_{\mathrm{t}}\right)$ 的一个无偏估计
        - 如果使用：$G_{t}=R_{t+1}+\gamma V\left(S_{t+1}\right)$ 来带入 $\mathrm{V}\left(\mathrm{s}_{\mathrm{t}}\right) \leftarrow \mathrm{V}\left(\mathrm{s}_{\mathrm{t}}\right)+\alpha\left(\mathrm{G}_{\mathrm{t}}-\mathrm{V}\left(\mathrm{s}_{\mathrm{t}}\right)\right)$
          - 那么，由于 $\mathrm{V}\left(\mathrm{s}_{\mathrm{t+1}}\right)$ 不是真切从轨迹中得到的，而是我们之前估计出来的，那么这个是对 $\mathrm{V}_{\pi}\left(\mathrm{s}_{\mathrm{t}}\right)$ 的一个有偏估计
          - 所以，这个估计是有一定的 bias 的。
      - 那么，为什么有 bias 还要使用 TD 呢？
        - 因为，与 MC 相比，TD 的 Variance 会小很多。
          - 因为，在 $G_{t}=R_{t+1}+\gamma R_{t+2}+\ldots+\gamma^{T-1} R_{T}$ 中，每一步都是一个 random，这个随机可能从系统来，从环境来，有很多不确定因素在里面，所以用 $G_t$ 估计的话，在 MC 里面，它的 Variance 会非常大。
          - 而 TD 中，使用 $G_{t}=R_{t+1}+\gamma V\left(S_{t+1}\right)$ 的话，只有下一步的 reward 中有  randomness 在。它的总的 variance 非常小。
        - 即：TD target Variance $<<$ MC target Variance
    - TD  MC 对比：
      - MC：
        - 要等到 episode 结束才能获得return
        - 只能使用完整的 episode
        - 高 variance, 零 bias 
        - 没有体现出马尔可夫性质
        - No Bootstrapping 
        - 收敛慢，steady
      - TD：
        - 每一步执行完都能获获得一个 return，是一个 online learning 的方法，每一步都可以 更新 $V(s)$ ，而 MC 是一个 offline learning 的方法。
        - 可以使用不完整的 episode，即不需要这个轨迹 终结
        - 低 variance，有 bias
        - 体现出了马尔可夫性质 (use MDP) 
          - 使用了 Bellman Expectation eq，用来代替 $G_t$。
          - 所以，如果过程是一个马尔科夫过程，用 TD 来，是更好的。如果不是，那么只能用 MC。
        - Bootstrapping (自助法，plug-in 原则)
          - MC 中没有把当前学到的东西用起来，而 TD 使用了已经学到的东西，即 $V\left(S_{t+1}\right)$
        - 收敛快，not steady
          - 收敛快因为 variance 低。
      - 通常认为 TD 比 MC 更好一些。
- 寻找最优策略问题：Unknown Environment MDP Control
  - 利用和探索 Exploration and Exploitation
    - Multi-Armed Bandit Problem
    - $\epsilon$-greedy strategy
  - 同策略学习和异策略学习 On Policy / Off Policy Learning
    - Monte Carlo Method
    - TD Method: Sarsa (on policy TD), Q-Learning (off policy TD)


- 基本思路: 广义策略迭代
  - 回顾策略迭代 for known environment MDP
    - 策略迭代：
      - 给定策略 $\pi$, 评估策略得到 $V_{\pi}(s)$
      - 改进策略: $\pi^{\prime}=\operatorname{greedy}\left(\mathrm{V}_{\pi}\right)=>\pi^{\prime} \geq \pi$
    - 如图：
        <p align="center">
            <img width="300" height="70%" src="http://images.iterate.site/blog/image/20200627/i3NsUbVy4kpq.png?imageslim">
        </p>
        <p align="center">
            <img width="200" height="70%" src="http://images.iterate.site/blog/image/20200627/67ricxHHG8BI.png?imageslim">
        </p>
  - 因此，可以类似的，进行 策略评估 加 策略改进：
    - 有如下问题：
      - 问题1：
        - 环境已知时，可以通过评估策略 $\pi$ 得到 $V_{\pi}(s)$，即解 Bellman Expectation Equation 方程即可。那么，环境未知时，无法解方程，怎么办：
          - 可以通过样本对 $V_{\pi}(s)$ 进行估计。
      - 问题2：
        - 环境已知时，我们通过 $\mathrm{V}(\mathrm{s})$ require model $\pi^{\prime}(s)=\underset{a \in \mathcal{A}}{\operatorname{argmax}} \mathcal{R}_{s}^{a}+\mathcal{P}_{s s^{\prime}}^{a} V\left(s^{\prime}\right)$ 来利用 $V_{\pi}(s)$ 改进 $\pi$，那么，环境未知时，即不知道 $\mathcal{R}_{s}^{a}$ 和 $\mathcal{P}_{s s^{\prime}}^{a}$ 怎么办：
          - 这时候，我们不 estimate 我们的 v-function ，而是 estimate 我们的 q-function： $Q(s, a)$，即 $\pi^{\prime}(s)=\underset{a \in \mathcal{A}}{\operatorname{argmax}} Q(s, a)$
  - 按照这个思路，流程如下：
    - 01 Initialize, for all $s \in \mathcal{S}, a \in \mathcal{A}(s)$
      - $Q(s, a) \leftarrow \text { arbitrary }$
      - $\pi(s) \leftarrow \text { arbitrary }$
      - $\text { Returns }(s, a) \leftarrow \text { empty }$
    - 02 Repeat forever:
    - 03 $\quad$ Generate an episode using exploring starts and $\pi$
    - 04 $\quad$ For each pair $s, a$ appearing in the episode:
    - 05 $\quad$ $\quad$ $R \leftarrow$ return following the first occurrence of $s, a$ 
    - 06 $\quad$ $\quad$ Append $R$ to $\operatorname{Returns}(s, a)$
    - 07 $\quad$ $\quad$ $Q(s, a) \leftarrow$ average $(\text {Returns}(s, a))$
    - 08 $\quad$ For each $s$ in the episode:
    - 09 $\quad$ $\quad$ $\pi(s) \leftarrow \arg \max _{a} Q(s, a)$
  - 那么，上面的思路有什么问题吗？
    - 有两个问题：
      - 问题1：
        - 如何保证每个状态行为对 $(s, a)$ 都可以被访问到？
          - 因为我们是 greedy 走的，即每次都是基于当前最好的策略产生一个样本，总是选最好的，那么可能就不会每个 $(s, a)$ 都访问到。
        - 所以：
          - 不能使用 greedy。而要确保历经每个状态行为对，即 $\pi(\mathrm{a} \mid \mathrm{s})>0$ for all $\mathrm{a}, \mathrm{s}$
          - 如图：
              <p align="center">
                  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200627/Vkd3c65bkNP1.png?imageslim">
              </p>
              <p align="center">
                  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200627/9083TtY3Ycsu.png?imageslim">
              </p>
      - 问题2：
        - 怎么能确保每次迭代的 $\pi^{\prime} \geq \pi \quad(\text { 回顾policy ordering })$？
      - 问题3：
        - 我们使用了 $\varepsilon$ -greedy 来得到我们的样本，但是，我们最终要衡量的是，greedy 的策略是不是好策略，这两个并不是同一个，怎么办？
    - 对于问题 1 的解决，引入了 探索和利用：
      - 实时在线决策
        - Exploitation: 基于之前所有的信息做出最优选择
        - Exploration: 收集更多信息
      - 如果是 greedy 的，那么总是 利用已知的最好的，而没有探索，所以需要一个 trade-off。因为：
        - 最好的长远策略可能需要牺牲短期利益
        - 只有收集到足够多的数据才能作出全局最好决策
      - 举例：
        - 多臂自动机(Multi-Armed Bandit)
            <p align="center">
                <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200627/lW3qz6fY17yE.png?imageslim">
            </p>
        - 对于一排老虎机，每个老虎机回报不同，而你的钱是有限的。 那么，
          - 纯粹的探索，是每个都玩一次，然后玩回报最多的那个。
            - 但是这个有个问题，每个老虎机每次回报的也不同，没准这个老虎机这次一下回报了很多，其实它平均下来很少。
          - 纯粹的利用，是只玩一个。
        - 那么，怎么求解这种情况的最优解呢？有三种方法：
          - Naive-Exploration: $\varepsilon$ -greedy (Add noise to greedy strategy)
              $$
              \pi(a \mid s) \leftarrow\left\{\begin{array}{c}
              1-\varepsilon+\frac{\varepsilon}{|A(s)|} \text { if } a=\arg \max _{a} Q(s, a) \\
              \frac{\varepsilon}{|A(s)|} \text { if } a \neq \arg \max _{a} Q(s, a)
              \end{array}\right.
              $$
            - 说明：
              - 本质上还是一个 greedy 的方法，对于最好的，下次玩的概率是 $1-\varepsilon+\frac{\varepsilon}{|A(s)|}$，对于其他的，下次玩的概率是：$\frac{\varepsilon}{|A(s)|}$
              - 相当于是在 greedy 上面加一个 noise。确保其他的有一定的几率探索到。
          - Thompson Sampling
          - Upper Confidence Bound(置信区间上界) UCB
            - Choose the arm with max value of $\bar{x}_{j}(t)+\sqrt{\frac{2 \ln t}{T_{j, t}}}$
              - 理解：
                - $t$ 时刻的每台老虎机 的 average return 是 $\bar{x}_{j}(t)$，但是，你选择下一台老虎机，还要平衡在 t 时刻这台老虎机选择过的次数 $T_{j, t}$，你选择过的次数越少，那么这台机器被选择的机会就更大一点。
            - 这个在 AlphaGo 中使用的 UCB 方法。当前使用的比较火的。
        - 多臂自动机的应用：
          - 推荐系统。
            - 在没有什么经验时，死命的推一个，或者每次都推不同的，都不是很好，与这种场景类似。
      - 因此，对于问题1，可以使用 $\varepsilon$ -greedy 或者 UCB 的方法来处理 基于一个策略生成样本的时候，每个 action 的选择 的问题。 
    - 对于问题 2 的解决：
      - 定理：
        - For any $\epsilon$ -greedy policy $\pi,$ the $\epsilon$ -greedy policy $\pi^{\prime}$ with respect to $q_{\pi}$ is an improvement, $v_{\pi^{\prime}}(s) \geq v_{\pi}(s)$
      - 证明：
        $$\begin{aligned}
        q_{\pi}\left(s, \pi^{\prime}(s)\right) &=\sum_{a \in \mathcal{A}} \pi^{\prime}(a \mid s) q_{\pi}(s, a) \\
        &=\epsilon / m \sum_{a \in \mathcal{A}} q_{\pi}(s, a)+(1-\epsilon) \max _{a \in \mathcal{A}} q_{\pi}(s, a) \\
        & \geq \epsilon / m \sum_{a \in \mathcal{A}} q_{\pi}(s, a)+(1-\epsilon) \sum_{a \in \mathcal{A}} \frac{\pi(a \mid s)-\epsilon / m}{1-\epsilon} q_{\pi}(s, a) \\
        &=\sum_{a \in \mathcal{A}} \pi(a \mid s) q_{\pi}(s, a)=v_{\pi}(s)
        \end{aligned}$$
      - 说明：
        - 第一行到第二行：根据 $\varepsilon$-greedy 公式，替换 $\pi^{\prime}(a \mid s)$
        - 第二行到第三行:$\sum_{a \in \mathcal{A}} \frac{\pi(a \mid s)-\epsilon / m}{1-\epsilon}$ 是一个 average，因为加起来是 1 ，max 肯定是大于这个 average。
        - 第三行到第四行：将 $(1-\epsilon)$ 乘进去。
        - $m$ 即状态数，等于 $|\mathcal{A}|$
      - 所以，$v_{\pi^{\prime}}(s) \geq v_{\pi}(s)$
    - 第三个问题：
      - 我们使用了 $\varepsilon$ -greedy 来得到我们的样本，但是，我们最终要衡量的是，greedy 的策略是不是好策略，这两个并不是同一个，怎么办？
      - 有两种方法解决：
        - On Policy Learning: 探索策略与评估策略为同一策略
          - "Learn on the job"
          - Learn about policy $\pi$ from experience sampled from $\pi$
        - Off Policy Learning:探索策略与评估策略为不同策略
          - "Look over someone's shoulder"
          - 可以从 policy $\mu$ 产生的样本中学习 policy $\pi$.
          - 可以从观察人类或者别的 agents 的行为来学习。
          - 可以重复使用从老的 policy $\pi_{1}, \pi_{2}, \ldots, \pi_{t-1}$ 中产生的样本。
          - Learn about optimal policy while following exploratory policy
          - Learn about multiple policies while following one policy
      - On-policy first-visit MC control (for $\varepsilon$ -soft policies), estimates $\pi \approx \pi$
        - 流程：
          - 01 Initialize, for all $s \in \mathcal{S}, a \in \mathcal{A}(s):$
          - 02 $\quad$ $Q(s, a) \leftarrow \text { arbitrary }$
          - 03 $\quad$ Returns $(s, a) \leftarrow$ empty list
          - 04 $\quad$ $\pi(a \mid s) \leftarrow \text { an arbitrary } \varepsilon \text { -soft policy }$
          - 05 Repeat forever:
          - 06 $\quad$ Generate an episode using $\pi$
          - 07 $\quad$ For each pair $s, a$ appearing in the episode:
          - 08 $\quad$ $\quad$ $G \leftarrow$ return following the first occurrence of $s, a$
          - 09 $\quad$ $\quad$ Append $G$ to $\operatorname{Returns}(s, a)$
          - 10 $\quad$ $\quad$ $Q(s, a) \leftarrow$ average $(\text { Returns }(s, a))$
          - 11 $\quad$ For each $s$ in the episode:
          - 12 $\quad$ $\quad$ $A^{*} \leftarrow \arg \max _{a} Q(s, a)$
          - 13 $\quad$ $\quad$ For all $a \in \mathcal{A}(s):$
          - 14 $\quad$ $\quad$ $\quad$ $\pi(a \mid s) \leftarrow\left\{\begin{array}{ll}1-\varepsilon+\varepsilon /|\mathcal{A}(s)| & \text { if } a=A^{*} \\\varepsilon /|\mathcal{A}(s)| & \text { if } a \neq A^{*}\end{array}\right.$
      - Sarsa (on-policy TD control) for estimating $Q \approx q$
        - 流程
          - 01 Initialize $Q(s, a), \forall s \in \mathcal{S}, a \in \mathcal{A}(s),$ arbitrarily, and $Q(\text {terminal}-\text {state}, \cdot)=0$
          - 02 Repeat (for each episode):
          - 03 $\quad$ Initialize $S$ 
          - 04 $\quad$ Choose $A$ from $S$ using policy derived from $Q$ (e.g., $\epsilon$-greedy ) 
          - 05 $\quad$ Repeat (for each step of episode):
          - 06 $\quad$ $\quad$ Take action $A,$ observe $R, S^{\prime}$ 
          - 07 $\quad$ $\quad$ Choose $A^{\prime}$ from $S^{\prime}$ using policy derived from $Q$ (e.g., $\epsilon$-greedy)
          - 08 $\quad$ $\quad$ $Q(S, A) \leftarrow Q(S, A)+\alpha\left[R+\gamma Q\left(S^{\prime}, A^{\prime}\right)-Q(S, A)\right]$
          - 09 $\quad$ $\quad$ $S\leftarrow S^{\prime} ; A \leftarrow A^{\prime} ;$
          - 10 $\quad$ until $S$ is terminal
        - 疑问：
          - 这个 $\alpha$ 和 $\gamma$ 的值怎么确定？
          - 为什么 blackjack 中的例子的 q(s,a) 有超过 1 的？
    - off policy
      - 先了解下重要性抽样 Importance Sampling：
        - 如图：
            <p align="center">
                <img width="50%" height="70%" src="http://images.iterate.site/blog/image/20200628/PcpNfviG12lU.png?imageslim">
            </p>
        - 比如说 我们要 evaluate $f(x)$，但是，每个 $x$ 的分布是 $p(x)$ 的，但是这个 $x$ 的 sample 不太好产生，比如说，高斯分布，比较好产生，那么我现在只有一个高斯的随机数发生器 $q(x)$，我怎么做呢？
        - 如下：
            $$\begin{aligned}
            \mathbb{E}_{X \sim P}[f(X)] &=\sum P(X) f(X) \\
            &=\sum Q(X) \frac{P(X)}{Q(X)} f(X) \\
            &=\mathbb{E}_{X \sim Q}\left[\frac{P(X)}{Q(X)} f(X)\right]
            \end{aligned}$$
        - 说明：
          - 这样，我每次使用高斯随机数发生器产生 x，然后把 x 带入到 $\frac{P(X)}{Q(X)} f(X)$ 里面，得到一个值，然后把所有的这些都加起来，就得到了期望。
        - 总结下，为什么要重要性抽样？
          - 原来的分布不好得到。所以，可以利用手里已有的随机数发生器。
          - 我可以设计出这样的重要性抽样来使我们的 variance 降低。可以使蒙特卡洛收敛更快。 
            - 因为我们做蒙特卡洛的话，蒙特卡洛的 variance 是通过中心极限定理告诉我们的。蒙特卡洛收敛是 $\sqrt{\frac{a}{N}}$ 这个常数 a 就是 f(x) 就是本身的 variance。（这个地方没听清楚，补充下）
      - 在 MDP 中的重要性抽样：
        - 由于，同样的 轨迹，在不同的 policy 下出现的概率是不同的。因为 policy 本质上是从 state 到 action 的映射。
        - 所以，这时候，在 policy $\pi$ 下，一个轨迹出现的概率为：
            $$\operatorname{Pr}\left(A_{t}, S_{t+1}, \cdots, S_{T}\right)=\prod_{k=t}^{T-1} \pi\left(A_{k} \mid S_{k}\right) p\left(S_{k+1} \mid S_{k}, A_{k}\right)$$
          - 说明，$p\left(S_{k+1} \mid S_{k}, A_{k}\right)$：给了一个 $S_{k}, A_{k}$ 之后，转换到 $S_{k+1}$ 的概率是系统内在的。
        - 在 policy $\mu$ 下，一个轨迹出现的概率为：
            $$\operatorname{Pr}\left(A_{t}, S_{t+1}, \cdots, S_{T}\right)=\prod_{k=t}^{T-1} \mu\left(A_{k} \mid S_{k}\right) p\left(S_{k+1} \mid S_{k}, A_{k}\right)$$
        - 假如我在 $\pi$ 下采样到了一个 轨迹，我想用这个轨迹样本来 estimate 我的 policy $\mu$ 的 q-function 或者 v-function，怎么做呢？
          - 这个地方就与重要性采样一样，把 $\mathbb{E}_{X \sim Q}\left[\frac{P(X)}{Q(X)} f(X)\right]$ 中的 $\frac{P(X)}{Q(X)}$ 算出来即可，即：
            $$\rho_{t}^{T}=\frac{\prod_{k=t}^{T-1} \pi\left(A_{k} \mid S_{k}\right) p\left(S_{k+1} \mid S_{k}, A_{k}\right)}{\prod_{k=t}^{T-1} \mu\left(A_{k} \mid S_{k}\right) p\left(S_{k+1} \mid S_{k}, A_{k}\right)}=\prod_{k=t}^{T-1} \frac{\pi\left(A_{k} \mid S_{k}\right)}{\mu\left(A_{k} \mid S_{k}\right)}$$
          - 说明：
            - $p\left(S_{k+1} \mid S_{k}, A_{k}\right)$ 是系统内在的，所以，可以抵消。
          - 我们使用重要性采样，主要原因是，想利用别人的 样本来 evaluate 自己的 策略，这里面是不能保证重要性采样的稳定性的。
        - 所以，可以把 $\rho_{t}^{T}$ 作为重要性权重，拿进来，就可以了。
          - 此时，有 Ordinary importance sampling：
            - 计算：
              $$V(s) \doteq \frac{\sum_{t \in \mathcal{T}(s)} \rho_{t: T(t)-1} G_{t}}{|\mathcal{T}(s)|}$$
            - 说明：
              - 比如说，我们在 $\pi$ 下产生一个 轨迹，然后我得到这个轨迹对于特定的 $(s,a)$ 的 return 是 $G_t$，由于我们要 estimate 的 是 $\mu$ 对应的 $V(s)$ 而不是 $\pi$ 对应的 $V(s)$，虽然我们 sample 的是 under  $\pi$ ，这样的话，我们把因子乘上 $G_t$，这是我们修正之后的 return ，这些 轨迹都这样处理后做 average。
              - $\mathcal{T}(s)$：即有 $\mathcal{T}$ 个这样的 $s$ 开头的轨迹。即 $s$ 在所有的轨迹中被 sample 了多少次。相当于代码里面的 count_sum 。
          - 但是，上面的做 average 的时候是有点问题的，因为每个 轨迹的权重不同，我们不把他们平均，所以，把上面的权重 $\rho_{t: T(t)-1}$ 加起来，作为权重。得到 Weighted importance sampling：
            - 计算：
                $$V(s) \doteq \frac{\sum_{t \in \mathcal{T}(s)} \rho_{t: T(t)-1} G_{t}}{\sum_{t \in \mathcal{T}(s)} \rho_{t: T(t)-1}}$$
            - 说明：
              - 这个是有一定 bias 的，但是会 reduce variance，会收敛很快。
          - Ordinary importance sampling 与 Weighted importance sampling 的使用：
            - 如图：
                <p align="center">
                    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200628/4SnqI5JQ1Qgl.png?imageslim">
                </p>
            - 说明：
              - 可见，如果使用 ordinary 的sampling 的方法，虽然没有 bias ，但是 variance 非常大。收敛的会比较慢。
              - 所以，通常我们使用 weighted importance sampling 的方法来更新 $V(s)$
      - Off-policy MC control, for estimating $\pi \approx \pi_{*}$
        - 对于在 Incremental MC 中的 $\rho_{t}^{T}$ 的使用：
            - 假如我们有一系列的 returns $\mathrm{G}_{1}, \mathrm{G}_{2}, \ldots, \mathrm{G}_{\mathrm{n}-1}$ with weight $W_{i}=\rho_{t: T(t)-1}$
            - 那么，这个时候的 MC Estimate 为：
                $$V_{n} \doteq \frac{\sum_{k=1}^{n-1} W_{k} G_{k}}{\sum_{k=1}^{n-1} W_{k}}, n \geq 2$$
            - 而，这个时候的 Incremental MC Estimate 为：
                $$
                V_{n+1} \doteq V_{n}+\frac{W_{n}}{C_{n}}\left[G_{n}-V_{n}\right], n \geq 1
                $$
                $$
                C_{n+1} \doteq C_{n}+W_{n+1}
                $$
              - 说明：
                - 原来的 Incremental MC Estimate 为：
                  - $\mathrm{N}(\mathrm{s}) \leftarrow \mathrm{N}(\mathrm{s})+1$
                  - $\mathrm{V}(\mathrm{s}) \leftarrow \mathrm{V}(\mathrm{s})+\left(\mathrm{G}_{\mathrm{t}}-\mathrm{V}(\mathrm{s})\right) / \mathrm{N}(\mathrm{s})$
        - 流程：
          - 01 Initialize, for all $s \in \mathcal{S}, a \in \mathcal{A}(s)$
          - 02 $\quad$ $Q(s, a) \leftarrow \text { arbitrary }$
          - 03 $\quad$ $C(s, a) \leftarrow 0$
          - 04 $\quad$ $\pi(s) \leftarrow \arg \max _{a} Q\left(S_{t}, a\right) \quad$ (with ties broken consistently)
          - 05 Repeat forever:
          - 06 $\quad$ $b \leftarrow$ any soft policy 
          - 07 $\quad$ Generate an episode using $b$
          - 08 $\quad$ $\quad$ $S_{0}, A_{0}, R_{1}, \ldots, S_{T-1}, A_{T-1}, R_{T}, S_{T}$
          - 09 $\quad$ $G \leftarrow 0$
          - 10 $\quad$ $W \leftarrow 1$
          - 11 $\quad$ For $t=T-1, T-2, \ldots$ downto 0
          - 12 $\quad$ $\quad$ $G \leftarrow \gamma G+R_{t+1}$
          - 13 $\quad$ $\quad$ $C\left(S_{t}, A_{t}\right) \leftarrow C\left(S_{t}, A_{t}\right)+W$
          - 14 $\quad$ $\quad$ $Q\left(S_{t}, A_{t}\right) \leftarrow Q\left(S_{t}, A_{t}\right)+\frac{W}{C\left(S_{t}, A_{t}\right)}\left[G-Q\left(S_{t}, A_{t}\right)\right]$
          - 15 $\quad$ $\quad$ $\pi\left(S_{t}\right) \leftarrow \arg \max _{a} Q\left(S_{t}, a\right) \quad$ (with ties broken consistently)
          - 16 $\quad$ $\quad$ If $A_{t} \neq \pi\left(S_{t}\right)$ then ExitForLoop
          - 17 $\quad$ $\quad$ $W \leftarrow W \frac{1}{b\left(A_{t} \mid S_{t}\right)}$
        - 说明：
          - $W_{i}=\rho_{t: T(t)-1}$
          - $W$ 是怎么更新的：$b$ 这个 policy 基本上是随机的，因为我们只是用它来产生序列。即 $\rho_{t}^{T}=\prod_{k=t}^{T-1} \frac{\pi\left(A_{k} \mid S_{k}\right)}{\mu\left(A_{k} \mid S_{k}\right)}$ 中的 $\pi$ 。而 上面流程中的 $\pi$ 就是这个式子中的 $\mu$ 是每次都要 update 的，是我们要 estimate 的，所以，每次得到一个新的 policy 之后，重新计算这个式子，就得到了新的 $W$
      - Off Policy TD (Q-learning)
        - (详细的我们放在下一章)
        - One - step $Q$ - learning:
          - $Q\left(s_{t}, a_{t}\right) \leftarrow Q\left(s_{t}, a_{t}\right)+\alpha\left[r_{t+1}+\gamma \max _{a} Q\left(s_{t+1}, a\right)-Q\left(s_{t}, a_{t}\right)\right]$
        - 如图：
            <p align="center">
                <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200628/UcYEPoFQXjQO.png?imageslim">
            </p>
        - 流程：
          - Initialize $Q(s, a)$ arbitrarily
          - Repeat (for each episode):
          - $\quad$ Initialize $s$ 
          - $\quad$ Repeat (for each step of episode):
          - $\quad$ $\quad$ Choose $a$ from $s$ using policy derived from $Q$ (e.g., $\varepsilon$ -greedy)
          - $\quad$ $\quad$ Take action $a,$ observe $r, s^{\prime}$
          - $\quad$ $\quad$ $Q(s, a) \leftarrow Q(s, a)+\alpha\left[r+\gamma \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime}\right)-Q(s, a)\right]$
          - $\quad$ $\quad$ $s \leftarrow s^{\prime}$
          - $\quad$ until $s$ is terminal