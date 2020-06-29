# Q-learning

- Q-Learning
- Deep Q-Learning 
- Breakout代码实战


- 回顾：
  - 给定状态 $s$ 和动作 $a$ ，给定策略 $\pi$，给定 discount factor $\gamma$，那么：
    - Q值函数被定义为 
      $$
      Q^{\pi}(s, a)=\mathbb{E}\left[R_{t+1}+\gamma R_{t+2}+\gamma^{2} R_{t+3}+\ldots \mid s, a\right]
      $$
    - Bellman Equation
      $$
      Q^{\pi}(s, a)=\mathbb{E}_{s^{\prime}, a^{\prime}}\left[R+\gamma Q^{\pi}\left(s^{\prime}, a^{\prime}\right) \mid s, a\right]
      $$
      - 即：将 Q 带入 Q。
  - SARSA：
    - 如图：
        <p align="center">
            <img width="30%" height="70%" src="http://images.iterate.site/blog/image/20200629/68vedSfRw5Fa.png?imageslim">
        </p>
    - $Q(S, A) \leftarrow Q(S, A)+\alpha\left(R+\gamma Q\left(S^{\prime}, A^{\prime}\right)-Q(S, A)\right)$
    - 说明：
      - 唯一准确的是 R，是你真正得到的值，我们是想使用这个 R 值调整我们的 Q(S,A) 的值。
- Q-learning：
  - 描述了 Quality of a state-action pair
    $$
    Q\left(S_{t}, A_{t}\right) \leftarrow Q\left(S_{t}, A_{t}\right)+\alpha\left(R_{t+1}+\gamma Q\left(S_{t+1}, A^{\prime}\right)-Q\left(S_{t}, A_{t}\right)\right)
    $$
  - 其中：
    - 我们在 $(S_t,A_t)$ 情况下，使用 $\epsilon$-greedy 的方式往前走，得到一个 $R_{t+1}$ 和 $S_{t+1}$ ，然后呢，我们在 $S_{t+1}$ 的情况下使用 greedy 的方式，得到一个 $A^{\prime}$。
    - 因为在 $S_{t+1}$ 的情况下，使用的是 greedy 方式的策略，即：
        $$
        \pi\left(S_{t+1}\right)=\underset{a^{\prime}}{\operatorname{argmax}} Q\left(S_{t+1}, a^{\prime}\right)
        $$
    - 则，上式可以写为：
        $$
        \begin{aligned}
        Q\left(S_{t}, A_{t}\right) &\leftarrow Q\left(S_{t}, A_{t}\right)+\alpha\left(R_{t+1}+\gamma Q\left(S_{t+1}, A^{\prime}\right)-Q\left(S_{t}, A_{t}\right)\right)
        \\&\leftarrow Q\left(S_{t}, A_{t}\right)+\alpha\left(R_{t+1}+\gamma Q\left(S_{t+1}, \underset{a^{\prime}}{\operatorname{argmax}} Q\left(S_{t+1}, a^{\prime}\right)\right)-Q\left(S_{t}, A_{t}\right)\right)
        \\&\leftarrow Q\left(S_{t}, A_{t}\right)+\alpha\left(R_{t+1}+\max _{a^{\prime}} \gamma Q\left(S_{t+1}, a^{\prime}\right)-Q\left(S_{t}, A_{t}\right)\right)
        \end{aligned}
        $$
  - 流程：
    - 01 Initialize $Q(s, a)$ arbitrarily
    - 02 Repeat (for each episode):
    - 03 $\quad$ Initialize $s$ 
    - 04 $\quad$ Repeat (for each step of episode):
    - 05 $\quad$ $\quad$ Choose $a$ from $s$ using policy derived from $Q$ (e.g., $\varepsilon$ -greedy)
    - 06 $\quad$ $\quad$ Take action $a,$ observe $r, s^{\prime}$
    - 07 $\quad$ $\quad$ $Q(s, a) \leftarrow Q(s, a)+\alpha\left[r+\gamma \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime}\right)-Q(s, a)\right]$
    - 08 $\quad$ $\quad$ $s \leftarrow s^{\prime}$
    - 09 $\quad$ until $s$ is terminal
  - 说明：
    - 第 05 行：$a$ 实际上可以用任何方式得到，使用 $\varepsilon$ -greedy 的方式得到实际上是比较合适一些。
    - 第 07 行：$\alpha$ 是一个比较偏经验的数字，可以选 0.1、0.01 等。
  - Q-learning 与 SARSA：
    - 并不一定那个比较好。
  - 这时候，如果 $(s,a)$ 状态太多太复杂怎么办？
    - 比如：
      - 围棋
      - Atari游戏 
      - 星际争霸
      - 无法列举所有状态
    - 那么，这时候，可以尝试构造一个函数，把 $Q(s, a)$ 拟合出来。（对于这一点，还没有很清楚）
    - 几种值函数逼近的方法（value function approximation)：
        <p align="center">
            <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200629/0JdEXNHvluVx.png?imageslim">
        </p>
    - 说明：
      - 第一种：给定一个 $s$ 返回一个 $v$ 值。
      - 第二种：给定一个 $s,a$ pair 返回一个 带 $w$ 参数的 $q$ 值。
      - 第三种：给定一个 $s$，它给每个 $s,a$ pair 都对应一个 $w$ 值。
    - 那么怎么拟合这个函数呢？
      - Linear Combination of Features
      - Neural Network
      - Decision Tree
      - Nearest Neighbor
      - 以及任何其他的函数都可能用来拟合值函数
    - 下面使用神经网络来表示 Q 值函数。
      - $Q(s, a, \mathbf{w}) \approx Q^{*}(s, a)$
      - 对于 Bellman Equation
        $$
        Q^{*}(s, a)=\mathbb{E}_{s^{\prime}}\left[r+\gamma \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime}\right)^{*} \mid s, a\right]
        $$
      - 所以，我们可以把右边部分当做 label，左边部分当做 prediction，当做一个 supervised learning 问题来训练。
      - 我们优化 Mean Squared Error Loss，用 stochastic gradient descent 来训练
        $$
        I=\left(r+\gamma \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime}, w\right)-Q(s, a, w)\right)^{2}
        $$
    - 存在的问题：
      - 用神经网络训练Q值函数可能会不收敛。（为什么可能会不收敛？）
    - 对于网络的优化：
      - Experience Replay：
        - 流程：
          - 使用 $\epsilon$ -greedy 策略来选择动作 $a_{t}$ 
          - 把一系列动作，状态变化，回报都存下来
              $$
              \left(s_{t}, a_{t}, r_{t+1}, s_{t+1}\right)
              $$
          - 更新Q-network参数的时候就先随机选取一些 transitions $\left(s, a, r, s^{\prime}\right)$
          - 然后训练的目标函数使用老的参数
              $$
              \mathcal{L}_{i}\left(w_{i}\right)=\mathbb{E}_{s, a, r, s^{\prime} \sim \mathcal{D}_{i}}\left[\left(r+\gamma \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime} ; w_{i}^{-}\right)-Q\left(s, a ; w_{i}\right)\right)^{2}\right]
              $$
            - 说明：
              - 注意这个地方是 $w_{i}^{-}$ 而不是 $w_{i}$。为什么呢？老师说了，但是没听清。补充一下。
        - 优点：
          - 每一个训练数据可以多次利用，数据利用率高
          - 随机采样出来的 experience 直接相关性小，可以降低训练的 variance。
            - 即：比如游戏中，当前这一帧的画面与下一帧的画面可能很相似。这样可能使训练时方差很大，在比较短的时间内，跑偏。
      - Double DQN
        - 构造两个Q-network 
          - 一个（当前的）Q-network用于选择动作 
          - 另一个（老的） Q-network用于评估动作
            $$
            \begin{array}{c}
            l=\left(r+\gamma Q\left(s^{\prime}, \operatorname{argmax}_{a^{\prime}} Q\left(s^{\prime}, a^{\prime}, \mathbf{w}\right), \mathbf{w}^{-}\right)-Q(s, a, \mathbf{w})\right)^{2} \\
            \mathcal{L}_{i}\left(w_{i}\right)=\mathbb{E}_{s, a, r, s^{\prime} \sim \mathcal{D}_{i}}\left[\left(r+\gamma \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime} ; w_{i}^{-}\right)-Q\left(s, a ; w_{i}\right)\right)^{2}\right]
            \end{array}
            $$
        - Double DQN 可以提升训练的稳定性，避开 overoptimistic value estimates. 如果用统一的 Qnetwork 来选取和评估(s, a)，误差容易被放大。
        - 流程：
          - 01 input $: \mathcal{D}$ -empty replay buffer; $\theta$ - initial network parameters, $\theta^{-}-$ copy of $\theta$ 
          - 02 input $: N_{r}-$ replay buffer maximum size; $N_{b}-$ training batch size; $N^{-}-$ target network replacement freq 
          - 03 for episode $e \in\{1,2, \ldots, M\}$ do
          - 04 $\quad$ Initialize frame sequence $\mathbf{x} \leftarrow()$ 
          - 05 $\quad$ for $t \in\{0,1, \ldots\}$ do
          - 06 $\quad$ $\quad$ Set state $s \leftarrow \mathbf{x},$ sample action $a \sim \pi_{\mathcal{B}}$ 
          - 07 $\quad$ $\quad$ Sample next frame $x^{t}$ from environment $\mathcal{E}$ given $(s, a)$ and receive reward $r,$ and append $x^{t}$ to $\mathbf{x}$
          - 08 $\quad$ $\quad$ if $|\mathbf{x}|>N_{f}$ then delete oldest frame $x_{t_{\min }}$ from x end 
          - 09 $\quad$ $\quad$ Set $s^{\prime} \leftarrow \mathbf{x},$ and add transition tuple $\left(s, a, r, s^{\prime}\right)$ to $\mathcal{D}$ 
          - 10 $\quad$ $\quad$ $\quad$ replacing the oldest tuple if $|\mathcal{D}| \geq N_{r}$ 
          - 11 $\quad$ $\quad$ Sample a minibatch of $N_{b}$ tuples $\left(s, a, r, s^{\prime}\right) \sim \operatorname{Unif}(\mathcal{D})$
          - 12 $\quad$ $\quad$ Construct target values, one for each of the $N_{b}$ tuples:
          - 13 $\quad$ $\quad$ Define $a^{\max }\left(s^{\prime} ; \theta\right)=\arg \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime} ; \theta\right)$
          - 14 $\quad$ $\quad$ $y_{j}=\left\{\begin{array}{ll}r & \text { if } s^{\prime} \text { is terminal } \\ r+\gamma Q\left(s^{\prime}, a^{\max }\left(s^{\prime} ; \theta\right) ; \theta^{-}\right), & \text {otherwise }\end{array}\right.$
          - 15 $\quad$ $\quad$ Do a gradient descent step with $\operatorname{loss}\left\|y_{j}-Q(s, a ; \theta)\right\|^{2}$ 
          - 16 $\quad$ $\quad$ Replace target parameters $\theta^{-} \leftarrow \theta$ every $N^{-}$ steps 
          - 17 $\quad$ end
          - 18 end
      - Prioritized Experience Replay
        - DQN的错误越大的有越高的 priority 被先做评估 
        - 训练加快 
        - 最终的policy也变好了
            $$
            \left|r+\gamma \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime}, \mathbf{w}^{-}\right)-Q(s, a, w)\right|
            $$
      - Dueling Network Architectures for Deep Reinforcement Learning
        - [论文](https://arxiv.org/pdf/1511.06581.pdf)
        - 把 Q-network 分成两个 channel
          - value function V(s), 与 action 无关，表示状态 $s$ 的好坏 
          - advantage function A(s, a), 才与动作有关，表示当前状态下执行 $a$ $动作的好坏
            $$
            A^{\pi}(s, a)=Q^{\pi}(s, a)-V^{\pi}(s)
            $$
        - 注意 $V^{\pi}(s)=\mathbb{E}_{a \sim \pi(s)}\left[Q^{\pi}(s, a)\right]$
        - 所以 $\mathbb{E}_{a \sim \pi(s)}\left[A^{\pi}(s, a)\right]=0$
        $$\begin{array}{l}
        Q(s, a ; \theta, \alpha, \beta)=V(s ; \theta, \beta)+A(s, a ; \theta, \alpha) \\
        Q(s, a ; \theta, \alpha, \beta)=V(s ; \theta, \beta)+ \left(A(s, a ; \theta, \alpha)-\max _{a^{\prime} \in|\mathcal{A}|} A\left(s, a^{\prime} ; \theta, \alpha\right)\right) \\
        Q(s, a ; \theta, \alpha, \beta)=V(s ; \theta, \beta)+\left(A(s, a ; \theta, \alpha)-\frac{1}{|\mathcal{A}|} \sum_{a^{\prime}} A\left(s, a^{\prime} ; \theta, \alpha\right)\right)
        \end{array}$$

        <p align="center">
            <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200629/i3TMerDQbVmK.png?imageslim">
        </p>
        - 说明:
          - 上面部分是原来的网络
          - 下面部分，做了两个channel。上面一个 channel 是 value，下面的是 advantage。
        