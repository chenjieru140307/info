# 策略梯度学习

- Policy Gradient
- Actor Critic
- AlphaGo Zero 
- Continuous Mountain Car代码实战


Q-learnging 的问题：

- 要学到某个状态下，做那些 action 回报比较高，那些 action 回报比较低。

增强学习的分类：

- Value based
  - 值函数
  - Q值函数数 
    - 对于 Q-learning 来说，值函数使用的神经网络来进行拟合。
- Policy Based 
  - 不需要值函数
  - 直接优化 Policy
    - P(a|s)
- Actor Critic
  - 学习值函数
  - 学习学习Policy
    - 是上面两种方法的合体



为什么想要避免训练 固定的 policy：

- 举例：
  - 如图：
    <p align="center">
        <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200629/xv4d7glcb9fw.png?imageslim">
    </p>
  - 说明：
    - 在上图中寻找财宝，如果寻找财宝的agent 无法区分两个灰色的格子，那么，如果策略是，遇到灰色的格子，往左走，结果，它就只能被困在左边的格子。
    - 所以，确定性策略的探索能力有点有限。


- The agent cannot differentiate the grey states Consider features of the following form (for all $N, E, S, W)$
    $$
    \phi(s, a)=\mathbf{1}(\text { wall to } N, a=\text { move } E)
    $$
- Compare value-based $\mathrm{RL}$, using an approximate value function
    $$
    Q_{\theta}(s, a)=f(\phi(s, a), \theta)
    $$
- To policy-based RL, using a parametrised policy
    $$
    \pi_{\theta}(s, a)=g(\phi(s, a), \theta)
    $$


对于Policy Network

- 在 Q-learning 中，我们是选择的，让 Q 值最大的这个策略，但是在 Policy Network 中，我们直接学习策略。
- 不需要优化 $Q$ 值函数，直接优化策路函数 $\pi$
$$
a=\pi(a \mid s, \mathbf{u}) \text { or } a=\pi(s, \mathbf{u})
$$
- 我们希望优化的是：expected reward：这个回报越大越好。
$$
L(\mathbf{u})=\mathbb{E}\left[r_{1}+\gamma r_{2}+\gamma^{2} r_{3}+\ldots \mid \pi(\cdot, \mathbf{u})\right]
$$

- 直接用SGD做优化




回如何确定一个策略 $\pi_{\theta}(s,a)$ 和参数 $\theta$ 如何找到最佳的 $\theta$

- 如何确定一个策略 $\pi_\theta$ 的好坏？
  - 我们可以优化一步动作下如回报（怎么优化一步动作下的回报？）
  - 可以 sample 一个 trajectory 然后优化整个回报
- 直接用SGD做优化


如果


- 如果我们只考虑一步 MDP 
  - 初始状态  $s\sim \mathrm{d}(\mathrm{s})$
  - 我们考虑一步就结束： $r=\mathcal{R}_{s, a}$
  - 那么，总的回报为：
    $$
    \begin{aligned}
    J(\theta) &=\mathbb{E}_{\pi_{\theta}}[r] \\
    &=\sum_{s \in \mathcal{S}} d(s) \sum_{a \in \mathcal{A}} \pi_{\theta}(s, a) \mathcal{R}_{s, a} \\
    \end{aligned}
    $$
  - 说明：
    - $d(s)$ 为所有的 $s$ 的情况，$\sum_{a \in \mathcal{A}} \pi_{\theta}(s, a) \mathcal{R}_{s, a}$ 为，某种 $s$ 情况下，使用策略 $\pi_{\theta}(s, a)$ 得到一个 $a$ 的 reward $\mathcal{R}_{s, a}$ 的合集。 即 所有的回报。
  - 这个时候，对于 $\theta$ 求偏导：
    $$
    \begin{aligned}
    \nabla_{\theta} J(\theta) &=\sum_{s \in \mathcal{S}} d(s) \sum_{a \in \mathcal{A}} \pi_{\theta}(s, a) \nabla_{\theta} \log \pi_{\theta}(s, a) \mathcal{R}_{s, a} \\
    &=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) r\right]
    \end{aligned}
    $$
  - 说明：
    - 第一行：$\begin{aligned}\nabla_{\theta} \pi_{\theta}(s, a) &=\pi_{\theta}(s, a) \frac{\nabla_{\theta} \pi_{\theta}(s, a)}{\pi_{\theta}(s, a)} \\&=\pi_{\theta}(s, a) \nabla_{\theta} \log \pi_{\theta}(s, a)\end{aligned}$ 因为 $d(\log_x)=\frac{dx}{x}$
    - 第一行到第二行：由于 $\mathbb{E}_{\pi_{\theta}}[r]    =\sum_{s \in \mathcal{S}} d(s) \sum_{a \in \mathcal{A}} \pi_{\theta}(s, a) \mathcal{R}_{s, a}$ 所以，可以将除了 $\nabla_{\theta} \log \pi_{\theta}(s, a)$ 的部分还原回去。
    - 第二行：含义为，对 $J(\theta)$ 的求导，等于对 在 $\pi_\theta$ 的策略下执行动作时，的 $\nabla_{\theta} \log \pi_{\theta}(s, a) r$ 回报。
      - 这个的好处是，如果你知道策略 $\pi_{\theta}$，那么，$\nabla_{\theta} \log \pi_{\theta}(s, a)$ 是可以求解出来的。
      - 而 $r$ 是可以 sample 出来的。就是，比如，我按照策略 $\pi_\theta$ 走，走 100 次，每次按照策略可能是 10% 往左走，90% 往右走，都可以，在实验的过程中知道，往左走，回报是 1，往右走回报是2，那么 $r$ 就是 $0.1*1+0.9*2$。
- 如果，我们考虑一整个trajectory
  - 我们使用 $\tau$ 来表示一条轨迹 $s_{0}, a_{0}, \ldots, s_{H}, a_{H} .$ 
  - 我们把整条轨迹的回报记为: $R(\tau)=\sum_{t=0}^{H} R\left(s_{t}, a_{t}\right)$
    - 疑问：为什么这个地方没有用 $\gamma$？如果这个地方没有 $\gamma$ ，那么，后面流程中的 $v_t$ 为什么又有 $\gamma$ 了？
  - 这个时候，我们的优化目标为 $J(\theta)$：
    $$
    J(\theta)=\mathrm{E}\left[\sum_{t=0}^{H} R\left(s_{t}, a_{t}\right) ; \pi_{\theta}\right]=\sum_{\tau} P(\tau ; \theta) R(\tau)
    $$
    - 说明：
      - $\mathrm{E}\left[\sum_{t=0}^{H} R\left(s_{t}, a_{t}\right) ; \pi_{\theta}\right]$ 即：我在 $\pi_\theta$ 策略下的时候，一条轨迹的总回报的期望。
      - $\sum_{\tau} P(\tau ; \theta) R(\tau)$ 即：我把每条轨迹的：轨迹在 $\theta$ 下出现的概率乘以 轨迹的回报，加起来。即 轨迹的回报的加权平均。
  - 那么，我们要找的就是，令 $J(\theta)$ 最大的 $\theta$：
    $$
    \max _{\theta} J(\theta)=\max _{\theta} \sum_{\tau} P(\tau ; \theta) R(\tau)
    $$
  - 这时候，我们可以对 $\theta$ 求偏导：
    $$
    \begin{aligned}
    \nabla_\theta J(\theta) &=\sum_{\tau}P(\tau;\theta)\nabla_\theta \log P(\tau;\theta)R(\tau)\\&\approx\frac{1}{m} \sum_{i=1}^{m} \nabla_{\theta} \log P\left(\tau^{(i)} ; \theta\right) R\left(\tau^{(i)}\right)\\ & =\hat{g}
    \end{aligned}
    $$
  - 说明：
    - 第一行：$d(\log_x)=\frac{dx}{x}$
    - 第一行到第二行：由于我们不知道 $P(\tau;\theta)$，但是，我们可以使用采样的方式得到 $m$ 个 $\tau$ 这样的 $m$ 个 $\nabla_{\theta} \log P\left(\tau^{(i)} ; \theta\right) R\left(\tau^{(i)}\right)$ 相加求平均，就等于 $\nabla_{\theta} \log P(\tau ; \theta) R(\tau)$ 的加权和了。
  - 我们将 $\nabla_{\theta} \log P\left(\tau^{(i)} ; \theta\right)$ 展开：
    $$\begin{aligned}
    \nabla_{\theta} \log P\left(\tau^{(i)} ; \theta\right) &=\nabla_{\theta} \log [\prod_{t=0}^{H} \underbrace{P\left(s_{t+1}^{(i)} \mid s_{t}^{(i)}, a_{t}^{(i)}\right)}_{\text {dynamics model }} \cdot \underbrace{\pi_{\theta}\left(a_{t}^{(i)} \mid s_{t}^{(i)}\right)}_{\text {policy }}] \\
    &=\nabla_{\theta}\left[\sum_{t=0}^{H} \log P\left(s_{t+1}^{(i)} \mid s_{t}^{(i)}, a_{t}^{(i)}\right)+\sum_{t=0}^{H} \log \pi_{\theta}\left(a_{t}^{(i)} \mid s_{t}^{(i)}\right)\right] \\
    &=\nabla_{\theta} \sum_{t=0}^{H} \log \pi_{\theta}\left(a_{t}^{(i)} \mid s_{t}^{(i)}\right)
    \end{aligned}$$
  - 说明：
    - 第一行：将 $P\left(\tau^{(i)} ; \theta\right)$ 写成连乘的形式。
      - $P\left(s_{t+1}^{(i)} \mid s_{t}^{(i)}, a_{t}^{(i)}\right)$ 为 在 $s_t$ 状态和 $a_t$ 行为下，环境转化为 $s_{t+1}$ 的概率
      - $\pi_{\theta}\left(a_{t}^{(i)} \mid s_{t}^{(i)}\right)$ 为 在 $s_t$ 状态下执行 $a_t$ 行为的概率。
    - 第一行到第二行：log 将连乘转化为 连加。
    - 第二行到第三行：由于 $P(s_{t+1}^{(i)} \mid s_{t}^{(i)}, a_{t}^{(i)})$ 是环境内在的，与策略的 $\theta$ 无关，所以，直接去掉。
  - 则，$\hat{g}$ 为：
    $$
    \begin{aligned}
    \hat{g}=&\frac{1}{m} \sum_{i=1}^{m} \nabla_{\theta} \log P\left(\tau^{(i)} ; \theta\right) R\left(\tau^{(i)}\right)\\ =&\frac{1}{m} \sum_{i=1}^{m} \left(\nabla_{\theta} \sum_{t=0}^{H} \log \pi_{\theta}\left(a_{t}^{(i)} \mid s_{t}^{(i)}\right) \right)R\left(\tau^{(i)}\right)\\ =&\frac{1}{m} \sum_{i=1}^{m} \sum_{t=0}^{H} \nabla_{\theta}  \log \pi_{\theta}\left(a_{t}^{(i)} \mid s_{t}^{(i)}\right) R\left(\tau^{(i)}\right)   
    \end{aligned}
    $$
  - 说明：
    - 第三行：由于 $\hat{g}$ 为近似的 $\theta$ 的偏导，那么，在梯度下降中，$\theta$ 可以按照如下进行更新：$\theta \leftarrow \theta+\alpha \hat{g}$
      - 此时，将 $\hat{g}$ 拆分：按照第三行：为，对每条轨迹的中，每个步骤的log 偏导的和，的加和。那么，对于每条轨迹，$\theta$ 可以如下更新：$\theta \leftarrow \theta+\sum_{t=0}^{H} \nabla_{\theta} \log \pi_{\theta}\left(a_{t}^{(i)} \mid s_{t}^{(i)}\right) R\left(\tau^{(i)}\right)$
      - 此时，继续拆分，可以将 $\sum_{t=0}^{H} \nabla_{\theta} \log \pi_{\theta}\left(a_{t}^{(i)} \mid s_{t}^{(i)}\right)$ 看做，对于轨迹中的每个步骤的 log 偏导的和，那么上面的更新中的求和拆分成循环：
        - $\text { for } t =1 \text { to } T-1 \text { do }$
        - $\quad$ $\theta \leftarrow \theta+\alpha \nabla_{\theta} \log \pi_{\theta}\left(s_{t}, a_{t}\right) v_{t}$
  - 则按照这种方式更新 $\theta$ 如下：
    - 函数：
      - 01 function REINFORCE 
      - 02 $\quad$ Initialise $\theta$ arbitrarily 
      - 03 $\quad$ for each episode $\left\{s_{1}, a_{1}, r_{2}, \ldots, s_{T-1}, a_{T-1}, r_{T}\right\} \sim \pi_{\theta}$ do
      - 04 $\quad$ $\quad$ $\text { for } t =1 \text { to } T-1 \text { do }$
      - 05 $\quad$ $\quad$ $\quad$ $\theta \leftarrow \theta+\alpha \nabla_{\theta} \log \pi_{\theta}\left(s_{t}, a_{t}\right) v_{t}$
      - 06 $\quad$ $\quad$ end for
      - 07 $\quad$ end for
      - 08 $\quad$ return $\theta$
      - 09 end function
    - 其中：
      - $v_1=R_2+\gamma R_3+\ldots$
      - $v_2=R_3+\gamma R_4+\ldots$
    - 疑问：
      - 为什么 这个地方是 $v_t$ 不是 $R\left(\tau^{(i)}\right)$？
      - 为什么上面的 $R$ 是从 $R_2$ 开始的？$R_0$ 和 $R_1$ 呢？
  - 总结：
  - Policy $\pi(\theta)$ 是一个神经网络
    - 用初始状态的回报作为优化的目标
        $$
        V_{\pi(\theta)}=\mathbb{E}_{\pi(\theta)}\left[r_{0}+\gamma r_{1}+\gamma^{2} r_{2}+\cdots\right]
        $$
    - Gradient ascent
        $$
        \theta_{t+1}=\theta_{t}+\alpha \nabla J\left(\theta_{t}\right)
        $$
    - 利用 policy gradient 拟合 gradient
        $$
        \nabla J(\theta)=\mathbb{E}_{\pi}\left[\gamma^{t} R_{t} \nabla_{\theta} \log \pi\left(a \mid s_{t}, \theta\right)\right]
        $$

Actor-critic：

- 上面的方法的问题：
  - Variance 非常大。
  - 怎么解决：
    - 可以借鉴 TD 中的，按照现在的策略运行一步，得到值函数，加上我现在的 reward，作为我总的值函数。（没有很清楚，看看）
  - Actor-critic：
    - 训练两个网络
      - Actor: 策略(policy)网络，选择下一个动作 
      - Critic: 评估 $Q(s,a)$ 的近似值，相当于策略评估
  - 流程：
    - 令：
      - $Q_{w}(s, a)=\phi(s, a)^{\top} w$
    - 函数：
      - 01 function QAC
      - 02 $\quad$ Initialise $s, \theta$ 
      - 03 $\quad$ Sample $a \sim \pi_{\theta}$ 
      - 04 $\quad$ for each step do 
      - 05 $\quad$ $\quad$ Sample reward $r=\mathcal{R}_{s}^{a} ;$ sample transition $s^{\prime} \sim \mathcal{P}_{s,}^{a}$
      - 06 $\quad$ $\quad$ Sample action $a^{\prime} \sim \pi_{\theta}\left(s^{\prime}, a^{\prime}\right)$
      - 07 $\quad$ $\quad$ $\delta=r+\gamma Q_{w}\left(s^{\prime}, a^{\prime}\right)-Q_{w}(s, a)$
      - 08 $\quad$ $\quad$ $\theta=\theta+\alpha \nabla_{\theta} \log \pi_{\theta}(s, a) Q_{w}(s, a)$
      - 09 $\quad$ $\quad$ $w \leftarrow w+\beta \delta \phi(s, a)$
      - 10 $\quad$ $\quad$ $a \leftarrow a^{\prime}, s \leftarrow s^{\prime}$
      - 11 $\quad$ end for 
      - 12 end function
  - 说明：
    - $Q_{w}(s, a)$ 即 Critic 网络，我们把它写成 $\phi(s, a)^{\top}$ 与 $w$ 相乘的线性模式。其中 $\phi(s, a)^{\top}$ 表示对 (s,a) 抽取特征为 $(x_1,x_2,\ldots,x_n)$，即 $\phi$ 是一个抽取特征的函数。
    - 03 行：我们基于策略 $\pi_\theta$ 抽样得到了一个 $a$
    - 05 行，我们执行这个 $a$ 后，得到了回报 $r=\mathcal{R}_{s}^{a}$，同时，环境变为了 $s^{\prime}$
    - 06 行，这时候，我们在 $s^{\prime}$ 的环境下，基于 $\pi_\theta$ 策略抽样得到了下一步的 $a^{\prime}$
    - 07 行：我们用 $r+\gamma Q_{w}\left(s^{\prime}, a^{\prime}\right)$ 与 $Q_{w}(s, a)$ 之间的差值作为 $\delta$，实际上，我们想要这个 $\delta$ 越小越好。
    - 08 行：这个地方我们对 $\theta$ 进行更新，在上面的 REINFORCE 的05 行，其中的 $v(t)$ 换成了这个地方的 $Q_{w}(s, a)$。
      - 为什么可以换？下面的 Compatible Function Approximation 定理说明了为什么可以替换。
    - 09 行：为什么 $\delta$ 可以放在这里？
      - 因为，如果我们算的是，真实的 $Q$ 与我们的当前的 $Q$ 的均方误差，而且，我们令 $r+\gamma Q_{w}\left(s^{\prime}, a^{\prime}\right)$ 近似于 真实的 Q，那么这个均方误差为：$\left(r+\gamma Q_{w}\left(s^{\prime}, a^{\prime}\right)-Q_{w}(s, a)\right)^2$，我们想令这个误差最小化，所以，对 w 求偏导，得：$2*\left(r+\gamma Q_{w}\left(s^{\prime}, a^{\prime}\right)-Q_{w}(s, a)\right)*\left(\phi(s,a)\right)$，因为 $r+\gamma Q_{w}\left(s^{\prime}, a^{\prime}\right)$ 我们认为是真实的 $Q$，所以，不对其中的 $w$ 求导。此时，这个式子就是 $2\delta \phi(s,a)$
  - Compatible Function Approximation 这是一个定理：
    - 定理
      - 如果 Value Function 与 policy 是 compatible 的：（这个是什么意思？为什么可以假设相等？）
        $$
        \nabla_{w} Q_{w}(s, a)=\nabla_{\theta} \log \pi_{\theta}(s, a)
        $$
      - 而且，我们的 value function 是最小化 MSE
        $$
        \varepsilon=\mathbb{E}_{\pi_{\theta}}\left[\left(Q^{\pi_{\theta}}(s, a)-Q_{w}(s, a)\right)^{2}\right]
        $$
        - 说明：
          - $Q^{\pi_{\theta}}(s, a)$ 就是我们的策略对应的 Q，也就是最好的 Q。
      - 那么，我们就可以用它来做 policy gradient
        $$
        \nabla_{\theta} J(\theta)=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) Q_{w}(s, a)\right]
        $$
    - 证明：    
      - 由于：
        $$\begin{aligned}
        \nabla_{w} \varepsilon &=0 \\
        \mathbb{E}_{\pi_{\theta}}\left[\left(Q^{\pi_{\theta}}(s, a)-Q_{w}(s, a)\right) \nabla_{w} Q_{w}(s, a)\right] &=0 \\
        \mathbb{E}_{\pi_{\theta}}\left[\left(Q^{\pi_{\theta}}(s, a)-Q_{w}(s, a)\right) \nabla_{\theta} \log \pi_{\theta}(s, a)\right] &=0 \\
        \mathbb{E}_{\pi_{\theta}}\left[Q^{\pi_{\theta}}(s, a) \nabla_{\theta} \log \pi_{\theta}(s, a)\right] &=\mathbb{E}_{\pi_{\theta}}\left[Q_{w}(s, a) \nabla_{\theta} \log \pi_{\theta}(s, a)\right]
        \end{aligned}$$
      - 则：
        $$
        \begin{aligned}
        \nabla_{\theta} J(\theta)=&\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) Q^{\pi_{\theta}}(s, a) \right]
        \\=&\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) Q_{w}(s, a)\right]
        \end{aligned}
        $$