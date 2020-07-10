
# cartpole

使用 policy gradient


## policy gradient

（这部分后续合并到 policy gradient 里面）

$\tau=(s_0,a_0,\dots,s_{T-1},a_{T-1}, s_T)$ 为 s,a 的序列，共 $T$ 步 注意，最终的 $s_T$ 之后，环境被重启。

$R(s_t,a_t)$ 是收到的 reward ，在依据  $s_t$ 执行 $a_t$ 之后。同样定义 打折回报：$R(\tau) := \sum_{t=0}^{T-1}\gamma^t R(s_t,a_t)$. 我们的目标就是最大化这个打折回报的期望：

$$ \max_\theta\mathbb{E}_{\pi_\theta}R(\tau), $$

$\pi_\theta$ 是一个带参数的策略。（实际上是一个神经网络。）。由于这个期望值是在通过策略 $\pi_\theta$ 执行得到的  $\tau$ 中出现的，所以，解决这个问题等价于找到最优 的 参数  $\theta$ 来提供最好的策略，来最大化这个奖励的期望。


可以通过梯度下降来做：假设我们知道怎么求着参数的梯度： $\nabla_\theta\mathbb{E}_{\pi_\theta}R(\tau)$，然后我们可以这样更新参数 $\theta$ :

$$ \theta\leftarrow\theta+\alpha\nabla_\theta\mathbb{E}_{\pi_\theta}R(\tau), $$

这里 $\alpha$ 是学习速率超参数。

用 $P(\tau\vert\theta)$ 表示在策略 $\pi_\theta$ 下出现轨迹 $\tau$ 的概率，这样我们可以这样求解梯度：

$$\begin{aligned}\nabla_\theta\mathbb{E}_{\pi_\theta}R(\tau) &=  \nabla_\theta\sum_\tau P(\tau\vert\theta)R(\tau) 
\\ &= \sum_{\tau}\nabla_\theta P(\tau\vert\theta)R(\tau) 
\\ &= \sum_{\tau}\frac{P(\tau\vert\theta)}{P(\tau\vert\theta)}\nabla_\theta P(\tau\vert\theta)R(\tau) 
\\ &= \sum_{\tau}P(\tau\vert\theta)\nabla_\theta\log P(\tau\vert\theta)R(\tau) 
\\ &= \mathbb{E}_{\pi_\theta}\left(\nabla_\theta\log P(\tau\vert\theta)R(\tau)\right) 
\end{aligned}$$

说明：

- 第一行：根据期望的定义。
- 第一行到第二行：交换 sum 和梯度。
- 第二行到第三行：同时乘以和除以 $P(\tau\vert\theta)$
- 第三行到第四行：由于 $\nabla_x\log(f(x))=\dfrac{\nabla_x f(x)}{f(x)}$，所以进行转换。
- 第四行到第五行：由期望的定义。

这时候，我们可以把 轨迹 $\tau$ 出现的概率展开：

$$P(\tau\vert\theta)=p(s_0)\prod_{t=0}^{T-1}p(s_{t+1}\vert s_t,a_t)\pi_\theta(a_t\vert s_t), $$


说明：

- $p(s_{t+1}\vert s_t,a_t)$ 是，在 $s_t$ 状态执行 $a_t$ 动作时，状态转化为 $s_{t+1}$ 的概率。
- $p(s_0)$ 是状态的开始的分布。

对一个轨迹的 $P(\tau\vert\theta)$ 的 $\log$ 进行求导：

$$\begin{aligned}\nabla_\theta\log P(\tau\vert\theta) &= 
\nabla_\theta\left(\log p(s_0)+\sum_{t=0}^{T-1}\left(\log p(s_{t+1}\vert s_t,a_t)+\log\pi_\theta(a_t\vert s_t)\right)\right) 
\\ &= \sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\vert s_t).
\end{aligned}$$

注意到，做了对 $\theta$ 求导之后，动态的模型 $p(s_{t+1}\vert s_t,a_t)$ 被消除了，这也就是，policy gradient 是一个 model-free 的方法。 

将上面的求导带入 $\nabla_\theta\mathbb{E}_{\pi_\theta}R(\tau)$，得到策略梯度的表达式：

$$\tag{1} \nabla_\theta\mathbb{E}_{\pi_\theta} R(\tau) = \mathbb{E}_{\pi_\theta}\left(\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\vert s_t)R(\tau)\right), $$

因为它是一个期望，所以，可以用 Monte Carlo 对轨迹进行采样来估计，也就是：（英文原文：because it is an expectation it can be estimated by Monte Carlo sampling of trajectories:）

$$ \nabla_\theta\mathbb{E}_{\pi_\theta} R(\tau)\approx\frac{1}{L}\sum_{\tau}\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\vert s_t)R(\tau), $$

这里：

- $L$ 是用来作为一次梯度更新的轨迹的个数。

等式（1）是策略梯度更新的基本表达式，该更新将观察到的奖励与 使用 用来得到这些奖励的策略 的概率相关联。但是，如果您查看它并考虑整个轨迹 $R(\tau)$ 的奖励的作用，您会发现一些奇怪的现象：在一条轨迹中，它作为 一个 grad-log-prob的乘积的综合，也就是说，整个 episode 中得到的 reward 被用来改变 在这个 episode 过程中，每个动作发生的概率。这样的话，即使某个 a 在执行前 的一个 reward 也会影响这个 a。但是呢，其实我们只想，将 a 执行后的一些 reward 用来更新这个 a。

我们将策略梯度写成如下：

$$\nabla_\theta\mathbb{E}_{\pi_\theta}R(\tau) = \mathbb{E}_{\pi_\theta}\left(\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\vert s_t)\sum_{t^\prime=t}^{T-1}\gamma^{t^\prime-t}R(s_{t^\prime},a_{t^\prime})\right), $$

这样，a 被强制限定为 只会被执行 a 之后的 rewards 影响。

这也是一种可以降低策略梯度方法的 方差的一种方法。


（好像下面的代码并不是对应这种 修改后的方法的）



## 实现







说明：

- 一般 $\pi_\theta$ 是神经网络。$\theta$ 表示权重。在这里，我们使用逻辑回归来参数化 左右移动的概率。
- $x$ 表示  length 4 observation vector
- 使用 $\pi_\theta(0\vert x)=\dfrac{1}{1+e^{-\theta\cdot x}}=\dfrac{e^{\theta\cdot x}}{1+e^{\theta\cdot x}}$ 来表示 action 0 的概率。即将 cart 向左移。
- $\pi_\theta(1\vert x)=1-\pi_\theta(0\vert x)=\dfrac{1}{1+e^{\theta\cdot x}}$
- 因此我们的策略使用的参数就是一个 长度为4 的 向量 $\theta$ 
- 我们先把 $\nabla_\theta\log\pi_\theta(a\vert x)$ 推导出来:
    $$\begin{aligned}\nabla_\theta\log\pi_\theta(0\vert x)  &= \nabla_\theta\left(\theta\cdot x-\log(1+e^{\theta\cdot x})\right) \\ &= x - \frac{xe^{\theta\cdot x}}{1+e^{\theta\cdot x}} \\ &= x - x\pi_\theta(0\vert x) \end{aligned}$$
    $$\begin{aligned}\nabla_\theta\log\pi_\theta(1\vert x) &= \nabla_\theta\left(-\log(1+e^{\theta\cdot x})\right) \\ &= -\frac{xe^{\theta\cdot x}}{1+e^{\theta\cdot x}} \\ &= -x\pi_\theta(0\vert x).\end{aligned}$$


代码如下：


```py
import numpy as np

import gym
from gym.wrappers.monitor import Monitor, load_results  # 用于保存和加载 训练后的 policy

# 为了可重复性
GLOBAL_SEED = 0
np.random.seed(GLOBAL_SEED)

# 环境
env = gym.make('CartPole-v0')

print(env.observation_space)
print(env.action_space)


# 策略逻辑
class LogisticPolicy:

    def __init__(self, theta, alpha, gamma):
        self.theta = theta  # 策略的参数 $\theta$
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # discount 因子

    def logistic(self, y):
        return 1 / (1 + np.exp(-y))

    def probs(self, x):
        # 返回两个 action 的概率。
        y = x @ self.theta
        prob0 = self.logistic(y)
        return np.array([prob0, 1 - prob0])

    def act(self, x):
        # 根据 action 的概率选择一个 action
        probs = self.probs(x)
        action = np.random.choice([0, 1], p=probs)
        return action, probs[action]

    def grad_log_p(self, x):
        # 计算 grad-log-probs $\nabla_{\theta} \log \pi_{\theta}\left(a_{t} \mid s_{t}\right)$
        y = x @ self.theta
        grad_log_p0 = x - x * self.logistic(y)
        grad_log_p1 = - x * self.logistic(y)
        return grad_log_p0, grad_log_p1

    def discount_rewards(self, rewards):
        # 计算打折的 rewards
        # calculate temporally adjusted, discounted rewards
        discounted_rewards = np.zeros(len(rewards))
        cumulative_rewards = 0
        for i in reversed(range(0, len(rewards))):
            cumulative_rewards = cumulative_rewards * self.gamma + rewards[i]
            discounted_rewards[i] = cumulative_rewards
        return discounted_rewards

    def grad_log_p_dot_rewards(self, grad_log_p, actions, discounted_rewards):
        # dot grads with future rewards for each action in episode
        return grad_log_p.T @ discounted_rewards

    def update(self, rewards, obs, actions):
        # 计算出一系列的 s,a pair 中， 执行某个 a 后的梯度。
        grad_log_p = np.array([self.grad_log_p(ob)[action] for ob, action
                               in zip(obs, actions)])
        assert grad_log_p.shape == (len(obs), 4)

        # 计算 打折的 rewards。
        # calculate temporaly adjusted, discounted rewards
        discounted_rewards = self.discount_rewards(rewards)

        # 将梯度乘以打折的 rewards，得到我们的策略梯度。
        dot = self.grad_log_p_dot_rewards(grad_log_p, actions, discounted_rewards)

        # 使用梯度更新参数 theta
        self.theta += self.alpha * dot


# 运行一个 episode
def run_episode(env, policy, render=False):
    observation = env.reset()
    totalreward = 0

    observations = []
    actions = []
    rewards = []
    probs = []

    done = False

    while not done:
        if render:
            env.render()

        observations.append(observation)

        action, prob = policy.act(observation)
        observation, reward, done, info = env.step(action)

        totalreward += reward
        rewards.append(reward)
        actions.append(action)
        probs.append(prob)

    return totalreward, np.array(rewards), np.array(observations), np.array(actions), np.array(probs)


def train(theta, alpha, gamma, Policy, MAX_EPISODES=1000, seed=None, evaluate=False):
    # initialize environment and policy
    env = gym.make('CartPole-v0')
    if seed is not None:
        env.seed(seed)
    episode_rewards = []
    policy = Policy(theta, alpha, gamma)

    for i in range(MAX_EPISODES):
        total_reward, rewards, observations, actions, probs = run_episode(env, policy, render=True)
        episode_rewards.append(total_reward)
        policy.update(rewards, observations, actions)
        print("EP: " + str(i) + " Score: " + str(total_reward))

    # 评估最后的策略 100 次
    if evaluate:
        env = Monitor(env, 'pg_cartpole/', video_callable=False, force=True)
        for _ in range(100):
            run_episode(env, policy, render=True)
        env.env.close()
    return episode_rewards, policy


# 训练
episode_rewards, policy = train(theta=np.random.rand(4),
                                alpha=0.002,
                                gamma=0.99,
                                Policy=LogisticPolicy,
                                MAX_EPISODES=2000,
                                seed=GLOBAL_SEED,
                                evaluate=True)

import matplotlib.pyplot as plt

plt.plot(episode_rewards)
plt.show()

results = load_results('pg_cartpole')
plt.hist(results['episode_rewards'], bins=20)
plt.show()
```

输出：

```txt
Box(4,)
Discrete(2)
EP: 0 Score: 20.0
EP: 1 Score: 29.0
EP: 2 Score: 12.0
EP: 3 Score: 19.0
EP: 4 Score: 20.0
EP: 5 Score: 17.0
EP: 6 Score: 41.0
..略..
EP: 1993 Score: 200.0
EP: 1994 Score: 200.0
EP: 1995 Score: 200.0
EP: 1996 Score: 200.0
EP: 1997 Score: 200.0
EP: 1998 Score: 200.0
EP: 1999 Score: 200.0
```

图像：

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200710/FBGsd9oAR662.png?imageslim">
</p>


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200710/RJqEGrU4EIdu.png?imageslim">
</p>

疑问：

- 为什么在训练 2000 次后，还会有无法保持平衡的情况？
- 好像上面的代码并不是对应那种，以 a 之后的 reward 来进行计算的方法的。


说明：

- 在 500 episodes 后，好像可以保持平衡了。
- 上图也展示了 pg 方法的高方差。1000 次 episodes 之后，仍然不是很稳定。
  - 这个可以使用一些小街桥：，比如：
    - 每10 episode 更新一次 gradient
    - 可以让 learning rate 动态减小。
    - 可以多跑一次训练流程使用不同的 random seeds，来得到平均水平。这个在对比不同的算法效果的时候很常用。
- grad_log_p 中：
  - `y = x @ self.theta` 计算出线性 的 y
  - `grad_log_p0 = x - x*self.logistic(y)` 根据 $\nabla_\theta\log\pi_\theta(0\vert x)= x - x\pi_\theta(0\vert x)$ 计算出 $\nabla_\theta\log\pi_\theta(0\vert x)$
  - `grad_log_p1 = - x*self.logistic(y)` 根据  $\nabla_\theta\log\pi_\theta(1\vert x) = -x\pi_\theta(0\vert x)$ 计算出 $\nabla_\theta\log\pi_\theta(1\vert x)$
- `plt.hist(results['episode_rewards'], bins=20)` 对应的图二说明：最后的策略可以在 100 个 episodes 里全部都成功的控制 cart 200 steps。


注意：

- 这里，我们是每个 episode 更新一次参数，实际中，我们最好一个 batch 更新一次。来保证训练的稳定性。





减少方差，也就是增加稳定性的方法：

- 消减方差的方法可以分为两类：
  - 简单的：
    - 以后续的回报进行计算。
    - 每个batch 更新一次参数，而不是一个 episode 更新一次参数。
    - 引入折扣因子 $\gamma$
    - 对 reward 进行归一化，即：减去平均值，然后除以标准差。
  - 比较复杂的:
    - 引入基线。多种选择，经常使用 action-value 或者 advantage function 和 actor-critic 方法。
    - 广义的 advantage 估计。Generalized advantage estimation (GAE)。这种引入了一些超参数来 在策略梯度中引入一些偏差，来控制方差。([see the paper](https://arxiv.org/abs/1506.02438)). I recommend [this blog post](https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/) 
    - [TRPO](https://arxiv.org/abs/1502.05477) 和  [PPO](https://arxiv.org/abs/1707.06347) 对 Vanilla Policy Gradient 进行了修改，使得策略不会变化太快。
- GAE 的[论文](https://arxiv.org/abs/1506.02438)对常用策略梯度进行了总结：
  - 策略梯度方法，通过重复的估计梯度 $g:=\nabla_{\theta} \mathbb{E}\left[\sum_{t=0}^{\infty} r_{t}\right] .$ 来最大话期望的 total reward。通用表达式如下：
  $$
  g=\mathbb{E}\left[\sum_{t=0}^{\infty} \Psi_{t} \nabla_{\theta} \log \pi_{\theta}\left(a_{t} \mid s_{t}\right)\right]
  $$
  - 说明：
    - 这里的 $\Psi_{t}$ 可以是下列之一：
      - $\sum_{t=0}^{\infty} r_{t}:$ total reward of the trajectory.
      - $\sum_{t^{\prime}=t}^{\infty} r_{t^{\prime}}:$ reward following action $a_{t}$
      - $\sum_{t^{\prime}=t}^{\infty} r_{t^{\prime}}-b\left(s_{t}\right):$ baselined version of
      previous formula.
      - $Q^{\pi}\left(s_{t}, a_{t}\right):$ state-action value function.
        - $Q^{\pi}\left(s_{t}, a_{t}\right):=\mathbb{E}_{s_{t+1: \infty},a_{t+1: \infty}}\left[\sum_{\infty}^{\infty} r_{t+l}\right]$
      - $r_{t}+V^{\pi}\left(s_{t+1}\right)-V^{\pi}\left(s_{t}\right):$ TD residual.
        - $V^{\pi}\left(s_{t}\right):=\mathbb{E}_{s_{t+1: \infty},a_{t: \infty}}\left[\sum_{\infty}^{\infty} r_{t+l}\right]$
      - $A^{\pi}\left(s_{t}, a_{t}\right):$ advantage function.
        - $A^{\pi}\left(s_{t}, a_{t}\right):=Q^{\pi}\left(s_{t}, a_{t}\right)-V^{\pi}\left(s_{t}\right), \quad \text { (Advantage function) }$



对于 softmax 和 logistic 的一个问题的理解：


- 简单介绍：
  - 逻辑回归中，我们使用了参数化的 logistic 函数 $f_\theta(x)=\dfrac{e^{\theta\cdot x}}{1+e^{\theta\cdot x}}$。这里 $\theta, x$ 是 $k$-维 的权重和特征向量。我们把  $f_\theta(x)$ 解释为 $p(y=1\vert \theta,x)$,也就是第一个类别的概率。然后，我们通过概率和为1得到第二个类别的概率。
  - 参数化的 softmax 函数为： $\sigma_\theta(x)_i=\dfrac{e^{\theta_i\cdot x}}{\sum_{j=1}^{m}e^{\theta_j\cdot x}}$ 。这里 $i=1,\dots,m$ 是 logistic 函数 在 m 个输出类别的一般化形式，这样将 logistic 扩展到了多类别问题。我们把 $\sigma_\theta(x)_i$ 解释为 $p(y=i\vert \theta_i, x)$ for $i=1,\dots,m$. 注意，softmax 对应于每个类别有不同的权重向量 $\theta_i$。
- 有个问题：
  - 有一个潜在的不同，很少有人去解释，当你使用 softmax 替代 logistic 在两个类别的分类问题中时，本质上，logistic 函数值维护一个 $k$ 长度的参数向量$\theta$ 来为了估计第一个类别的概率。而 softmax 会对应每个类别维护一个向量参数，因此，会有 $2k$ 个参数，这意味着，softmax 公式有冗余的参数，这就是 过度参数化。
  - 详细如下：
    - 对于 m 类别问题，每个类别的概率为：
      $$ p(y=i\vert \theta_i,x) = \dfrac{e^{\theta_i\cdot x}}{\sum_{j=1}^{m}e^{\theta_j\cdot x}}, \text{ for $i=1,\dots,m$}. $$
    - 假定 $\phi$ 为某个固定向量，注意到，如果把 $\theta_i$ 替换为 $\theta_i-\phi$，概率变为：
      $$\begin{aligned}\dfrac{e^{(\theta_i-\phi)\cdot x}}{\sum_{j=1}^{m}e^{(\theta_j-\phi)\cdot x}} &= \dfrac{e^{-\phi\cdot x}e^{\theta_i\cdot x}}{e^{-\phi\cdot x}\sum_{j=1}^{m}e^{\theta_j\cdot x}} \\ &= p(y=i\vert \theta_i,x).\end{aligned}$$
    - 特别的，我们可以通过设定 $\phi=\theta_0$ 来使得第一个参数矢量为 0，这样可以消掉 $k$ 个冗余的参数，在二分类问题中这样做，可以把原始的 logistic 函数还原出来。
    - 即：
      - 设定 $\theta_0\to 0$ and $\theta_1\to\theta_1-\theta_0=:\theta$ 会使得 $p_0=\dfrac{1}{1+e^{\theta\cdot x}}$ 和 $p_1=\dfrac{e^{\theta\cdot x}}{1+e^{\theta\cdot x}}$ 变为一般化的 logistic 函数。
  - 注意：
    - 虽然，softmax 在 m 类别的问题中也可以这样做来消除一些参数，但是，最好不要，因为是一些参数为 0 会使得代码有点麻烦，而且，会使得计算梯度的时候有些麻烦。 For more discussion on the softmax function look [here](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/) and for more details concerning the overparametrization of softmax look [here](http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression).




效率优化：

- 对 discount_rewards 部分的代码的修改：
  - 原始的：
    - 上述代码中，discount_rewards 的计算，有个循环，遇到循环是可以考虑转化为矢量计算的。
        ```python
        def discount_rewards(rewards, gamma):
            discounted_rewards = np.zeros(len(rewards))
            cumulative_rewards = 0
            for i in reversed(range(0, len(rewards))):
                cumulative_rewards = cumulative_rewards * gamma + rewards[i]
                discounted_rewards[i] = cumulative_rewards
            return discounted_rewards
        ```
  - 改为矩阵计算：
    - 我们令 $\mathbf{r,\hat{r}}$ 为包含 original rewards 和 disounted rewards 的矢量，可以写出： $\mathbf{\hat{r}}=\mathbf{\Gamma r}$，这里：
        $$\mathbf{\Gamma} = \begin{bmatrix}1 & \gamma & \gamma^2 & \cdots & \gamma^{n-1} \\ 0 & 1 & \gamma & \cdots & \gamma^{n-2} \\ \vdots & & \ddots & \\ 0 & 0 & \cdots & 1 & \gamma \\ 0 & 0 & 0 & \cdots & 1 \end{bmatrix}.$$
    - 这个矩阵即是：[Toeplitz matrix](https://en.wikipedia.org/wiki/Toeplitz_matrix) (可以在 [Wolfram Alpha](https://www.wolframalpha.com/input/?i=[[1,a,a^2],[0,1,a],[0,0,1]]) 中输入一个矩阵的例子，然后它会告诉你这个矩阵的信息。
    - Scipy 有一个函数 `scipy.linalg.toeplitz` 可以用来构建这种矩阵，
    - 重写代码如下：
        ```python
        import scipy as sp
        import scipy.linalg

        def toeplitz_discount_rewards(rewards, gamma):
            n = len(rewards)
            c = np.zeros_like(rewards)
            c[0] = 1

            r = np.array([gamma**i for i in range(n)])
            matrix = sp.linalg.toeplitz(c, r)
            discounted_rewards = matrix @ rewards
            return discounted_rewards
        ```
    - 实际运行，统计时间，结果，这种改写反而更加耗费时间。。
    - 因为，在构建矩阵时，会有一些时间消耗，所以，原来的代码效率反而比这个高些。
  - 使用 `scipy.signal`
    - 实现如下：
        ```python
        import scipy.signal

        def magic_discount_rewards(rewards, gamma):
            return sp.signal.lfilter([1], [1, float(-gamma)], rewards[::-1], axis=0)[::-1]
        ```
    - 说明：
      - 使用了 scipy 的 signal processing 类库，使用一个数字滤波器在 1D 的数字序列上。[see the docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html)
      - 是原始的实现方式的 1/3 时间。非常快。当这种类型的计算经常出现时，使用这个是很好的。
