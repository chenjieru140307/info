# blackjack 21 点

- 基于 gym 框架，搭建了 blackjack env。

## 实现


blackjack.py

```py
import gym
from gym import spaces
from gym.utils import seeding


def cmp(a, b):
    return int((a > b)) - int((a < b))


# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def draw_card(np_random):
    return np_random.choice(deck)


def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]


class BlackjackEnv(gym.Env):
    """Simple blackjack environment
    Blackjack is a card game where the goal is to obtain cards that sum to as
    near as possible to 21 without going over.  They're playing against a fixed
    dealer.
    Face cards (Jack, Queen, King) have point value 10.
    Aces can either count as 11 or 1, and it's called 'usable' at 11.
    This game is placed with an infinite deck (or with replacement).
    The game starts with each (player and dealer) having one face up and one
    face down card.
    The player can request additional cards (hit=1) until they decide to stop
    (stick=0) or exceed 21 (bust).
    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust the player wins.
    If neither player nor dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.  The reward for winning is +1,
    drawing is 0, and losing is -1.
    The observation of a 3-tuple of: the players current sum,
    the dealer's one showing card (1-10 where 1 is ace),
    and whether or not the player holds a usable ace (0 or 1).
    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto (1998).
    https://webdocs.cs.ualberta.ca/~sutton/book/the-book.html
    """

    def __init__(self, natural=False):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2)))
        self._seed()

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        # Start the first game
        self.reset()  # Number of
        self.nA = 2

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        if action:  # hit: add a card to players hand and return
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -1
            else:
                done = False
                reward = 0
        else:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1:
                reward = 1.5
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))

    def reset(self):
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)

        # Auto-draw another card if the score is less than 12
        while sum_hand(self.player) < 12:
            self.player.append(draw_card(self.np_random))

        return self._get_obs()
```

plotting.py

```py
import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


def plot_cost_to_go_mountain_car(env, estimator, num_tiles=20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')
    ax.set_title("Mountain \"Cost To Go\" Function")
    fig.colorbar(surf)
    plt.show()


def plot_value_function(V, title="Value Function"):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))


def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        plt.close(fig3)
    else:
        plt.show(fig3)

    return fig1, fig2, fig3
```



main1.py：

```py
## Monte Carlo Prediction

import matplotlib
from collections import defaultdict
from blackjack import BlackjackEnv
import plotting

# matplotlib.style.use('ggplot')

env = BlackjackEnv()


def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    returns_sum = defaultdict(float)  # G_t 的加和
    returns_count = defaultdict(float)

    # the final value function
    V = defaultdict(float)

    # 每个 episode 就是一个序列
    for i_episode in range(1, num_episodes + 1):
        episode = []
        state = env.reset()
        for t in range(100):
            action = policy(state)
            next_state, reward, done, info = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        # 将 episode 里的 state 全部拿出来。
        states_in_episode = set([tuple(x[0]) for x in episode])
        for state in states_in_episode:
            # 第一次在 episode 中遇到这个 state
            first_occurence_idx = next(i for i, x in enumerate(episode) if x[0] == state)
            # 计算这个 state 对应的 G_t
            G = sum([x[2] * (discount_factor ** i) for i, x in enumerate(episode[first_occurence_idx:])])

            returns_sum[state] += G
            returns_count[state] += 1.0
            V[state] = returns_sum[state] / returns_count[state]
    return V


# 根据 state 的一个简单的策略
# score 小于 20 就叫牌。
def sample_policy(state):
    score, dealer_score, usable_ace = state
    return 0 if score >= 20 else 1


vv = mc_prediction(sample_policy, env, num_episodes=10000)
plotting.plot_value_function(vv, title="10000 steps")
```

main2.py：

```py
import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict
from blackjack import BlackjackEnv
import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()


def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(state):
        probs = np.ones(nA, dtype=float) * epsilon / nA # 每个 action 的概率
        best_action = np.argmax(Q[state])
        probs[best_action] += (1.0 - epsilon) # 最好的 action 的概率修正下。
        return probs

    return policy_fn


def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # policy 使用嵌套函数的好处是，每次调用这个 policy ，这个 policy 都已经根据 Q 更新了。
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(1, num_episodes + 1):
        episode = []
        state = env.reset()
        for t in range(100):
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            first_occurence_idx = next(i for i, x in enumerate(episode) if x[0] == state and x[1] == action)
            G = sum([x[2] * (discount_factor ** i) for i, x in enumerate(episode[first_occurence_idx:])])
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]

    return Q, policy


Q, policy = mc_control_epsilon_greedy(env, num_episodes=500000, epsilon=0.1)

V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")
```

main3.py：

```py
import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd

from collections import defaultdict
from blackjack import BlackjackEnv
import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()


def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(state):
        probs = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[state])
        probs[best_action] += (1.0 - epsilon)
        return probs

    return policy_fn


def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        state = env.reset()
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        for t in itertools.count():
            next_state, reward, done, _ = env.step(action)
            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
            td_target = reward + discount_factor * Q[next_state][next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                break
            action = next_action
            state = next_state
    return Q


Q = sarsa(env, 500000)
print(Q)

V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
print(V)
plotting.plot_value_function(V, title="Optimal Value Function sarsa")
```

mian4.py

```py
import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict
from blackjack import BlackjackEnv
import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()


# 每个 state 概率相同
def create_random_policy(nA):
    probs = np.ones(nA, dtype=float) / nA

    def policy_fn(state):
        return probs

    return policy_fn

# 是 greedy policy ，如果 q 最大就是 1.0
def create_greedy_policy(Q):
    def policy_fn(state):
        probs = np.zeros_like(Q[state], dtype=float)
        best_action = np.argmax(Q[state])
        probs[best_action] = 1.0
        return probs

    return policy_fn


def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # cumulative denominator across all episodes
    C = defaultdict(lambda: np.zeros(env.action_space.n))

    target_policy = create_greedy_policy(Q)

    for i_episode in range(1, num_episodes + 1):
        episode = []
        state = env.reset()
        for t in range(100):
            probs = behavior_policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        G = 0.0
        W = 1.0

        for t in range(len(episode))[::-1]:
            state, action, reward = episode[t]
            G = discount_factor * G + reward
            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])

            if action != np.argmax(target_policy(state)):
                break
            W = W * 1. / behavior_policy(state)[action] # 1.0 是我们的 greedy的 policy

    return Q, target_policy


random_policy = create_random_policy(env.action_space.n)
Q, policy = mc_control_importance_sampling(env, num_episodes=500000, behavior_policy=random_policy)

V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")
```


如果使用 main1.py：

- 说明：
  - 策略为简单的策略。
- 输出：
  - 10000 steps：No Usable Ace：
    <p align="center">
        <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200627/AKkQVElBr6Pa.png?imageslim">
    </p>
  - 10000 steps：Usable Ace：
    <p align="center">
        <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200627/qaezIETdg3Qd.png?imageslim">
    </p>

如果使用 main2.py：

- 说明：
  - 策略为 mc_control_epsilon_greedy
- 输出：
  - Optimal Value Function：No Usable Ace：
    <p align="center">
        <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200628/4hvczEMz3Vh7.png?imageslim">
    </p>
  - Optimal Value Function：Usable Ace：
    <p align="center">
        <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200628/TOEVe8lUCDPi.png?imageslim">
    </p>

如果使用 main3.py：


- 说明：
  - 策略为 sarsa
- 输出：
  - Optimal Value Function：No Usable Ace：
    <p align="center">
        <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200628/RXAUITX9ftaj.png?imageslim">
    </p>
  - Optimal Value Function：Usable Ace：
    <p align="center">
        <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200628/wbvCS1ewzN8J.png?imageslim">
    </p>
- 疑问：
  - 为什么是这样的图像？为什么值超过了 1？是我使用的不对吗？


如果使用 main4.py：

- 说明：
  - 策略为 off policy mc_control_importance_sampling 即 
- 输出：
  - Optimal Value Function：No Usable Ace：
    <p align="center">
        <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200628/S1hhL8xS7mYY.png?imageslim">
    </p>
  - Optimal Value Function：Usable Ace：
    <p align="center">
        <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200628/g8A39A18azEM.png?imageslim">
    </p>