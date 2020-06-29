# gym

介绍：

- OpenAI gym是一个用于开发和比较RL算法的工具包, 基准测试平台
- 文档: https://gym.openai.com/docs/
- gym开源库： 包含一个测试问题集，每个问题为一个环境env, 环境有共享的接口，
- 公许用户设计通用的算法。
- Openai gym服务：提供站点和API允许用户对训续的算法进行性能比较。
- 目前支持 python, tensorflow, theano
- gym 的核心接口是 Env， 包含几个核心方法如下：
  - reset(self):重置环境的状态，返回观察。
  - step(self, action):推进一个时间步长，返回 observation, reward, done, info
  - render(self, mode='human', close=False):重绘环境的一坝。


举例：

```py
import gym


env = gym.make("CartPole-v0")

n_episode = 20
for i in range(n_episode):
    observation = env.reset()
    for t in range(1000):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
```

输出：

```txt
[-0.00234875  0.0222368  -0.00262931 -0.03965835]
...略...
[ 0.07920393  0.77285071 -0.20510276 -1.52091507]
Episode finished after 15 timesteps
...略...
Episode finished after 38 timesteps
```

图像：

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200627/5CvO8pMXYBVV.png?imageslim">
</p>

说明：

- `env.action_space.sample()` 随机抽取的一个 action。