# TRFL

松露传送门：


https://github.com/deepmind/trfl/




今天，DeepMind开源了一个基于TensorFlow的**强化学习库**，名字叫**TRFL**。

思路是**模块化**，强调灵活度：如果把造**智能体**想象成搭积木，许多关键的、常用的木块都在这里集合了：

比如，**DQN** (深度Q网络) 、**DDPG** (深度确定策略梯度)，以及**IMPALA** (重要性加权演员学习者架构) ，都是DeepMind功勋卓著的组件。

库里面的组件，虽然来源各不相同，但都经过严密测试，因而相对可靠；并且只要一个**API**，对开发者比较友好。

DeepMind团队自身做研究，也***\*严重依赖\****这个库。

# 为了那些难以发觉的Bug

这个库，写作TRFL，读作“Truffle”。翻译成中文叫“**松露**”。

那么，松露为何而生？

![640?wx_fmt=jpeg](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_jpg/YicUhk5aAGtBs3a3XcLO9BPDmQ1XNczsuMPqLhkjKhwwucOU90BQCfjHCC9Qwptl5BI2sBJUdoKRrrUweMIeAmg/640?wx_fmt=jpeg)

## 交互Bug很隐秘

深度强化学习智能体，里面常常包含大量的**交互组件**：

至少要有**环境**，加上**价值网络**或者**策略网络**；

通常，还会有环境学习模型 (Learned Model) 、伪奖励函数 (Pseudo-Reward Functions) 、或者重播系统 (Replay System) 这样的部分。

可是，交互组件到底用什么方式交互？论文里一般没有细致的讨论，***\*有bug也很难发现\****。

![640?wx_fmt=png](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtBs3a3XcLO9BPDmQ1XNczsuPguPYuYykZJbNNh7nZ0j7IG8dItIPKeQELroGW7CDicvnN0mvKda0Pg/640?wx_fmt=png)

为此，OpenAI写过一篇博客，研究了10个热门的强化学习智能体，发现6个都有隐藏bug。

虽然，用一个开源的、完整的智能体，对复现研究成果是有帮助的，但灵活度不够，要**修改**就很难了。

所以，才有了松露。

## 损失函数模块化

深度强化学习 (DRL) ，依赖**价值网络**或**策略网络**的不断更新。

DeepMind团队发现，比起传统的RL更新，损失函数更加**模块化**，更容易结合到监督/无监督的目标里去。

松露里包含了许多**损失函数**和**运算**，全部在纯TensorFlow里实现。

不是完整算法，但是各自经过严密测试，可以用来搭成完整的智能体。

![640?wx_fmt=jpeg](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_jpg/YicUhk5aAGtBs3a3XcLO9BPDmQ1XNczsuxmr8hnxqxbv7Yyh4MwYVzxaeGRnsxyr743wK508DvIylapNWyFqq9g/640?wx_fmt=jpeg)

并且，只要**一个API**来解决各种核心组件，即便各自来源是天南地北，也很容易互相组合。

# 松露，营养很丰富

松露里的许多函数和运算，既可以用在经典RL算法里，也可以用在尖端技术上。

## 基于价值

针对基于价值的强化学习，松露提供了各种TensorFlow运算，用于在离散动作空间 (Discrete Action Spaces) 里学习：时间差分法，Sarsa，Q学习，以及它们的变体。还有**连续控制算法** (比如DPG) 需要的运算。

除此之外，也有学习**分布式价值函数** (Distributional Value Function) 用的运算。

以上运算都支持**批量** (Batches) ，返回的损失可以用TensorFlow优化器来最小化。不论是Transition的批量，还是Trajectory的批量。

## 基于策略

针对基于策略的强化学习，这里既有工具可以轻松实现**在线**方法，比如A2C ，也支持**离线**的修正技术，比如v-trace。

另外，连续动作里**策略梯度**的计算，松露也支持。

最后的最后，松露还提供辅助的**伪奖励函数** (Pseudo-Reward Functions) ，用来提升数据效率。

# 开源了，并待续

如今，松露已经开源了，传送门在文底。

不过，团队在博客里写到，这不是一次性发布。

因为，DeepMind在做研究的过程中，也***\*非常\*******\*依赖\****这个库，所以会持续对它进行维护，也会随时添加新功能。

当然，团队也欢迎强化学习界的小伙伴们，为松露添砖加瓦。

# 多巴胺也是强化学习库

![640?wx_fmt=png](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtBs3a3XcLO9BPDmQ1XNczsuIMlfnFf5PzibzIoOjARf973lk6GSof0hSjjiabJCUpkCnwkTzAy9murw/640?wx_fmt=png)

如果你还记得，今年8月**谷歌**开源了强化学习框架Dopamine，中文叫**多巴胺**，也是基于TensorFlow。

名字取自人类大脑**奖励机制**中的主角物质多巴胺，为了表达神经科学和强化学习之间的缘分联系。

多巴胺框架，也是强调**灵活性**、**稳定性**和**复现性**。

![640?wx_fmt=png](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/YicUhk5aAGtBs3a3XcLO9BPDmQ1XNczsuCRaHoBechIuicv7VICIu29qCDf339xoEk62HgO6ia1uljDR3rfyNCfbg/640?wx_fmt=png)

至于，多巴胺和松露之间有怎样的关系，或者怎样的差别，如果你也好奇的话，可以自行探索一下。

