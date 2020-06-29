

# MDP Planing


注意：

- 前提是，state 和 action 是离散的，所以可以使用矩阵来描述。


（补充例子，或者代码）

给定模型时，就是，已经知道 $\mathcal{R}$ 和 $\mathcal{P}_{\mathrm{ss}^{\prime}}^{a}$，此时，有两类问题：

- 第一类问题：策略评估: 给定策略 $\pi$， 计算累计回报期望值
  - 求解 Bellman Expectation Eq. 就行。
    $$
    v_{\pi}=\mathcal{R}^{\pi}+\gamma \mathcal{P}^{\pi} v_{\pi}
    $$
  - 本质上就是求解一个线性方程组。
- 第二类问题：寻找最优策略：
极大化从任何状态开始的累计回报期望值 $\quad v_{*}(s)=\max _{a} q_{*}(s, a)$
  - 这是一个典型的动态规划的问题
    - 动态规划有两个特征：
      - 最优子结构： 优化问题可以分解为子优化问题
      - 子问题重桑：子优化问题的解可以重复使用
  - 有两种方法：
    - 值迭代
    - 策略迭代

对于策略评估问题：

- 求解：$\quad v_{\pi}=\mathcal{R}^{\pi}+\gamma \mathcal{P}^{\pi} v_{\pi}$
  - 方法1：直接求解求解矩阵方程 $v_{\pi}=\left(I-\gamma \mathcal{P}^{\pi}\right)^{-1} \mathcal{R}^{\pi}$
    - 如果直接求解，需要求解逆矩阵，代价比较大，不推荐。
  - 方法2：迭代求解(不动点迭代, fixed-point iteration)
    - 方式：
      - 方式1：同步的 Synchronous
        - 每次，每次存两个 v 的copy，
      - 方式2：异步的 Asynchronous
        - 只保存一个 v 的 copy。
    - 理解：
      - 理解，不动点迭代，即所有得 v s 全部等于 0，给定一个 $v_\pi$ 然后带进去，得到一个新得 $v_\pi$，最终会收敛，收敛得到的就是，在这个 pi 的情况下，对应的 v function。
      - 为什么会收敛？ (Contraction Mapping Theorem)
        $$
        v_{k+1}(s)=\sum_{a \in \mathcal{A}} \pi(a \mid s)\left(\mathcal{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \mathcal{S}} \mathcal{P}_{s s^{\prime}}^{a} v_{k}\left(s^{\prime}\right)\right)
        $$
        - 比如 x=f(x) ，那么 这样迭代的收敛的前提时 f(x) 是一个收缩的操作，即: f(x)-f(y) <= 某个 matirx 下 (x-y)*常数。

对于寻找最优策略问题：

- 值迭代方法：
  - 理解
    - 先求解 Bellman Optimality Equation $v_{*}(s)=\max _{a} \mathcal{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \mathcal{S}} \mathcal{P}_{s s^{\prime}}^{a} v_{*}\left(s^{\prime}\right)$
        - 把 $v_{*}(s)$ 和 $q_{*}(s,a)$ 先求出来
        - 求解的过程中，不需要明确的 policy。
    - 然后，使用求解出来的 $v_{*}(s)$ 和 $q_{*}(s,a)$ 带入下面式子得到最优 policy：
      $$\quad \pi_{*}(a \mid s)=\left\{\begin{array}{ll}1 & \text { if } a=\underset{a \in \mathcal{A}}{\operatorname{argmax}} q_{*}(s, a) \\ 0 & \text { otherwise }\end{array}\right.$$
  - 流程：
    - 01 For each state $s,$ initialize $V(s):=0$
    - 02 Repeat until convergence{
    - 03 $\quad$ For every state, update $V(s):=R(s)+\max _{a \in A} \gamma \sum_{s^{\prime}} P_{s a}\left(s^{\prime}\right) V\left(s^{\prime}\right)$
    - 04 }
    - 05 Upon convergence, use $\pi^{*}(s)=\arg \max _{a \in A} \sum_{s^{\prime} \in S} P_{s a}\left(s^{\prime}\right) V^{*}\left(s^{\prime}\right)$
  - 算法评估：
    - 时间复杂度：对于每个循环： $\mathrm{O}\left(|\mathrm{A}||\mathrm{S}|^{2}\right)$
    - 迭代次数：$Poly(|A|, |S|, 1/(1-\gamma))$
- 策略迭代方法：（没明白，补充例子）
  - 理解：
    - 先给定一个策略 $\pi$, 评估策略得到了 $V _{\pi}(s)$
    - 然后，我们想办法改进这个策略：$\pi^{\prime}=\operatorname{greedy}\left(\vee_{\pi}\right)=>\pi^{\prime} \geq \pi$
      - 还是使用 greedy 的方法来改进，我当前得到了每个 state 的 value function，那么我选择那个 action 呢？我选取使我下一步 的 $V _{\pi}(s)$ 最大的。这样就得到了新得策略。
    - 如图：
      <p align="center">
          <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200627/WGtlT0yrbl2w.png?imageslim">
      </p>
      <p align="center">
          <img width="50%" height="70%" src="http://images.iterate.site/blog/image/20200627/PRSMII5y1mVL.png?imageslim">
      </p>
  - 与值迭代的比较：
    - 值迭代，没有指定 policy ，而策略迭代，是从指定一个明确的 policy 开始的。
  - 流程：
    - 01 Initialize $\pi$ randomly.
    - 02 Repeat until convergence {
    - 03 $\quad$ Let $V:=V^{\pi}$
    - 04 $\quad$ For each state $s,$ let $\pi(s):=\arg \max _{a \in A} \sum_{s^{\prime}} P_{s a}\left(s^{\prime}\right) V\left(s^{\prime}\right)$
    - 05 }
  - 算法评估：
    - 时间复杂度：对于每个循环：$\mathrm{O}\left(|\mathrm{S}|^{3}+|\mathrm{A}||\mathrm{S}|^{2}\right)$
      - $|S|^{3}$ 对应策略评估, $|A||S|^{2}$ 对应提升。
    - 迭代次数：未知，但是，实际应用中比值迭代更快些。所以用的更多一些。