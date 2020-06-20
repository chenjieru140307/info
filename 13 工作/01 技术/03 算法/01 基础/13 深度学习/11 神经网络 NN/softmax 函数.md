
softmax 函数



- $P(i) = \frac{e^{(\theta_i^T x)}}{\sum_{k=1}^{K} e^{(\theta_i^T x)}}$
  - 其中
    - $\theta_i$ 和 $x$ 是列向量
    - $\theta_i^T x$ 可能被换成函数关于 $x$ 的函数 $f_i(x)$
- 说明：
  - 通过 softmax 函数，可以使得 $P(i)$ 的范围在 $[0,1]$ 之间。
    - 使得范围在 $[0,1]$  之间的方法有很多，为啥要在前面加上以 $e$ 的幂函数的形式呢？
    - 参考 logistic 函数：$P(i) = \frac{1}{1+e^{(-\theta_i^T x)}}$
      - 这个函数的作用就是使得 $P(i)$ 在负无穷到 $0$ 的区间趋向于 $0$， 在 $0$ 到正无穷的区间趋向 $1$。
    - 同样的，softmax 函数加入了 $e$ 的幂函数正是为了两极化：
      - 正样本的结果将趋近于 1，
      - 而负样本的结果趋近于 0。
    - 这样为多类别提供了方便（可以把 $P(i)$ 看做是样本属于类别的概率）。
    - 可以说，Softmax 函数是 logistic 函数的一种泛化。

- 应用：
  - 多用于多分类神经网络输出。
    - Softmax 函数可以把它的输入，通常被称为 logits 或者 logit scores，处理成 0 到 1 之间，并且能够把输出归一化到和为 1。这意味着 softmax 函数与分类的概率分布等价。它是一个网络预测多分类问题的最佳输出激活函数。

- 举例：
  - 如图：

    <p align="center">
        <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20190722/j5uziqVDMLas.png?imageslim">
    </p>

    <p align="center">
        <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20190722/tpiMHpc7MX4g.png?imageslim">
    </p>
  - 说明：
    - 此处，softmax 直白来说就是将原来输出是 $3,1,-3​$ 通过 softmax 函数的作用，映射成为 $(0,1)$ 的值，而这些值的累和为 $1​$（满足概率的性质），那么我们就可以将它理解成概率，在最后选取输出结点的时候，我们就可以选取概率最大（也就是值对应最大的）结点，作为我们的预测目标。
