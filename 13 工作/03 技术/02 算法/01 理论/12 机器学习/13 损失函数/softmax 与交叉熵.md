# Softmax 与交叉熵

可以合并到 CNN 部分。

主要内容：

- Softmax 是如何把 CNN 的输出转变成概率
- 以及交叉熵是如何为优化过程提供度量。



## Softmax函数



Softmax 函数接收一个 这N维向量作为输入，然后把每一维的值转换成（0，1）之间的一个实数，它的公式如下面所示：

$$
p_{i}=\frac{e^{a_{i}}}{\sum_{k=1}^{N} e_{k}^{a}}
$$

正如它的名字一样，Softmax 函数是一个“软”的最大值函数，它不是直接取输出的最大值那一类作为分类结果，同时也会考虑到其它相对来说较小的一类的输出。



说白了，Softmax 可以将全连接层的输出映射成一个概率的分布，我们训练的目标就是让属于第k类的样本经过 Softmax 以后，第 k 类的概率越大越好。这就使得分类问题能更好的用统计学方法去解释了。



使用 Python，我们可以这么去实现 Softmax 函数：

```py
def softmax(X):
    exps=np.exp(X)
    return exps/np.sum(exps)
```



我们需要注意的是，在 numpy 中浮点类型是有数值上的限制的，对于`float64`，它的上限是 $10^{308}$ 。对于指数函数来说，这个限制很容易就会被打破，如果这种情况发生了 python 便会返回 `nan`。



为了让 Softmax 函数在数值计算层面更加稳定，避免它的输出出现 `nan`这种情况，一个很简单的方法就是对输入向量做一步归一化操作，仅仅需要在分子和分母上同乘一个常数 `C`，如下面的式子所示

$$
\begin{aligned} p_{j} &=\frac{e^{a_{i}}}{\sum_{k=1}^{N} e^{a_{k}}} \\ &=\frac{C e^{a_{i}}}{C \sum_{k=1}^{N} e^{a_{k}}} \\ &=\frac{e^{a_{i}+\log (C)}}{\sum_{k=1}^{N} e^{a_{k}+\log (C)}} \end{aligned}
$$


理论上来说，我们可以选择任意一个值作为 $\log (C)$，但是一般我们会选择 $\log (C)=-\max (a)$ ，通过这种方法就使得原本非常大的指数结果变成 0，避免出现 `nan` 的情况。



同样使用 Python，改进以后的 Softmax 函数可以这样写：

```py
def stable_softmax(X):
    exps=np.exp(X-np.max(X))
    return exps / np.sum(exps)
```

## Softmax 函数的导数



通过上文我们了解到，Softmax 函数可以将样本的输出转变成概率密度函数，由于这一很好的特性，我们就可以把它加装在神经网络的最后一层，随着迭代过程的不断深入，它最理想的输出就是样本类别的 One-hot 表示形式。

求 Softmax 函数的导数：


$$
\frac{\partial p_{j}}{\partial a_{j}}=\frac{\partial \frac{e^{a_{i}}}{\sum_{k=1}^{N} e^{a_{k}}}}{\partial a_{j}}
$$

- 根据商的求导法则，对于 $f(x)=\frac{g(x)}{h(x)}$ 其导数为 $f^{\prime}(x)=\frac{g^{\prime}(x) h(x)-h^{\prime}(x) g(x)}{h(x)^{2}}$  。
- 对于我们来说$g(x)=e^{a_{i}}$，$h(x)=\sum_{k=1}^{N} e^{a_{k}}$ 。
- 在 $h(x)$中，$\frac{\partial}{\partial e^{a_{j}}}$  一直都是 $e^{a_{j}}$，但是在 $g(x)$ 中，当且仅当 $i=j$ 的时候，$\frac{\partial}{\partial e^{a_{j}}}$ 才为$e^{a_{j}}$。
- 具体的过程，我们看一下下面的步骤：
- 如果 $i=j$，

$$
\begin{aligned} \frac{\partial \frac{e^{a_{i}}}{\sum_{k=1}^{N} e^{a_{k}}}}{\partial a_{j}} &=\frac{e^{a_{i}} \sum_{k=1}^{N} e^{a_{k}}-e^{a_{j}} e^{a_{i}}}{\left(\sum_{k=1}^{N} e^{a_{k}}\right)^{2}} \\ &=\frac{e^{a_{i}}\left(\sum_{k=1}^{N} e^{a_{k}}-e^{a_{j}}\right)}{\left(\sum_{k=1}^{N} e^{a_{k}}\right)^{2}} \\ &=\frac{e^{a_{j}}}{\sum_{k=1}^{N} e^{a_{k}}} \times \frac{\left(\sum_{k=1}^{N} e^{a_{k}}-e^{a_{j}}\right)}{\sum_{k=1}^{N} e^{a_{k}}} \\ &=p_{i}\left(1-p_{j}\right) \end{aligned}
$$

- 如果$i \neq j$：

$$
\begin{aligned} \frac{\partial \frac{e^{a_{i}}}{\sum_{k=1}^{N} e^{a_{k}}}}{\partial a_{j}} &=\frac{0-e^{a_{j}} e^{a_{i}}}{\left(\sum_{k=1}^{N} e^{a_{k}}\right)^{2}} \\ &=\frac{-e^{a_{j}}}{\sum_{k=1}^{N} e^{a_{k}}} \times \frac{e^{a_{i}}}{\sum_{k=1}^{N} e^{a_{k}}} \\ &=-p_{j} \cdot p_{i} \end{aligned}
$$

- 所以 Softmax 函数的导数如下：



$$
\frac{\partial p_{j}}{\partial a_{j}}=\left\{\begin{array}{ll}p_{i}\left(1-p_{j}\right) & \text { if } i=j \\ -p_{j} \cdot p_{i} & \text { if } i \neq j\end{array}\right.
$$



## 交叉熵损失函数



下面我们来看一下对模型优化真正起到作用的损失函数——交叉熵损失函数。

交叉熵函数体现了模型输出的概率分布和真实样本的概率分布的相似程度。它的定义式就是这样：


$$
H(y, p)=-\sum_{i} y_{i} \log \left(p_{i}\right)
$$



在分类问题中，交叉熵函数已经大范围的代替了均方误差函数。

也就是说，在输出为概率分布的情况下，就可以使用交叉熵函数作为理想与现实的度量。

这也就是为什么它可以作为有 Softmax 函数激活的神经网络的损失函数。



我们来看一下，在 Python 中是如何实现交叉熵函数的：


```py
def cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_clases)
    y is labels (num_examples x 1)
    """
    m=y.shape[0]
    p=softmax(X)
    log_likelihood=-np.log(p[range(m),y])
    loss=np.sum(log_likelihood)/m
    return loss
```




## 交叉熵损失函数的求导



就像我们之前所说的，Softmax 函数和交叉熵损失函数是一对好兄弟，我们用上之前推导 Softmax 函数导数的结论，配合求导交叉熵函数的导数：

- 已知：

$$
L=-\sum_{i} y_{i} \log \left(p_{i}\right)
$$

- 对 $o_{i}$ 进行求导：

$$
\begin{aligned} \frac{\partial L}{\partial o_{i}} &=-\sum_{k} y_{k} \frac{\partial \log \left(p_{k}\right)}{\partial o_{i}} \\ &=-\sum_{k} y_{k} \frac{\partial \log \left(p_{k}\right)}{\partial p_{k}} \times \frac{\partial p_{k}}{\partial o_{i}} \\ &=-\sum y_{k} \frac{1}{p_{k}} \times \frac{\partial p_{k}}{\partial o_{i}} \end{aligned}
$$


- 带入上面的 Softmax 函数的导数 $\frac{\partial p_{k}}{\partial o_{i}}$ 后如下:

$$
\begin{aligned} \frac{\partial L}{\partial o_{i}} &=-\sum y_{k} \frac{1}{p_{k}} \times \frac{\partial p_{k}}{\partial o_{i}} \\ &=-\sum_{k = i} y_{k} \frac{1}{p_{k}} \times p_i(1-p_k)-\sum_{k \neq i} y_{k} \frac{1}{p_{k}} \times (-p_{k} \cdot p_{i}) \\ &=-y_{i}\left(1-p_{i}\right)-\sum_{k \neq i} y_{k} \frac{1}{p_{k}}\left(-p_{k} \cdot p_{i}\right) \\ &=-y_{i}\left(1-p_{i}\right)+\sum_{k \neq i} y_{k} \cdot p_{i} \\ &=-y_{i}+y_{i} p_{i}+\sum_{k \neq i} y_{k} \cdot p_{i} \\ &=p_{i}\left(y_{i}+\sum_{k \neq i} y_{k}\right)-y_{i} \end{aligned}
$$



- 由于 $y$ 代表标签的 One-hot 编码，因此 $\sum_{k} y_{k}=1$，即 $y_{i}+\sum_{k \neq i} y_{k}=1$。
- 因此我们就得到：


$$
\frac{\partial L}{\partial o_{i}}=p_{i}-y_{i}
$$



可以看到，这个结果真的太简单了，不得不佩服发明它的大神们！

最后，我们把它转换成代码：

```py
def delta_cross_entropy(X,y):
    m=y.shape[0]
    grad=softmax(X)
    grad[range(m),y]-=1
    grad=grad/m
    return grad
```


