
# LSTM

- 缘由：
  - RNN在处理长期依赖（时间序列上距离较远的节点）时会遇到巨大的困难，因为计算距离较远的节点之间的联系时会涉及雅可比矩阵的多次相乘，会造成梯度消失或者梯度膨胀的现象。
  - 为了解决该问题，研究人员提出了许多解决办法，例如：
    - ESN（Echo State Network）
    - 增加有漏单元（Leaky Units）等等。
  - 其中最成功，应用最广泛的就是门限 RNN（Gated RNN）
    - LSTM 就是门限 RNN 中最著名的一种。
      - 有漏单元通过设计连接间的权重系数，从而允许 RNN 累积距离较远节点间的长期联系；
      - 门限 RNN 泛化了这样的思想，允许在不同时刻改变该系数，且允许网络忘记当前已经累积的信息。



理解：

- 细胞状态：
  - LSTM 的关键，是细胞状态。细胞状态类似于传送带。直接在整个链上运行，只有一些少量的线性交互。信息在上面流传保持不变会很容易。
  - 如图：
    <p align="center">
        <img width="90%" height="70%" src="http://images.iterate.site/blog/image/20190722/VlMgEOhJfwer.png?imageslim">
    </p>
  - 说明:
    - 水平线在图上方贯穿运行。
- 门 结构
  - LSTM 使用 门 结构，来去除或者增加信息到细胞状态。
  - 门是一种让信息选择式通过的方法。他们包含一个 sigmoid 神经网络层和一个 pointwise 乘法操作。
  - 如图：
      <p align="center">
          <img width="25%" height="70%" src="http://images.iterate.site/blog/image/20190722/laxCsNPQF6j1.png?imageslim">
      </p>
  - LSTM 拥有三个门，分别是忘记层门，输入层门和输出层门，来保护和控制细胞状态。
    - 忘记层门
      - 作用对象：细胞状态 。
      - 作用：将细胞状态中的信息选择性的遗忘。
      - 门的操作：
        - 读取 $h_{t-1}$ 和 $x_t$，输出一个在 0 到 1 之间的数值给每个在细胞状态 $C_{t-1}​$ 中的数字。
          - 图示
            <p align="center">
                <img width="90%" height="70%" src="http://images.iterate.site/blog/image/20190722/V6sL7vH5wrNI.png?imageslim">
            </p>
          - 其中:
            $$f_{t}=\sigma\left(W_{f} \cdot\left[h_{t-1}, x_{t}\right]+b_{f}\right)$$
          - 说明：
            - 1 表示“完全保留”
            - 0 表示“完全舍弃”
    - 输入层门
      - 作用对象：细胞状态
      - 作用：将新的信息选择性的记录到细胞状态中。
      - 门的操作：
        - sigmoid 层称 “输入门层” 决定什么值我们将要更新。
        - tanh 层创建一个新的候选值向量 $\tilde{C}_t$ 加入到状态中。
          - 图示：
              <p align="center">
                  <img width="90%" height="70%" src="http://images.iterate.site/blog/image/20190722/fxs7siIdGXKX.png?imageslim">
              </p>
          - 其中:
            $$\begin{aligned} i_{t} &=\sigma\left(W_{i} \cdot\left[h_{t-1}, x_{t}\right]+b_{i}\right) \\ \tilde{C}_{t} &=\tanh \left(W_{C} \cdot\left[h_{t-1}, x_{t}\right]+b_{C}\right) \end{aligned}$$
        - 将 $c_{t-1}$ 更新为 $c_{t}$。将旧状态与 $f_t$ 相乘，丢弃掉我们确定需要丢弃的信息。接着加上 $i_t * \tilde{C}_t$ 得到新的候选值，根据我们决定更新每个状态的程度进行变化。
          - 图示：
              <p align="center">
                  <img width="90%" height="70%" src="http://images.iterate.site/blog/image/20190722/ncrqo41x9B13.png?imageslim">
              </p>
          - 其中：
              $$C_{t}=f_{t} * C_{t-1}+i_{t} * \tilde{C}_{t}$$

    - 输出层门
      - 作用对象：隐层 $h_t$
      - 作用：确定输出什么值。
      - 门的操作：
        - 通过 sigmoid 层来确定细胞状态的哪个部分将输出。
        - 把细胞状态通过 tanh 进行处理，并将它和 sigmoid 门的输出相乘，最终我们仅仅会输出我们确定输出的那部分。
          - 图示：
              <p align="center">
                  <img width="90%" height="70%" src="http://images.iterate.site/blog/image/20190722/TNk0b9xHhVid.png?imageslim">
              </p>
          - 其中：
              $$\begin{array}{l}
              o_{t}=\sigma\left(W_{o}\left[h_{t-1}, x_{t}\right]+b_{o}\right) \\
              h_{t}=o_{t} * \tanh \left(C_{t}\right)
              \end{array}$$




变体：

- 增加 peephole 连接
  - 在正常的 LSTM 结构中，增加 peephole 连接，可以门层接受细胞状态的输入。
  - 如图：
    <p align="center">
        <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20190722/d2hM9ygWk95W.png?imageslim">
    </p>
- 对忘记门和输入门进行同时确定
  - 不同于之前是分开确定什么忘记和需要添加什么新的信息，这里是一同做出决定。
  - 如图：
    <p align="center">
        <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20190722/tiuQH4V9v6hg.png?imageslim">
    </p>
- Gated Recurrent Unit(GRU)
  - 将忘记门和输入门合成了一个单一的更新门，同样还混合了细胞状态和隐藏状态，和其他一些改动。
    - 最终的模型比标准的 LSTM 模型要简单，是非常流行的变体。
  - 如图：
    <p align="center">
        <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20190722/VkiuuFi127pk.png?imageslim">
    </p>
  - 理解：
    - 如图
      <p align="center">
          <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20190616/Mp0GAYolBRXg.png?imageslim">
      </p>
    - 其中：

      $$
      z_{t}=\sigma\left(W^{(z)} x_{t}+U^{(z)} h_{t-1}\right)
      $$
      $$
      r_{t}=\sigma\left(W^{(r)} x_{t}+U^{(r)} h_{t-1}\right)
      $$
      $$
      \tilde{h}_{t}=\tanh \left(W x_{t}+r_{t} U h_{t-1}\right)
      $$
      $$
      h_{t}=z_{t} h_{t-1}+\left(1-z_{t}\right) \tilde{h}_{t}
      $$

    - 说明：
      - GRU 首先根据当前输入向量 word vector 在前一个隐藏层的状态中计算出 update gate 和 reset gate。
      - 再根据 reset gate、当前 word vector 以及前一个隐藏层计算新的记忆单元内容（New Memory Content）。
      - 当 reset gate 为 1 的时候，前一个隐藏层计算新的记忆单元内容忽略之前的所有记忆单元内容，最终的记忆是之前的隐藏层与新的记忆单元内容的结合。

  - LSTM 与 GRU 的比较：
    - 图示：

    <p align="center">
        <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20190722/FCe1Me8dLstj.png?imageslim">
    </p>

    - 说明:
      - new memory 都是根据之前 state 及 input 进行计算，但是 GRUs 中有一个 reset gate 控制之前 state 的进入量，而在 LSTMs 里没有类似 gate；
      - 产生新的 state 的方式不同，LSTMs 有两个不同的 gate，分别是 forget gate (f gate) 和 input gate(i gate)，而 GRUs 只有一种 update gate(z gate)；
      - LSTMs 对新产生的 state 可以通过 output gate(o gate) 进行调节，而 GRUs 对输出无任何调节。



RNN 和 LSTM 比较：

- 重复的模块。所有 RNN 都具有一种重复神经网络模块的链式的形式。
  - 在标准的 RNN 中：
    - 这个重复的模块只有一个非常简单的结构，例如一个 tanh 层。
    - 如图：
        <p align="center">
            <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20190722/YBxm7aruD4IX.png?imageslim">
        </p>
  - 在 LSTM 中
    - LSTM 中重复的模块中，四个激活函数以一种非常特殊的方式进行交互。
    - 如图：
        <p align="center">
            <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20190722/TFMApF799vve.png?imageslim">
        </p>
    - 说明：
      - 图标含义如下：
        <p align="center">
            <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20190722/KVeAuD5zxml8.png?imageslim">
        </p>


LSTM 的改进与扩展：

- Bidirectional LSTMs
  - 介绍：
    - 与 bidirectional RNNs 类似，bidirectional LSTMs 有两层 LSTMs。
    - 一层处理过去的训练信息，另一层处理将来的训练信息。
  - 在 bidirectional LSTMs 中，通过前向 LSTMs 获得前向隐藏状态，后向 LSTMs 获得后向隐藏状态，当前隐藏状态是前向隐藏状态与后向隐藏状态的组合。
- Stacked LSTMs
  - 与 deep rnns 类似，stacked LSTMs 通过将多层 LSTMs 叠加起来得到一个更加复杂的模型。
  - 不同于 bidirectional LSTMs，stacked LSTMs 只利用之前步骤的训练信息。
- CNN-LSTMs
  - 为了同时利用 CNN 以及 LSTMs 的优点，CNN-LSTMs 被提出。
  - 在该模型中，CNN 用于提取对象特征，LSTMs 用于预测。CNN 由于卷积特性，其能够快速而且准确地捕捉对象特征。LSTMs 的优点在于能够捕捉数据间的长时依赖性。
