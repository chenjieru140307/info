

# RNN 的一些结构

RNN 的一些结构：



- sequence-to-sequence 结构
  - 最经典的 RNN 结构
  - 对于一个样本序列，进行训练时，每输入一个特征，输出一个特征。
  - 如图：
    <p align="center">
        <img width="60%" height="70%" src="http://images.iterate.site/blog/image/20190722/KNBKJr9i7jf1.jpg?imageslim">
    </p>
  - 其中：
    - 图中的圆圈表示向量，箭头表示对向量做变换。
    - 权重：
      - $x$ 到 $h$ 是 $U$
      - $h$ 到 $h$ 是 $W$
      - $h$ 到 $y$ 是 $V$
    - 输入为 $x_1,x_2,x_3,x_4$，
    - 输出为 $y_1,y_2,y_3,y_4$
      - $y_{1}=\operatorname{Soft} \max \left(V h_{1}+c\right)$
    - 隐状态 $h$（hidden state），$h​$ 可对序列数据提取特征，接着再转换为输出。
      - $h_{1}=f\left(U x_{1}+W h_{0}+b\right)$
      - $h_{2}=f\left(U x_{2}+W h_{1}+b\right)$
  - 可见，这个结构的 RNN 输入和输出等长。
- vector-to-sequence 结构
  - 输入是一个单独的值，输出是一个序列。
  - 模型：有两种主要建模方式。（这两种到底什么情况下使用的？为什么是合理的？）
    - 方式一：可只在其中的某一个序列进行计算，比如序列第一个进行输入计算，其建模方式如下：

    <p align="center">
        <img width="60%" height="70%" src="http://images.iterate.site/blog/image/20190722/QKja1vcVY5TE.jpg?imageslim">
    </p>

    - 方式二：把输入信息 $X$ 作为每个阶段的输入，其建模方式如下：

    <p align="center">
        <img width="60%" height="70%" src="http://images.iterate.site/blog/image/20190722/u9128GeVywMk.jpg?imageslim">
    </p>

  - 应用：
    - 从图像生成文字，输入为图像的特征，输出为一段句子
    - 根据图像生成语音或音乐，输入为图像特征，输出为一段语音或音乐。
- sequence-to-vector 结构
  - 输入是一个序列，输出是一个单独的值，此时通常在最后的一个序列上进行输出变换
  - 模型：

  <p align="center">
      <img width="60%" height="70%" src="http://images.iterate.site/blog/image/20190722/TzdcWvAoaw9Y.jpg?imageslim">
  </p>

  - 应用：
    - 输出一段文字，判断其所属类别
    - 输入一个句子，判断其情感倾向
    - 输入一段视频，判断其所属类别。
- Encoder-Decoder 结构
  - 缘由：
    - 原始的 sequence-to-sequence 结构的 RNN 要求序列等长，然而我们遇到的大部分问题序列都是不等长的，如机器翻译中，源语言和目标语言的句子往往并没有相同的长度。
  - 模型：
    - 先将输入数据编码成一个上下文向量 $c$，这部分称为 Encoder。
      - 示意图：
        <p align="center">
          <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200625/LFkhMnWQl6jT.png?imageslim">
        </p>
      - 得到 $c$ 有多种方式：
        - 最简单的方法就是把 Encoder 的最后一个隐状态赋值给 $c$
          - $c=h_{4}$
        - 还可以对最后的隐状态做一个变换得到 $c$
          - $c=q\left(h_{4}\right)$
        - 也可以对所有的隐状态做变换。
          - $c=q\left(h_{1}, h_{2}, h_{3}, h_{4}\right)$
        - 其示意如下所示：
    - 再用另一个 RNN 网络（我们将其称为 Decoder）对 $c​$ 进行编码。有两种方式：
      - 方法一：将 $c​$ 作为初始状态输入到 Decoder，示意图如下：
        <p align="center">
            <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20190722/XQhbHtVEOEMn.jpg?imageslim">
        </p>
      - 方法二：将 $c$ 作为 Decoder 的每一步输入，示意图如下所示：
        <p align="center">
            <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20190722/ke5ViPSbAStS.jpg?imageslim">
        </p>
  - 理解：
    - 对于：
      $$
      \begin{array}{l}{\text { Source }=\left\langle\mathbf{x}_{1}, \mathbf{x}_{2} \dots \mathbf{x}_{\mathbf{m}}\right\rangle} \\ {\text { Target }=\left\langle\mathbf{y}_{1}, \mathbf{y}_{2} \dots \mathbf{y}_{\mathbf{n}}\right\rangle}\end{array}
      $$

    - Encoder：
      - 将 Source 编码为中间语义表示 $C$：$\mathbf{C}=\mathcal{F}\left(\mathbf{x}_{1}, \mathbf{x}_{2} \dots \mathbf{x}_{\mathbf{m}}\right)$
    - Decoder：
      - 根据中间语义表示 $C$ 和之前已经生成的历史信息 $y_{1}, y_{2} \dots \dots y_{i-1}$ 来生成 $i$ 时刻要生成的单词 $y_i$：$\mathbf{y}_{\mathbf{i}}=\boldsymbol{g}\left(\mathbf{C}, \mathbf{y}_{\mathbf{1}}, \mathbf{y}_{\mathbf{2}} \dots \mathbf{y}_{\mathbf{i}-\mathbf{1}}\right)$
    - 每个 $y_i$ 都依次这么产生，那么看起来就是整个系统根据输入句子 Source 生成了目标句子 Target。
  - 应用：
    - 在文本处理领域：
      - 机器翻译：输入中文，输出英文。
      - 文本摘要：输入文章，输出描述语句。
      - 阅读理解，输入文章，输出问题答案。
      - 问答系统或者对话机器人：输入问题，输出回答
    - 对于语音识别来说
      - 语音识别，输入 语音序列，输出文字序列。
    - 对于“图像描述”任务来说
      - Encoder部分的输入是一副图片，Decoder的输出是能够描述图片语义内容的一句描述语。
  - 使用时注意：
    - 一般而言，文本处理和语音识别的 Encoder 部分通常采用 RNN 模型，图像处理的 Encoder 一般采用 CNN 模型。
