
## DataLoader



**`torch.utils.data.DataLoader` 参数说明如下：**

- dataset：加载数据的数据集。
- batch_size：加载批训练的数据个数。
- shuffle:True在每个 epoch 重新排列的数据。
- sampler：从数据集中提取样本。
- batch_sampler：一次返回一批索引。
- num_workers：用于数据加载的子进程数。0表示数据将在主进程中加载。
- collate_fn：合并样本列表以形成小批量。<span style="color:red;">这个是什么意思？</span>
- pin_memory：如果为 True，数据加载器在返回前将张量复制到 CUDA 固定内存中。<span style="color:red;">没懂，为什么要将张量复制到 CUDA 固定内存中？如果数据是从 generator 中临时生成的，这个也要 true 吗？</span>
- drop_last：如果数据集大小不能被 batch_size整除，设置为 True，可删除最后一个不完整的批处理。如果设为 False，并且数据集的大小不能被 batch_size 整除，则最后一个 batch 将更小。


## 二维卷积层 Conv2d

**nn.Conv2d：**

参数 kernel_size、stride、padding、dilation 也可以是一个 int 的数据，此时卷积 height 和 width 值相同；也可以是一个 tuple 数组，tuple 的第一维度表示 height 的数值，tuple 的第二维度表示 width 的数值。

参数说明如下。

- in_channels(int)：输入信号的通道。
- out_channels(int)：卷积产生的通道。
- kerner_size(int or tuple)：卷积核的尺寸。
- stride(int or tuple,optional)：卷积步长。
- padding(int or tuple,optional)：是否对输入数据填充 0。Padding可以将输入数据的区域改造成是卷积核大小的整数倍，这样对不满足卷积核大小的部分数据就不会忽略了。通过 padding 参数指定填充区域的高度和宽度。
- dilation(int or tuple,`optional``)：卷积核之间的空格。
- groups(int,optional)：将输入数据分成组，in_channels应该被组数整除。group=1，输出是所有的输入的卷积；group=2，此时相当于有并排的两个卷积层，每个卷积层计算输入通道的一半，并且产生的输出是输出通道的一半，随后将这两个输出连接起来。
- bias(bool,optional)：如果 bias=True，添加偏置。




