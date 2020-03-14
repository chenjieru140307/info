
# Faster R-CNN


- [faster-rcnn 之 RPN网络的结构解析](https://blog.csdn.net/sloanqin/article/details/51545125)
- [faster-rcnn中，对 RPN 的理解](https://blog.csdn.net/ying86615791/article/details/72788414)



目的：

- 为解决 Fast R-CNN 缺陷，实现 two stage 的全网络结构

地址：

- <https://arxiv.org/pdf/1506.01497.pdf>

介绍：

- 设计辅助生成样本的 RPN（Region Proposal Networks）网络，将算法结构分为两个部分
  - 先由 RPN 网络判断候选框是否为目标
  - 再经分类定位的多任务损失判断目标类型

效果：

- 整个网络流程都能共享卷积神经网络提取的的特征信息，节约计算成本，且解决 Fast R-CNN 算法生成正负样本候选框速度慢的问题，同时避免候选框提取过多导致算法准确率下降。

缺点：

- 由于 RPN 网络可在固定尺寸的卷积特征图中生成多尺寸的候选框，导致出现可变目标尺寸和固定感受野不一致的现象。<span style="color:red;">？</span>



