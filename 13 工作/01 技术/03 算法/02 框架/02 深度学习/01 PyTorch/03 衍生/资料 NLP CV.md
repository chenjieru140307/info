
## NLP&PyTorch实战

- [Pytorch text](https://github.com/pytorch/text)

  ：Torchtext是一个非常好用的库，可以帮助我们很好的解决文本的预处理问题。此github存储库包含两部分：

  - torchText.data：文本的通用数据加载器、抽象和迭代器（包括词汇和词向量）
  - torchText.datasets：通用NLP数据集的预训练加载程序 我们只需要通过pip install torchtext安装好torchtext后，便可以开始体验Torchtext 的种种便捷之处。

- [Pytorch-Seq2seq](https://github.com/IBM/pytorch-seq2seq)：Seq2seq是一个快速发展的领域，新技术和新框架经常在此发布。这个库是在PyTorch中实现的Seq2seq模型的框架，该框架为Seq2seq模型的训练和预测等都提供了模块化和可扩展的组件，此github项目是一个基础版本，目标是促进这些技术和应用程序的开发。

- [BERT NER](https://github.com/kamalkraj/BERT-NER)：BERT是2018年google 提出来的预训练语言模型，自其诞生后打破了一系列的NLP任务，所以其在nlp的领域一直具有很重要的影响力。该github库是BERT的PyTorch版本，内置了很多强大的预训练模型，使用时非常方便、易上手。

- [Fairseq](https://github.com/pytorch/fairseq)：Fairseq是一个序列建模工具包，允许研究人员和开发人员为翻译、总结、语言建模和其他文本生成任务训练自定义模型，它还提供了各种Seq2seq模型的参考实现。该github存储库包含有关入门、训练新模型、使用新模型和任务扩展Fairseq的说明，对该模型感兴趣的小伙伴可以点击上方链接学习。

- [Quick-nlp](https://github.com/outcastofmusic/quick-nlp)：Quick-nlp是一个深受fast.ai库启发的深入学习Nlp库。它遵循与Fastai相同的API，并对其进行了扩展，允许快速、轻松地运行NLP模型。

- [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)：这是OpenNMT的一个PyTorch实现，一个开放源码的神经网络机器翻译系统。它的设计是为了便于研究，尝试新的想法，以及在翻译，总结，图像到文本，形态学等许多领域中尝试新的想法。一些公司已经证明该代码可以用于实际的工业项目中，更多关于这个github的详细信息请参阅以上链接。

## CV&PyTorch实战

- [pytorch vision](https://github.com/pytorch/vision)：Torchvision是独立于pytorch的关于图像操作的一些方便工具库。主要包括：vision.datasets 、vision.models、vision.transforms、vision.utils 几个包，安装和使用都非常简单，感兴趣的小伙伴们可以参考以上链接。

- [OpenFacePytorch](https://github.com/thnkim/OpenFacePytorch)：此github库是OpenFace在Pytorch中的实现，代码要求输入的图像要与原始OpenFace相同的方式对齐和裁剪。

- [TorchCV](https://github.com/donnyyou/torchcv)：TorchCV是一个基于PyTorch的计算机视觉深度学习框架，支持大部分视觉任务训练和部署，此github库为大多数基于深度学习的CV问题提供源代码，对CV方向感兴趣的小伙伴还在等什么？

- [Pytorch-cnn-finetune](https://github.com/creafz/pytorch-cnn-finetune)：该github库是利用pytorch对预训练卷积神经网络进行微调，支持的架构和模型包括：ResNet 、DenseNet、Inception v3 、VGG、SqueezeNet 、AlexNet 等。

- Pt-styletransfer

  ：这个github项目是Pytorch中的神经风格转换，具体有以下几个需要注意的地方：

  - StyleTransferNet作为可由其他脚本导入的类；
  - 支持VGG（这是在PyTorch中提供预训练的VGG模型之前）
  - 可保存用于显示的中间样式和内容目标的功能
  - 可作为图像检查图矩阵的函数
  - 自动样式、内容和产品图像保存
  - 一段时间内损失的Matplotlib图和超参数记录，以跟踪有利的结果

- [Face-alignment](https://github.com/1adrianb/face-alignment#face-recognition)：Face-alignment是一个用 pytorch 实现的 2D 和 3D 人脸对齐库，使用世界上最准确的面对齐网络从 Python 检测面部地标，能够在2D和3D坐标中检测点。该github库详细的介绍了使用Face-alignment进行人脸对齐的基本流程，欢迎感兴趣的同学学习。