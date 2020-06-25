
# RBF 网络

RBF (Radial Basis Function，径向基函数) 


RBF 网络：

- 是一种单隐层前馈神经网络，它使用径向基函数作为隐层神经元激活函数，而输出层则是对隐层神经元输出的线性组合。
- 假定输入为 $d$ 维向量 $\boldsymbol{x}$ ，输出为实值，则 RBF 网络可表示为


$$
\varphi(\boldsymbol{x})=\sum_{i=1}^{q} w_{i} \rho\left(\boldsymbol{x}, \boldsymbol{c}_{i}\right)
$$

- 说明：
  - $q$ 为隐层神经元个数
  - $\boldsymbol{c}_{i}$ 和 $w_i$ 分别是第 $i$ 个隐层神经元所对应的中心和权重，
  - $\rho\left(\boldsymbol{x}, \boldsymbol{c}_{i}\right)$ 是径向基函数，这是某种沿径向对称的标量函数，通常定义为样本 $\boldsymbol{x}$ 到数据中心 $\boldsymbol{c}_{i}$ 之间欧氏距离的单调函数。（什么时径向基函数？）
    - 常用的高斯径向基函数形如：
    $$
    \rho\left(\boldsymbol{x}, \boldsymbol{c}_{i}\right)=e^{-\beta_{i}\left\|\boldsymbol{x}-c_{i}\right\|^{2}}
    $$

- 效果:
  - 具有足够多隐层神经元的 RBF 网络能以任意精度逼近任意连续函数.
- 训练：
  - 第一步，确定神经元中心 $\boldsymbol{c}_{i}$ ，常用的 方式包括随机采样、聚类等;
  - 第二步，利用 BP 算法等来确定参数 $w_i$ 和 $\beta_i$ 。


