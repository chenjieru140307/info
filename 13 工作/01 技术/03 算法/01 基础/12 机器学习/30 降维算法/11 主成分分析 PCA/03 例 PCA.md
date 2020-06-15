

# PCA 例子

代码：

```py
import numpy as np

np.random.seed(123)

# 样本生成
mu_vec1 = np.array([0, 0, 0])
cov_mat1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T
assert class1_sample.shape == (3, 20)  # 检验数据的维度是否为 3*20，若不为 3*20，则抛出异常
print(class1_sample)
mu_vec2 = np.array([1, 1, 1])
cov_mat2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T
assert class2_sample.shape == (3, 20)

# 作图查看原始数据的分布
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 10
ax.plot(class1_sample[0, :], class1_sample[1, :], class1_sample[2, :], 'o', markersize=8, color='blue', alpha=0.5,
        label='class1')
ax.plot(class2_sample[0, :], class2_sample[1, :], class2_sample[2, :], '^', markersize=8, alpha=0.5, color='red',
        label='class2')
plt.title('Samples for class 1 and class 2')
ax.legend(loc='upper right')
plt.show()

# 样本混合
all_samples = np.concatenate((class1_sample, class2_sample), axis=1)
assert all_samples.shape == (3, 40)

# 计算各特征均值
mean_x = np.mean(all_samples[0, :])
mean_y = np.mean(all_samples[1, :])
mean_z = np.mean(all_samples[2, :])
mean_vector = np.array([[mean_x], [mean_y], [mean_z]])
print('Mean Vector:\n', mean_vector)

# 计算 归一化后的 x*xT
scatter_matrix = np.zeros((3, 3))
for i in range(all_samples.shape[1]):
    scatter_matrix += (all_samples[:, i].reshape(3, 1) - mean_vector).dot(
        (all_samples[:, i].reshape(3, 1) - mean_vector).T)
print('Scatter Matrix:\n', scatter_matrix)

# 计算协方差矩阵
cov_mat = np.cov([all_samples[0, :], all_samples[1, :], all_samples[2, :]])
print('Covariance Matrix:\n', cov_mat)

# 计算相应的特征向量和特征值
eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)
eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
for i in range(len(eig_val_sc)):
    eigvec_sc = eig_vec_sc[:, i].reshape(1, 3).T
    eigvec_cov = eig_vec_cov[:, i].reshape(1, 3).T
    assert eigvec_sc.all() == eigvec_cov.all()
print('Eigenvector {}: \n{}'.format(i + 1, eigvec_sc))
print('Eigenvalue {} from scatter matrix: {}'.format(i + 1, eig_val_sc[i]))
print('Eigenvalue {} from covariance matrix: {}'.format(i + 1, eig_val_cov[i]))
print('Scaling factor: ', eig_val_sc[i] / eig_val_cov[i])
print(40 * '-')

# 快速验证一下特征值-特征向量的计算是否正确，
# 是不是满足方程 $\boldsymbol{\Sigma} \boldsymbol{v}=\lambda \boldsymbol{v}$（其中 $\mathbf{\Sigma}$ 为协方差矩阵，$v$ 为特征向量，lambda为特征值）
for i in range(len(eig_val_sc)):
    eigv = eig_vec_sc[:, i].reshape(1, 3).T
    np.testing.assert_array_almost_equal(scatter_matrix.dot(eigv), eig_val_sc[i] * eigv, decimal=6, err_msg='',
                                         verbose=True)

# 可视化特征向量
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(all_samples[0, :], all_samples[1, :], all_samples[2, :], 'o', markersize=8, color='green', alpha=0.2)
ax.plot([mean_x], [mean_y], [mean_z], 'o', markersize=10, color='red', alpha=0.5)
for v in eig_vec_sc.T:
    a = Arrow3D([mean_x, v[0]], [mean_y, v[1]], [mean_z, v[2]], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
    ax.add_artist(a)
ax.set_xlabel('x_values')
ax.set_ylabel('y_values')
ax.set_zlabel('z_values')
plt.title('Eigenvectors')
plt.show()

# 根据特征值对特征向量降序排列
eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:, i]) for i in range(len(eig_val_sc))]  # 生成（特征向量，特征值）元祖
eig_pairs.sort(key=lambda x: x[0], reverse=True)  # 对（特征向量，特征值）元祖按照降序排列
for i in eig_pairs:
    print(i[0])

# 选出前 k 个特征值最大的特征向量
matrix_w = np.hstack((eig_pairs[0][1].reshape(3, 1), eig_pairs[1][1].reshape(3, 1)))
print('Matrix W:\n', matrix_w)

# 将样本转化为新的特征空间
matrix_w = np.hstack((eig_pairs[0][1].reshape(3, 1), eig_pairs[1][1].reshape(3, 1)))
print('Matrix W:\n', matrix_w)
transformed = matrix_w.T.dot(all_samples)
assert transformed.shape == (2, 40), "The matrix is not 2x40 dimensional."
plt.plot(transformed[0, 0:20], transformed[1, 0:20], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
plt.plot(transformed[0, 20:40], transformed[1, 20:40], '^', markersize=7, color='red', alpha=0.5, label='class2')
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.title('Transformed samples with class labels')
plt.show()
```


输出：

```txt
[[-1.0856306  -1.50629471 -2.42667924 -0.8667404   1.49138963 -0.43435128
   1.0040539   1.49073203 -1.25388067 -1.4286807  -0.25561937 -0.69987723
   0.00284592  0.28362732 -0.39089979 -0.01183049  0.97873601 -1.03878821
   0.02968323  1.75488618]
 [ 0.99734545 -0.57860025 -0.42891263 -0.67888615 -0.638902    2.20593008
   0.3861864  -0.93583387 -0.6377515  -0.14006872 -2.79858911  0.92746243
   0.68822271 -0.80536652  0.57380586  2.39236527  2.23814334  1.74371223
   1.06931597  1.49564414]
 [ 0.2829785   1.65143654  1.26593626 -0.09470897 -0.44398196  2.18678609
   0.73736858  1.17582904  0.9071052  -0.8617549  -1.7715331  -0.17363568
  -0.87953634 -1.72766949  0.33858905  0.41291216 -1.29408532 -0.79806274
   0.89070639  1.06939267]]
Mean Vector:
 [[0.42227489]
 [0.73574907]
 [0.39040387]]
Scatter Matrix:
 [[55.60258006  6.48293267 -0.30031338]
 [ 6.48293267 72.01573076  8.65376765]
 [-0.30031338  8.65376765 48.96338303]]
Covariance Matrix:
 [[ 1.42570718  0.16622904 -0.00770034]
 [ 0.16622904  1.8465572   0.22189148]
 [-0.00770034  0.22189148  1.25547136]]
Eigenvector 3: 
[[ 0.25371897]
 [-0.35516527]
 [ 0.89971346]]
Eigenvalue 3 from scatter matrix: 45.46258745029219
Eigenvalue 3 from covariance matrix: 1.1657073705203127
Scaling factor:  38.99999999999999
----------------------------------------
76.65867415791243
54.4604322442545
45.46258745029219
Matrix W:
 [[ 0.27846954  0.92632683]
 [ 0.91759056 -0.17856424]
 [ 0.28369398 -0.33171284]]
Matrix W:
 [[ 0.27846954  0.92632683]
 [ 0.91759056 -0.17856424]
 [ 0.28369398 -0.33171284]]
```

图像：

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200615/GDX00C1dQFQs.png?imageslim">
</p>

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200615/8vhw6EihgFlC.png?imageslim">
</p>

<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200615/AdgDQ7pD2Xgu.png?imageslim">
</p>

说明：


- 首先随机生成 40*3 维的符合多元高斯分布数据。数据被分为两类。
- scatter_matrix 公式：
  - $S=\sum_{k=1}^{n}\left(\boldsymbol{x}_{k}-\boldsymbol{m}\right)\left(\boldsymbol{x}_{k}-\boldsymbol{m}\right)^{T}$
  - 其中 m 是向量的均值：$\boldsymbol{m}=\frac{1}{n} \sum_{k=1}^{n} \boldsymbol{x}_{k}$（mean_vector）
- 计算协方差矩阵
  - 如果不计算散布矩阵的话，也可以用 python 里内置的 numpy.cov() 函数直接计算协方差矩阵。因为散步矩阵和协方差矩阵非常类似，散布矩阵乘以（1/N-1）就是协方差，所以他们的特征空间是完全等价的（特征向量相同，特征值用一个常数（1/N-1，这里是 1/39）等价缩放了）。
  - 协方差矩阵如下所示：
  - $\Sigma_{i}=\left[\begin{array}{ccc}{\sigma_{11}^{2}} & {\sigma_{12}^{2}} & {\sigma_{13}^{2}} \\ {\sigma_{21}^{2}} & {\sigma_{22}^{2}} & {\sigma_{23}^{2}} \\ {\sigma_{31}^{2}} & {\sigma_{32}^{2}} & {\sigma_{33}^{2}}\end{array}\right]$
- 计算相应的特征向量和特征值
  - 从结果可以发现，通过散布矩阵和协方差矩阵计算的特征空间相同，协方差矩阵的特征值*39 = 散布矩阵的特征值
- 根据特征值对特征向量降序排列
  - 我们的目标是减少特征空间的维度，即通过 PCA 方法将特征空间投影到一个小一点的子空间里，其中特征向量将会构成新的特征空间的轴。然而，特征向量只会决定轴的方向，他们的单位长度都为 1，可以用代码检验一下。
  - 因此，对于低维的子空间来说，决定丢掉哪个特征向量，就必须参考特征向量相应的特征值。通俗来说，如果一个特征向量的特征值特别小，那它所包含的数据分布的信息也很少，那么这个特征向量就可以忽略不计了。常用的方法是根据特征值对特征向量进行降序排列，选出前 k 个特征向量


- 选出前 k 个特征值最大的特征向量
  - 本文的例子是想把三维的空间降维成二维空间，现在我们把前两个最大特征值的特征向量组合起来，生成 d*k 维的特征向量矩阵 W
- 将样本转化为新的特征空间
  - 最后一步，把 2*3维的特征向量矩阵 W 带到公式 $\boldsymbol{y}=\boldsymbol{W}^{T} \times \boldsymbol{x}$ 中，将样本数据转化为新的特征空间



python 中 PCA 模块：

- matplotlib：matplotlib.mlab.PCA()
- sklearn：Dimensionality reduction 专门讲 PCA，包括传统的 PCA，以及增量 PCA，核 PCA 等等，除了 PCA 以外，还有 ZCA 白化等等，在图像处理中也经常会用到。
