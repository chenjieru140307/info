
## Hessian 矩阵

以下均假设连续可导：


$$
f^{\prime \prime}(x) ; \quad \mathbf{H}(\mathbf{x})=\nabla^{2} f(\mathbf{x})=\left[\begin{array}{cccc}{\frac{\partial^{2} f(\mathbf{x})}{\partial x_{1}^{2}}} & {\frac{\partial^{2} f(\mathbf{x})}{\partial x_{1} \partial x_{2}}} & {\cdots} & {\frac{\partial^{2} f(\mathbf{x})}{\partial x_{1} \partial x_{n}}} \\ {\frac{\partial^{2} f(\mathbf{x})}{\partial x_{2} \partial x_{1}}} & {\frac{\partial^{2} f(\mathbf{x})}{\partial x_{2}^{2}}} & {\cdots}& {\cdots} \\ {\cdots} & {\cdots}& {\cdots}& {\cdots}\\ {\frac{\partial^{2} f(\mathbf{x})}{\partial x_{n} \partial x_{1}}} & {\frac{\partial^{2} f(\mathbf{x})}{\partial x_{n} \partial x_{2}}} & {\cdots} & {\frac{\partial^{2} f(\mathbf{x})}{\partial x_{n}^{2}}}\end{array}\right]
$$


## 二次型的梯度


若 $x=\left[x_{1}, x_{2}, \cdots, x_{n}\right]$，则：

$$
\frac{\partial x^{\mathrm{T}}}{\partial x}=I
$$

式子中，$I$ 为单位矩阵，这是一个非常有用的结果。

## 一个推论

若 $A$ 与 $y$ 均与向量 $x$ 无关，则：


$$
\frac{\partial x^{\mathrm{T}} A y}{\partial x}=\frac{\partial x^{T}}{\partial x} A y=A y
$$

## 一个推论


注意到 $\boldsymbol{y}^{\mathrm{T}} \boldsymbol{A} \boldsymbol{x}=\left\langle\boldsymbol{A}^{\mathrm{T}} \boldsymbol{y}, \boldsymbol{x}\right\rangle=\left\langle\boldsymbol{x}, \boldsymbol{A}^{\mathrm{T}} \boldsymbol{y}\right\rangle=\boldsymbol{x}^{\mathrm{T}} \boldsymbol{A}^{\mathrm{T}} \boldsymbol{y}$，故：


$$
\frac{\partial y^{\mathrm{T}} A x}{\partial x}=\frac{\partial x^{\mathrm{T}} A^{T} y}{\partial x}=A^{\mathrm{T}} y
$$

## 一个推论

由于 $x^{\mathrm{T}} A x=\sum_{i=1}^{n} \sum_{j=1}^{n} A_{i j} x_{i} x_{j}$，可求出梯度 $\frac{\partial x^{\mathrm{T}} A x}{\partial x}$ 的第 $k$ 个分量为：

$$
\left[\frac{\partial x^{\mathrm{T}} A x}{\partial x}\right]_{k}=\frac{\partial}{\partial x_{k}} \sum_{i=1}^{n} \sum_{j=1}^{n} A_{i j} x_{i} x_{j}=\sum_{i=1}^{n} A_{i k} x_{i}+\sum_{j=1}^{n} A_{k j} x_{j}
$$

即，有：

$$
\frac{\partial x^{\mathrm{T}} A x}{\partial x}=A x+A^{\mathrm{T}} x
$$

特别的，若 $A$ 为对称矩阵，则：

$$
\frac{\partial x^{\mathrm{T}} A x}{\partial x}=2 A x
$$
