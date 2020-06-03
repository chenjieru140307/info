# Scipy

1. `Scipy`的核心计算部分是一些`Fortran`数值计算库：
   - 线性代数使用`LAPACK`库
   - 快速傅立叶变换使用`FFTPACK`库
   - 常微分方程求解使用`ODEPACK`库
   - 非线性方程组求解以及最小值求解使用`MINPACK` 库

## 一、 常数和特殊函数

### 1. constants 模块

1. `scipy`的`constants`模块包含了众多的物理常数：
   - `constants.c`：真空中的光速
   - `constants.h`：普朗克常数
   - `constants.g`：重力加速度
   - `constants.m_e`：电子质量 <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200602/rcGmcMnoTndt.JPG">
   </p>
   
2. 在字典`constants.physical_constants`中，以物理常量名为键，对应的值是一个含有三元素的元组，分别为：常量值、单位、误差。 <p align="center">
  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/PpYNGgiXpqIC.JPG">
</p>

3. `constants`模块还包含了许多单位信息，它们是 1 单位的量转换成标准单位时的数值：
   - `C.mile`：一英里对应的米
   - `C.inch`：一英寸对应的米
   - `C.gram`：一克等于多少千克
   - `C.pound`：一磅对应多少千克 <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200602/WcAKvssv8qzr.JPG">
   </p>
   

### 2. special 模块

1. `scipy`的`special`模块是个非常完整的函数库，其中包含了基本数学函数、特殊数学函数以及`numpy`中出现的所有函数。这些特殊函数都是`ufunc`函数，支持数组的广播运算。

2. `gamma`函数：`special.gamma(x)`。其数学表达式为：

   $$\Gamma(z)=\int_{0}^{+\infty} t^{z-1}e^{-t}dt$$

<p align="center">
  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/Vgr0F8L9KrWs.JPG">
</p>


`gamma`函数是阶乘函数在实数和复数系上的扩展，增长的非常迅速。`1000`的阶乘已经超过了双精度浮点数的表示范围。为了计算更大范围，可以使用 `gammaln`函数来计算$\ln(|\Gamma(z)|)$的值： `special.gammaln(x)` <p align="center">
  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/TSeBSI6QlQ3i.JPG">
</p>


计算雅可比椭圆函数：`sn,cn,dn,phi=special.ellipj(u,m)`，其中：

- `sn`=$\sin(\phi)$
- `cn`=$\cos(\phi)$
- `dn`=$\sqrt{1-m\sin^{2}(\phi)}$
- `phi`=$\phi$
- `u`=$\int_{0}^{\phi}\frac {1}{\sqrt{1-m\sin^{2}(\phi)}}d\theta$

<p align="center">
  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/aS4s7PMzLt15.JPG">
</p>


`special`模块的某些函数并不是数学意义上的特殊函数。如`log1p(x)`计算的是$\log(1+x)$。这是因为浮点数精度限制，无法精确地表示非常接近 1的实数。因此$\log(1+10^{-20})$的值为 0 。但是$\log1p(1+10^{-20})$的值可以计算。 <p align="center">
  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/tjU2lr4kBvLk.JPG">
</p>


## 二、 拟合与优化

1. `scipy`的`optimize`模块提供了许多数值优化算法。

2. 求解非线性方程组：

   ```
     scipy.optimize.fsolve(func, x0, args=(), fprime=None, full_output=0, col_deriv=0,
   ```

   ```
     xtol=1.49012e-08, maxfev=0, band=None, epsfcn=None, factor=100, diag=None)
   ```

   - `func`：是个可调用对象，它代表了非线性方程组。给他传入方程组的各个参数，它返回各个方程残差（残差为零则表示找到了根）
   - `x0`：预设的方程的根的初始值
   - `args`：一个元组，用于给`func`提供额外的参数。
   - `fprime`：用于计算`func`的雅可比矩阵（按行排列）。默认情况下，算法自动推算
   - `full_output`：如果为`True`，则给出更详细的输出
   - `col_deriv`：如果为`True`，则计算雅可比矩阵更快（按列求导）
   - `xtol`：指定算法收敛的阈值。当误差小于该阈值时，算法停止迭代
   - `maxfev`：设定算法迭代的最大次数。如果为零，则为 `100*(N+1)`，`N`为`x0`的长度
   - `band`：If set to a two-sequence containing the number of sub- and super-diagonals within the band of the Jacobi matrix, the Jacobi matrix is considered banded (only for fprime=None)
   - `epsfcn`：采用前向差分算法求解雅可比矩阵时的步长。
   - `factor`：它决定了初始的步长
   - `diag`：它给出了每个变量的缩放因子

   返回值：

   - `x`：方程组的根组成的数组
   - `infodict`：给出了可选的输出。它是个字典，其中的键有：
     - `nfev`：`func`调用次数
     - `njev`：雅可比函数调用的次数
     - `fvec`：最终的`func`输出
     - `fjac`：the orthogonal matrix, q, produced by the QR factorization of the final approximate Jacobian matrix, stored column wise
     - `r`：upper triangular matrix produced by QR factorization of the same matrix
   - `ier`：一个整数标记。如果为 1，则表示根求解成功
   - `mesg`：一个字符串。如果根未找到，则它给出详细信息

   假设待求解的方程组为：

   $$f_1(x_1,x_2,x_3)=0\\ f_2(x_1,x_2,x_3)=0\\ f_3(x_1,x_2,x_3)=0$$

   那么我们的`func`函数为：

   ```
   xxxxxxxxxx
   ```

   ```
     def func(x):
   ```

   ```
       x1,x2,x3=x.tolist() # x 为向量，形状为 (3,)
   ```

   ```
       return np.array([f1(x1,x2,x3),f2(x1,x2,x3),f3(x1,x2,x3)])
   ```

   > 数组的`.tolist()`方法能获得标准的`python`列表

   而雅可比矩阵为：

   $$J=\begin{bmatrix} \frac{\partial f_1}{\partial x_1}&\frac{\partial f_1}{\partial x_2}&\frac{\partial f_1}{\partial x_3}\\ \frac{\partial f_2}{\partial x_1}&\frac{\partial f_2}{\partial x_2}&\frac{\partial f_2}{\partial x_3}\\ \frac{\partial f_3}{\partial x_1}&\frac{\partial f_3}{\partial x_2}&\frac{\partial f_3}{\partial x_3}\\ \end{bmatrix}$$

   ```
   xxxxxxxxxx
   ```

   ```
     def fprime(x):
   ```

   ```
       x1,x2,x3=x.tolist() # x 为向量，形状为 (3,)
   ```

   ```
        return np.array([[df1/dx1,df1/dx2,df1/df3],
   ```

   ```
            [df2/dx1,df2/dx2,df2/df3],
   ```

   ```
            [df3/dx1,df3/dx2,df3/df3]])
   ```

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/qHDXNzbY5vYn.JPG">
  </p>
  

3. 最小二乘法拟合数据：

   ```
   xxxxxxxxxx
   ```

   ```
     scipy.optimize.leastsq(func, x0, args=(), Dfun=None, full_output=0, col_deriv=0,
   ```

   ```
     ftol=1.49012e-08, xtol=1.49012e-08, gtol=0.0, maxfev=0, epsfcn=None, 
   ```

   ```
     factor=100, diag=None)
   ```

   - `func`：是个可调用对象，给出每次拟合的残差。最开始的参数是待优化参数；后面的参数由`args`给出
   - `x0`：初始迭代值
   - `args`：一个元组，用于给`func`提供额外的参数。
   - `Dfun`：用于计算`func`的雅可比矩阵（按行排列）。默认情况下，算法自动推算。它给出残差的梯度。最开始的参数是待优化参数；后面的参数由`args`给出
   - `full_output`：如果非零，则给出更详细的输出
   - `col_deriv`：如果非零，则计算雅可比矩阵更快（按列求导）
   - `ftol`：指定相对的均方误差的阈值
   - `xtol`：指定参数解收敛的阈值
   - `gtol`：Orthogonality desired between the function vector and the columns of the Jacobian
   - `maxfev`：设定算法迭代的最大次数。如果为零：如果为提供了`Dfun`，则为 `100*(N+1)`，`N`为`x0`的长度；如果未提供`Dfun`,则为`200*(N+1)`
   - `epsfcn`：采用前向差分算法求解雅可比矩阵时的步长。
   - `factor`：它决定了初始的步长
   - `diag`：它给出了每个变量的缩放因子

   返回值：

   - `x`：拟合解组成的数组
   - `cov_x`：Uses the fjac and ipvt optional outputs to construct an estimate of the jacobian around the solution
   - `infodict`：给出了可选的输出。它是个字典，其中的键有：
     - `nfev`：`func`调用次数
     - `fvec`：最终的`func`输出
     - `fjac`：A permutation of the R matrix of a QR factorization of the final approximate Jacobian matrix, stored column wise.
     - `ipvt`：An integer array of length N which defines a permutation matrix, p, such that fjac*p = q*r, where r is upper triangular with diagonal elements of nonincreasing magnitude
   - `ier`：一个整数标记。如果为 1/2/3/4，则表示拟合成功
   - `mesg`：一个字符串。如果解未找到，则它给出详细信息

   假设我们拟合的函数是$f(x,y;a,b,c)=0$，其中$a,b,c$为参数。假设数据点的横坐标为$X$，纵坐标为$Y$，那么我们可以给出`func`为：

   ```
   xxxxxxxxxx
   ```

   ```
     def func(p,x,y):
   ```

   ```
       a,b,c=p.tolist() # 这里p 为数组，形状为 (3,)； x,y 也是数组，形状都是 (N,)
   ```

   ```
       return f(x,y;a,b,c))
   ```

   其中 `args=(X,Y)`

   而雅可比矩阵为$[\frac{\partial f}{\partial a},\frac{\partial f}{\partial b},\frac{\partial f}{\partial c}]$，给出`Dfun`为：

   ```
   xxxxxxxxxx
   ```

   ```
     def func(p,x,y):
   ```

   ```
       a,b,c=p.tolist()
   ```

   ```
       return np.c_[df/da,df/db,df/dc]# 这里p为数组，形状为 (3,)；x,y 也是数组，形状都是 (N,)
   ```

   其中 `args=(X,Y)`

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/bYG9dekyyIUT.JPG">
  </p>
  

4. `scipy`提供了另一个函数来执行最小二乘法的曲线拟合：

   ```
   xxxxxxxxxx
   ```

   ```
    scipy.optimize.curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, 
   ```

   ```
       check_finite=True, bounds=(-inf, inf), method=None, **kwargs)
   ```

   - `f`：可调用函数，它的优化参数被直接传入。其第一个参数一定是`xdata`，后面的参数是待优化参数
   - `xdata`：`x`坐标
   - `ydata`：`y`坐标
   - `p0`：初始迭代值
   - `sigma`：`y`值的不确定性的度量
   - `absolute_sigma`： If False, sigma denotes relative weights of the data points. The returned covariance matrix pcov is based on estimated errors in the data, and is not affected by the overall magnitude of the values in sigma. Only the relative magnitudes of the sigma values matter.If True, sigma describes one standard deviation errors of the input data points. The estimated covariance in pcov is based on these values.
   - `check_finite`：如果为`True`，则检测输入中是否有`nan`或者`inf`
   - `bounds`：指定变量的取值范围
   - `method`：指定求解算法。可以为 `'lm'/'trf'/'dogbox'`
   - `kwargs`：传递给 `leastsq/least_squares`的关键字参数。

   返回值：

   - `popt`：最优化参数
   - `pcov`：The estimated covariance of popt.

   假设我们拟合的函数是$y=f(x;a,b,c)$，其中$a,b,c$为参数。假设数据点的横坐标为$X$，纵坐标为$Y$，那么我们可以给出`func`为：

   ```
   xxxxxxxxxx
   ```

   ```
     def func(x,a,b,c):
   ```

   ```
       return f(x;a,b,c)#x 为数组，形状为 (N,)
   ```

<p align="center">
  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200602/yFrcHNTPm8b7.JPG">
</p>


5. 求函数最小值：

   ```
   xxxxxxxxxx
   ```

   ```
     scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None, 
   ```

   ```
     hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
   ```

   - `fun`：可调用对象，待优化的函数。最开始的参数是待优化的自变量；后面的参数由`args`给出
   - `x0`：自变量的初始迭代值
   - `args`：一个元组，提供给`fun`的额外的参数
   - `method`：一个字符串，指定了最优化算法。可以为：`'Nelder-Mead'`、`'Powell'`、`'CG'`、`'BFGS'`、`'Newton-CG'`、`'L-BFGS-B'`、`'TNC'`、`'COBYLA'`、`'SLSQP'`、 `'dogleg'`、`'trust-ncg'`
   - `jac`：一个可调用对象（最开始的参数是待优化的自变量；后面的参数由`args`给出），雅可比矩阵。只在`CG/BFGS/Newton-CG/L-BFGS-B/TNC/SLSQP/dogleg/trust-ncg`算法中需要。如果`jac`是个布尔值且为`True`，则会假设`fun`会返回梯度；如果是个布尔值且为`False`，则雅可比矩阵会被自动推断（根据数值插值）。
   - `hess/hessp`：可调用对象（最开始的参数是待优化的自变量；后面的参数由`args`给出），海森矩阵。只有`Newton-CG/dogleg/trust-ncg`算法中需要。二者只需要给出一个就可以，如果给出了`hess`，则忽略`hessp`。如果二者都未提供，则海森矩阵自动推断
   - `bounds`：一个元组的序列，给定了每个自变量的取值范围。如果某个方向不限，则指定为`None`。每个范围都是一个`(min,max)`元组。
   - `constrants`：一个字典或者字典的序列，给出了约束条件。只在`COBYLA/SLSQP`中使用。字典的键为：
     - `type`：给出了约束类型。如`'eq'`代表相等；`'ineq'`代表不等
     - `fun`：给出了约束函数
     - `jac`：给出了约束函数的雅可比矩阵（只用于`SLSQP`）
     - `args`：一个序列，给出了传递给`fun`和`jac`的额外的参数
   - `tol`：指定收敛阈值
   - `options`：一个字典，指定额外的条件。键为：
     - `maxiter`：一个整数，指定最大迭代次数
     - `disp`：一个布尔值。如果为`True`，则打印收敛信息
   - `callback`：一个可调用对象，用于在每次迭代之后调用。调用参数为`x_k`，其中`x_k`为当前的参数向量

   返回值：返回一个`OptimizeResult`对象。其重要属性为：

   - `x`：最优解向量
   - `success`：一个布尔值，表示是否优化成功
   - `message`：描述了迭代终止的原因

   假设我们要求解最小值的函数为：$f(x,y)=(1-x)^{2}+100(y-x^{2})^{2}$，则雅可比矩阵为：

   $$\left[\frac{\partial f(x,y)}{\partial x},\frac{\partial f(x,y)}{\partial y}\right]$$

   则海森矩阵为：

   $$\begin{bmatrix} \frac{\partial^{2} f(x,y)}{\partial x^{2}}&\frac{\partial^{2} f(x,y)}{\partial x\partial y}\\ \frac{\partial^{2} f(x,y)}{\partial y\partial x}&\frac{\partial^{2} f(x,y)}{\partial y^{2}} \end{bmatrix}$$

   于是有：

   ```
   xxxxxxxxxx
   ```

   ```
     def fun(p):
   ```

   ```
       x,y=p.tolist()#p 为数组，形状为 (2,)
   ```

   ```
       return f(x,y)
   ```

   ```
     def jac(p):
   ```

   ```
       x,y=p.tolist()#p 为数组，形状为 (2,)
   ```

   ```
       return np.array([df/dx,df/dy])
   ```

   ```
     def hess(p):
   ```

   ```
       x,y=p.tolist()#p 为数组，形状为 (2,)
   ```

   ```
       return np.array([[ddf/dxx,ddf/dxdy],[ddf/dydx,ddf/dyy]])
   ```

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/GrlRjUx1qSby.JPG">
  </p>
  

6. 常规的最优化算法很容易陷入局部极值点。`basinhopping`算法是一个寻找全局最优点的算法。

   ```
   xxxxxxxxxx
   ```

   ```
   scipy.optimize.basinhopping(func, x0, niter=100, T=1.0, stepsize=0.5, 
   ```

   ```
     minimizer_kwargs=None,take_step=None, accept_test=None, callback=None, 
   ```

   ```
     interval=50, disp=False, niter_success=None)
   ```

   - `func`：可调用函数。为待优化的目标函数。最开始的参数是待优化的自变量；后面的参数由`minimizer_kwargs`字典给出
   - `x0`：一个向量，设定迭代的初始值
   - `niter`：一个整数，指定迭代次数
   - `T`：一个浮点数，设定了“温度”参数。
   - `stepsize`：一个浮点数，指定了步长
   - `minimizer_kwargs`：一个字典，给出了传递给`scipy.optimize.minimize`的额外的关键字参数。
   - `take_step`：一个可调用对象，给出了游走策略
   - `accept_step`：一个可调用对象，用于判断是否接受这一步
   - `callback`：一个可调用对象，每当有一个极值点找到时，被调用
   - `interval`：一个整数，指定`stepsize`被更新的频率
   - `disp`：一个布尔值，如果为`True`，则打印状态信息
   - `niter_success`：一个整数。Stop the run if the global minimum candidate remains the same for this number of iterations.

   返回值：一个`OptimizeResult`对象。其重要属性为：

   - `x`：最优解向量
   - `success`：一个布尔值，表示是否优化成功
   - `message`：描述了迭代终止的原因

   假设我们要求解最小值的函数为：$f(x,y)=(1-x)^{2}+100(y-x^{2})^{2}$，于是有：

   ```
   xxxxxxxxxx
   ```

   ```
     def fun(p):
   ```

   ```
       x,y=p.tolist()#p 为数组，形状为 (2,)
   ```

   ```
       return f(x,y)
   ```

   <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200602/niRF5NW8EJRv.JPG">
   </p>
   

## 三、线性代数

1. `numpy`和`scipy`都提供了线性代数函数库`linalg`。但是`scipy`的线性代数库比`numpy`更加全面。

2. `numpy`中的求解线性方程组：`numpy.linalg.solve(a, b)`。而`scipy`中的求解线性方程组：

   ```
   xxxxxxxxxx
   ```

   ```
   scipy.linalg.solve(a, b, sym_pos=False, lower=False, overwrite_a=False, 
   ```

   ```
     overwrite_b=False, debug=False, check_finite=True)
   ```

   - `a`：方阵，形状为 `(M,M)`
   - `b`：一维向量，形状为`(M,)`。它求解的是线性方程组$\mathbf A \mathbf x=\mathbf b$。如果有$k$个线性方程组要求解，且 `a`，相同，则 `b`的形状为 `(M,k)`
   - `sym_pos`：一个布尔值，指定`a`是否正定的对称矩阵
   - `lower`：一个布尔值。如果`sym_pos=True`时：如果为`lower=True`，则使用`a`的下三角矩阵。默认使用`a`的上三角矩阵。
   - `overwrite_a`：一个布尔值，指定是否将结果写到`a`的存储区。
   - `overwrite_b`：一个布尔值，指定是否将结果写到`b`的存储区。
   - `check_finite`：如果为`True`，则检测输入中是否有`nan`或者`inf`

   返回线性方程组的解。

   通常求解矩阵$\mathbf A^{-1}\mathbf B$，如果使用`solve(A,B)`，要比先求逆矩阵、再矩阵相乘来的快。

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/FoPsuBIgQBCP.JPG">
  </p>
  

3. 矩阵的`LU`分解：

   ```
   xxxxxxxxxx
   ```

   ```
   scipy.linalg.lu_factor(a, overwrite_a=False, check_finite=True)
   ```

   - `a`：方阵，形状为`(M,M)`，要求非奇异矩阵
   - `overwrite_a`：一个布尔值，指定是否将结果写到`a`的存储区。
   - `check_finite`：如果为`True`，则检测输入中是否有`nan`或者`inf`

   返回:

   - `lu`：一个数组，形状为`(N,N)`，该矩阵的上三角矩阵就是`U`，下三角矩阵就是`L`（`L`矩阵的对角线元素并未存储，因为它们全部是1）
   - `piv`：一个数组，形状为`(N,)`。它给出了`P`矩阵：矩阵`a`的第 `i`行被交换到了第`piv[i]`行

   矩阵`LU`分解：

   $$\mathbf A=\mathbf P \mathbf L \mathbf U$$

   其中：$\mathbf P$为转置矩阵，该矩阵任意一行只有一个1，其他全零；任意一列只有一个1，其他全零。$\mathbf L$为单位下三角矩阵（对角线元素为1），$\mathbf U$为上三角矩阵（对角线元素为0）

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/sgTiM9LKj6QX.JPG">
  </p>
   

4. 当对矩阵进行了`LU`分解之后，可以方便的求解线性方程组。

   ```
   xxxxxxxxxx
   ```

   ```
   scipy.linalg.lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True)
   ```

   - `lu_and_piv`：一个元组，由`lu_factor`返回
   - `b`：一维向量，形状为`(M,)`。它求解的是线性方程组$\mathbf A \mathbf x=\mathbf b$。如果有$k$个线性方程组要求解，且 `a`，相同，则 `b`的形状为 `(M,k)`
   - `overwrite_b`：一个布尔值，指定是否将结果写到`b`的存储区。
   - `check_finite`：如果为`True`，则检测输入中是否有`nan`或者`inf`
   - `trans`：指定求解类型.
     - 如果为 0 ，则求解：$\mathbf A \mathbf x=\mathbf b$
     - 如果为 1 ，则求解：$\mathbf A^{T} \mathbf x=\mathbf b$
     - 如果为 2 ，则求解：$\mathbf A^{H} \mathbf x=\mathbf b$

5. `lstsq`比`solve`更一般化，它不要求矩阵$\mathbf A$是方阵。 它找到一组解$\mathbf x$，使得$||\mathbf b-\mathbf A\mathbf x||$最小，我们称得到的结果为最小二乘解。

   ```
   xxxxxxxxxx
   ```

   ```
   scipy.linalg.lstsq(a, b, cond=None, overwrite_a=False, overwrite_b=False,
   ```

   ```
     check_finite=True, lapack_driver=None)
   ```

   - `a`：为矩阵，形状为`(M,N)`
   - `b`：一维向量，形状为`(M,)`。它求解的是线性方程组$\mathbf A \mathbf x=\mathbf b$。如果有$k$个线性方程组要求解，且 `a`，相同，则 `b`的形状为 `(M,k)`
   - `cond`：一个浮点数，去掉最小的一些特征值。当特征值小于`cond * largest_singular_value`时，该特征值认为是零
   - `overwrite_a`：一个布尔值，指定是否将结果写到`a`的存储区。
   - `overwrite_b`：一个布尔值，指定是否将结果写到`b`的存储区。
   - `check_finite`：如果为`True`，则检测输入中是否有`nan`或者`inf`
   - `lapack_driver`：一个字符串，指定求解算法。可以为：`'gelsd'/'gelsy'/'gelss'`。默认的`'gelsd'`效果就很好，但是在许多问题上`'gelsy'`效果更好。

   返回值：

   - `x`：最小二乘解，形状和`b`相同
   - `residures`：残差。如果$rank(\mathbf A)$大于`N`或者小于`M`，或者使用了`gelsy`，则是个空数组；如果`b`是一维的，则它的形状是`(1,)`；如果`b`是二维的，则形状为`(K,)`
   - `rank`：返回矩阵`a`的秩
   - `s`：`a`的奇异值。如果使用`gelsy`，则返回`None`

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/GEgysDYERQMu.JPG">
  </p>
  

6. 求解特征值和特征向量：

   ```
   xxxxxxxxxx
   ```

   ```
   scipy.linalg.eig(a, b=None, left=False, right=True, overwrite_a=False, 
   ```

   ```
     overwrite_b=False, check_finite=True)
   ```

   - `a`：一个方阵，形状为`(M,M)`。待求解特征值和特征向量的矩阵。
   - `b`：默认为`None`，表示求解标准的特征值问题：$\mathbf A\mathbf x=\lambda\mathbf x$。 也可以是一个形状与`a`相同的方阵，此时表示广义特征值问题：$\mathbf A\mathbf x=\lambda\mathbf B\mathbf x$
   - `left`：一个布尔值。如果为`True`，则计算左特征向量
   - `right`：一个布尔值。如果为`True`，则计算右特征向量
   - `overwrite_a`：一个布尔值，指定是否将结果写到`a`的存储区。
   - `overwrite_b`：一个布尔值，指定是否将结果写到`b`的存储区。
   - `check_finite`：如果为`True`，则检测输入中是否有`nan`或者`inf`

   返回值：

   - `w`：一个一维数组，代表了`M`特特征值。
   - `vl`：一个数组，形状为`(M,M)`，表示正则化的左特征向量（每个特征向量占据一列，而不是一行）。仅当`left=True`时返回
   - `vr`：一个数组，形状为`(M,M)`，表示正则化的右特征向量（每个特征向量占据一列，而不是一行）。仅当`right=True`时返回

   > `numpy`提供了`numpy.linalg.eig(a)`来计算特征值和特征向量

   右特征值：$\mathbf A\mathbf x_r=\lambda\mathbf x_r$；左特征值：$\mathbf A^{H}\mathbf x_l=conj(\lambda)\mathbf x_l$，其中$conj(\lambda)$为特征值的共轭。

   令$\mathbf P=[\mathbf x_{r1},\mathbf x_{r2},\cdots,\mathbf x_{rM}]$，令

   $$\mathbf \Sigma=\begin{bmatrix} \lambda_1&0&0&\cdots&0\\ 0&\lambda_2&0&\cdots&0\\ \vdots&\vdots&\vdots&\ddots&\vdots\\ 0&0&0&\cdots&\lambda_M\\ \end{bmatrix}$$

   则有：

   $$\mathbf A \mathbf P=\mathbf P \mathbf \Sigma\Longrightarrow \mathbf A=\mathbf P \mathbf \Sigma\mathbf P ^{-1}$$

   <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200602/LowMUh5l11pX.JPG">
   </p>
   

7. 矩阵的奇异值分解： 设矩阵$\mathbf A$为$M\times N$阶的矩阵，则存在一个分解，使得：$\mathbf A=\mathbf U \mathbf \Sigma \mathbf V^{H}$，其中$\mathbf U$为$M \times M$阶酉矩阵；$\mathbf \Sigma$为半正定的$M\times N$阶的对焦矩阵； 而$\mathbf V$为$N \times N$阶酉矩阵。

  $\mathbf \Sigma$对角线上的元素为$\mathbf A$的奇异值，通常按照从大到小排列。

   ```
   xxxxxxxxxx
   ```

   ```
   scipy.linalg.svd(a, full_matrices=True, compute_uv=True, overwrite_a=False, 
   ```

   ```
     check_finite=True, lapack_driver='gesdd')
   ```

   - `a`：一个矩阵，形状为`(M,N)`，待分解的矩阵。
   - `full_matrices`：如果为`True`，则$\mathbf U$的形状为`(M,M)`、$\mathbf V^{H}$的形状为`(N,N)`；否则$\mathbf U$的形状为`(M,K)`、$\mathbf V^{H}$的形状为`(K,N)`，其中 `K=min(M,N)`
   - `compute_uv`：如果`True`，则结果中额外返回`U`以及`Vh`；否则只返回奇异值
   - `overwrite_a`：一个布尔值，指定是否将结果写到`a`的存储区。
   - `overwrite_b`：一个布尔值，指定是否将结果写到`b`的存储区。
   - `check_finite`：如果为`True`，则检测输入中是否有`nan`或者`inf`
   - `lapack_driver`：一个字符串，指定求解算法。可以为：`'gesdd'/'gesvd'`。默认的`'gesdd'`。

   返回值：

   - `U`：$\mathbf U$矩阵
   - `s`：奇异值，它是一个一维数组，按照降序排列。长度为 `K=min(M,N)`
   - `Vh`：就是$\mathbf V^{H}$矩阵

   > 判断两个数组是否近似相等 `np.allclose(a1,a2)`（主要是浮点数的精度问题）

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/UTqh0PjqzDmg.JPG">
  </p>
  

## 四、 统计

1. `scipy`中的`stats`模块中包含了很多概率分布的随机变量
   - 所有的连续随机变量都是`rv_continuous`的派生类的对象
   - 所有的离散随机变量都是`rv_discrete`的派生类的对象

### 1. 连续随机变量

1. 查看所有的连续随机变量：

   ```
   xxxxxxxxxx
   ```

   ```
   [k for k,v in stats.__dict__.items() if isinstance(v,stats.rv_continuous)]
   ```

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/Hzry3HItXWOV.JPG">
  </p>
  

2. 连续随机变量对象都有如下方法：

   - `rvs(*args, **kwds)`：获取该分布的一个或者一组随机值
   - `pdf(x, *args, **kwds)`：概率密度函数在`x`处的取值
   - `logpdf(x, *args, **kwds)` ：概率密度函数在`x`处的对数值
   - `cdf(x, *args, **kwds)`：累积分布函数在`x`处的取值
   - `logcdf(x, *args, **kwds)`：累积分布函数在`x`处的对数值
   - `sf(x, *args, **kwds)` ：生存函数在`x`处的取值，它等于`1-cdf(x)`
   - `logsf(x, *args, **kwds)` ：生存函数在`x`处的对数值
   - `ppf(q, *args, **kwds)`：累积分布函数的反函数
   - `isf`(q, *args, **kwds) ：生存函数的反函数
   - `moment(n, *args, **kwds)` n-th order non-central moment of distribution.
   - `stats(*args, **kwds)`：计算随机变量的期望值和方差值等统计量
   - `entropy(*args, **kwds)` ：随机变量的微分熵
   - `expect([func, args, loc, scale, lb, ub, ...])`：计算$func(\cdot)$的期望值
   - `median(*args, **kwds)`：计算该分布的中值
   - `mean(*args, **kwds)`：计算该分布的均值
   - `std(*args, **kwds)`：计算该分布的标准差
   - `var(*args, **kwds)`：计算该分布的方差
   - `interval(alpha, *args, **kwds)`：Confidence interval with equal areas around the median.
   - `__call__(*args, **kwds)`：产生一个参数冻结的随机变量
   - `fit(data, *args, **kwds)` ：对一组随机取样进行拟合，找出最适合取样数据的概率密度函数的系数
   - `fit_loc_scale(data, *args)`：Estimate loc and scale parameters from data using 1st and 2nd moments.
   - `nnlf(theta, x)`：返回负的似然函数

   其中的`args/kwds`参数可能为（具体函数具体分析）：

   - `arg1, arg2, arg3,...`: array_like.The shape parameter(s) for the distribution
   - `loc` : array_like.location parameter (default=0)
   - `scale` : array_like.scale parameter (default=1)
   - `size` : int or tuple of ints.Defining number of random variates (default is 1).
   - `random_state` : None or int or np.random.RandomState instance。If int or RandomState, use it for drawing the random variates. If None, rely on self.random_state. Default is None.

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/0hA4ceaGkWFa.JPG">
  </p>
  

3. 这些连续随机变量可以像函数一样调用，通过`loc`和`scale`参数可以指定随机变量的偏移和缩放系数。

   - 对于正态分布的随机变量而言，这就是期望值和标准差

### 2. 离散随机变量

1. 查看所有的连续随机变量：

   ```
   xxxxxxxxxx
   ```

   ```
     [k for k,v in stats.__dict__.items() if isinstance(v,stats.rv_discrete)]
   ```

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/sgL5tWhzSaFr.JPG">
  </p>
  
2. 离散随机变量对象都有如下方法：

   - `rvs(, loc=0, size=1)`：生成随机值
   - `pmf(x, , loc=0)`：概率密度函数在`x`处的值
   - `logpmf(x, , loc=0)`：概率密度函数在`x`处的对数值
   - `cdf(x, , loc=0)`：累积分布函数在`x`处的取值
   - `logcdf(x, , loc=0)`：累积分布函数在`x`处的对数值
   - `sf(x, , loc=0)` ：生存函数在`x`处的值
   - `logsf(x, , loc=0, scale=1)`：生存函数在`x`处的对数值
   - `ppf(q, , loc=0)` ：累积分布函数的反函数
   - `isf(q, , loc=0)` ：生存函数的反函数
   - `moment(n, , loc=0)`：non-central n-th moment of the distribution. May not work for array arguments.
   - `stats(, loc=0, moments='mv')` ：计算期望方差等统计量
   - `entropy(, loc=0)`：计算熵
   - `expect(func=None, args=(), loc=0, lb=None, ub=None, conditional=False)`：Expected value of a function with respect to the distribution. Additional kwd arguments passed to integrate.quad
   - `median(, loc=0)`：计算该分布的中值
   - `mean(, loc=0)`：计算该分布的均值
   - `std(, loc=0)`：计算该分布的标准差
   - `var(, loc=0)`：计算该分布的方差
   - `interval(alpha, , loc=0)` Interval that with alpha percent probability contains a random realization of this distribution.
   - `__call__(, loc=0)`：产生一个参数冻结的随机变量

3. 我们也可以通过`rv_discrete`类自定义离散概率分布：

   ```
   xxxxxxxxxx
   ```

   ```
     x=range(1,7)
   ```

   ```
     p=(0.1,0.3,0.1,0.3,0.1,0.1)
   ```

   ```
     stats.rv_discrete(values=(x,p))
   ```

   只需要传入一个`values`关键字参数，它是一个元组。元组的第一个成员给出了随机变量的取值集合，第二个成员给出了随机变量每一个取值的概率

### 3. 核密度估计

1. 通常我们可以用直方图来观察随机变量的概率密度。但是直方图有个缺点：你选取的直方图区间宽度不同，直方图的形状也发生变化。核密度估计就能很好地解决这一问题。

2. 核密度估计的思想是：对于随机变量的每一个采样点$x_i$，我们认为它代表一个以该点为均值、$s$为方差的一个正态分布的密度函数$f_i(x_i;s)$。将所有这些采样点代表的密度函数叠加并归一化，则得到了核密度估计的一个概率密度函数：

   $$\frac 1N \sum_{i=1}^{N}f_i(x_i;s)$$

   其中：

   - 归一化操作就是$\frac 1N$，因为每个点代表的密度函数的积分都是 1
   -$s$就是带宽参数，它代表了每个正态分布的形状

   如果采用其他的分布而不是正态分布，则得到了其他分布的核密度估计。

3. 核密度估计的原理是：如果某个样本点出现了，则它发生的概率就很高，同时跟他接近的样本点发生的概率也比较高。

4. 正态核密度估计：

  ```
  xxxxxxxxxx
  ```

  ```
  class scipy.stats.gaussian_kde(dataset, bw_method=None)
  ```

  参数：

  - `dataset`：被估计的数据集。
  - `bw_method`：用于设定带宽$s$。可以为：
    - 字符串：如`'scott'/'silverman'`。默认为`'scott'`
    - 一个标量值。此时带宽是个常数
    - 一个可调用对象。该可调用对象的参数是`gaussian_kde`，返回一个标量值

  属性：

  - `dataset`：被估计的数据集
  - `d`：数据集的维度
  - `n`：数据点的个数
  - `factor`：带宽
  - `covariance`：数据集的相关矩阵

  方法：

  - `evaluate(points)`：估计样本点的概率密度
  - `__call__(points)`：估计样本点的概率密度
  - `pdf(x)`：估计样本的概率密度

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/3KlKhMJyWMno.JPG">
  </p>
  

5. 带宽系数对核密度估计的影响：当带宽系数越大，核密度估计曲线越平滑。 <p align="center">
  <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200602/Rrx7gLTdkpLL.JPG">
</p>


### 4. 常见分布

1. 二项分布：假设试验只有两种结果：成功的概率为$p$，失败的概率为$1-p$。 则二项分布描述了独立重复地进行$n$次试验中，成功$k$次的概率。

   - 概率质量函数：

     $$f(k;n,p)=\frac{n!}{k!(n-k)!}p^{k}(1-p)^{n-k}$$

   - 期望：$np$

   - 方差：$np(1-p)$

   `scipy.stats.binom`使用`n`参数指定$n$；`p`参数指定$p$；`loc`参数指定平移

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/m0rfptPWPGy3.JPG">
  </p>
  

2. 泊松分布：泊松分布使用$\lambda$描述单位时间（或者单位面积）中随机事件发生的平均次数（只知道平均次数，具体次数是个随机变量）。

   - 概率质量函数：

     $$f(k;\lambda)=\frac{e^{-\lambda}\lambda^{k}}{k!}$$

     其物理意义是：单位时间内事件发生$k$次的概率

   - 期望：$\lambda$

   - 方差：$\lambda$

   在二项分布中，如果$n$很大，而$p$很小。乘积$np$可以视作$\lambda$，此时二项分布近似于泊松分布。

   泊松分布用于描述单位时间内随机事件发生的次数的分布的情况。

   `scipy.stats.poisson`使用`mu`参数来给定$\lambda$，使用 `loc` 参数来平移。

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/xXs5e3wNUOul.JPG">
  </p>
  

3. 用均匀分布模拟泊松分布：

   ```
   xxxxxxxxxx
   ```

   ```
     def make_poisson(lmd,tm):
   ```

   ```
       '''
   ```

   ```
       用均匀分布模拟泊松分布。 lmd为 lambda 参数； tm 为时间
   ```

   ```
       '''
   ```

   ```
       t=np.random.uniform(0,tm,size=lmd*tm) # 获取 lmd*tm 个事件发生的时刻
   ```

   ```
       count,tm_edges=np.histogram(t,bins=tm,range=(0,tm))#获取每个单位时间内，事件发生的次数
   ```

   ```
       max_k= lmd *2 # 要统计的最大次数
   ```

   ```
       dist,count_edges=np.histogram(count,bins=max_k,range=(0,max_k),density=True)
   ```

   ```
       x=count_edges[:-1]
   ```

   ```
       return x,dist,stats.poisson.pmf(x,lmd)  
   ```

   该函数首先随机性给出了 `lmd*tm`个事件发生的时间（时间位于区间`[0,tm]`）内。然后统计每个单位时间区间内，事件发生的次数。然后统计这些次数出现的频率。最后将这个频率与理论上的泊松分布的概率质量函数比较。

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/AoKP1WpndUm4.JPG">
  </p>
  

4. 指数分布：若事件服从泊松分布，则该事件前后两次发生的时间间隔服从指数分布。由于时间间隔是个浮点数，因此指数分布是连续分布。

   - 概率密度函数：$f(t;\lambda)=\lambda e^{-\lambda t}$，$t$为时间间隔
   - 期望：$\frac 1\lambda$
   - 方差：$\frac {1}{\lambda^{2}}$

   在`scipy.stats.expon`中，`scale`参数为$\frac {1}{\lambda}$；而`loc`用于对函数平移

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/25fUwvUJS3mO.JPG">
  </p>
  

5. 用均匀分布模拟指数分布：

   ```
   xxxxxxxxxx
   ```

   ```
    def make_expon(lmd,tm):
   ```

   ```
       '''
   ```

   ```
       用均匀分布模拟指数分布。 lmd为 lambda 参数； tm 为时间 
   ```

   ```
       '''
   ```

   ```
       t=np.random.uniform(0,tm,size=lmd*tm) # 获取 lmd*tm 个事件发生的时刻
   ```

   ```
       sorted_t=np.sort(t) #时刻升序排列
   ```

   ```
       delt_t=sorted_t[1:]-sorted_t[:-1] #间隔序列
   ```

   ```
       dist,edges=np.histogram(delt_t,bins="auto",density=True)
   ```

   ```
       x=edges[:-1]
   ```

   ```
       return x,dist,stats.expon.pdf(x,loc=0,scale=1/lmd) #scale 为 1/lambda
   ```

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/dJsM1pnjuPgb.JPG">
  </p>
  

6. 伽玛分布：若事件服从泊松分布，则事件第$i$次发生和第$i+k$次发生的时间间隔为伽玛分布。由于时间间隔是个浮点数，因此指数分布是连续分布。

   - 概率密度函数：$f(t;\lambda,k)=\frac{t^{(k-1)}\lambda^{k}e^{(-\lambda t)}}{\Gamma(k)}$，$t$为时间间隔
   - 期望：$\frac{k}{\lambda}$
   - 方差：$\frac{k}{\lambda^{2}}$

   在`scipy.stats.gamma`中，`scale`参数为$\frac {1}{\lambda}$；而`loc`用于对函数平移，参数`a`指定了$k$ <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/5LVKvCzOR6cy.JPG">
   </p>
   

7. 用均匀分布模拟伽玛分布：

   ```
   xxxxxxxxxx
   ```

   ```
     def make_gamma(lmd,tm,k):
   ```

   ```
       '''
   ```

   ```
       用均匀分布模拟伽玛分布。 lmd为 lambda 参数； tm 为时间；k 为 k 参数
   ```

   ```
       '''
   ```

   ```
       t=np.random.uniform(0,tm,size=lmd*tm) # 获取 lmd*tm 个事件发生的时刻
   ```

   ```
       sorted_t=np.sort(t) #时刻升序排列
   ```

   ```
       delt_t=sorted_t[k:]-sorted_t[:-k] #间隔序列
   ```

   ```
       dist,edges=np.histogram(delt_t,bins="auto",density=True)
   ```

   ```
       x=edges[:-1]
   ```

   ```
       return x,dist,stats.gamma.pdf(x,loc=0,scale=1/lmd,a=k) #scale 为 1/lambda,a 为 k
   ```

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/gYLCDzf94XK2.JPG">
  </p>
  

## 五、数值积分

1. `scipy`的`integrate`模块提供了集中数值积分算法，其中包括对常微分方程组`ODE`的数值积分。

### 1. 积分

1. 数值积分函数：

   ```
   xxxxxxxxxx
   ```

   ```
   scipy.integrate.quad(func, a, b, args=(), full_output=0, epsabs=1.49e-08, 
   ```

   ```
     epsrel=1.49e-08, limit=50, points=None, weight=None, wvar=None,
   ```

   ```
     wopts=None, maxp1=50, limlst=50)
   ```

   - `func`：一个`Python`函数对象，代表被积分的函数。如果它带有多个参数，则积分只在第一个参数上进行。其他参数，则由`args`提供
   - `a`：积分下限。用`-numpy.inf`代表负无穷
   - `b`：积分上限。用`numpy.inf`代表正无穷
   - `args`：额外传递的参数给`func`
   - `full_output`：如果非零，则通过字典返回更多的信息
   - 其他参数控制了积分的细节。参考官方手册

   返回值：

   - `y`：一个浮点标量值，表示积分结果
   - `abserr`：一个浮点数，表示绝对误差的估计值
   - `infodict`：一个字典，包含额外的信息

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/QUEFVNmzvApL.JPG">
  </p>
  

2. 多重定积分可以通过多次调用`quad()`实现。为了调用方便，`integrate`模块提供了`dblquad()`计算二重定积分，提供了`tplquad()`计算三重定积分。

3. 二重定积分：

   ```
   xxxxxxxxxx
   ```

   ```
   scipy.integrate.dblquad(func, a, b, gfun, hfun, args=(),
   ```

   ```
     epsabs=1.49e-08, epsrel=1.49e-08)
   ```

   - `func`：一个`Python`函数对象，代表被积分的函数。它至少有两个参数：`y`和`x`。其中`y`为第一个参数，`x`为第二个参数。这两个参数为积分参数。如果有其他参数，则由`args`提供
   - `a`：`x`的积分下限。用`-numpy.inf`代表负无穷
   - `b`：`x`的积分上限。用`numpy.inf`代表正无穷
   - `gfun`：`y`的下边界曲线。它是一个函数或者`lambda`表达式，参数为`x`,返回一个浮点数。
   - `hfun`：`y`的上界曲线。它是一个函数或者`lambda`表达式，参数为`x`,返回一个浮点数。
   - `args`：额外传递的参数给`func`
   - `epsabs`：传递给`quad`
   - `epsrel`：传递给`quad`

   返回值：

   - `y`：一个浮点标量值，表示积分结果
   - `abserr`：一个浮点数，表示绝对误差的估计值

   <p align="center">
     <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200602/yPYCoShKqYKI.JPG">
   </p>
   

4. 三重定积分：

   ```
   xxxxxxxxxx
   ```

   ```
   scipy.integrate.tplquad(func, a, b, gfun, hfun, qfun, rfun, args=(), 
   ```

   ```
     epsabs=1.49e-08, epsrel=1.49e-08)
   ```

   - `func`：一个`Python`函数对象，代表被积分的函数。它至少有三个参数：`z`、`y`和`x`。其中`z`为第一个参数，`y`为第二个参数，`x`为第三个参数。这三个参数为积分参数。如果有其他参数，则由`args`提供
   - `a`：`x`的积分下限。用`-numpy.inf`代表负无穷
   - `b`：`x`的积分上限。用`numpy.inf`代表正无穷
   - `gfun`：`y`的下边界曲线。它是一个函数或者`lambda`表达式，参数为`x`,返回一个浮点数。
   - `hfun`：`y`的上界曲线。它是一个函数或者`lambda`表达式，参数为`x`,返回一个浮点数。
   - `qfun`：`z`的下边界曲面。它是一个函数或者`lambda`表达式，第一个参数为`x`，第二个参数为`y`，返回一个浮点数。
   - `rfun`：`z`的上边界曲面。它是一个函数或者`lambda`表达式，第一个参数为`x`，第二个参数为`y`，返回一个浮点数。
   - `args`：额外传递的参数给`func`
   - `epsabs`：传递给`quad`
   - `epsrel`：传递给`quad`

   返回值：

   - `y`：一个浮点标量值，表示积分结果
   - `abserr`：一个浮点数，表示绝对误差的估计值

### 2. 求解常微分方程组

1. 求解常微分方程组用：

   ```
   xxxxxxxxxx
   ```

   ```
   scipy.integrate.odeint(func, y0, t, args=(), Dfun=None, col_deriv=0, full_output=0, 
   ```

   ```
     ml=None, mu=None, rtol=None, atol=None, tcrit=None, h0=0.0, hmax=0.0, 
   ```

   ```
     hmin=0.0, ixpr=0, mxstep=0, mxhnil=0, mxordn=12, mxords=5, printmessg=0)
   ```

   - `func`：梯度函数。第一个参数为`y`，第二个参数为`t0`，即计算`t0`时刻的梯度。其他的参数由`args`提供
   - `y0`：初始的`y`
   - `t`：一个时间点序列。
   - `args`：额外提供给`func`的参数。
   - `Dfun`：`func`的雅可比矩阵，行优先
   - `col_deriv`：一个布尔值。如果`Dfun`未给出，则算法自动推导。该参数决定了自动推导的方式
   - `full_output`：如果`True`，则通过字典返回更多的信息
   - `printmessg`：布尔值。如果为`True`，则打印收敛信息
   - 其他参数用于控制求解的细节

   返回值：

   - `y`：一个数组，形状为 `(len(t),len(y0)`。它给出了每个时刻的`y`值
   - `infodict`：一个字典，包含额外的信息

  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/D3gK2E2Dasqw.JPG">
  </p>
  <p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200603/5nQkNHtacCEE.JPG">
  </p>
  
## 六、 稀疏矩阵

1. 稀疏矩阵是那些矩阵中大部分为零的矩阵。这种矩阵只用保存非零元素的相关信息，从而节约了内存的使用。`scipy.sparse`提供了多种表示稀疏矩阵的格式。`scipy.sparse.lialg`提供了对稀疏矩阵进行线性代数运算的函数。`scipy.sparse.csgraph`提供了对稀疏矩阵表示的图进行搜索的函数。
2. `scipy.sparse`中有多种表示稀疏矩阵的格式：
   - `dok_matrix`：采用字典保存矩阵中的非零元素：字典的键是一个保存元素（行，列）信息的元组，对应的值为矩阵中位于（行，列）中的元素值。这种格式很适合单个元素的添加、删除、存取操作。通常先逐个添加非零元素，然后转换成其他支持高效运算的格式
   - `lil_matrix`：采用两个列表保存非零元素。`data`保存每行中的非零元素，`row`保存非零元素所在的列。
   - `coo_matrix`：采用三个数组`row/col/data`保存非零元素。这三个数组的长度相同，分别保存元素的行、列和元素值。`coo_matrix`不支持元素的存取和增删，一旦创建之后，除了将之转换成其他格式的矩阵，几乎无法对其进行任何操作和矩阵运算。