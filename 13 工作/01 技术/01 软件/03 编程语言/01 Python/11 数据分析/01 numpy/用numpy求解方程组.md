
## 用numpy求解方程组

**举例1：**

- 求解矩阵向量方程 $A x=b$
- 其中：
  - $A=\left[\begin{array}{lll}
2 & 1 & -2 \\
3 & 0 & 1 \\
1 & 1 & -1
\end{array}\right]$
  - $\mathbf{b}=\left[\begin{array}{c}
-3 \\
5 \\
-2
\end{array}\right]$


```python
import numpy as np

A = np.array([[2, 1, -2], [3, 0, 1], [1, 1, -1]])
b = np.transpose(np.array([[-3, 5, -2]]))
x = np.linalg.solve(A, b)
print(x)
```

输出：

```txt
[[ 1.]
 [-1.]
 [ 2.]]
```
