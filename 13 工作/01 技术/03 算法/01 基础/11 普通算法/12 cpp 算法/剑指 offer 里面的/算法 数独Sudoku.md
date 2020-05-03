


数独问题：

解数独问题，初始化时的空位用‘.’表示。
 每行、每列、每个九宫内，都是 1-9这 9 个数字。


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/a4imgAGeFE.png?imageslim">
</p>



数独 Sudoku 分析
 若当前位置是空格，则尝试从 1 到 9 的所有
数；如果对于 1 到 9 的某些数字，当前是合法
的，则继续尝试下一个位置——调用自身即
可。


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/4b0g2d0aIL.png?imageslim">
</p>

代码如下：


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/7H40d32i4A.png?imageslim">
</p>

非递归数独 Sudoku


<p align="center">
    <img width="70%" height="70%" src="http://images.iterate.site/blog/image/180727/59ELEa1JiC.png?imageslim">
</p>

COMMENT：

REF：




  1. 七月在线 算法



