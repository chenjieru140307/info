# fire


在讲解主程序`main.py`之前，我们先来看看2017年3月谷歌开源的一个命令行工具`fire` [官网](https://github.com/google/python-fire)，通过`pip install fire`即可安装。下面来看看`fire`的基础用法，假设`example.py`文件内容如下：

```python
import fire

def add(x, y):
  return x + y
  
def mul(**kwargs):
    a = kwargs['a']
    b = kwargs['b']
    return a * b

if __name__ == '__main__':
  fire.Fire()
```

那么我们可以使用：

```bash
python example.py add 1 2 # 执行add(1, 2)
python example.py mul --a=1 --b=2 # 执行mul(a=1, b=2), kwargs={'a':1, 'b':2}
python example.py add --x=1 --y==2 # 执行add(x=1, y=2)
```

可见，只要在程序中运行`fire.Fire()`，即可使用命令行参数`python file <function> [args,] {--kwargs,}`。fire还支持更多的高级功能，具体请参考[官方指南](https://github.com/google/python-fire/blob/master/doc/guide.md)。


在主程序`main.py`中，主要包含四个函数，其中三个需要命令行执行，`main.py`的代码组织结构如下：

```python
def train(**kwargs):
    """
    训练
    """
    pass
	 
def val(model, dataloader):
    """
    计算模型在验证集上的准确率等信息，用以辅助训练
    """
    pass

def test(**kwargs):
    """
    测试（inference）
    """
    pass

def help():
    """
    打印帮助的信息 
    """
    print('help')

if __name__=='__main__':
    import fire
    fire.Fire()
```

根据fire的使用方法，可通过 `python main.py <function> --args=xx` 的方式来执行训练或者测试。