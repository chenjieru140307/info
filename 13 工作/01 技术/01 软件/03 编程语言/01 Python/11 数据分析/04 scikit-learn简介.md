# scikit-learn

作用：

- 包含了很多监督式学习和非监督式学习的模型，可以实现分类，聚类，预测等任务。




举例：

数据下载：

- [数据下载](https://www.kaggle.com/c/titanic/data)

```py
import numpy as np
import pandas as pd

train = pd.read_csv('./datasets/titanic/train.csv')
test = pd.read_csv('./datasets/titanic/test.csv')

print(train.head())
print(train.isnull().sum())
print(test.isnull().sum())


impute_value = train['Age'].median()
train['Age'] = train['Age'].fillna(impute_value)
test['Age'] = test['Age'].fillna(impute_value)

train['IsFemale'] = (train['Sex'] == 'female').astype(int)
test['IsFemale'] = (test['Sex'] == 'female').astype(int)
print(train.head())

predictors = ['Pclass', 'IsFemale', 'Age']
X_train = train[predictors].values
X_test = test[predictors].values
y_train = train['Survived'].values
print(X_train[:5])
print(y_train[:5])


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
print(model)
y_predict = model.predict(X_test)
print(y_predict[:10])


from sklearn.linear_model import LogisticRegressionCV
model_cv = LogisticRegressionCV(10)
model_cv.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
model = LogisticRegression(C=10)
print(model)
scores = cross_val_score(model, X_train, y_train, cv=4)
print(scores)
```


输出：

```txt
   PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
0            1         0       3  ...   7.2500   NaN         S
1            2         1       1  ...  71.2833   C85         C
2            3         1       3  ...   7.9250   NaN         S
3            4         1       1  ...  53.1000  C123         S
4            5         0       3  ...   8.0500   NaN         S

[5 rows x 12 columns]
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
PassengerId      0
Pclass           0
Name             0
Sex              0
Age             86
SibSp            0
Parch            0
Ticket           0
Fare             1
Cabin          327
Embarked         0
dtype: int64
   PassengerId  Survived  Pclass  ... Cabin Embarked  IsFemale
0            1         0       3  ...   NaN        S         0
1            2         1       1  ...   C85        C         1
2            3         1       3  ...   NaN        S         1
3            4         1       1  ...  C123        S         1
4            5         0       3  ...   NaN        S         0

[5 rows x 13 columns]
[[ 3.  0. 22.]
 [ 1.  1. 38.]
 [ 3.  1. 26.]
 [ 1.  1. 35.]
 [ 3.  0. 35.]]
[0 1 1 1 0]
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
[0 0 0 0 1 0 1 0 1 0]
LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
[0.77578475 0.79820628 0.77578475 0.78828829]
```




- statsmodels 和 scikit-learn 通常不能应付缺失值，所以我们先检查一下哪些列有缺失值：

- 对于这样的数据集，通常的任务是预测一个乘客最后是否生还。在训练集上训练模型，在测试集上验证效果。
- 上面的 Age 这一列有缺失值，这里我们简单的用中位数来代替缺失值
- 对于 Sex 列，我们将其变为 IsFemale，用整数来表示性别
- `predictors = ['Pclass', 'IsFemale', 'Age']` 决定一些模型参数并创建 numpy 数组：
- 这里我们用逻辑回归模型（LogisticRegression）
- 如果我们有测试集的真是结果的话，可以用来计算准确率或其他一些指标：(y_true == y_predcit).mean()
- 交叉验证（cross-validation）。把训练集分为几份，每一份上又取出一部分作为测试样本，这些被取出来的测试样本不被用于训练，但我们可以在这些测试样本上验证当前模型的准确率或均方误差（mean squared error），而且还可以在模型参数上进行网格搜索（grid search）。
  - 一些模型，比如逻辑回归，自带一个有交叉验证的类。LogisticRegressionCV类可以用于模型调参，使用的时候需要指定正则化项 C，来控制网格搜索的程度：
- 如果想要自己来做交叉验证的话，可以使用 cross_val_score 函数，可以用于数据切分。比如，把整个训练集分为 4 个不重叠的部分。
- 默认的评价指标每个模型是不一样的，但是可以自己指定评价函数。交差验证的训练时间较长，但通常能得到更好的模型效果。
