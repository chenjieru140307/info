# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## 用决策树模型完成分类问题
# %% [markdown]
# #### 把需要的工具库import进来

# %%
#用于数据处理和分析的工具包
import pandas as pd
#引入用于数据预处理/特征工程的工具包
from sklearn import preprocessing
#import决策树建模包
from sklearn import tree

# %% [markdown]
# #### 读取数据

# %%
adult_data = pd.read_csv('./DecisionTree.csv')


# %%
#读取前5行，了解一下数据
adult_data.head(5)


# %%
adult_data.info()


# %%
adult_data.shape


# %%
adult_data.columns

# %% [markdown]
# #### 区分一下特征(属性)和目标

# %%
feature_columns = [u'workclass', u'education', u'marital-status', u'occupation', u'relationship', u'race', u'gender', u'native-country']
label_column = ['income']


# %%
#区分特征和目标列
features = adult_data[feature_columns]
label = adult_data[label_column]


# %%
features.head(2)


# %%
label.head(2)

# %% [markdown]
# #### 特征处理/特征工程

# %%
features = pd.get_dummies(features)


# %%
features.head(2)

# %% [markdown]
# ### 构建模型

# %%
#初始化一个决策树分类器
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)
#用决策树分类器拟合数据
clf = clf.fit(features.values, label.values)


# %%
clf


# %%
clf.predict(features.values)

# %% [markdown]
# ### 可视化一下这颗决策树

# %%
import pydotplus


# %%
from IPython.display import display, Image


# %%
dot_data = tree.export_graphviz(clf, 
                                out_file=None, 
                                feature_names=features.columns,
                                class_names = ['<=50k', '>50k'],
                                filled = True,
                                rounded =True
                               )


# %%
graph = pydotplus.graph_from_dot_data(dot_data)
display(Image(graph.create_png()))


# %%


