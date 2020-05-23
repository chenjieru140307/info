# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### 构建随机森林回归模型
# %% [markdown]
# #### 0.import工具库

# %%
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston

# %% [markdown]
# #### 1.加载数据

# %%
boston_house = load_boston()


# %%
boston_feature_name = boston_house.feature_names
boston_features = boston_house.data
boston_target = boston_house.target


# %%
boston_feature_name


# %%
print(boston_house.DESCR)


# %%
boston_features[:5,:]


# %%
boston_target

# %% [markdown]
# ### 构建模型

# %%
help(RandomForestRegressor)


# %%
rgs = RandomForestRegressor(n_estimators=15)  ##随机森林模型
rgs = rgs.fit(boston_features, boston_target)


# %%
rgs


# %%
rgs.predict(boston_features)


# %%
from sklearn import tree


# %%
rgs2 = tree.DecisionTreeRegressor()           ##决策树模型，比较两个模型的预测结果！
rgs2.fit(boston_features, boston_target)


# %%
rgs2.predict(boston_features)


# %%


