# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## machine learning for credit scoring
# 
# 
# Banks play a crucial role in market economies. They decide who can get finance and on what terms and can make or break investment decisions. For markets and society to function, individuals and companies need access to credit. 
# 
# Credit scoring algorithms, which make a guess at the probability of default, are the method banks use to determine whether or not a loan should be granted. This competition requires participants to improve on the state of the art in credit scoring, by predicting the probability that somebody will experience financial distress in the next two years. [Dataset](https://www.kaggle.com/c/GiveMeSomeCredit)
# 
# Attribute Information:
# 
# |Variable Name	|	Description	|	Type|
# |----|----|----|
# |SeriousDlqin2yrs	|	Person experienced 90 days past due delinquency or worse 	|	Y/N|
# |RevolvingUtilizationOfUnsecuredLines	|	Total balance on credit divided by the sum of credit limits	|	percentage|
# |age	|	Age of borrower in years	|	integer|
# |NumberOfTime30-59DaysPastDueNotWorse	|	Number of times borrower has been 30-59 days past due |	integer|
# |DebtRatio	|	Monthly debt payments	|	percentage|
# |MonthlyIncome	|	Monthly income	|	real|
# |NumberOfOpenCreditLinesAndLoans	|	Number of Open loans |	integer|
# |NumberOfTimes90DaysLate	|	Number of times borrower has been 90 days or more past due.	|	integer|
# |NumberRealEstateLoansOrLines	|	Number of mortgage and real estate loans	|	integer|
# |NumberOfTime60-89DaysPastDueNotWorse	|	Number of times borrower has been 60-89 days past due |integer|
# |NumberOfDependents	|	Number of dependents in family	|	integer|
# 
# %% [markdown]
# Read the data into Pandas 

# %%
import pandas as pd
pd.set_option('display.max_columns', 500)
import zipfile
with zipfile.ZipFile('KaggleCredit2.csv.zip', 'r') as z:   ##读取zip里的文件
    f = z.open('KaggleCredit2.csv')
    data = pd.read_csv(f, index_col=0)
data.head()


# %%
data.shape

# %% [markdown]
# Drop na

# %%
data.isnull().sum(axis=0)


# %%
data.dropna(inplace=True)   ##去掉为空的数据
data.shape

# %% [markdown]
# Create X and y

# %%
y = data['SeriousDlqin2yrs']
X = data.drop('SeriousDlqin2yrs', axis=1)


# %%
y.mean() ##求取均值

# %% [markdown]
# # 练习1
# 
# 把数据切分成训练集和测试集

# %%
from sklearn import model_selection
x_tran,x_test,y_tran,y_test=model_selection.train_test_split(X,y,test_size=0.2)
print(x_test.shape)

# %% [markdown]
# # 练习2
# 使用logistic regression/决策树/SVM/KNN...等sklearn分类算法进行分类，尝试查sklearn API了解模型参数含义，调整不同的参数。

# %%
from sklearn.linear_model import LogisticRegression
## https://blog.csdn.net/sun_shengyun/article/details/53811483
lr=LogisticRegression(multi_class='ovr',solver='sag',class_weight='balanced')
lr.fit(x_tran,y_tran)
score=lr.score(x_tran,y_tran)
print(score) ##最好的分数是1

# %% [markdown]
# # 练习3
# 在测试集上进行预测，计算准确度

# %%
from sklearn.metrics import accuracy_score
## https://blog.csdn.net/qq_16095417/article/details/79590455
train_score=accuracy_score(y_tran,lr.predict(x_tran))
test_score=lr.score(x_test,y_test)
print('训练集准确率：',train_score)
print('测试集准确率：',test_score)

# %% [markdown]
# # 练习4
# 查看sklearn的官方说明，了解分类问题的评估标准，并对此例进行评估。

# %%
##召回率
from sklearn.metrics import recall_score
train_recall=recall_score(y_tran,lr.predict(x_tran),average='macro')
test_recall=recall_score(y_test,lr.predict(x_test),average='macro')
print('训练集召回率：',train_recall)
print('测试集召回率：',test_recall)

# %% [markdown]
# # 练习5
# 
# 银行通常会有更严格的要求，因为fraud带来的后果通常比较严重，一般我们会调整模型的标准。<br>
# 比如在logistic regression当中，一般我们的概率判定边界为0.5，但是我们可以把阈值设定低一些，来提高模型的“敏感度”，试试看把阈值设定为0.3，再看看这时的评估指标(主要是准确率和召回率)。
# 
# tips:sklearn的很多分类模型，predict_prob可以拿到预估的概率，可以根据它和设定的阈值大小去判断最终结果(分类类别)

# %%
import numpy as np
y_pro=lr.predict_proba(x_test) ##获取预测概率值
y_prd2 = [list(p>=0.3).index(1) for i,p in enumerate(y_pro)]   ##设定0.3阈值，把大于0.3的看成1分类。
train_score=accuracy_score(y_test,y_prd2)
print(train_score)


# %%


