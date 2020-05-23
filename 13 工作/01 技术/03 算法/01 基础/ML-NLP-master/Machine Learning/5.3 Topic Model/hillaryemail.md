# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # LDA模型应用：一眼看穿希拉里的邮件
# 
# 我们拿到希拉里泄露的邮件，跑一把LDA，看看她平时都在聊什么。
# 
# 首先，导入我们需要的一些库

# %%
import numpy as np
import pandas as pd
import re

# %% [markdown]
# 然后，把希婆的邮件读取进来。
# 
# 这里我们用pandas。不熟悉pandas的朋友，可以用python标准库csv

# %%
df = pd.read_csv("HillaryEmails.csv")
# 原邮件数据中有很多Nan的值，直接扔了。
df = df[['Id','ExtractedBodyText']].dropna()

# %% [markdown]
# ### 文本预处理：
# 
# 上过我其他NLP课程的同学都知道，文本预处理这个东西，对NLP是很重要的。
# 
# 我们这里，针对邮件内容，写一组正则表达式：
# 
# （不熟悉正则表达式的同学，直接百度关键词，可以看到一大张Regex规则表）

# %%
def clean_email_text(text):
    text = text.replace('\n'," ") #新行，我们是不需要的
    text = re.sub(r"-", " ", text) #把 "-" 的两个单词，分开。（比如：pre-processing ==> pre processing）
    text = re.sub(r"\d+/\d+/\d+", "", text) #日期，对主体模型没什么意义
    text = re.sub(r"[0-2]?[0-9]:[0-6][0-9]", "", text) #时间，没意义
    text = re.sub(r"[\w]+@[\.\w]+", "", text) #邮件地址，没意义
    text = re.sub(r"/[a-zA-Z]*[:\//\]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", "", text) #网址，没意义
    pure_text = ''
    # 以防还有其他特殊字符（数字）等等，我们直接把他们loop一遍，过滤掉
    for letter in text:
        # 只留下字母和空格
        if letter.isalpha() or letter==' ':
            pure_text += letter
    # 再把那些去除特殊字符后落单的单词，直接排除。
    # 我们就只剩下有意义的单词了。
    text = ' '.join(word for word in pure_text.split() if len(word)>1)
    return text

# %% [markdown]
# 好的，现在我们新建一个colum，并把我们的方法跑一遍：

# %%
docs = df['ExtractedBodyText']
docs = docs.apply(lambda s: clean_email_text(s))  

# %% [markdown]
# 好，来看看长相：

# %%
docs.head(1).values

# %% [markdown]
# 我们直接把所有的邮件内容拿出来。

# %%
doclist = docs.values

# %% [markdown]
# ### LDA模型构建：
# 
# 好，我们用Gensim来做一次模型构建
# 
# 首先，我们得把我们刚刚整出来的一大波文本数据
# ```
# [[一条邮件字符串]，[另一条邮件字符串], ...]
# ```
# 
# 转化成Gensim认可的语料库形式：
# 
# ```
# [[一，条，邮件，在，这里],[第，二，条，邮件，在，这里],[今天，天气，肿么，样],...]
# ```
# 
# 引入库：

# %%
from gensim import corpora, models, similarities
import gensim

# %% [markdown]
# 为了免去讲解安装NLTK等等的麻烦，我这里直接手写一下**停止词列表**：
# 
# 这些词在不同语境中指代意义完全不同，但是在不同主题中的出现概率是几乎一致的。所以要去除，否则对模型的准确性有影响

# %%
stoplist = ['very', 'ourselves', 'am', 'doesn', 'through', 'me', 'against', 'up', 'just', 'her', 'ours', 
            'couldn', 'because', 'is', 'isn', 'it', 'only', 'in', 'such', 'too', 'mustn', 'under', 'their', 
            'if', 'to', 'my', 'himself', 'after', 'why', 'while', 'can', 'each', 'itself', 'his', 'all', 'once', 
            'herself', 'more', 'our', 'they', 'hasn', 'on', 'ma', 'them', 'its', 'where', 'did', 'll', 'you', 
            'didn', 'nor', 'as', 'now', 'before', 'those', 'yours', 'from', 'who', 'was', 'm', 'been', 'will', 
            'into', 'same', 'how', 'some', 'of', 'out', 'with', 's', 'being', 't', 'mightn', 'she', 'again', 'be', 
            'by', 'shan', 'have', 'yourselves', 'needn', 'and', 'are', 'o', 'these', 'further', 'most', 'yourself', 
            'having', 'aren', 'here', 'he', 'were', 'but', 'this', 'myself', 'own', 'we', 'so', 'i', 'does', 'both', 
            'when', 'between', 'd', 'had', 'the', 'y', 'has', 'down', 'off', 'than', 'haven', 'whom', 'wouldn', 
            'should', 've', 'over', 'themselves', 'few', 'then', 'hadn', 'what', 'until', 'won', 'no', 'about', 
            'any', 'that', 'for', 'shouldn', 'don', 'do', 'there', 'doing', 'an', 'or', 'ain', 'hers', 'wasn', 
            'weren', 'above', 'a', 'at', 'your', 'theirs', 'below', 'other', 'not', 're', 'him', 'during', 'which']

# %% [markdown]
# 人工分词：
# 
# 这里，英文的分词，直接就是对着空白处分割就可以了。
# 
# 中文的分词稍微复杂点儿，具体可以百度：CoreNLP, HaNLP, 结巴分词，等等
# 
# 分词的意义在于，把我们的长长的字符串原文本，转化成有意义的小元素：

# %%
texts = [[word for word in doc.lower().split() if word not in stoplist] for doc in doclist]

# %% [markdown]
# 这时候，我们的texts就是我们需要的样子了：

# %%
texts[0]

# %% [markdown]
# ### 建立语料库
# 
# 用词袋的方法，把每个单词用一个数字index指代，并把我们的原文本变成一条长长的数组：

# %%
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# %% [markdown]
# 给你们看一眼：

# %%
corpus[13]

# %% [markdown]
# 这个列表告诉我们，第14（从0开始是第一）个邮件中，一共6个有意义的单词（经过我们的文本预处理，并去除了停止词后）
# 
# 其中，36号单词出现1次，505号单词出现1次，以此类推。。。
# 
# 接着，我们终于可以建立模型了：

# %%
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)

# %% [markdown]
# 我们可以看到，第10号分类，其中最常出现的单词是：

# %%
lda.print_topic(10, topn=5)

# %% [markdown]
# 我们把所有的主题打印出来看看

# %%
lda.print_topics(num_topics=20, num_words=5)

# %% [markdown]
# ### 接下来：
# 
# 通过
# ```
# lda.get_document_topics(bow)
# ```
# 或者
# ```
# lda.get_term_topics(word_id)
# ```
# 
# 两个方法，我们可以把新鲜的文本/单词，分类成20个主题中的一个。
# 
# *但是注意，我们这里的文本和单词，都必须得经过同样步骤的文本预处理+词袋化，也就是说，变成数字表示每个单词的形式。*
# 
# ### 作业：
# 
# 我这里有希拉里twitter上的几条(每一空行是单独的一条)：
# 
# ```
# To all the little girls watching...never doubt that you are valuable and powerful & deserving of every chance & opportunity in the world.
# 
# I was greeted by this heartwarming display on the corner of my street today. Thank you to all of you who did this. Happy Thanksgiving. -H
# 
# Hoping everyone has a safe & Happy Thanksgiving today, & quality time with family & friends. -H
# 
# Scripture tells us: Let us not grow weary in doing good, for in due season, we shall reap, if we do not lose heart.
# 
# Let us have faith in each other. Let us not grow weary. Let us not lose heart. For there are more seasons to come and...more work to do
# 
# We have still have not shattered that highest and hardest glass ceiling. But some day, someone will
# 
# To Barack and Michelle Obama, our country owes you an enormous debt of gratitude. We thank you for your graceful, determined leadership
# 
# Our constitutional democracy demands our participation, not just every four years, but all the time
# 
# You represent the best of America, and being your candidate has been one of the greatest honors of my life
# 
# Last night I congratulated Donald Trump and offered to work with him on behalf of our country
# 
# Already voted? That's great! Now help Hillary win by signing up to make calls now
# 
# It's Election Day! Millions of Americans have cast their votes for Hillary—join them and confirm where you vote
# 
# We don’t want to shrink the vision of this country. We want to keep expanding it
# 
# We have a chance to elect a 45th president who will build on our progress, who will finish the job
# 
# I love our country, and I believe in our people, and I will never, ever quit on you. No matter what
# 
# ```
# 
# 各位同学请使用训练好的LDA模型，判断每句话各自属于哪个potic
# 
# 么么哒

# %%


