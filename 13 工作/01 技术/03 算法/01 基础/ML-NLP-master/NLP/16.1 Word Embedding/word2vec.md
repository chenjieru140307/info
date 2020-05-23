# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## word2vec训练中文模型
# ------
# ### 1.准备数据与预处理
# 首先需要一份比较大的中文语料数据，可以考虑中文的维基百科（也可以试试搜狗的新闻语料库）。
# 
# 中文维基百科的打包文件地址为 
# 链接: https://pan.baidu.com/s/1H-wuIve0d_fvczvy3EOKMQ 提取码: uqua 
# 
# 百度网盘加速下载地址：https://www.baiduwp.com/?m=index
# 
# 中文维基百科的数据不是太大，xml的压缩文件大约1G左右。首先用处理这个XML压缩文件。
# 
# **注意输入输出地址**

# %%
import logging
import os.path
import sys
from gensim.corpora import WikiCorpus
if __name__ == '__main__':
    
    # 定义输入输出
    basename = "F:/temp/DL/"
    inp = basename+'zhwiki-latest-pages-articles.xml.bz2'
    outp = basename+'wiki.zh.text'
    
    program = os.path.basename(basename)
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    # check and process input arguments
    if len(sys.argv) < 3:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    
    space = " "
    i = 0
    output = open(outp, 'w',encoding='utf-8')
    wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        output.write(space.join(text) + "\n")
        i = i + 1
        if (i % 10000 == 0):
            logger.info("Saved " + str(i) + " articles")
    output.close()
    logger.info("Finished Saved " + str(i) + " articles")

# %% [markdown]
# 
# ### 2.训练数据
# Python的话可用jieba完成分词，生成分词文件wiki.zh.text.seg 
# 接着用word2vec工具训练： 
# 
# **注意输入输出地址**

# %%
import logging
import os.path
import sys
import multiprocessing
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# 定义输入输出
basename = "F:/temp/DL/"
inp = basename+'wiki.zh.text'
outp1 = basename+'wiki.zh.text.model'
outp2 = basename+'wiki.zh.text.vector'

program = os.path.basename(basename)
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))
# check and process input arguments
if len(sys.argv) < 4:
    print(globals()['__doc__'] % locals())

model = Word2Vec(LineSentence(inp), size=400, window=5, min_count=5,
        workers=multiprocessing.cpu_count())
# trim unneeded model memory = use(much) less RAM
#model.init_sims(replace=True)
model.save(outp1)
model.save_word2vec_format(outp2, binary=False)

输出如下：
2019-05-08 22:28:25,638: INFO: running c:\users\mantch\appdata\local\programs\python\python35\lib\site-packages\ipykernel_launcher.py -f C:\Users\mantch\AppData\Roaming\jupyter\runtime\kernel-b1f915fd-fdb2-43fc-bcf3-b361fb4a7c3d.json
2019-05-08 22:28:25,640: INFO: collecting all words and their counts
2019-05-08 22:28:25,642: INFO: PROGRESS: at sentence #0, processed 0 words, keeping 0 word types
Automatically created module for IPython interactive environment
2019-05-08 22:28:27,887: INFO: PROGRESS: at sentence #10000, processed 4278620 words, keeping 2586311 word types
2019-05-08 22:28:29,666: INFO: PROGRESS: at sentence #20000, processed 7491125 words, keeping 4291863 word types
2019-05-08 22:28:31,445: INFO: PROGRESS: at sentence #30000, processed 10424455 words, keeping 5704507 word types
2019-05-08 22:28:32,854: INFO: PROGRESS: at sentence #40000, processed 13190001 words, keeping 6983862 word types
2019-05-08 22:28:34,125: INFO: PROGRESS: at sentence #50000, processed 15813238 words, keeping 8145905 word types
2019-05-08 22:28:35,353: INFO: PROGRESS: at sentence #60000, processed 18388731 words, keeping 9198885 word types
2019-05-08 22:28:36,544: INFO: PROGRESS: at sentence #70000, processed 20773000 words, keeping 10203788 word types
2019-05-08 22:28:37,652: INFO: PROGRESS: at sentence #80000, processed 23064544 words, keeping 11144885 word types
2019-05-08 22:28:39,490: INFO: PROGRESS: at sentence #90000, processed 25324650 words, keeping 12034343 word types
2019-05-08 22:28:40,688: INFO: PROGRESS: at sentence #100000, processed 27672540 words, keeping 12878856 word types
2019-05-08 22:28:41,871: INFO: PROGRESS: at sentence #110000, processed 29985282 words, keeping 13688622 word types
2019-05-08 22:28:42,944: INFO: PROGRESS: at sentence #120000, processed 32025045 words, keeping 14477767 word types
2019-05-08 22:28:44,048: INFO: PROGRESS: at sentence #130000, processed 34267390 words, keeping 15309447 word types
2019-05-08 22:28:45,197: INFO: PROGRESS: at sentence #140000, processed 36451394 words, keeping 16090548 word types
2019-05-08 22:28:46,345: INFO: PROGRESS: at sentence #150000, processed 38671717 words, keeping 16877015 word types
2019-05-08 22:28:47,483: INFO: PROGRESS: at sentence #160000, processed 40778409 words, keeping 17648492 word types
2019-05-08 22:28:48,655: INFO: PROGRESS: at sentence #170000, processed 43154040 words, keeping 18308373 word types
2019-05-08 22:28:49,759: INFO: PROGRESS: at sentence #180000, processed 45231681 words, keeping 19010906 word types
2019-05-08 22:28:50,826: INFO: PROGRESS: at sentence #190000, processed 47190144 words, keeping 19659373 word types
2019-05-08 22:28:51,886: INFO: PROGRESS: at sentence #200000, processed 49201934 words, keeping 20311518 word types
2019-05-08 22:28:52,856: INFO: PROGRESS: at sentence #210000, processed 51116197 words, keeping 20917125 word types
2019-05-08 22:28:53,859: INFO: PROGRESS: at sentence #220000, processed 53321151 words, keeping 21513016 word types
2019-05-08 22:28:54,921: INFO: PROGRESS: at sentence #230000, processed 55408211 words, keeping 22207241 word types
2019-05-08 22:28:59,645: INFO: PROGRESS: at sentence #240000, processed 57442276 words, keeping 22849499 word types
2019-05-08 22:29:00,988: INFO: PROGRESS: at sentence #250000, processed 59563975 words, keeping 23544817 word types
2019-05-08 22:29:02,292: INFO: PROGRESS: at sentence #260000, processed 61764248 words, keeping 24222911 word types
2019-05-08 22:29:03,654: INFO: PROGRESS: at sentence #270000, processed 63938511 words, keeping 24906453 word types
2019-05-08 22:29:04,900: INFO: PROGRESS: at sentence #280000, processed 66096661 words, keeping 25519781 word types
2019-05-08 22:29:06,057: INFO: PROGRESS: at sentence #290000, processed 67947209 words, keeping 26062482 word types
2019-05-08 22:29:07,229: INFO: PROGRESS: at sentence #300000, processed 69927780 words, keeping 26649878 word types
2019-05-08 22:29:08,506: INFO: PROGRESS: at sentence #310000, processed 71800313 words, keeping 27230264 word types
2019-05-08 22:29:09,836: INFO: PROGRESS: at sentence #320000, processed 73942427 words, keeping 27850568 word types
2019-05-08 22:29:11,419: INFO: PROGRESS: at sentence #330000, processed 75859220 words, keeping 28467061 word types
2019-05-08 22:29:12,379: INFO: collected 28914285 word types from a corpus of 77273203 raw words and 338042 sentences# %% [markdown]
# 
# ### 3.测试结果

# %%

# 测试结果
import gensim

# 定义输入输出
basename = "F:/temp/DL/"
model = basename+'wiki.zh.text.model'

model = gensim.models.Word2Vec.load(model)

result = model.most_similar(u"足球")
for e in result:
    print(e[0], e[1])


# %%
result = model.most_similar(u"男人")
for e in result:
    print(e[0], e[1])


# %%


