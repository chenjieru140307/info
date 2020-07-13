
# jieba

地址：

- [jieba](https://github.com/fxsjy/jieba)


举例：

```py
# encoding=utf-8
import jieba

seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式

seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
print(", ".join(seg_list))

seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
print(", ".join(seg_list))

# 词性标注
import jieba.posseg as pseg
words = pseg.cut("我爱北京天安门") #jieba默认模式
for word, flag in words:
   print('%s %s' % (word, flag))

# Tokenize：返回词语在原文的起止位置
result = jieba.tokenize(u'永和服装饰品有限公司') # 默认模式
for tk in result:
    print("word %s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))

result = jieba.tokenize(u'永和服装饰品有限公司', mode='search') # 搜索模式
for tk in result:
    print("word %s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))
```

输出：

```txt
Building prefix dict from the default dictionary ...
Loading model from cache C:\Users\wanfa\AppData\Local\Temp\jieba.cache
Loading model cost 0.644 seconds.
Prefix dict has been built successfully.
Full Mode: 我/ 来到/ 北京/ 清华/ 清华大学/ 华大/ 大学
Default Mode: 我/ 来到/ 北京/ 清华大学
他, 来到, 了, 网易, 杭研, 大厦
小明, 硕士, 毕业, 于, 中国, 科学, 学院, 科学院, 中国科学院, 计算, 计算所, ，, 后, 在, 日本, 京都, 大学, 日本京都大学, 深造
我 r
爱 v
北京 ns
天安门 ns
word 永和		 start: 0 		 end:2
word 服装		 start: 2 		 end:4
word 饰品		 start: 4 		 end:6
word 有限公司		 start: 6 		 end:10
word 永和		 start: 0 		 end:2
word 服装		 start: 2 		 end:4
word 饰品		 start: 4 		 end:6
word 有限		 start: 6 		 end:8
word 公司		 start: 8 		 end:10
word 有限公司		 start: 6 		 end:10
```


## suggest_freq

```py
import jieba
jieba.load_userdict("dict.txt")
# jieba.suggest_freq('确与', tune=True)
[jieba.suggest_freq(line.strip(), tune=True) for line in open("dict.txt", 'r', encoding='utf8')]

if __name__ == "__main__":
    string = "测试正确与否。"
    words=list(jieba.cut(string))
    print(words)
```

