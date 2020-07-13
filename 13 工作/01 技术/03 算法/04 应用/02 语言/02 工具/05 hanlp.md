# hanlp

介绍：

- HanLP是由一系列模型与算法组成的Java工具包，目标是普及自然语言处理在生产环境中的应用。HanLP具备功能完善、性能高效、架构清晰、语料时新、可自定义的特点。
- 功能：中文分词 词性标注 命名实体识别 依存句法分析 关键词提取新词发现 短语提取 自动摘要 文本分类 拼音简繁
- 比 jieba 功能要强一些。

地址：

- [github](https://github.com/hankcs/HanLP)
- [网盘](https://pan.baidu.com/s/1GSKoh8974aXcqZPfC4pEuA) 提取码：2gvz 内有 wiki.zh.zip

安装：

- `pip install hanlp`

举例：

```py
import hanlp

# 分词
tokenizer = hanlp.utils.rules.tokenize_english
print(tokenizer("Don't go gentle into that good night."))
tokenizer = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')
print(tokenizer('商品和服务'))

# 词性标注
tagger = hanlp.load(hanlp.pretrained.pos.CTB5_POS_RNN_FASTTEXT_ZH)
print(tagger(['我', '的', '希望', '是', '希望', '和平']))
tagger = hanlp.load(hanlp.pretrained.pos.PTB_POS_RNN_FASTTEXT_EN)
print(tagger([['I', 'banked', '2', 'dollars', 'in', 'a', 'bank', '.'],
              ['Is', 'this', 'the', 'future', 'of', 'chamber', 'music', '?']]))

# 命名实体识别
recognizer = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)
print(recognizer([list('上海华安工业（集团）公司董事长谭旭光和秘书张晚霞来到美国纽约现代艺术博物馆参观。'),
                  list('萨哈夫说，伊拉克将同联合国销毁伊拉克大规模杀伤性武器特别委员会继续保持合作。')]))
recognizer = hanlp.load(hanlp.pretrained.ner.CONLL03_NER_BERT_BASE_UNCASED_EN)
print(recognizer(["President", "Obama", "is", "speaking", "at", "the", "White", "House"]))


# 依存句法分析
syntactic_parser = hanlp.load(hanlp.pretrained.dep.PTB_BIAFFINE_DEP_EN)
print(syntactic_parser([('Is', 'VBZ'), ('this', 'DT'), ('the', 'DT'), ('future', 'NN'), ('of', 'IN'), ('chamber', 'NN'), ('music', 'NN'), ('?', '.')]))
syntactic_parser = hanlp.load(hanlp.pretrained.dep.CTB7_BIAFFINE_DEP_ZH)
print(syntactic_parser([('蜡烛', 'NN'), ('两', 'CD'), ('头', 'NN'), ('烧', 'VV')]))


# 语义依存分析
semantic_parser = hanlp.load(hanlp.pretrained.sdp.SEMEVAL15_PAS_BIAFFINE_EN)
print(semantic_parser([('Is', 'VBZ'), ('this', 'DT'), ('the', 'DT'), ('future', 'NN'), ('of', 'IN'), ('chamber', 'NN'), ('music', 'NN'), ('?', '.')]))
semantic_parser = hanlp.load(hanlp.pretrained.sdp.SEMEVAL16_NEWS_BIAFFINE_ZH)
print(semantic_parser([('蜡烛', 'NN'), ('两', 'CD'), ('头', 'NN'), ('烧', 'VV')]))


# 流水线
pipeline = hanlp.pipeline() \
    .append(hanlp.utils.rules.split_sentence, output_key='sentences') \
    .append(tokenizer, output_key='tokens') \
    .append(tagger, output_key='part_of_speech_tags') \
    .append(syntactic_parser, input_key=('tokens', 'part_of_speech_tags'), output_key='syntactic_dependencies') \
    .append(semantic_parser, input_key=('tokens', 'part_of_speech_tags'), output_key='semantic_dependencies')
print(pipeline)
text="Jobs and Wozniak co-founded Apple in 1976 to sell Wozniak's Apple I personal computer.Together the duo gained fame and wealth a year later with the Apple II."
print(pipeline(text))
text=[
    "HanLP是一系列模型与算法组成的自然语言处理工具包，目标是普及自然语言处理在生产环境中的应用。",
    "HanLP具备功能完善、性能高效、架构清晰、语料时新、可自定义的特点。",
    "内部算法经过工业界和学术界考验，配套书籍《自然语言处理入门》已经出版。" ]
print(pipeline(text))
```

注意：

- 上面的程序，因为 load 过多模型，导致内存不足，所以没法运行。（有没有好的解决办法？好像没找到 unload 函数）


使用自定义字典：

```py

from hanlp.common.trie import Trie

import hanlp

tokenizer = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')
text = 'NLP统计模型没有加规则，聪明人知道自己加。英文、数字、自定义词典统统都是规则。'
print(tokenizer(text))

trie = Trie()
trie.update({'规则': 'test1', '词典': 'test2', '自定义': 'test3'})


def split_sents(text: str, trie: Trie):
    words = trie.parse_longest(text)
    sents = []
    pre_start = 0
    offsets = []
    for word, value, start, end in words:
        if pre_start != start:
            sents.append(text[pre_start: start])
            offsets.append(pre_start)
        pre_start = end
    if pre_start != len(text):
        sents.append(text[pre_start:])
        offsets.append(pre_start)
    return sents, offsets, words


print(split_sents(text, trie))


def merge_parts(parts, offsets, words):
    items = [(i, p) for (i, p) in zip(offsets, parts)]
    items += [(start, [word]) for (word, value, start, end) in words]
    # In case you need the tag, use the following line instead
    # items += [(start, [(word, value)]) for (word, value, start, end) in words]
    return [each for x in sorted(items) for each in x[1]]


tokenizer = hanlp.pipeline() \
    .append(split_sents, output_key=('parts', 'offsets', 'words'), trie=trie) \
    .append(tokenizer, input_key='parts', output_key='tokens') \
    .append(merge_parts, input_key=('tokens', 'offsets', 'words'), output_key='merged')

print(tokenizer(text))
```

输出：

```txt
['NLP', '统计', '模型', '没有', '加', '规则', '，', '聪明人', '知道', '自己', '加', '。', '英文', '、', '数字', '、', '自定义', '词典', '统统', '都', '是', '规则', '。']
(['NLP统计模型没有加', '，聪明人知道自己加。英文、数字、', '统统都是', '。'], [0, 12, 33, 39], [('规则', 'test1', 10, 12), ('自定义', 'test3', 28, 31), ('词典', 'test2', 31, 33), ('规则', 'test1', 37, 39)])
{
  "parts": [
    "NLP统计模型没有加",
    "，聪明人知道自己加。英文、数字、",
    "统统都是",
    "。"
  ],
  "offsets": [
    0,
    12,
    33,
    39
  ],
  "words": [
    ["规则", "test1", 10, 12],
    ["自定义", "test3", 28, 31],
    ["词典", "test2", 31, 33],
    ["规则", "test1", 37, 39]
  ],
  "tokens": [
    ["NLP", "统计", "模型", "没有", "加"],
    ["，", "聪明人", "知道", "自己", "加", "。", "英文", "、", "数字", "、"],
    ["统统", "都", "是"],
    ["。"]
  ],
  "merged": [
    "NLP",
    "统计",
    "模型",
    "没有",
    "加",
    "规则",
    "，",
    "聪明人",
    "知道",
    "自己",
    "加",
    "。",
    "英文",
    "、",
    "数字",
    "、",
    "自定义",
    "词典",
    "统统",
    "都",
    "是",
    "规则",
    "。"
  ]
}
```




hanlp 自带的几个例子挺好的，可以都放进来。其他的 demo 好像都没有很靠谱。






## 词性列表

（这个词性列表为什么于 hanlp 使用 tagger 做出来的完全不同？而且连大小写都不同？确认下。）

| **Hanlp** | **词性列表**                                        |
| -------------------- | ---------------------------------------- |
| a      |  形容词       |
| ad     |   副形词      |
| ag     |   形容词性语素 |
| al     |   形容词性惯用语 |
| an     |   名形词      |
| b      |  区别词       |
| begin  |      仅用于始##始 |
| bg     |   区别语素    |
| bl     |   区别词性惯用语 |
| c      |  连词         |
| cc     |   并列连词    |
| d      |  副词         |
| dg     |   辄,俱,复之类的副词 |
| dl     |   连语        |
| e      |  叹词         |
| end    |    仅用于终##终 |
| f      |  方位词       |
| g      |  学术词汇     |
| gb     |   生物相关词汇 |
| gbc    |    生物类别   |
| gc     |   化学相关词汇 |
| gg     |   地理地质相关词汇 |
| gi     |   计算机相关词汇 |
| gm     |   数学相关词汇 |
| gp     |   物理相关词汇 |
| h      |  前缀         |
| i      |  成语         |
| j      |  简称略语     |
| k      |  后缀         |
| l      |  习用语       |
| m      |  数词         |
| mg     |   数语素      |
| Mg     |   甲乙丙丁之类的数词 |
| mq     |   数量词      |
| n      |  名词         |
| nb     |   生物名      |
| nba    |    动物名     |
| nbc    |    动物纲目   |
| nbp    |    植物名     |
| nf     |   食品，比如“薯片” |
| ng     |   名词性语素  |
| nh     |   医药疾病等健康相关名词 |
| nhd    |    疾病       |
| nhm    |    药品       |
| ni     |   机构相关（不是独立机构名） |
| nic    |    下属机构   |
| nis    |    机构后缀   |
| nit    |    教育相关机构 |
| nl     |   名词性惯用语 |
| nm     |   物品名      |
| nmc    |    化学品名   |
| nn     |   工作相关名词 |
| nnd    |    职业       |
| nnt    |    职务职称   |
| nr     |   人名        |
| nr1    |    复姓       |
| nr2    |    蒙古姓名   |
| nrf    |    音译人名   |
| nrj    |    日语人名   |
| ns     |   地名        |
| nsf    |    音译地名   |
| nt     |   机构团体名  |
| ntc    |    公司名     |
| ntcb   |     银行      |
| ntcf   |     工厂      |
| ntch   |     酒店宾馆  |
| nth    |    医院       |
| nto    |    政府机构   |
| nts    |    中小学     |
| ntu    |    大学       |
| nx     |   字母专名    |
| nz     |   其他专名    |
| o      |  拟声词       |
| p      |  介词         |
| pba    |    介词“把”   |
| pbei   |     介词“被”  |
| q      |  量词         |
| qg     |   量词语素    |
| qt     |   时量词      |
| qv     |   动量词      |
| r      |  代词         |
| rg     |   代词性语素  |
| Rg     |   古汉语代词性语素 |
| rr     |   人称代词    |
| ry     |   疑问代词    |
| rys    |    处所疑问代词 |
| ryt    |    时间疑问代词 |
| ryv    |    谓词性疑问代词 |
| rz     |   指示代词    |
| rzs    |    处所指示代词 |
| rzt    |    时间指示代词 |
| rzv    |    谓词性指示代词 |
| s      |  处所词       |
| t      |  时间词       |
| tg     |   时间词性语素 |
| u      |  助词         |
| ud     |   助词        |
| ude1   |     的 底     |
| ude2   |     地        |
| ude3   |     得        |
| udeng  |      等 等等 云云 |
| udh    |    的话       |
| ug     |   过          |
| uguo   |     过        |
| uj     |   助词        |
| ul     |   连词        |
| ule    |    了 喽      |
| ulian  |      连 （“连小学生都会”） |
| uls    |    来讲 来说 而言 说来 |
| usuo   |     所        |
| uv     |   连词        |
| uyy    |    一样 一般 似的 般 |
| uz     |   着          |
| uzhe   |     着        |
| uzhi   |     之        |
| v      |  动词         |
| vd     |   副动词      |
| vf     |   趋向动词    |
| vg     |   动词性语素  |
| vi     |   不及物动词（内动词） |
| vl     |   动词性惯用语 |
| vn     |   名动词      |
| vshi   |     动词“是”  |
| vx     |   形式动词    |
| vyou   |     动词“有”  |
| w      |  标点符号     |
| wb     |   百分号千分号，全角：`％ ‰`   半角：`%` |
| wd     |   逗号，全角：`，` 半角：`,` |
| wf     |   分号，全角：`；` 半角： `;` |
| wh     |   单位符号，全角：`￥ ＄ ￡   °  ℃`  半角：`$` |
| wj     |   句号，全角：。 |
| wky    |    右括号，全角：`） 〕   ］ ｝ 》  】 〗 〉 半角： )  ] { >` |
| wkz    |    左括号，全角：（ 〔   ［  ｛  《  【  〖 〈  半角：( [ { < |
| wm     |   冒号，全角：`：` 半角： `:` |
| wn     |   顿号，全角：`、` |
| wp     |   破折号，全角：`——   －－  ——－`  半角：`—  —-` |
| ws     |   省略号，全角：`……`   `…` |
| wt     |   叹号，全角：`！` |
| ww     |   问号，全角：`？` |
| wyy    |    右引号，全角：`” ’ 』` |
| wyz    |    左引号，全角：`“ ‘ 『` |
| x      |  字符串       |
| xu     |   网址URL     |
| xx     |   非语素字    |
| y      |  语气词(delete yg) |
| yg     |   语气语素    |
| z      |  状态词       |
| zg     |   状态词      |

 