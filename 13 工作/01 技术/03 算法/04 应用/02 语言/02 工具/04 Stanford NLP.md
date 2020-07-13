# Stanford NLP

（这个是不是已经过期了，github 最后更新是 2 年前）

作用：

- Stanford NLP提供了一系列自然语言分析工具。它能够给出基本的词形，词性，不管是公司名还是人名等，格式化的日期，时间，量词，并且能够标记句子的结构，语法形式和字词依赖，指明那些名字指向同样的实体，指明情绪，提取发言中的开放关系等。
- 功能：
  - 一个集成的语言分析工具集；
  - 进行快速，可靠的任意文本分析；
  - 整体的高质量的文本分析;
  - 支持多种主流语言;
  - 多种编程语言的易用接口;
  - 方便的简单的部署web服务。
- 支持中文。


地址：

- [官网](https://stanfordnlp.github.io/CoreNLP/download.html)
- [网盘](https://pan.baidu.com/s/1CYS7idcyzKB8zanNcWogvw) 提取码：02d4 内有:
  - stanford-corenlp-latest
  - stanford-corenlp-4.0.0-models-chinese.jar
- [可以看下](https://github.com/Lynten/stanford-corenlp)


安装：

- 安装 java
  - [地址](https://www.oracle.com/java/technologies/javase-downloads.html)
  - 如图：
    <p align="center">
      <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200713/TcVXv861VTfe.png?imageslim">
    </p>
  - 注意：
    - 在 oracle 的官网下载 64 位 java，java 的官网只有 32 位 java 的下载，使用 32 位的 java 时，创建 StanfordCoreNLP 使用 `memory='4g',quiet=False` 时会打印出 error：
      ```txt
      Error: Could not create the Java Virtual Machine.
      Error: A fatal exception has occurred. Program will exit.
      Invalid maximum heap size: -Xmx6g
      The specified size exceeds the maximum representable size.
      ```
    - 如果将 java 安装到自定义的路径下，那么需要将对应的 bin 路径如 `D:\01.ProgramFiles\Java\14.0.1\bin` 添加到环境变量的 Path 中。
    - 安装完 java 可能需要重启电脑。
- 安装 stanford nlp 自然语言处理包： `pip install stanfordcorenlp`
- 下载两个东西：
  - 一个是 Stanford CoreNLP 文件，一个是 中文 jar 包。
  - [地址](https://stanfordnlp.github.io/CoreNLP/download.html)
  - 如图：
    <p align="center">
        <img width="70%" height="70%" src="http://images.iterate.site/blog/image/20200711/spOLmHV3nx45.png?imageslim">
    </p>
  - 这两个包可以通过上面的百度网盘下载。使用官网下载基本速度为0。
- 把加压后的 Stanford CoreNLP 文件夹和下载的stanford-chinese-corenlp-2018-02-27-models.jar放在同一目录下
- 在Python中引用模型：
  - 举例：
    ```py

    from stanfordcorenlp import StanfordCoreNLP
    nlp = StanfordCoreNLP(r'path', lang='zh')
    ```
  - 说明：
    - path 为解压后的 Stanford CoreNLP 文件夹的路径。


举例：

```py
from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('D:\stanford-corenlp', lang='zh')

fin = open('news.txt', 'r', encoding='utf8')
fner = open('ner.txt', 'w', encoding='utf8')
ftag = open('pos_tag.txt', 'w', encoding='utf8')
for line in fin:
    line = line.strip()
    if len(line) < 1:
        continue

    fner.write(" ".join([each[0] + "/" + each[1] for each in nlp.ner(line) if len(each) == 2]) + "\n") # nlp.ner 可以识别出 人名 地点 国家 等  命名实体识别
    ftag.write(" ".join([each[0] + "/" + each[1] for each in nlp.pos_tag(line) if len(each) == 2]) + "\n") # nlp.pos_tag 可以用来划分 名词 动词 等。
fner.close()
ftag.close()
nlp.close()
```

其中：

news.txt：

```txt
所属行业： 快速消费品(食品,饮料,化妆品)
业务部 采购主管
本人是从黄金珠宝营业员开始做起,现从事黄金珠宝和化妆品的招商工作,曾参与1999年和2004年3家商场的招商工作,在公司我能比较完美的贯彻和执行公司的营销策略和销售计划,并能提出合理化建议,在工作中能与供应商保持良好的合作关系.
RESUMEDOCSSTARTFLAG	销售总监	半年市场部总监工作，之后公司开始华东地区大客户销售业务，转为华东区
销售总监，创建易车网华东地区大客户销售团队，并带领团队完成公司下达
的销售任务。
主要收获：半年市场总监工作经验，成功稳固公司在上海的品牌形象和客户关系，有效
利用各种手段推广公司品牌。
两年半网络广告销售从业及管理经验，成功组建
易车网华东区域大客户销售团队，以直客为导向，兼顾渠道。
有丰富的互联网
广告知识，同时也有华东区域的所有汽车客户及周边产品的直客关系，及相
关的渠道关系。
RESUMEDOCSSTARTFLAG	市场部高级经理	负责东方网旗下的<东方社区>刊物的市场推广及合作
主要收获: 了解纸媒体特点,熟悉期刊编辑出版流程,并结合活动推广刊物
同时也参与
部分平面广告销售工作,熟悉平面广告销售的所有流程
RESUMEDOCSSTARTFLAG	事业发展部主管	（自 2004 年 3 月——2005 年 3 月，在共青团上海市委管理信息部挂职锻炼）网站编辑;
公司内部活动策划组织,项目管理;
协调. 联络并保持与团市委机关的良好合作关系
主要收获: 对网络与信息化有了进一步的认识,了解网站的运营模式
在团市委机关挂职
期间,加强了自身思想道德等各方面的学习,积累了广泛的人脉,多次参与
一些市大型活动的策划组织及执行,列: 上海 IT 青年十大新锐评选活动. 上
海市信息化青年人才协会第一次会员大会--暨“世博会与信息化”青年论
坛等等,使自身各方面综合素质有了一个更全面的提高
RESUMEDOCSSTARTFLAG	市场公关专员	策划组织各类活动,包括户外拓展训练,团队建设等;
策划组织公司内部各项活动,落实凝聚力工程;
组织各类公关接待及外联工作;
各类活动的主持及执行等事宜
主要收获: 积累了更丰富的策划经验,多次的主持经历让我在语言表达方面有了进一步
的提高,不断接触到的不同人群,也让我积累了良好的社会关系,现场的活
动执行,更让我在面对突发事件时,有了冷静思考的能力
```


输出：

ner.txt：

```txt
所/O 属/O 行业/O ：/O 快速/O 消费品/O (/O 食品/O ,/O 饮料/O ,/O 化妆品/O )/O
业务部/O 采购/TITLE 主管/TITLE
本人/O 是/O 从/O 黄金/O 珠宝/O 营业员/O 开始/O 做起/O ,/O 现/O 从事/O 黄金/O 珠宝/O 和/O 化妆品/O 的/O 招商/O 工作/O ,/O 曾/O 参与/O 1999年/DATE 和/O 2004年/DATE 3/NUMBER 家/O 商场/O 的/O 招商/O 工作/O ,/O 在/O 公司/O 我/O 能/O 比较/O 完美/O 的/O 贯彻/O 和/O 执行/O 公司/O 的/O 营销/O 策略/O 和/O 销售/O 计划/O ,/O 并/O 能/O 提出/O 合理化/O 建议/O ,/O 在/O 工作/O 中/O 能/O 与/O 供应商/O 保持/O 良好/O 的/O 合作/O 关系/O ./O
RESUMEDOCSSTARTFLAG/O 销售/O 总监/TITLE 半/NUMBER 年/MISC 市场部/O 总监/TITLE 工作/O ，/O 之后/O 公司/O 开始/O 华东/LOCATION 地区/O 大/O 客户/O 销售/O 业务/O ，/O 转为/O 华东区/O
销售/O 总监/TITLE ，/O 创建/O 易车网/O 华东/LOCATION 地区/O 大/O 客户/O 销售/O 团队/O ，/O 并/O 带领/O 团队/O 完成/O 公司/O 下达/O
的/O 销售/O 任务/O 。/O
主要/O 收获/O ：/O 半/NUMBER 年/MISC 市场/TITLE 总监/TITLE 工作/O 经验/O ，/O 成功/O 稳固/O 公司/O 在/O 上海/STATE_OR_PROVINCE 的/O 品牌/O 形象/O 和/O 客户/O 关系/O ，/O 有效/O
利用/O 各/O 种/O 手段/O 推广/O 公司/O 品牌/O 。/O
两/NUMBER 年/MISC 半/NUMBER 网络/O 广告/O 销售/O 从业/O 及/O 管理/O 经验/O ，/O 成功/O 组建/O
易车网/O 华东/LOCATION 区域/O 大/O 客户/O 销售/O 团队/O ，/O 以/O 直客/O 为/O 导向/O ，/O 兼顾/O 渠道/O 。/O
有/O 丰富/O 的/O 互联网/O
广告/O 知识/O ，/O 同时/O 也/O 有/O 华东/LOCATION 区域/O 的/O 所有/O 汽车/O 客户/O 及/O 周边/O 产品/O 的/O 直客/O 关系/O ，/O 及/O 相/O
关/O 的/O 渠道/O 关系/O 。/O
RESUMEDOCSSTARTFLAG/ORGANIZATION 市场部/ORGANIZATION 高级/TITLE 经理/TITLE 负责/O 东方网/ORGANIZATION 旗下/O 的/O </O 东方/O 社区/O >/O 刊物/O 的/O 市场/O 推广/O 及/O 合作/O
主要/O 收获/O :/O 了解/O 纸/O 媒体/O 特点/O ,/O 熟悉/O 期刊/O 编辑/TITLE 出版/O 流程/O ,/O 并/O 结合/O 活动/O 推广/O 刊物/O
同时/O 也/O 参与/O
部分/NUMBER 平面/O 广告/O 销售/O 工作/O ,/O 熟悉/O 平面/O 广告/O 销售/O 的/O 所有/O 流程/O
RESUMEDOCSSTARTFLAG/O 事业/O 发展部/O 主管/TITLE （/O 自/O 2004/NUMBER 年/MISC 3/NUMBER 月/MISC ——/O 2005/NUMBER 年/MISC 3/NUMBER 月/MISC ，/O 在/O 共青团/ORGANIZATION 上海/ORGANIZATION 市委/ORGANIZATION 管理/ORGANIZATION 信息部/ORGANIZATION 挂职/O 锻炼/O ）/O 网站/O 编辑/TITLE ;/O
公司/O 内部/O 活动/O 策划/O 组织/O ,/O 项目/O 管理/O ;/O
协调/O ./O 联络/O 并/O 保持/O 与/O 团市委/ORGANIZATION 机关/O 的/O 良好/O 合作/O 关系/O
主要/O 收获/O :/O 对/O 网络/O 与/O 信息化/O 有/O 了/O 进一步/O 的/O 认识/O ,/O 了解/O 网站/O 的/O 运营/O 模式/O
在/O 团市委/ORGANIZATION 机关/O 挂职/O
期间/O ,/O 加强/O 了/O 自身/O 思想/O 道德/O 等/O 各/O 方面/O 的/O 学习/O ,/O 积累/O 了/O 广泛/O 的/O 人脉/O ,/O 多/NUMBER 次/O 参与/O
一些/NUMBER 市大型/O 活动/O 的/O 策划/O 组织/O 及/O 执行/O ,/O 列/O :/O 上海/STATE_OR_PROVINCE IT/O 青年/O 十/NUMBER 大/MISC 新锐/MISC 评选/O 活动/O ./O 上/O
海市/ORGANIZATION 信息化/ORGANIZATION 青年/ORGANIZATION 人才/ORGANIZATION 协会/ORGANIZATION 第一/ORDINAL 次/O 会员/O 大会/O --/O 暨/O “/O 世博会/MISC 与/O 信息化/O ”/O 青年/O 论/O
坛/O 等等/O ,/O 使/O 自身/O 各/O 方面/O 综合/O 素质/O 有/O 了/O 一/NUMBER 个/O 更/O 全面/O 的/O 提高/O
RESUMEDOCSSTARTFLAG/O 市场/O 公关/O 专员/TITLE 策划/O 组织/O 各/O 类/O 活动/O ,/O 包括/O 户外/O 拓展/O 训练/O ,/O 团队/O 建设/O 等/O ;/O
策划/O 组织/O 公司/O 内部/O 各/O 项/O 活动/O ,/O 落实/O 凝聚力/O 工程/O ;/O
组织/O 各/O 类/O 公关/O 接待/O 及/O 外联/O 工作/O ;/O
各/O 类/O 活动/O 的/O 主持/TITLE 及/O 执行/O 等/O 事宜/O
主要/O 收获/O :/O 积累/O 了/O 更/O 丰富/O 的/O 策划/O 经验/O ,/O 多/NUMBER 次/O 的/O 主持/TITLE 经历/O 让/O 我/O 在/O 语言/O 表达/O 方面/O 有/O 了/O 进一步/O
的/O 提高/O ,/O 不断/O 接触/O 到/O 的/O 不同/O 人群/O ,/O 也/O 让/O 我/O 积累/O 了/O 良好/O 的/O 社会/O 关系/O ,/O 现场/O 的/O 活/O
动/O 执行/O ,/O 更/O 让/O 我/O 在/O 面对/O 突发/O 事件/O 时/O ,/O 有/O 了/O 冷静/O 思考/O 的/O 能力/O
```

pos_tag.txt：

```txt
所/AD 属/VV 行业/NN ：/PU 快速/AD 消费品/NN (/PU 食品/NN ,/PU 饮料/NN ,/PU 化妆品/NN )/PU
业务部/NN 采购/NN 主管/NN
本人/PN 是/VC 从/P 黄金/NN 珠宝/NN 营业员/NN 开始/VV 做起/VV ,/PU 现/AD 从事/VV 黄金/NN 珠宝/NN 和/CC 化妆品/NN 的/DEG 招商/NN 工作/NN ,/PU 曾/AD 参与/VV 1999年/NT 和/CC 2004年/NT 3/CD 家/M 商场/NN 的/DEG 招商/NN 工作/NN ,/PU 在/P 公司/NN 我/PN 能/VV 比较/AD 完美/VA 的/DEC 贯彻/NN 和/CC 执行/NN 公司/NN 的/DEG 营销/NN 策略/NN 和/CC 销售/NN 计划/NN ,/PU 并/CC 能/VV 提出/VV 合理化/JJ 建议/NN ,/PU 在/P 工作/NN 中/LC 能/VV 与/P 供应商/NN 保持/VV 良好/VA 的/DEC 合作/NN 关系/NN ./PU
RESUMEDOCSSTARTFLAG/NR 销售/VV 总监/NN 半/CD 年/M 市场部/NN 总监/NN 工作/NN ，/PU 之后/AD 公司/NN 开始/VV 华东/NR 地区/NN 大/JJ 客户/NN 销售/NN 业务/NN ，/PU 转为/VV 华东区/NR
销售/NN 总监/NN ，/PU 创建/VV 易车网/NN 华东/NR 地区/NN 大/JJ 客户/NN 销售/NN 团队/NN ，/PU 并/CC 带领/VV 团队/NN 完成/VV 公司/NN 下达/VV
的/DEG 销售/NN 任务/NN 。/PU
主要/JJ 收获/NN ：/PU 半/CD 年/M 市场/NN 总监/NN 工作/NN 经验/NN ，/PU 成功/AD 稳固/VV 公司/NN 在/P 上海/NR 的/DEG 品牌/NN 形象/NN 和/CC 客户/NN 关系/NN ，/PU 有效/VA
利用/VV 各/DT 种/M 手段/NN 推广/NN 公司/NN 品牌/NN 。/PU
两/CD 年/M 半/CD 网络/NN 广告/NN 销售/NN 从业/JJ 及/CC 管理/NN 经验/NN ，/PU 成功/AD 组建/VV
易车网/NR 华东/NR 区域/NN 大/JJ 客户/NN 销售/NN 团队/NN ，/PU 以/P 直客/NN 为/VC 导向/NN ，/PU 兼顾/VV 渠道/NN 。/PU
有/VE 丰富/JJ 的/DEG 互联网/NN
广告/NN 知识/NN ，/PU 同时/AD 也/AD 有/VE 华东/NR 区域/NN 的/DEG 所有/DT 汽车/NN 客户/NN 及/CC 周边/NN 产品/NN 的/DEG 直客/NN 关系/NN ，/PU 及/CC 相/AD
关/VV 的/DEC 渠道/NN 关系/NN 。/PU
RESUMEDOCSSTARTFLAG/NR 市场部/NN 高级/JJ 经理/NN 负责/VV 东方网/NN 旗下/NN 的/DEG </PU 东方/NN 社区/NN >/PU 刊物/NN 的/DEG 市场/NN 推广/NN 及/CC 合作/NN
主要/JJ 收获/NN :/PU 了解/VV 纸/NN 媒体/NN 特点/NN ,/PU 熟悉/VV 期刊/NN 编辑/VV 出版/NN 流程/NN ,/PU 并/AD 结合/VV 活动/NN 推广/NN 刊物/NN
同时/AD 也/AD 参与/VV
部分/CD 平面/JJ 广告/NN 销售/NN 工作/NN ,/PU 熟悉/VV 平面/JJ 广告/NN 销售/VV 的/DEC 所有/NN 流程/NN
RESUMEDOCSSTARTFLAG/NR 事业/NN 发展部/NN 主管/NN （/PU 自/P 2004/CD 年/M 3/CD 月/NN ——/PU 2005/CD 年/M 3/CD 月/NN ，/PU 在/P 共青团/NR 上海/NR 市委/NN 管理/NN 信息部/NN 挂职/VV 锻炼/VV ）/PU 网站/NN 编辑/NN ;/PU
公司/NN 内部/NN 活动/NN 策划/NN 组织/NN ,/PU 项目/NN 管理/NN ;/PU
协调/NN ./PU 联络/VV 并/CC 保持/VV 与/CC 团市委/NN 机关/NN 的/DEG 良好/JJ 合作/NN 关系/NN
主要/JJ 收获/NN :/PU 对/P 网络/NN 与/CC 信息化/NN 有/VE 了/AS 进一步/JJ 的/DEG 认识/NN ,/PU 了解/VV 网站/NN 的/DEC 运营/NN 模式/NN
在/P 团市委/NN 机关/NN 挂职/VV
期间/NN ,/PU 加强/VV 了/AS 自身/PN 思想/NN 道德/NN 等/ETC 各/DT 方面/NN 的/DEG 学习/NN ,/PU 积累/VV 了/AS 广泛/VA 的/DEC 人脉/NN ,/PU 多/CD 次/M 参与/VV
一些/CD 市大型/NN 活动/NN 的/DEC 策划/NN 组织/NN 及/CC 执行/NN ,/PU 列/VV :/PU 上海/NR IT/NR 青年/NN 十/CD 大/JJ 新锐/NN 评选/NN 活动/NN ./PU 上/VV
海市/NR 信息化/JJ 青年/NN 人才/NN 协会/NN 第一/OD 次/M 会员/NN 大会/NN --/PU 暨/CC “/PU 世博会/NN 与/CC 信息化/NN ”/PU 青年/NN 论/NN
坛/NN 等等/ETC ,/PU 使/VV 自身/PN 各/DT 方面/NN 综合/JJ 素质/NN 有/VE 了/AS 一/CD 个/M 更/AD 全面/JJ 的/DEG 提高/NN
RESUMEDOCSSTARTFLAG/NR 市场/NN 公关/NN 专员/NN 策划/NN 组织/NN 各/DT 类/M 活动/NN ,/PU 包括/VV 户外/JJ 拓展/NN 训练/NN ,/PU 团队/NN 建设/NN 等/ETC ;/PU
策划/NN 组织/NN 公司/NN 内部/NN 各/DT 项/M 活动/NN ,/PU 落实/VV 凝聚力/NN 工程/NN ;/PU
组织/VV 各/DT 类/M 公关/NN 接待/VV 及/CC 外联/NN 工作/NN ;/PU
各/DT 类/M 活动/NN 的/DEG 主持/NN 及/CC 执行/NN 等/ETC 事宜/NN
主要/JJ 收获/NN :/PU 积累/VV 了/AS 更/AD 丰富/VA 的/DEC 策划/NN 经验/NN ,/PU 多/CD 次/M 的/DEG 主持/NN 经历/NN 让/VV 我/PN 在/P 语言/NN 表达/NN 方面/NN 有/VE 了/AS 进一步/AD
的/DEG 提高/NN ,/PU 不断/AD 接触/VV 到/VV 的/DEC 不同/JJ 人群/NN ,/PU 也/AD 让/VV 我/PN 积累/VV 了/AS 良好/VA 的/DEC 社会/NN 关系/NN ,/PU 现场/NN 的/DEG 活/NN
动/VV 执行/VV ,/PU 更/AD 让/VV 我/PN 在/P 面对/VV 突发/JJ 事件/NN 时/LC ,/PU 有/VE 了/AS 冷静/NN 思考/VV 的/DEC 能力/NN
```

疑问：

- 为什么这个词性标注与课程中的那套小写的词性标注有很大的不同呢？stanford nlp 现在的词性标注列表是什么？