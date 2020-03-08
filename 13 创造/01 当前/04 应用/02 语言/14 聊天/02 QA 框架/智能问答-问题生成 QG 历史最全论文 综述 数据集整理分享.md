Question Generation（问题生成），简单理解就是“主动提问”的AI应用场景，是Question Answer（QA）一个子领域。QG 的应用还是挺广泛的，像是为 QA 任务产生训练数据、自动合成 FAQ 文档、自动辅导系统（automatic tutoring systems）等。

传统工作主要是利用句法树或者知识库，基于规则来产生问题。如基于语法（Heilman and Smith, 2010; Ali et al., 2010; Kumar et al., 2015），基于语义（Mannem et al., 2010; Lindberg et al., 2013），大多是利用规则操作句法树来形成问句。还有是基于模板（templates），定好 slot，然后从文档中找到实体来填充模板（Lindberg et al., 2013; Chali and Golestanirad, 2016）。

**本文整理了QG相关的经典、前沿、综述性的论文，涉及篇章级问题生成、基于知识图谱问题生成等，以及一些该领域的公开数据集，评评测指标，分享给大家。**

资源整理自网络，源地址：[https://github.com/bisheng/QuestionGeneration](https://link.zhihu.com/?target=https%3A//github.com/bisheng/QuestionGeneration)

**篇章级别QG**

Harvesting paragraph-level question-answer pairs from wikipedia. Xinya Du, Claire Cardie. ACL, 2018. paper code

Leveraging Context Information for Natural Question Generation. Linfeng Song, Zhiguo Wang, Wael Hamza, Yue Zhang, Daniel Gildea. ACL, 2018. paper code

Paragraph-level Neural Question Generation with Maxout Pointer and Gated Self-attention Networks. Yao Zhao, Xiaochuan Ni, Yuanyuan Ding, Qifa Ke. EMNLP, 2018. paper code

Capturing Greater Context for Question Generation. Luu Anh Tuan, Darsh J Shah, Regina Barzilay. arxiv, 2019. paper code

**基于知识图谱QG，KBQG**

**2014-2019**

Difficulty-controllable Multi-hop Question Generation From Knowledge Graphs. Vishwajeet Kumar, Yuncheng Hua, Ganesh Ramakrishnan, et al. EMNLP, 2019. paper code&dataset

Difficulty-level Modeling of Ontology-based Factual Questions.

Question Difficulty Estimation in Community Question Answering Services.

Domain-specific question generation from a knowledge base.

Generating Quiz Questions from knowledge graphs.

Knowledge Questions from Knowledge Graphs.

Question Generation from a Knowledge Base with Web Exploration.

Question generation from a knowledge base.

Question generation from concept maps.

**2008-2013**

A similarity-based theory of controlling MCQ difficulty. Tahani Alsubait, Bijan Parsia, Ulrike Sattler IEEE, 2013. paper

**2014-2019**

Let's Ask Again: Refine Network for Automatic Question Generation. Nema P, Mohankumar A K, Khapra M M, et al. arXiv, 2019. paper

Difficulty Controllable Generation of Reading Comprehension Questions. Yifan Gao, Lidong Bing, Wang Chen, et al. IJCAI, 2019. paper

Generating Question-Answer Hierarchies. Kalpesh Krishna and Mohit Iyyer. ACL, 2019. paper code

Improving Generative Visual Dialog by Answering Diverse Questions. Murahari V, Chattopadhyay P, Batra D, et al. arXiv, 2019. paper

Reverse SQL Question Generation Algorithm in the DBLearn Adaptive E-Learning System. Atchariyachanvanich K, Nalintippayawong S, Julavanich T. IEEE, 2019. paper

Interconnected Question Generation with Coreference Alignment and Conversation Flow Modeling. Yifan Gao, Piji Li, Irwin King, et al. ACL, 2019. paper code

Cross-Lingual Training for Automatic Question Generation. Kumar V, Joshi N, Mukherjee A, et al. ACL, 2019. paper dataset

Multi-hop Reading Comprehension through Question Decomposition and Rescoring. Sewon Min, Victor Zhong, Luke Zettlemoyer, et al. ACL, 2019. paper

Learning to Ask Unanswerable Questions for Machine Reading Comprehension. Haichao Zhu, Li Dong, Furu Wei, et al. ACL, 2019.

Reinforced Dynamic Reasoning for Conversational Question Generation. Boyuan Pan, Hao Li, Ziyu Yao, et al. ACL, 2019. paper code dataset

Asking the Crowd: Question Analysis, Evaluation and Generation for Open Discussion on Online Forums. Zi Chai, Xinyu Xing, Xiaojun Wan, et al. ACL, 2019.

Self-Attention Architectures for Answer-Agnostic Neural Question Generation. Thomas Scialom, Benjamin Piwowarski and Jacopo Staiano. ACL, 2019.

Evaluating Rewards for Question Generation Models. Tom Hosking and Sebastian Riedel. NAACL, 2019. paper

Difficulty controllable question generation for reading comprehension. Gao Y, Wang J, Bing L, et al. IJCAI, 2019. paper

Weak Supervision Enhanced Generative Network for Question Generation. Yutong Wang, Jiyuan Zheng, Qijiong Liu, et al. IJCAI, 2019. paper

Answer-based Adversarial Training for Generating Clarification Questions. Rao S, Daumé III H. NAACL, 2019. paper code

Information Maximizing Visual Question Generation. Krishna, Ranjay, Bernstein, Michael, Fei-Fei, Li. arXiv, 2019. paper

Learning to Generate Questions by Learning What not to Generate. Liu B, Zhao M, Niu D, et al. WWW, 2019. paper

Joint Learning of Question Answering and Question Generation. Sun Y, Tang D, Duan N, et al. IEEE, 2019. paper dataset

Domain-specific question-answer pair generation. Beason W A, Chandrasekaran S, Gattiker A E, et al. Google Patents, 2019. paper

Anaphora Reasoning Question Generation Using Entity Coreference. Hasegawa, Kimihiro, Takaaki Matsumoto, and Teruko Mitamura. 2019. paper

Improving Neural Question Generation using Answer Separation. Kim Y, Lee H, Shin J, et al. AAAI, 2019. paper

A novel framework for Automatic Chinese Question Generation based on multi-feature neural network mode Zheng H T, Han J, Chen J Y, et al. Comput. Sci. Inf. Syst., 2018. paper

Visual question generation as dual task of visual question answering. Li Y, Duan N, Zhou B, et al. IEEE, 2018. paper

Answer-focused and Position-aware Neural Question Generation. Sun X, Liu J, Lyu Y, et al. EMNLP, 2018. paper

Automatic Question Generation using Relative Pronouns and Adverbs. Khullar P, Rachna K, Hase M, et al. ACL, 2018. paper

Learning to ask good questions: Ranking clarification questions using neural expected value of perfect information Rao S, Daumé III H. arXiv, 2018. paper dataset

Soft layer-specific multi-task summarization with entailment and question generation. Guo H, Pasunuru R, Bansal M. arXiv, 2018. paper

Leveraging context information for natural question generation Song L, Wang Z, Hamza W, et al. ACL, 2018. paper code

Learning to Ask Questions in Open-domain Conversational Systems with Typed Decoders. Wang Y, Liu C, Huang M, et al. arXiv, 2018. paper code dataset

Did the model understand the question? Mudrakarta P K, Taly A, Sundararajan M, et al. arXiv, 2018. paper code dataset

Know What You Don't Know: Unanswerable Questions for SQuAD. Rajpurkar P, Jia R, Liang P. arXiv, 2018. paper code&dataset

Harvesting paragraph-level question-answer pairs from wikipedia. Du X and Cardie C. arXiv, 2018. paper code&dataset

Teaching Machines to Ask Questions. Kaichun Yao, Libo Zhang, Tiejian Luo, et al. IJCAI, 2018. paper

Question Generation using a Scratchpad Encoder. Benmalek R Y, Khabsa M, Desu S, et al. 2018. paper

Learning to collaborate for question answering and asking. Tang D, Duan N, Yan Z, et al. NAACL, 2018. paper

A Question Type Driven Framework to Diversify Visual Question Generation Zhihao Fan, Zhongyu Wei, Piji Li, et al. IJCAI,2018. paper

Neural Generation of Diverse Questions using Answer Focus, Contextual and Linguistic Features. Harrison V, Walker M. arXiv,2018. paper

Learning to Ask: Neural Question Generation for Reading Comprehension. Xinya Du, Junru Shao, Claire Cardie. ACL, 2017. paper code

Neural question generation from text: A preliminary study. Zhou Q, Yang N, Wei F, et al. NLPCC, 2017. paper

Question answering and question generation as dual tasks. Tang D, Duan N, Qin T, et al. arXiv, 2017. paper

Creativity: Generating diverse questions using variational autoencoders. Jain U, Zhang Z, Schwing A G. IEEE,2017. paper

A joint model for question answering and question generation. Wang, Tong, Xingdi Yuan, and Adam Trischler. arXiv, 2017. paper

Neural models for key phrase detection and question generation. Subramanian S, Wang T, Yuan X, et al. arXiv, 2017. paper

Machine comprehension by text-to-text neural question generation. Yuan X, Wang T, Gulcehre C, et al. arXiv, 2017. paper

Question generation for question answering. Duan N, Tang D, Chen P, et al. EMNLP,2017. paper

Ranking automatically generated questions using common human queries. Chali Y, Golestanirad S. INLG, 2016. paper

Generating Factoid Questions With Recurrent Neural Networks: The 30M Factoid Question-Answer Corpus. Serban I V, García-Durán A, Gulcehre C, et al. arXiv, 2016. paper dataset

Towards Topic-to-Question Generation. XYllias Chali, Sadid A. Hasan. Computational Linguistics, 2015. paper

Literature review of automatic question generation systems. Rakangor, Sheetal, and Y. Ghodasara. International Journal of Scientific and Research Publications,2015. paper

Revup: Automatic gap-fill question generation from educational texts. Kumar G, Banchs R and D'Haro L F. ACL, 2015. paper

Deep questions without deep understanding. Labutov I, Basu S and Vanderwende L. ACL, 2015. paper

Ontology-based multiple choice question generation. Al-Yahya, Maha. The Scientific World Journal, 2014. paper

Linguistic considerations in automatic question generation. Mazidi, Karen, and Rodney D. Nielsen. ACL, 2014. paper

Automatic question generation for educational applications–the state of art. Le, Nguyen-Thinh, Tomoko Kojiri, and Niels Pinkwart. ACMKE, 2014. paper

**2008-2013**

Generating natural language questions to support learning on-line. Lindberg D, Popowich F, Nesbit J, et al. ENLG, 2013. paper

Question generation for French: collating parsers and paraphrasing questions. Bernhard, Delphine, et al. Dialogue & Discourse,2012. paper dataset1 dataset2

Question generation from concept maps. Olney A M, Graesser A C, Person N K. Dialogue & Discourse, 2012. paper

Towards automatic topical question generation. Chali, Yllias, and Sadid A. Hasan. COLING,2012. paper dataset

Question generation based on lexico-syntactic patterns learned from the web. Curto, Sérgio, Ana Cristina Mendes, and Luisa Coheur. Dialogue & Discourse,2012. paper

G-Asks: An intelligent automatic question generation system for academic writing support. Liu, Ming, Rafael A. Calvo, and Vasile Rus. Dialogue & Discourse, 2012. paper

Semantics-based question generation and implementation. Yao, Xuchen, Gosse Bouma, and Yi Zhang. Dialogue & Discourse,2012. paper system dataset1 dataset2 dataset3 dataset4

Mind the gap: learning to choose gaps for question generation. Becker, Lee, Sumit Basu, and Lucy Vanderwende. NAACL,2012. paper dataset

OntoQue: a question generation engine for educational assesment based on domain ontologies. Al-Yahya, Maha. IEEE, 2011. paper

Automatic gap-fill question generation from text books. Agarwal M, Mannem P. the 6th Workshop on Innovative Use of NLP for Building Educational Applications,2011. paper

Automatic question generation using discourse cues. Agarwal, Manish, Rakshit Shah, and Prashanth Mannem. the 6th Workshop on Innovative Use of NLP for Building Educational Applications,2011. paper

Automatic factual question generation from text. Heilman, Michael. Language Technologies Institute School of Computer Science Carnegie Mellon University 2011. paper

Question generation and answering. Linnebank, Floris, Jochem Liem, and Bert Bredeweg. DynaLearn, EC FP7 STREP project,2010. paper

Question generation from paragraphs at UPenn: QGSTEC system description. Mannem, Prashanth, Rashmi Prasad, and Aravind Joshi. QG2010: The Third Workshop on Question Generation,2010. paper

Question generation with minimal recursion semantics. Yao, Xuchen, and Yi Zhang. QG2010: The Third Workshop on Question Generation. 2010. paper

Natural language question generation using syntax and keywords. Kalady S, Elikkottil A, Das R. QG2010: The Third Workshop on Question Generation, 2010. paper

Automatic question generation for literature review writing support. Liu, Ming, Rafael A. Calvo, and Vasile Rus. International Conference on Intelligent Tutoring Systems,2010. paper

Overview of the first question generation shared task evaluation challenge. Rus, Vasile, et al. the Third Workshop on Question Generation, 2010. paper

Question generation in the CODA project. Piwek, Paul, and Svetlana Stoyanchev. no conference, 2010. paper

The first question generation shared task evaluation challenge. Rus V, Wyse B, Piwek P, et al. INLG, 2010. paper

Extracting simplified statements for factual question generation. Heilman, Michael, and Noah A. Smith. QG2010: The Third Workshop on Question Generation, 2010. paper system

Good Question! Statistical Ranking for Question Generation. Heilman, Michael and Smith, Noah A. ACL, 2010.paper dataset1 dataset2

Automation of question generation from sentences. Ali, H., Chali, Y., Hasan, S. A. QG2010: The Third Workshop on Question Generation 2010. paper

Question Generation via Overgenerating Transformations and Ranking. Michael Heilman, Noah A. Smith. CARNEGIE-MELLON UNIV PITTSBURGH PA LANGUAGE TECHNOLOGIES INST, 2009. paper

Automatic question generation and answer judging: a q&a game for language learning. Yushi Xu, Anna Goldie, Stephanie Seneff. SLaTE, 2009. paper

**评测**

Unifying Human and Statistical Evaluation for Natural Language Generation. Tatsunori B. Hashimoto, Hugh Zhang, Percy Liang. NAACL, 2019. paper code

Evaluating Rewards for Question Generation Models. Hosking T, Riedel S. arXiv, 2019. paper

The price of debiasing automatic metrics in natural language evaluation. Arun Tejasvi Chaganty, Stephen Mussmann, Percy Liang arXiv, 2018. paper code

BLEU: a Method for Automatic Evaluation of Machine Translation. Kishore Papineni, Salim Roukos, Todd Ward, Wei-Jing Zhu. ACL, 2002. paper

Evaluating question answering over linked data. Lopez V, Unger C, Cimiano P, et al. WWW, 2013. paper

The Meteor metric for automatic evaluation of machine translation. Lavie A, Denkowski M J. Machine translation, 2009. paper

Rouge: A package for automatic evaluation of summaries. Lin, Chin-Yew. Text Summarization Branches Out, 2004. paper

**数据集**

Program induction by rationale generation: Learning to solve and explain algebraic word problems. Ling W, Yogatama D, Dyer C, et al. arXiv, 2017. paper code

On Generating Characteristic-rich Question Sets for QA Evaluation. Su Y, Sun H, Sadler B, et al. EMNLP, 2016. paper code

Squad: 100,000+ questions for machine comprehension of text. Rajpurkar P, Zhang J, Lopyrev K, et al. arXiv, 2016. paper dataset

Who did what: A large-scale person-centered cloze dataset Onishi T, Wang H, Bansal M, et al. arXiv, 2016. paper dataset

Teaching machines to read and comprehend Hermann K M, Kocisky T, Grefenstette E, et al. NIPS, 2015. paper code

Mctest: A challenge dataset for the open-domain machine comprehension of text. Richardson M, Burges C J C, and Renshaw E. EMNLP, 2013. paper dataset

The Value of Semantic Parse Labeling for Knowledge Base Question Answering. Yih W, Richardson M, Meek C, et al. ACL, 2016. paper dataset

Semantic Parsing on Freebase from Question-Answer Pairs. Berant J, Chou A, Frostig R, et al. EMNLP, 2013. paper