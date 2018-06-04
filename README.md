# Text_Sentiment_Analyzer
Self-learning record of text sentiment analysis

Environment：python3.5 + windows10

1、create_new_sentiment_dict_and_predict_by_using_rules 项目                                                                               

搜集了知网、百度、搜狗、台湾大学等多个平台的情感词典，对它们进行合并去重后得到新的较为完整的情感词典集合；
在处理文本问题上，首先对文本评论进行分词、去停用词等预处理，其次采用基于规则的方法（统计文本中情感词数及否定词和程度副词情况），对文本评论进行情感分析。

2、create_new_sentiment_dict_by_using_LDA 项目

以微博评论语料为基础，采用LDA主题模型的方法来寻找和情感种子词语义相近的词作为新的情感词，其中情感种子词是人工从知网情感词表里挑选的情感强度较明显的单词（包含正负向情感词）。

3、create_new_sentiment_by_using_Word2Vec 项目

对爬虫获取的25000条微博评论进行文本预处理，提取候选情感词（使用TF/IDF方法）；并从知网情感词典中人工挑选出正负向种子情感词，以及用于后期测试的情感词。利用Word2Vec模型对语料进行词向量训练，结合词向量之间的余弦距离来计算词之间的语义相似度，从而对候选情感词进行情感极性判断；并以候选词与种子词之间的相似度来量化候选词的情感强度。
