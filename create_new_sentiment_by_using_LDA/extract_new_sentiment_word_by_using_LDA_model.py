# coding: utf-8

# Function: 使用LDA主题模型的方法来对一定规模的微博评论进行情感词提取。
#           
# （1）装载所需的语料，将每条微博评论当成一个文档统并存储为矩阵的形式（每行代表一个文档，每列代表文档中的词）；
# （2）文本预处理操作，主要包括分词、去停用词、去标点符号等；
# （3）统计并得到文档的TF-IDF值矩阵；
# （4）设置主题数目K，用TF-IDF矩阵训练LDA模型至收敛，分别得到（文档--主题）以及（主题--词）的概率分布；
# （5）统计得到文档矩阵中所有出现过的词集C，载入知网的正负情感词典并与C取交集作为种子情感词集P和N；
# （6）分别计算非种子词与正负种子词集权重概率的绝对距离，得到两个距离列表D1,D2；
# （7）选取阈值threshold，判断非种子词的情感极性，得到新的情感词，并将其保存为文件导出；
# （8）评估新情感词的质量（评估方法待确定）
# 
# Version: 1.1
# Date: 2018.4.26
# Environment：python3.5 + windows10

import jieba
import jieba.analyse
import yaml
import os
import re
import lda
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import feature_extraction    
from sklearn.feature_extraction.text import TfidfTransformer    
from sklearn.feature_extraction.text import CountVectorizer  


class LDA_Model(object):
     
    def compute_tfidf_matrix(self, corpus, Mode=True):
        '''
            计算词频矩阵
        '''
        if Mode == True:
            # 采用 tf-idf 矩阵
            vectorizer = CountVectorizer()  
            # 统计每个词语的tf-idf权值  
            transformer = TfidfTransformer()  
            #第一个 fit_transform 是计算 tf-idf；第二个 fit_transform 是将文本转为词频矩阵  
            tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))  
            weight = tfidf.toarray()
            
        elif Mode == False:
            # 采用 TF 矩阵
            vectorizer = CountVectorizer()  
            X = vectorizer.fit_transform(corpus)  
            weight = X.toarray()
        else:
            raise KeyError('Mode value is illegal.')
        
        all_words = vectorizer.get_feature_names()
        
        return weight, all_words, vectorizer
        
    
    def train(self, weight, n_topics=100, epochs=500,):
        '''
            建立LDA主题模型，设置主题数为100，训练迭代次数为500.
        '''
        
        model = lda.LDA(n_topics=n_topics, n_iter=epochs, random_state=1)  
        model.fit(np.asarray(weight.astype(np.int32)))
        
        return model
    
    
    def topic_word_distribute(self, model, vectorizer, k=10):
        '''
            主题-单词（topic-word）分布
        '''
        
        topic_word = model.topic_word_  
        print("type(topic_word): {}".format(type(topic_word)))  
        print("shape: {}".format(topic_word.shape))  
        
        # 计算每个主题中的概率最大的前K个单词
        word = vectorizer.get_feature_names()  

        n = k
        for i, topic_dist in enumerate(topic_word):  
            topic_words = np.array(word)[np.argsort(topic_dist)][:-(n+1):-1]  
            print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))
        
        return topic_word
    
    
    def text_topic_distribute(self, model, corpus, k=5):
        '''
            文档-主题（Document-Topic）分布
        '''
        
        doc_topic = model.doc_topic_  

        print('type(doc_topic): {}'.format(type(doc_topic)))
        print('Shape: {}'.format(doc_topic.shape))
        print('\n')

        for i in range(len(corpus)):
            most_topic = np.argsort(-doc_topic[i])[0:k]
            print('doc: {} topic: {}'.format(i, most_topic))
        
        return doc_topic
    
    
    def plot_topic_word_stem_leaf(self, model, n=5):
        '''
            作枝叶图分析, 统计得到构成每个主题的单词的权重分布
        '''
        
        topic_word = model.topic_word_
        f, ax= plt.subplots(n, 1, figsize=(8, 6), sharex=True)  
        
        for i, k in enumerate(range(n)):  
            ax[i].stem(topic_word[k,:], linefmt='r-',  
                       markerfmt='bo', basefmt='w-')  
            ax[i].set_xlim(-10, 10000)  
            ax[i].set_ylim(0, 0.23)  
            ax[i].set_ylabel("Prob")  
            ax[i].set_title("topic {}".format(k))  

        ax[n-1].set_xlabel("word")  
        plt.tight_layout()
        plt.show()
        
    
    def plot_text_topic_stem_leaf(self, model, n=5):
        '''
            作图计算每个文档具体分布在哪个主题
        '''
        
        doc_topic = model.doc_topic_
        ff, axx = plt.subplots(n, 1, figsize=(8, 6), sharex=True)
        
        for i, k in enumerate(range(n)):
            axx[i].stem(doc_topic[k,:], linefmt='r-',
                      markerfmt='bo', basefmt='w-')
            axx[i].set_xlim(-1.0, 20)
            axx[i].set_ylim(0, 1.0)
            axx[i].set_ylabel('Prob')
            axx[i].set_title('Document {}'.format(k))

        axx[n-1].set_xlabel('topic')
        plt.tight_layout()
        plt.show()
    
    
    def get_topic_word_proba_matrix(self, topic_word, pos_seed, neg_word, all_words):
        '''
            建立候选词矩阵和种子情感词矩阵，其中矩阵元素为对应的概率权重
        '''
        
        topic_word_frame = pd.DataFrame(topic_word, columns=all_words)
        candidate_word = [word for word in all_words if word not in pos_seed and word not in neg_seed]

        topic_word_can_matrix = topic_word_frame[candidate_word]
        topic_word_pos_matrix = topic_word_frame[pos_seed]
        topic_word_neg_matrix = topic_word_frame[neg_seed]

        print('shape of topic_word_can_matrix: ', topic_word_can_matrix.shape)
        print('shape of topic_word_pos_matrix: ', topic_word_pos_matrix.shape)
        print('shape of topic_word_neg_matrix: ', topic_word_neg_matrix.shape)

        return topic_word_can_matrix, topic_word_pos_matrix, topic_word_neg_matrix
    
    
    def get_topic_word_proba_matrix(self, topic_word, pos_seed, neg_seed, all_words):
        '''
            获取主题-单词概率权重矩阵
        '''
        
        topic_word_frame = pd.DataFrame(topic_word, columns=all_words)
        candidate_word = [word for word in all_words if word not in pos_seed and word not in neg_seed]

        topic_word_can_matrix = topic_word_frame[candidate_word]
        topic_word_pos_matrix = topic_word_frame[pos_seed]
        topic_word_neg_matrix = topic_word_frame[neg_seed]

        print('shape of topic_word_can_matrix: ', topic_word_can_matrix.shape)
        print('shape of topic_word_pos_matrix: ', topic_word_pos_matrix.shape)
        print('shape of topic_word_neg_matrix: ', topic_word_neg_matrix.shape)

        return topic_word_can_matrix, topic_word_pos_matrix, topic_word_neg_matrix, candidate_word
    

        
def load_txt_file(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as f:
        words = []
        for value in f.readlines():
            words.append(value.strip(' \n'))
    return words
    

def select_seed_sentiment_word(all_words):
                               
        '''
            以语料为基础，从情感词典中挑选种子词.
        '''
        file_path = os.path.dirname(os.path.abspath(__file__))
        
        # load hownet dict
        pos_word = load_txt_file(os.path.join(file_path, 'conf', 'hownet_pos.txt'), encoding='gbk')
        neg_word = load_txt_file(os.path.join(file_path, 'conf', 'hownet_neg.txt'), encoding='gbk')

        # select the sentiment word from the all words by using hownet dict
        pos_seed = [value for value in all_words if value in pos_word and value not in neg_word]
        neg_seed = [value for value in all_words if value in neg_word and value not in pos_seed]

        print('number of positive seed is: ', len(pos_seed))
        print('number of negative seed is: ', len(neg_seed))

        return pos_seed, neg_seed       
        


def compute_similarity(A, B):
    '''
        计算每个非种子情感词与Seed1和Seed2的平均距离，其中采用绝对距离度量;
        并以词汇在某个主题下的概率权重为元素进行计算.
    '''
    
    # A,B are two Dataframe
    m, n = A.shape
    p, q = B.shape

    result = []
    for i in range(n):
        res_temp = None
        for j in range(q):
            temp = A.iloc[:, i] - B.iloc[:, j]
            res_temp = temp.abs().sum() / q
        result.append(res_temp)

    return result



def extract_and_save_new_word(pos_dist, neg_dist, topic_word_neg_matrix, 
                              thresh_pos=0.0000114, thresh_neg=0.0000310):

    '''
        提取新的情感词,选取阈值，对candidate_word是否为情感词进行判断
    '''

    pos = [word for i, word in enumerate(candidate_word) if pos_dist[i] < thresh_pos]
    neg = [word for i, word in enumerate(candidate_word) if neg_dist[i] < thresh_neg]

    print('extracting new positive word number is: ',len(pos))
    print('extracting new negative word number is: ' ,len(neg))
    
    # save the new sentiment word as txt file
    
    new_pos_word = [re.sub('[A-Za-z0-9]', '', value) for value in pos if re.sub('[A-Za-z0-9]', '', value) != '']
    new_neg_word = [re.sub('[A-Za-z0-9]', '', value) for value in neg if re.sub('[A-Za-z0-9]', '', value) != '']
    
    file_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(file_path, 'output', 'new_pos_word.txt'), 'w') as fp:
        fp.write('\n'.join(new_pos_word))

    with open(os.path.join(file_path, 'output', 'new_neg_word.txt'), 'w') as fn:
        fn.write('\n'.join(new_neg_word))
   


if __name__ == '__main__':
    
    # 读取分词后的语料
    file_path = os.path.dirname(os.path.abspath(__file__))
    corpus = load_txt_file(os.path.join(file_path, 'conf', 'corpus.txt'))
    
    # 将文本中的词语转换为词频矩阵；矩阵元素a[i][j] 表示j词在i类文本下的词频  
    LDA_ = LDA_Model()
    weight, all_words, vectorizer = LDA_.compute_tfidf_matrix(corpus)
    
    # 从知网的情感词典中挑选在语料中出现的词作为种子词
    pos_seed, neg_seed = select_seed_sentiment_word(all_words)

    # 构建LDA模型并训练    
    model = LDA_.train(n_topics=100 ,epochs=500, weight=weight)
    
    # 获取主题概率分布矩阵
    topic_word_can_matrix, topic_word_pos_matrix, topic_word_neg_matrix, candidate_word = LDA_.get_topic_word_proba_matrix(topic_word=model.topic_word_, 
                                     all_words=all_words, pos_seed=pos_seed, neg_seed=neg_seed)
    

    # 计算每个非种子情感词与 pos_seed 和 neg_seed 的平均距离
    pos_dist = compute_similarity(topic_word_can_matrix, topic_word_pos_matrix)
    neg_dist = compute_similarity(topic_word_can_matrix, topic_word_neg_matrix)

    
#    # 保存pos_dist和neg_dist，省去每次重复计算的时间
#    with open(os.path.join(file_path, 'output', 'pos_dist.pickle'), 'wb') as f1:
#        pickle.dump(pos_dist, f1)
#    
#    with open(os.path.join(file_path, 'output', 'neg_dist.pickle'), 'wb') as f2:
#        pickle.dump(neg_dist, f2)
        

    # 提取新的情感词
    extract_and_save_new_word(pos_dist, neg_dist, topic_word_neg_matrix)

    # 绘制主题--词汇统计分布图
    LDA_.plot_topic_word_stem_leaf(model=model)

    # 绘制文档--主题统计分布图
    LDA_.plot_text_topic_stem_leaf(model=model)

    # 获取文档--主题概率分布
    doc_topic_distribute_proba_matrix = LDA_.text_topic_distribute(corpus=corpus, model=model)

    # 获取主题--词汇概率分布
    topic_word_distribute_proba_matrix = LDA_.topic_word_distribute(model=model, vectorizer=vectorizer)

