# coding: utf-8
import jieba
import jieba.analyse
import yaml
import os
import pickle
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


'''
  Function:
 （1）首先，提取每个用户的中文评论；
 （2）其次，用结巴分词工具对中文评论语料进行分词；
 （3）然后，去除分词结果中的停用词；
 （4）接着，按单词出现频次来提取干净语料中的关键词，并过滤掉已经在基础情感词表中的关键词，再作为候选情感词.
 （5）最后，人工挑选情感强度较明显的正负种子情感词组成各自集合，并滤除种子词集中未出现在语料中的词。
'''


def load_yaml_file():
    '''
        as a tool to load yaml file 
    '''
    
    file_path = os.path.dirname(os.path.abspath(__file__)) 
    with open(file_path + '/conf' + '/stop_words.yaml', 'r') as f1:
        stopwords = list(set(yaml.load(f1)))
    
    with open(file_path + '/conf' + '/sentiment_keywords.yaml', 'r') as f2: 
        sentiment_keywords = yaml.load(f2)
    
    with open(file_path + '/conf' + '/seed_sentiment_words.yaml', 'r') as f3:
        seed_dict = yaml.load(f3)
    
    return stopwords, sentiment_keywords, seed_dict

    

def load_user_library(file):
    '''
        Load user dictionary to increase segmentation accuracy
    '''
    
    if isinstance(file, str):
        jieba.load_userdict(file)
    elif isinstance(file, list):
        for value in file:
            jieba.add_word(value.lower())
    else:
        raise IOError('input file format is illegal.')
    
    

def cut_sentence(file_path):
    '''
        get user's comment
    '''
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            txt_temp = f.readlines()
            txt_data = []
            for line in txt_temp:
                txt_data.append(line.split('\t')[2])
        
        return txt_data
        
    except:
        raise TypeError('the file format is illegal.')



def segmentation(sentence, para='list'):
    '''
        use jieba tool to cut sentence
    '''
    
    if para == 'str':
        seg_list = jieba.cut(sentence)
        seg_result = ' '.join(seg_list)
        return seg_result
    
    elif para == 'list':
        seg_result = jieba.lcut(sentence)
        return seg_result
    else:
        raise KeyError('the input type is illegal.')


def sentence_filter_stopwords(word_list, stopwords, Mode=True):
    '''
        filter stop words when the sentence has been cutted and get the clear words
        clean_word_list: [[], [], []], 嵌套列表，每一个内层列表存放了一条经过处理后的评论.
    '''
    
    file_path = os.path.dirname(os.path.abspath(__file__)) 

    # filter stopwords
    clean_word_list = []
    for word in word_list:
        if word in stopwords:
            continue
        else:
            clean_word_list.append(word)      
            
    # 选择是否保存文本预处理后的语料，Mode=False默认不保存，True为保存        
    if Mode == True:
        with open(os.path.join(file_path, 'intermediate_data', 'corpus.pickle'), 'wb') as f:
            pickle.dump(clean_word_list, f, protocol=2)
    elif Mode == False:
        pass
    else:
        raise IOError('the input value is illegal.')

    return clean_word_list	
    

def key_words_extract(comment, clean_word_list, sentiment_keywords, K=2, Mode=False):
    '''
         extract key words as candidate word from clear words to build up the origin sentiment dict
         sentence: [], 列表的每个元素是原始的一条评论.
    '''
    
    file_path = os.path.dirname(os.path.abspath(__file__)) 
        
    # 提取出现频率较高的词作为关键词
    words = []
    for sentence in comment:
        key_words = jieba.analyse.extract_tags(sentence=sentence, topK=K)
        words.append(key_words)
        print('extract word job has done!')
        
    words = [line for lines in words for line in lines]
    
    clean_dict = {key: 0 for line in clean_word_list for key in line}
    
    # 过滤已经出现在基础情感词表中的关键词
    candidate_sentiment_words = []
    for word in words:
        if word in sentiment_keywords['positive'] and word in clean_dict.keys():
            continue
        elif word in sentiment_keywords['negative'] and word in clean_dict.keys():
            continue
        else:
            candidate_sentiment_words.append(word)
            
    # 选择是否保存文本预处理后的关键字，Mode=False默认不保存，True为保存 
    if Mode == True:
        with open(os.path.join(file_path, 'intermediate_data', 'candidate_sentiment_words.pickle'), 'wb') as f:
            pickle.dump(candidate_sentiment_words, f)
    elif Mode == False:
        pass 
    
    else:
        raise IOError('the input value is illegal.')
       
    return candidate_sentiment_words



def seed_sentiment_word_extract(clean_word_list, seed_dict):
    '''
        从基础情感词表中筛选在语料中出现过的情感词，以此得到正负情感种子词集合.
    '''
    
    # 将嵌套列表[[], []]整理成单层列表，方便获取所有的干净词
    clean_word = [line for lines in clean_word_list for line in lines]

    file_path = os.path.dirname(os.path.abspath(__file__))
        
    # filter seed sentiment words in clean words
    seed_pos = [v for v in seed_dict['seed_positive'] if v in clean_word]
    seed_neg = [u for u in seed_dict['seed_negative'] if u in clean_word]   

    return seed_pos, seed_neg
    
  
def plot_normal_curve(nums):
    data = np.array(nums)
    mu = np.mean(data)
    sigma = np.std(data, ddof=1)
    num_bins = 200
    # plot the hist image
    n, bins, patches = plt.hist(data, bins=num_bins, normed=True, facecolor = 'k', alpha = 0.5)
    
    # plot the distribution curve
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--')
    plt.xlabel('Expectation')
    plt.ylabel('Probability')
    plt.title('histogram of normal distribution: $\mu = 0$, $\sigma=1$')

    plt.subplots_adjust(left = 0.15)
#     axes = plt.gca()  
#     axes.set_xlim([-100,300])  
    #axes.set_ylim([0,1]) 
    plt.axvline(mu)
    plt.show() 
    
    
    
def process_main():
    '''
        输入一批文本评论，对其进行批量预处理.
    '''
    
    # 从爬虫抓取的数据中切分出文本评论
    comment = cut_sentence(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'weibo_media_comment'))
    
    # use jieba tool to cut words
    comment_words = list(map(segmentation, comment))
    
    # load yaml files
    stopwords, sentiment_keywords, seed_dict = load_yaml_file()
    
    # filter stopwords
    clean_word_list = list(map(sentence_filter_stopwords, comment_words, len(comment_words) * [stopwords]))
    
    # extract candidate words
    clean_word = [line for lines in clean_word_list for line in lines]
    candidate_sentiment_words = key_words_extract(comment, clean_word_list, sentiment_keywords)
    
    seed_pos, seed_neg = seed_sentiment_word_extract(clean_word_list, seed_dict)
    
    plot_normal_curve([len(i) for i in comment_words])
    
    return clean_word_list, clean_word, candidate_sentiment_words, sentiment_keywords, seed_pos, seed_neg
    
        
        
if __name__ == '__main__':
    
    clean_word_list, clean_word, candidate_sentiment_words, sentiment_keywords, seed_pos, seed_neg = process_main()