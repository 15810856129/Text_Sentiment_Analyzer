# coding: utf-8
'''
     Created at 2018.4.9
     Author：baby_qian
     Filename：Sentiment_predict_by_new_rules.py
     Version: V1.0
     
     Function： 使用基于情感词统计规则的方法进行文本情感分析.
'''

import os
import jieba
import yaml


def _load_user_library(file):
    '''
        Load user dictionary to increse segmentation accuracy
    '''
    
    if isinstance(file, str):
        jieba.load_userdict(file)
    elif isinstance(file, list):
        for value in file:
            jieba.add_word(value.lower())
    else:
        raise KeyError("input type is illegal" )
        
    
def _load_sentiment_dict():
    '''
        导入准备好的情感词典
    '''
    
    file_path = os.path.dirname(os.path.abspath(__file__))
       
    with open(os.path.join(file_path, 'output', 'new_sentiment_keywords.yaml'), 'r', encoding='utf-8') as f1:
        sentiment_dict = yaml.load(f1)
    with open(os.path.join(file_path, 'output', 'new_inverse_keywords.yaml'), 'r', encoding='utf-8') as f2:
        inverse_dict = yaml.load(f2)
    with open(os.path.join(file_path, 'output', 'new_degree_keywords.yaml'), 'r', encoding='utf-8') as f3:
        degree_dict = yaml.load(f3)    
    with open(os.path.join(file_path, 'output', 'new_stop_words.yaml'), 'r', encoding='utf-8') as f4:
        stop_dict = yaml.load(f4)
    
    return sentiment_dict, inverse_dict, degree_dict, stop_dict



def sentence_cut(sentence):
    '''
        对一整句话按标点符号先进行切分.
    '''
    
    start = 0
    i = 0   # i is the position of words
    token = 'meaningless'
    sents = []
    punt_list = ',.!?;~，。！？；～… '
    for word in sentence:
        if word not in punt_list:
            i += 1
            token = list(sentence[start:i+2]).pop()
        elif word in punt_list and token in punt_list:
            i += 1
            token = list(sentence[start:i+2]).pop()
        else:
            sents.append(sentence[start:i+1])
            start = i+1
            i += 1
    if start < len(sentence):
        sents.append(sentence[start:])
        
    return sents


    
def segmentation(sentence, para='str'):
    '''
        使用结巴分词工具切分一段话.
    '''
    
    if para == 'str':
        seg_list = jieba.lcut(sentence)
        return seg_list
    elif para == 'list':
        seg_list = []
        for line in sentence:
            for value in jieba.lcut(line):
                seg_list.append(value)
        return seg_list



def senstence_filter_stopwords(word_list, stop_dict):
    '''
        分词后过滤掉停用词.
    '''
    
    clean_word_list = []
    for word in word_list:
        if word in stop_dict:
            continue
        else:
            clean_word_list.append(word.strip())
            
    return clean_word_list
    


def transform_to_positive_num(poscount, negcount):
    '''
        Function of transforming negative score to positive score
        first and second position represented positive and negative socre, respectively.
        Example: [5, -2] →  [7, 0]; [-4, 8] →  [0, 12]
    '''
    
    pos_count = 0
    neg_count = 0
    if poscount >= 0 and negcount < 0:
        pos_count = poscount - negcount
        neg_count = 0
    elif poscount < 0 and negcount >= 0:
        pos_count = 0
        neg_count = negcount - poscount
    elif poscount < 0 and negcount < 0:
        pos_count = -poscount
        neg_count = -negcount
    else:
        pos_count = poscount
        neg_count = negcount
        
    return [pos_count, neg_count]




def match(word, sentiment_value, degree_dict, inverse_dict):
    '''
        compute single sentence sentiment score
        compute positive and negative scores for each sentence, expressed as [poscount, negcount]
    '''
    
    if word in degree_dict:
        sentiment_value *= degree_dict[word]
    elif word in inverse_dict:
        sentiment_value *= -1 * inverse_dict[word]
    
    return sentiment_value

    
    
def single_sentence_sentiment_score(review, sentiment_dict, stop_dict, degree_dict, inverse_dict):
    '''
        posdict and negdict are sentiment positive dict and negative dict, respectively
    '''
    
    single_review_senti_score = []
    cuted_review = sentence_cut(review)

    for sent in cuted_review:
        sentence_list = segmentation(sent)
        clean_word_list = senstence_filter_stopwords(sentence_list, stop_dict)
        
        single_sentence_score = []
        i = 0   # word position
        s = 0
        poscount = 0   # count a positive word
        negcount = 0   # count a negative word

        for word in clean_word_list:
            if word in sentiment_dict['positive']:
                poscount += 1
                # backwards search
                for w in clean_word_list[s:i]:
                    poscount = match(w, poscount, degree_dict, inverse_dict)
                a = i + 1

            elif word in sentiment_dict['negative']:
                negcount += 1
                # backwards search
                for w in clean_word_list[s:i]:
                    negcount = match(w, negcount, degree_dict, inverse_dict)
                a = i + 1

            # Match "!" in the review, every "!" has a weight of +2
            elif word == '!' or word == '！':
                for w in clean_word_list[::-1]:
                    if w in sentiment_dict['positive']:
                        poscount += 2
                        break
                    elif w in sentiment_dict['negative']:
                        negcount += 2
                        break
            i += 1

        single_sentence_score.append(transform_to_positive_num(poscount=poscount, negcount=negcount))
        single_review_senti_score.append(single_sentence_score)
        
    return single_review_senti_score
    

    
def main_predict_by_rules(review):
    '''
        主程序入口
    '''
    
    sentiment_dict, inverse_dict, degree_dict, stop_dict = _load_sentiment_dict()
    score = single_sentence_sentiment_score(review, sentiment_dict, inverse_dict, degree_dict, stop_dict)
    avg_score = sum([value for lines in score for line in lines for value in line]) / len(score)
    
    return score, avg_score
    
         
        
if __name__ == '__main__':
    
    review = '文学创造正式以这样的属性，在向人们展现真理的同时，也向人们呈示着意义，并以审美情感诉诸人们的心灵和激发人们的情绪的方式，发挥着它的审美意识形态作用情感评价”与“诗意的裁判”含义相通。'
    score, avg_score = main_predict_by_rules(review)

    

