# coding: utf-8
'''
     Created at 2018.4.9
     Author：baby_qian
     Filename：create_new_sentiment_keywords.py
     Version: V1.0
     
     Function： Create new sentiment dictionary
'''

import os
import sys 
import yaml



def get_txt_data(filepath, encodings='utf-8'):
    '''
        读取txt文件
    '''
    txt_file = open(filepath, 'r', encoding=encodings)
    txt_tmp1 = txt_file.readlines()
    txt_tmp2 = ''.join(txt_tmp1)
    txt_data = txt_tmp2.split('\n')
    txt_data_temp = []
    for line in txt_data:
        if line != '' or ' ':
            txt_data_temp.append(line.strip('\ufeff\n'))
    txt_file.close()
    return txt_data_temp

    
def load_sentiment_dict():
    '''
        导入搜集的各个平台的情感词典
    '''
    
    file_path = os.path.dirname(os.path.abspath(__file__))
    # 载入百度的情感词典
    baidu_pos_dict = get_txt_data(os.path.join(file_path, 'sentiment_keywords', 'posdict.txt'))
    baidu_neg_dict = get_txt_data(os.path.join(file_path, 'sentiment_keywords', 'negdict.txt'))
    
    # 载入知网的情感词典
    hownet_p1 = get_txt_data(os.path.join(file_path, 'sentiment_keywords', '正面情感词语.txt'), encodings='gbk')
    hownet_p2 = get_txt_data(os.path.join(file_path, 'sentiment_keywords', '正面评价词语.txt'), encodings='gbk')
    hownet_n1 = get_txt_data(os.path.join(file_path, 'sentiment_keywords', '负面情感词语.txt'), encodings='gbk')
    hownet_n2 = get_txt_data(os.path.join(file_path, 'sentiment_keywords', '负面评价词语.txt'), encodings='gbk')
    
    # 载入台湾大学的情感词典
    ntusd_pos = get_txt_data(os.path.join(file_path, 'sentiment_keywords', 'ntusd-positive.txt'))
    ntusd_neg = get_txt_data(os.path.join(file_path, 'sentiment_keywords', 'ntusd-negative.txt'))
    
    # 合并成正负情感两类词典，并赋予权值1
    pos = baidu_pos_dict + hownet_p1 + hownet_p2 + ntusd_pos
    neg = baidu_neg_dict + hownet_n1 + hownet_n2 + ntusd_neg

    # 导入搜集的一个较完整的情感词典
    with open(os.path.join(file_path, 'sentiment_keywords', 'sentiment_keywords.yaml'), 'r', encoding='utf-8') as f:
        _sentiment_dict = yaml.load(f)
    
    # 保存为字典结构
    sentiment_dict = {}
    sentiment_dict['positive'] = {key: 1 for key in pos if key != ''}
    sentiment_dict['negative'] = {key: -1 for key in neg if key != ''}

    sentiment_dict['positive'].update(_sentiment_dict['positive'])
    sentiment_dict['negative'].update(_sentiment_dict['negative'])
    
    # 保存输出到output文件夹
    with open(os.path.join(file_path, 'output', 'new_sentiment_keywords.yaml'), 'w') as f:
        yaml.dump(sentiment_dict, f)
    print('The number of the positive sentiment words is :', len(sentiment_dict['positive']))
    print('The number of the negative sentiment words is :', len(sentiment_dict['negative']))
    
    return sentiment_dict
    


def load_inverse_dict():
    '''
        导入搜集的否定词，并组合在一起.
    '''
    
    file_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(file_path, 'sentiment_inverse_keywords', 'inverse_keywords.yaml'), 'r', encoding='utf-8') as f:
        inverse_dict = yaml.load(f)
    
    # 合并搜集到的否定词（否定词权值设置为 1）
    inverse_list = get_txt_data(os.path.join(file_path, 'sentiment_inverse_keywords', 'inverse.txt'))
    inverse_dict.update({key: 1.0 for key in inverse_list if key != ''})
    
    # 保存合并后的否定词字典
    with open(os.path.join(file_path, 'output', 'new_inverse_keywords.yaml'), 'w') as f:
        yaml.dump(inverse_dict, f)
    print('The number of the inverse dict length is:', len(inverse_dict))
    
    return inverse_dict


def combine_degree_dicts():
    '''
        合并多个程度副词字典
    '''
    
    #导入搜集的程度副词词典
    file_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(file_path, 'sentiment_degree_keywords', 'degree_keywrods.yaml'), 'r', encoding='utf-8') as ff:
        degree_dict = yaml.load(ff)
    
    most_degree = get_txt_data(os.path.join(file_path, 'sentiment_degree_keywords', 'most.txt'), encodings='utf-8')
    very_degree = get_txt_data(os.path.join(file_path, 'sentiment_degree_keywords', 'very.txt'), encodings='utf-8')
    more_degree = get_txt_data(os.path.join(file_path, 'sentiment_degree_keywords', 'more.txt'), encodings='utf-8')
    ish_degree = get_txt_data(os.path.join(file_path, 'sentiment_degree_keywords', 'ish.txt'), encodings='utf-8')
    insuf_degree = get_txt_data(os.path.join(file_path, 'sentiment_degree_keywords', 'insufficiently.txt'), encodings='utf-8')
    
    # 合并多个程度副词表为字典
    # 权值设置为：most：2.0  very:  1.5 more: 1.25  ish: 0.5  insuf: 0.25
    degree_dict.update({key: 2.0 for key in most_degree if key != ''})
    degree_dict.update({key: 1.5 for key in very_degree if key != ''})
    degree_dict.update({key: 1.25 for key in more_degree if key != ''})
    degree_dict.update({key: 0.5 for key in ish_degree if key != ''})
    degree_dict.update({key: 0.25 for key in insuf_degree if key != ''})
    
    # 保存合并后的程度词字典
    with open(os.path.join(file_path, 'output', 'new_degree_keywords.yaml'), 'w') as f:
        yaml.dump(degree_dict, f)
    print('The number of the degree dict length is:', len(degree_dict))
    
    return degree_dict
    


def combine_stopwords_dict():
    '''
        合并从多个平台搜集的多个停用词表，并组合成字典的形式保存.
    '''
    # load all of the stopwords dicts
    file_path = os.path.dirname(os.path.abspath(__file__))    
    stopwords_1 = get_txt_data(os.path.join(file_path, 'sentiment_stopwords', 'stopword.txt'))
    stopwords_2 = get_txt_data(os.path.join(file_path, 'sentiment_stopwords', '百度停用词列表.txt'), encodings='gbk')
    stopwords_3 = get_txt_data(os.path.join(file_path, 'sentiment_stopwords', '哈工大停用词表.txt'), encodings='gbk')
    stopwords_4 = get_txt_data(os.path.join(file_path, 'sentiment_stopwords', '四川大学机器智能实验室停用词库.txt'), encodings='gbk')
    stopwords_5 = get_txt_data(os.path.join(file_path, 'sentiment_stopwords', '中文停用词库.txt'), encodings='gbk')
    
    # 合并停用词表到一个字典中(停用词的权值全部设置为 0.0)
    stopwords_dict = {}
    stopwords_dict.update({key: 0.0 for key in stopwords_1 if key != ''})
    stopwords_dict.update({key: 0.0 for key in stopwords_2 if key != ''})
    stopwords_dict.update({key: 0.0 for key in stopwords_3 if key != ''})
    stopwords_dict.update({key: 0.0 for key in stopwords_4 if key != ''})
    stopwords_dict.update({key: 0.0 for key in stopwords_5 if key != ''})
    
    # 保存合并后的停用词字典
    with open(os.path.join(file_path, 'output', 'new_stop_words.yaml'), 'w') as f:
        yaml.dump(stopwords_dict, f)
    print('The stopwords dict length is:', len(stopwords_dict))
    
    return stopwords_dict
    
def main_create_new_sentiment_dict():
    '''
        提供一个生成新的情感词典的主函数.
    '''
    
    sentiment_dict = load_sentiment_dict()
    inverse_dict = load_inverse_dict()
    degree_dict = combine_degree_dicts()
    stopwords_dict = combine_stopwords_dict()
    
    return sentiment_dict, inverse_dict, degree_dict, stopwords_dict
    

if __name__ == '__main__':
    
    sentiment_dict, inverse_dict, degree_dict, stopwords_dict = main_create_new_sentiment_dict()

