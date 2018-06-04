# -*- coding: utf-8 -*-
'''
    Function: 对干净的微博语料进行词向量训练，然后依据从语料中选取的情感种子词，计算与其语义相近的词，
              扩充为新的情感词典；并选取知网的部分情感词进行测试。
              
    （1）装载所需的语料；
    （2）制作好每部分需要的词典；
    （3）训练Word2Vec模型；
    （4）利用余弦相似度计算得到和情感种子词语义相近的词；
    （5）以选取的种子词为基准（正负向情感种子词强度设置为±1），计算候选情感词的情感强度；
    （6）人工选取知网的部分情感词作为测试，评估性能.
'''
import gensim
import os
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from text_preprocess import process_main


def load_file_and_filter(clean_word):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    try:
        with open(file_path + '/hownet_pos.txt', 'r') as f:
            hownet_pos_words = list(set([line.strip(' \n') for line in f.readlines() if line.strip(' \n') in clean_word]))
        
        with open(file_path + '/hownet_neg.txt', 'r') as ff:
            hownet_neg_words = list(set([line.strip(' \n') for line in ff.readlines() if line.strip(' \n') in clean_word]))    
        
        return hownet_pos_words, hownet_neg_words
        
    except:
        raise TypeError('the open file type is illegal.')



class Word2Vec_Model(object):
    
    def train(self, word_list, model_save_path, size=100, window=5, min_count=1, sg=0, alpha=0.1):
        '''
            create Word2Vec model
        '''
        
        model = gensim.models.Word2Vec(word_list, alpha=alpha, size=size, window=window, min_count=min_count, sg=sg)
        
        # save the word2vec model 
        model.save(model_save_path)
        
        return model
    
    
    
    def cos_similarity(self, array, seed_vector_pos, seed_vector_neg):
        '''
            compute the cosin similarity between a word vector and seed words vectors
        '''
        # seed_vector is m x n, while m is the number of seed, n is the seed vector dimension expressed by word2vec.
    
        n1, _ = seed_vector_pos.shape
        n2, _ = seed_vector_neg.shape
        n = min(n1, n2)
        print(n)
        
        A = array
        m, _ = A.shape
        result = []
        pos_avg = []
        neg_avg = []

        for j in range(m):

            distance_pos = []
            distance_neg = []

            for i in range(n):
                temp = np.dot(A[j,:], seed_vector_pos[i,:])
                tmp = np.linalg.norm(A[j,:]) * np.linalg.norm(seed_vector_pos[i,:])

                temp_1 = np.dot(A[j,:], seed_vector_neg[i,:])
                tmp_1 = np.linalg.norm(A[j,:]) * np.linalg.norm(seed_vector_neg[i,:])

                distance_pos.append(temp / tmp)
                distance_neg.append(temp_1 / tmp_1)

            try:
                result.append(sum(distance_pos) - sum(distance_neg))
                pos_avg.append(sum(distance_pos) / len(distance_pos))
                neg_avg.append(sum(distance_neg) / len(distance_neg))
            
            except:
                raise ZeroDivisionError('The denominator in division is 0.')

        return [np.array(result), pos_avg, neg_avg ]

    
    
    # normalized a numpy array 
    def normalized(self, nums):
        if len(nums) <= 1:
            raise IOError('please input correct nums !')
        else:
            max_ = nums.max()
            min_ = nums.min()
            
            try:
                return (nums - min_) / (max_ - min_)
            except:
                raise ZeroDivisionError('The denominator in division is 0.')
    
    
    # select a threshold to judge the hownet words belong to positive and negative sentiment or others
    def threshold_map(self, nums, k_min=0.3, k_mid=0.6):
        # 0, 1, 2分别表示负、正、中向情感
        
        result = []
        for i in range(len(nums)):
            if nums[i] <= k_min:
                result.append(0)
            elif k_min < nums[i] <= k_mid:
                result.append(2)
            else:
                result.append(1)

        return result
    
    
    # predict the words label
    def predict(self, similarity_word, k_min=0.3, k_mid=0.6):
        
        pred = self.threshold_map(similarity_word, k_min=k_min, k_mid=k_mid)

        print('the number of predict positive is :', pred.count(1))
        print('the number of predict negative is :', pred.count(0))
        
        return pred
    
        
    def validation(self, pred, label):
        
        # compute the classification report
        target_names = ['Positive words', 'Negative words', 'Neutral words']
        print(classification_report(label, pred, target_names=target_names))
        
    
    # show the test result
    def show_result(self, predict, hownet_pos_words, hownet_neg_words):
    
        j = 0
        for i in range(len(predict)):
            sum_words = hownet_pos_words + hownet_neg_words
            if predict[i] == 1.0 and i >= len(hownet_pos_words):
                j = j + 1
                print('The negative predict to positive words:', sum_words[i])
            elif predict[i] == 0.0 and i < len(hownet_pos_words):
                j = j + 1
                print('The positive predict to negative words:', sum_words[i])
            elif predict[i] == 2.0:
                j = j + 1
                print('The neural words:', sum_words[i])
            elif predict[i] == 1.0 and i < len(hownet_pos_words) or predict[i] == 0.0 and i >= len(hownet_pos_words):
                continue
            else:
                raise IOError('the predicted value have an illegal value.')

        print('The number of incorrect prdedict words:', j)
        
        
    
    # observe some words and their latest words
    def find_similarity_word(self, strings, model):
        for value in strings:
            temp = model.similar_by_word(value)
            print(value)
            print(temp)

            
            
def compute_intensity(candidate_sentiment_words, nums1, nums2):
    '''
        计算单词的情感强度值
    '''
    
    # nums1 and nums2 are a list of same length
    result = []
    for i in range(len(nums1)):
        temp = 0
        if nums1[i] > 0 and nums2[i] >= 0:
            temp = max(nums1[i], nums2[i])
            if nums1[i] >= nums2[i]:
                result.append(temp)
            else:
                result.append(-temp)
                
        elif nums1[i] <= 0 and nums2[i] < 0:
            temp = max(nums1[i], nums2[i])
            if nums1[i] >= nums2[i]:
                result.append(abs(temp))
            else:
                result.append(temp)
            
        elif nums1[i] <= 0 and nums2[i] >= 0:
            temp = max(abs(nums1[i]), abs(nums2[i]))
            result.append(-temp)
            
        elif nums1[i] > 0 and nums2[i] < 0:
            temp = max(abs(nums1[i]), abs(nums2[i]))
            result.append(temp)
        else:
            raise ValueError('the result have an illegal value.')
    
    return result