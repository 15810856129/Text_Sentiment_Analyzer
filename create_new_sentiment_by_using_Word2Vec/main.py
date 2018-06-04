# -*- coding: utf-8 -*-
import os
from text_preprocess import process_main
from word2vec import Word2Vec_Model
from word2vec import load_file_and_filter
from word2vec import compute_intensity
import pickle



def main(k_min=0.4, k_mid=0.5):
    
    # 文本预处理
    clean_word_list, clean_word, candidate_sentiment_words, sentiment_keywords, seed_pos, seed_neg = process_main()
     
    # 训练 word2vec 模型
    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'w2v_model')
    word2vec = Word2Vec_Model()
    model = word2vec.train(clean_word_list, model_save_path)
    
    # 获得单词的向量化表示
    hownet_pos_words, hownet_neg_words = load_file_and_filter(clean_word)
    seed_vector_pos = model[seed_pos]
    seed_vector_neg = model[seed_neg]
    
    # 以知网的情感词典作为测试集，通过计算测试集单词与人工挑选的种子情感词的相似度来判别测试集的情感极性
    hownet_vector = model[hownet_pos_words + hownet_neg_words]
      
    # compute the similarity between the hownet words and seed words
    similarity_hownet = word2vec.normalized(word2vec.cos_similarity(hownet_vector, seed_vector_pos, seed_vector_neg)[0])
    
    # compute the classification report
    pred = word2vec.predict(similarity_hownet, k_min=k_min, k_mid=k_mid)
    label = [1 for i in range(len(hownet_pos_words))] + [0 for i in range(len(hownet_neg_words))]
    
    # validation result
    word2vec.validation(pred, label)
    
    # show the test result which word has been predicted wrong
    word2vec.show_result(predict=pred, hownet_neg_words=hownet_neg_words, hownet_pos_words=hownet_pos_words)
    
    # 计算候选情感词与种子情感词的余弦相似度
    candidate_sentiment_words = [line for line in candidate_sentiment_words if line in model.wv.vocab]
    candidate_vector = model[candidate_sentiment_words] 
    candidate_absolute, candidate_pos_avg, candidate_neg_avg = word2vec.cos_similarity(candidate_vector, seed_vector_pos, seed_vector_neg) 
    candidate_intensity = compute_intensity(candidate_sentiment_words, candidate_pos_avg, candidate_neg_avg)
    
    # 预测候选词的情感极性
    predict = word2vec.predict(candidate_absolute, k_min=k_min, k_mid=k_mid)
    
    # 得到新的情感词及其情感强度
    new_sentiment_dict = {}
    new_sentiment_dict['positive'] = {key: value for i, (key, value) in enumerate(zip(candidate_sentiment_words, candidate_intensity)) if predict[i] == 1}
    new_sentiment_dict['negative'] = {key: value for i, (key, value) in enumerate(zip(candidate_sentiment_words, candidate_intensity)) if predict[i] == 0}
  
    # 合并原有的情感词典和新扩充的词典
    all_senitment_dict = {}
    all_senitment_dict['positive'] = dict(sentiment_keywords['positive'], **new_sentiment_dict['positive'])
    all_senitment_dict['negative'] = dict(sentiment_keywords['negative'], **new_sentiment_dict['negative'])
    
    # 保存合并后的情感词典
    file_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(file_path, 'output', 'all_sentiment_dict.pkl'), 'wb') as f:
        pickle.dump(all_senitment_dict, f)

    return all_senitment_dict, new_sentiment_dict, sentiment_keywords, pred
        
    

if __name__ == '__main__':
    
    all_senitment_dict, new_sentiment_dict, sentiment_keywords, pred = main()