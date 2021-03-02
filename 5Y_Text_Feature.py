# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 19:28:14 2020

@author: ksh
"""

import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from konlpy.tag import Okt ; okt = Okt()
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder ; LE = LabelEncoder()
from keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing import sequence


data_file_path = r'C:' #data_file_path
w2v_name = '5Y_w2v_model.model' #w2v_model
train_name = '5_year_train_data.xlsx' #train_file
test_name = '3_final_test.xlsx' #test_file
word2vec_model = Word2Vec.load(os.path.join(data_file_path, w2v_name))
index2word_set = set(word2vec_model.wv.index2word)


def load_data(base_dir, file_name):
    f_path = os.path.join(base_dir, file_name)
    data = pd.read_excel(f_path, encoding = "cp949")
    return data


def convert_to_char(data):
    return [ord(xx) for xx in data]


def convert_to_word(data):
    global word_maxlen 
    global emd_size 
    feature_vector = np.zeros((word_maxlen, emd_size), dtype = np.float64)
    sentence = [xx for xx in okt.morphs(data) if xx in index2word_set]
    if len(sentence) > word_maxlen:
        sentence = sentence[:word_maxlen]
    for j, word in enumerate(sentence):
        if word in index2word_set:
            feature_vector[j, :]= np.stack(word2vec_model.wv.word_vec(word))
            
    return feature_vector


if __name__ == "__main__":
    #parameter
    char_maxlen = 48
    word_maxlen = 27
    emd_size = 200
    
    # 데이터 불러오기
    raw_train_data = load_data(data_file_path, train_name)
    raw_test_data = load_data(data_file_path, test_name)
    raw_train_data = shuffle(raw_train_data)
    
    # 데이터 전처리
    raw_test_data["Text"] = raw_test_data["Text"].apply(lambda x : np.nan if type(x) != str else x)
    raw_test_data.dropna(inplace = True)
    
    
    # 데이터 라벨 인코딩
    raw_train_data["Label"] = LE.fit_transform(raw_train_data["Label"])
    raw_test_data["Label"] = LE.transform(raw_test_data["Label"])
    
    #print(LE.classes_)
    y_train_data = to_categorical(raw_train_data["Label"])
    y_test = to_categorical(raw_test_data["Label"])
    
    #Char
    char_tmp1 = raw_train_data["Text"].map(convert_to_char)
    char_tmp2 = raw_test_data["Text"].map(convert_to_char)
    char_train_data = sequence.pad_sequences(char_tmp1, maxlen = char_maxlen)
    char_test_data = sequence.pad_sequences(char_tmp2,  maxlen = char_maxlen)
    
    #Word  
    word_tmp1 = raw_train_data["Text"].map(convert_to_word)
    word_tmp2 =  raw_test_data["Text"].map(convert_to_word)
    word_train_data = np.zeros(len(word_tmp1))
    word_test_data = np.zeros(len(word_tmp2))
    word_train_data = np.stack(word_tmp1)
    word_test_data = np.stack(word_tmp2)