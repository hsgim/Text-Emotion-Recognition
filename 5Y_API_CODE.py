# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 19:19:27 2020

@author: ksh
"""

from konlpy.tag import Twitter ; tw = Twitter()
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from attention_layer_ import AttentionLayer
from gensim.models import Word2Vec
import json, os, re
import numpy as np
from collections import OrderedDict
from tensorflow.keras.models import load_model
from konlpy.tag import Twitter ; tw = Twitter()
from flask import Flask, request, jsonify
from gensim.models import Word2Vec   



# =============================================================================
model_file = (r'C:\Users\ywy\Desktop\iitp_5', '5Y_final_model.h5')
word2vec_file = (r'C:\Users\ywy\Desktop\iitp_5', '5Y_final_w2v.model')
char_maxlen = 48
word_maxlen = 27
model = load_model(model_file[1], custom_objects={"AttentionLayer": AttentionLayer})
word2vec_model = Word2Vec.load(word2vec_file[1])
index2word_set = set(word2vec_model.wv.index2word)
#emotion_list = ['H_happiness', 'A_anger', 'D_disgust', 'F_fear', 'N_neutral', 'S_sadness', 'P_surprise']
emotion_list = {'ANGER': 10002, 'DISGUST' : 10003, 'FEAR' : 10004, 'HAPPINESS' : 10001, 'NEUTRAL' : 10005, 'SADNESS' : 10006, 'SURPRISE' : 10007}
# =============================================================================


def jsonify_or_dump(inp):
    try:
        return jsonify(inp)
    except:
        return json.dumps(inp)

def predict(comments):
    ret = model.predict(comments)
    
    return ret
    
def convert_to_word(data):
    feature_vector = np.zeros((word_maxlen, 200), dtype = np.float64)
    sentence = [xx for xx in tw.morphs(data, norm = True, stem = True) if xx in index2word_set]
    if len(sentence) > word_maxlen:
        sentence = sentence[:word_maxlen]
    for j, word in enumerate(sentence):
        if word in index2word_set:
            feature_vector[j, :]= np.stack(word2vec_model.wv.word_vec(word))
            
    return feature_vector

def sentiment_analysis(text="문장입력") :
    text = text.lower()    
    comment = [ord(xx) for xx in re.sub('[^\da-z가-힣 ]', '', text.strip())][:char_maxlen]
    
    for tmp_i, tmp_x in enumerate(comment):
        tmp_warning = True
        for s_num, e_num, t_num in [(32,32,0), (48,57,1), (97,122,1+10), (44032,55203,1+10+26)]:
            if tmp_x>=s_num and tmp_x<=e_num:
                comment[tmp_i] = tmp_x-s_num+t_num
                tmp_warning = False
                break
        if tmp_warning:
            print(tmp_i, ''.join([chr(xx) for xx in comment]))
            raise Warning
            
    if len(comment) <= char_maxlen:
        comment = ([ord(' ')-32] * (char_maxlen - len(comment))) + comment
    
    comment2 = convert_to_word(text).reshape(1,27,200)
    
    ret = predict([np.array([comment]).reshape(1,48), comment2])
    
    file_data = OrderedDict()
    file_data['predict_emotion'] = [{'code': x[1], 'description': x[0], 'percentage':str(round(ret[0][i] * 100, 4))+'%'} for i,x in enumerate(emotion_list.items())]
    file_data["predict_emotion"].insert(0, file_data["predict_emotion"].pop(3))
    return jsonify_or_dump(file_data)



app = Flask(__name__)
@app.route('/')
def hello():
    return "IBIS LAB  -  API testing......<br/><br/>[Character-based Multi-category Sentiment Analysis on Korean sentences using Deep Learing Algorithms]<br/><br/>URL : http://ibis.hanyang.ac.kr/sentiment_analysis?text=문장입력"
@app.route('/sentiment_analysis', methods = ['GET'])
def response():
    if request.method == 'GET':
        if 'text' in request.args:
            text = request.args['text']
            return sentiment_analysis(text)
    return jsonify_or_dump({'raw_comment': 'error'})
if __name__ == '__main__':
    app.run(host = '166.104.158.193')














