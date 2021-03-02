import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import json, os, re
import numpy as np
from collections import OrderedDict
from flask import Flask, request, jsonify

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import gluonnlp as nlp
from tqdm import tqdm

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load kobert_model, vocab, tokenizer
bertmodel, vocab = get_pytorch_kobert_model()
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

model_file = (r'model_path', '201223_hyu_emotion_recognition.pt')

emotion_list = {'ANGER': 10002, 'DISGUST' : 10003, 'FEAR' : 10004, 'HAPPINESS' : 10001, 'NEUTRAL' : 10005, 'SADNESS' : 10006, 'SURPRISE' : 10007}


# =============================================================================
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=7,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)
    
def data_preprocess(test_file):
    result_dt = test_file

    result_dt['pred_Label']='9' #입력을 위해 임시로 label을 달아줍니다
    
    prepro_data = [list(result_dt.iloc[i,:2]) for i in range(len(result_dt))]
    
    return result_dt, prepro_data


def jsonify_or_dump(inp):
    try:
        return jsonify(inp)
    except:
        return json.dumps(inp)



def sentiment_analysis(text="문장입력") :
    # text = text.lower() #
    #load model
    model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
    model.load_state_dict(torch.load(os.path.join(model_file[0], model_file[1])))	
    
    test = pd.DataFrame([text], columns=['reviews'])
    result_dt, prepro_dt = data_preprocess(test)
    
    #setting parameters
    max_len = 48
    batch_size = 8
    
    data_test = BERTDataset(prepro_dt, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size)
    
    #predict
    model.eval()
    
    pred_feature = []
    
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        
        # max_vals, max_indices = torch.max(out, 1)
        # pred_answer.append(max_indices.cpu().clone().numpy())
        
        prob = F.softmax(out, dim=1)
        pred_feature.append(prob.cpu().detach().numpy()) 
    
    file_data = OrderedDict()
    for i,x in enumerate(emotion_list.items()): 
        print(round(pred_feature[0][0][i]*100,4))
        
    file_data['predict_emotion'] = [{'code': x[1], 'description': x[0], 'percentage':str(round(pred_feature[0][0][i] * 100, 4))+'%'} for i,x in enumerate(emotion_list.items())]
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