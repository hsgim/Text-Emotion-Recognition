import os
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import gluonnlp as nlp
from tqdm import tqdm

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

data_file_path = r'' #file_path
test_file_name = '5th_test_data_sample.xlsx'
result_file_name = '5th_test_data_sample_result.xlsx'
model_name = '201223_hyu_emotion_recognition.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

# load kobert_model, vocab, tokenizer
bertmodel, vocab = get_pytorch_kobert_model()
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

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
    sent_to_index = {'Angry':'0', 
                     'Disgust':'1',
                     'Fear':'2',
                     'Happiness':'3',
                     'Neutral':'4',
                     'Sadness':'5',
                     'Surprise':'6'}
    
    test_file.iloc[:,1] = test_file.iloc[:,1].apply(lambda x : sent_to_index.get(str(x)))

    result_dt = test_file

    result_dt['pred_Label']='9' #입력을 위해 임시로 label을 달아줍니다
    # dt=dt[['Text', 'pred_Label', 'Label']] 
    list(result_dt.iloc[1,:])
    prepro_dt = [list(result_dt.iloc[i,:2]) for i in range(len(result_dt))]
    
    return result_dt, prepro_dt


if __name__ == "__main__":    
    #load model
    model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
    model.load_state_dict(torch.load(os.path.join(data_file_path, model_name)))	
    model.eval()
    
    test = pd.read_excel(os.path.join(data_file_path, test_file_name))
    result_dt, prepro_dt = data_preprocess(test)
    
    #setting parameters
    max_len = 48
    batch_size = 8
    
    data_test = BERTDataset(prepro_dt, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size)
    
    #predict
    model.eval()
    
    pred_answer = []
    pred_feature = []
    
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        
        max_vals, max_indices = torch.max(out, 1)
        pred_answer.append(max_indices.cpu().clone().numpy())
        
        prob = F.softmax(out, dim=1)
        pred_feature.append(prob.cpu().detach().numpy()) 
    
    #save predict value
    answer_ls = []
    for i in pred_answer:
        answer_ls.extend(i)
    
    feature_ls = []
    for i in pred_feature:
        feature_ls.extend(i)
    
    result_dt['pred_Label']=answer_ls
    
    feature = pd.DataFrame(feature_ls, columns=['Angry',
                                                'Disgust', 
                                                'Fear', 
                                                'Happiness', 
                                                'Neutral', 
                                                'Sadness', 
                                                'Surprise'
                                                ])
    
    index_to_sent = {'0':'Angry', 
                     '1':'Disgust',
                     '2':'Fear',
                     '3':'Happiness',
                     '4':'Neutral',
                     '5':'Sadness',
                     '6':'Surprise'}
    
    result_dt['pred_Label'] = result_dt['pred_Label'].apply(lambda x : index_to_sent.get(str(x)))
    result_dt['Label'] = result_dt['Label'].apply(lambda x : index_to_sent.get(str(x)))
    final_result = pd.concat([result_dt, feature], axis=1)
    
    final_result.to_excel(result_file_name, index=False)

