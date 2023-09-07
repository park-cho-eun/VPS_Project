#!/usr/bin/env python
# coding: utf-8

# preprocessing

# In[1]:


import pandas as pd
import re
import kss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import sent_tokenize
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
import torch.nn.functional as F


# In[2]:


data1=pd.read_excel('C:/Users/ssb70/OneDrive/바탕 화면/간호법/data1.xlsx')
data2=pd.read_excel('C:/Users/ssb70/OneDrive/바탕 화면/간호법/data2.xlsx')


# In[3]:


data1.columns=data1.iloc[0]
data1=data1[1:]
data2.columns=data2.iloc[0]
data2=data2[1:]


# In[4]:


print(len(data1),len(data2),len(data1)+len(data2))


# In[5]:


data=pd.concat([data1,data2],axis=0,ignore_index=True)
print(len(data))


# In[6]:


data.head(5)


# In[7]:


data['내용_전처리(특수문자)']=None


# In[8]:


#제목은 특수문자가 많은 내용을 내포하므로 내용만 전처리('.' 은 제외(문장토큰화시 필요))
import re

data['내용'] = data['내용'].astype(str)
for i, text in enumerate(data['내용']):
    # 마침표를 제외한 다른 특수 문자 제거
    data.loc[i, '내용_전처리(특수문자)'] = re.sub(r'[^\w\s.]', '', text)


# In[9]:


data['slicing']=None


# 문장 slicing 하기

# In[10]:


for i, text in tqdm(enumerate(data['내용_전처리(특수문자)'])):
    texts=sent_tokenize(text)
    data.at[i,'slicing']=texts


# 기사 개수가 1줄인 기사는 제외 (별 의미없음)

# In[11]:


length=[]
for i in range(len(data)):
    a=len(data.loc[i,'slicing'])
    length.append(a)


# In[12]:


no_article_idx=[]
for i in range(len(data)):
    if len(data.loc[i,'slicing'])==1:
        no_article_idx.append(i)


# In[13]:


len(no_article_idx)


# In[14]:


data=data.drop(no_article_idx)
data.reset_index(drop=True, inplace=True)


# In[15]:


data['embedding']=None


# In[16]:


class PositionalEncoding(nn.Module):
    
    def __init__(self, seq_len, d_model, n, device):
        
        super(PositionalEncoding, self).__init__() # nn.Module 초기화
        
        # encoding : (seq_len, d_model)
        self.encoding = torch.zeros(seq_len, d_model, device=device)
        self.encoding.requires_grad = False
        
        # (seq_len, )
        pos = torch.arange(0, seq_len, device=device)
        # (seq_len, 1)         
        pos = pos.float().unsqueeze(dim=1) # int64 -> float32 
        
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        
        self.encoding[:, ::2] = torch.sin(pos / (n ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (n ** (_2i / d_model)))
        
        
    def forward(self, x):
        # x.shape : (batch, seq_len) or (batch, seq_len, d_model)
        seq_len = x.size()[1] 
        # return : (seq_len, d_model)
        # return matrix will be added to x by broadcasting
        return self.encoding[:seq_len, :]
    
    
tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-Medium", do_lower_case=False)
model = AutoModel.from_pretrained("snunlp/KR-Medium")


# In[ ]:


for idx in tqdm(range(len(data))) :
    title = tokenizer(data.iloc[idx,2], padding='longest', return_tensors='pt') # 제목 토큰화
    tokenized = tokenizer(data.iloc[idx,5], padding='longest', return_tensors='pt') # 내용 토큰화됨
    try :
        title = model(**title).pooler_output # 제목 embedding
        tokenized = model(**tokenized).pooler_output # 내용 각 문장들 embedding
        # 제목 semantic을 내용에 발라주기
        for _ in range(6) :
            tokenized = (F.softmax(F.cosine_similarity(title, tokenized), dim=0).reshape(-1,1) + 1) * tokenized
        # 문장 positional encoding
        seq_len = len(tokenized)
        pos_encoder = PositionalEncoding(seq_len=seq_len, d_model=768, n=10000, device='cpu')
        tokenized += pos_encoder(tokenized)
        # document embedding vector 준비
        dummy = torch.rand((768)).reshape(1,-1)
        total = torch.cat([dummy, title, tokenized])
        # document embedding
        for _ in range(6) :
            cosine_similarities = F.cosine_similarity(total.unsqueeze(1), total.unsqueeze(0), dim=2)
            attention_weights = F.softmax(cosine_similarities, dim=1)
            total = torch.matmul(attention_weights, total)
        data.at[idx,'embedding'] = total[0].tolist()
    except :
        print(idx, 'error')

