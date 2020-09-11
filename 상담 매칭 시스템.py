import tensorflow as tf
import os
import pandas as pd
import numpy as np  
import re
import pickle

from konlpy.tag import Okt
from collections import Counter

import keras as keras
from keras.models import load_model
from keras import backend as K
from keras import Input, Model
from keras import optimizers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import codecs
from tqdm import tqdm
import shutil

from keras_bert import load_trained_model_from_checkpoint, load_vocabulary
from keras_bert import Tokenizer
from keras_bert import AdamWarmup, calc_train_steps
from keras_bert import get_custom_objects
from keras_radam import RAdam

import warnings
import tensorflow as tf
warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
path = "drive/My Drive/Colab Notebooks/naver_sentiment"
text_similarity_bert_model = load_model("/content/drive/My Drive/Colab Notebooks/naver_sentiment/[QQP]text_similarity_final.h5", custom_objects= get_custom_objects(), compile=False)
bert_model = load_model(path+"/BERT Category Classification Model Final.h5", custom_objects= get_custom_objects(), compile=False)
data = ((pd.read_excel("drive/My Drive/Colab Notebooks/naver_sentiment/윌슨데이터/new_상담자1.xlsx")).drop_duplicates()).reset_index(drop=True)

f = lambda x: len(x)
data_length = data['data'].astype(str).apply(f)
data_length.head()

plt.figure(figsize=(12,5))
plt.hist(data_length, bins=200, alpha=0.5, color='orange', label='word')
plt.yscale('log', nonposy='clip')
plt.title('Log-histogram of length of data')
plt.xlabel('Length of data')
plt.ylabel('Number of data')

data= data.replace(4, 3)

fig, axe = plt.subplots(ncols=1)
fig.set_size_inches(6, 3)
sns.countplot(data['label'])

data_label = []
category = ['연애', '진로', '자존감', '대인관계']

print("data set\n")
for i in range(4):
  x = data['label'].value_counts()[i]
  data_label.append(x)
  print('{} :{}개'.format(category[i],data_label[i]))

love = data.loc[data['label'] == 0]

love

f = lambda x: len(x)
love_length = love['data'].astype(str).apply(f)
love_length.head()

plt.figure(figsize=(12,5))
plt.hist(love_length, bins=200, alpha=0.5, color='orange', label='word')
plt.yscale('log', nonposy='clip')
plt.title('Log-histogram of length of data')
plt.xlabel('Length of data')
plt.ylabel('Number of data')

course = data.loc[data['label'] == 1]

course

f = lambda x: len(x)
course_length = course['data'].astype(str).apply(f)
course_length.head()

plt.figure(figsize=(12,5))
plt.hist(course_length, bins=200, alpha=0.5, color='orange', label='word')
plt.yscale('log', nonposy='clip')
plt.title('Log-histogram of length of data')
plt.xlabel('Length of data')
plt.ylabel('Number of data')

self_esteem = data.loc[data['label'] == 2]

self_esteem

f = lambda x: len(x)
self_esteem_length = self_esteem['data'].astype(str).apply(f)
self_esteem_length.head()

plt.figure(figsize=(12,5))
plt.hist(self_esteem_length, bins=200, alpha=0.5, color='orange', label='word')
plt.yscale('log', nonposy='clip')
plt.title('Log-histogram of length of data')
plt.xlabel('Length of data')
plt.ylabel('Number of data')

relationship = data.loc[data['label'] == 3]

relationship

f = lambda x: len(x)
relationship_length = relationship['data'].astype(str).apply(f)
relationship_length.head()

plt.figure(figsize=(12,5))
plt.hist(relationship_length, bins=200, alpha=0.5, color='orange', label='word')
plt.yscale('log', nonposy='clip')
plt.title('Log-histogram of length of data')
plt.xlabel('Length of data')
plt.ylabel('Number of data')

new_requester = "어릴 때부터 취업보다는 창업을 하고 싶다고 생각했는데 너무 막연하네요. 혹시 창업에 관해서 조언해주실 사람 있나요?" # 상담 요청자 데이터

okt = Okt()

noun = okt.nouns(new_requester)

word = []
for n in noun:
  if len(n) > 1:
    word.append(n)


count = Counter(word)

#명사 빈도 카운트
word_list = count.most_common(10)

keword = ''
word_list_length = len(word_list)
for i in range(word_list_length):
  if i == word_list_length-1:
    keword += word_list[i][0]
  else:
    keword += word_list[i][0] + '|'

keword

SEQ_LEN = 300

pretrained_path ="drive/My Drive/Colab Notebooks/naver_sentiment/bert"
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

DATA_COLUMN = "data"
LABEL_COLUMN = "label"

token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        if "_" in token:
          token = token.replace("_","")
          token = "##" + token
        token_dict[token] = len(token_dict)

token_dict

class inherit_Tokenizer(Tokenizer):
  def _tokenize(self, text):
        if not self._cased:
            text = text
            
            text = text.lower()
        spaced = ''
        for ch in text:
            if self._is_punctuation(ch) or self._is_cjk_character(ch):
                spaced += ' ' + ch + ' '
            elif self._is_space(ch):
                spaced += ' '
            elif ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch
        tokens = []
        for word in spaced.strip().split():
            tokens += self._word_piece_tokenize(word)
        return tokens

tokenizer = inherit_Tokenizer(token_dict)

print(tokenizer.tokenize("오늘은 비가 와요."))

print(tokenizer.encode("오늘은 비가 와요.","김재희",))

def predict_convert_data(data_df):
    global tokenizer
    indices = []
    
    ids, segments = tokenizer.encode(data_df, max_len=SEQ_LEN)
    indices.append(ids)
        
    items = indices
        
    indices = np.array(indices)
    return [indices, np.zeros_like(indices)]

def predict_load_data(x): #Pandas Dataframe을 인풋으로 받는다
    data_df = x

    data_x = predict_convert_data(data_df)

    return data_x

new_data = predict_load_data(new_requester) # 새로 들어온 데이터 버트입력형식으로 바꿈

new_data

#예측
preds = bert_model.predict(new_data)

preds

preds = np.argmax (preds, axis = 1) # 가장 높은 인덱스 추출

preds

label = preds.tolist() #numpy to list

label

category = ''

if label[0] == 0:
  category = '연애'
elif label[0] == 1:
  category = '진로'
elif label[0] == 2:
  category = '자존감'
elif label[0] == 3:
  category = '대인관계'

data_category = [love, course, self_esteem, relationship]

new_dictionary = {'data':new_requester, 'category':category, 'label':label[0]}

new_dictionary

def convert_data1(data, data_df):
    global tokenizer
    indices, indices1 = [], []
    for i in tqdm(range(len(data_df))): #tqdm : for문 상태바 라이브러리
        ids, segments = tokenizer.encode(data, data_df[DATA_COLUMN][i], max_len=300)
        indices.append(ids)
        indices1.append(segments)    

    indices = np.array(indices)
    indices1 = np.array(indices1)
    return [indices, indices1]

def predict_load_data1(requester,pandas_dataframe):
    data = requester
    data_df = pandas_dataframe    
    
    data_df[DATA_COLUMN] = data_df[DATA_COLUMN].astype(str)
    data_x = convert_data1(data, data_df)

    return data_x

sorted_category = data_category[label[0]]

sorted_category

sorted_category = sorted_category.loc[sorted_category['data'].str.contains(keword, na = False)]

select_category = sorted_category.reset_index(drop=True) # 인덱스 reset

select_category

similarity_set = predict_load_data1(new_requester, select_category) # 문장 유사도를 위한 버트 input 데이터 생성

similarity_set[0][0]

similarity_set[1][0]

text_similarity_preds = text_similarity_bert_model.predict(similarity_set)

text_similarity_preds

import copy


preds = copy.deepcopy(text_similarity_preds)
text_similarity_rank = []
result = []
result_dictionary = {'index': -1, 'data': ' ', 'category': ' ' , 'label' : -1}
result_dictionary1 = {'index': -1, 'data': ' ', 'category': ' ' , 'label' : -1}
result_dictionary2 = {'index': -1, 'data': ' ', 'category': ' ' , 'label' : -1}
dic_list = [result_dictionary, result_dictionary1, result_dictionary2]
for i in range(3):
  x = np.argmax(preds)
  text_similarity_rank.append(x)
  print(select_category[x:x+1])
  result.append(select_category[x:x+1])
  preds[text_similarity_rank[i]] = [0]
  dic_list[i]['index'] = result[i].index.start
  dic_list[i]['data'] = result[i].iloc[0,0]
  dic_list[i]['category'] = result[i].iloc[0,1]
  dic_list[i]['label'] = result[i].iloc[0,2]

result_dictionary

result_dictionary1

result_dictionary2