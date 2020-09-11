import tensorflow as tf
import os
import pandas as pd
import numpy as np  
import re
import pickle

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

#Keras 코드 수행 시에 출력되는 Warning을 포함한 기타 메시지 제거.
import warnings
import tensorflow as tf
warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

path = "drive/My Drive/Colab Notebooks/naver_sentiment"

data = (pd.read_excel("drive/My Drive/Colab Notebooks/naver_sentiment/윌슨데이터/w_min.xlsx").drop_duplicates()).reset_index(drop=True)

love = data.loc[data['label']==0] #카테고리 별로 자료 분류
course = data.loc[data['label']==1]
self_esteem = data.loc[data['label']==2]
relationship = data.loc[data['label']==3]

course

testset_num = 20

love_test = love.sample(n=testset_num, random_state= 1)
course_test = course.sample(n=testset_num, random_state= 1)
self_esteem_test = self_esteem.sample(n=testset_num, random_state= 1)
relationship_test = relationship.sample(n=testset_num, random_state= 1)

love_test

love = pd.concat([love, love_test])
love = love.reset_index(drop=True)

course = pd.concat([course, course_test])
course = course.reset_index(drop=True)

self_esteem = pd.concat([self_esteem, self_esteem_test])
self_esteem = self_esteem.reset_index(drop=True)

relationship = pd.concat([relationship, relationship_test])
relationship = relationship.reset_index(drop=True)

course

love_gpby = love.groupby(list(love.columns))
course_gpby = course.groupby(list(course.columns))
self_esteem_gpby = self_esteem.groupby(list(self_esteem.columns))
relationship_gpby = relationship.groupby(list(relationship.columns))

love_idx = [x[0] for x in love_gpby.groups.values() if len(x) == 1]
course_idx = [x[0] for x in course_gpby.groups.values() if len(x) == 1]
self_esteem_idx = [x[0] for x in self_esteem_gpby.groups.values() if len(x) == 1]
relationship_idx = [x[0] for x in relationship_gpby.groups.values() if len(x) == 1]

love_train = love.reindex(love_idx)
course_train = course.reindex(course_idx)
self_esteem_train = self_esteem.reindex(self_esteem_idx)
relationship_train = relationship.reindex(relationship_idx)

course_train.reindex()

train =  pd.concat([love_train, course_train, self_esteem_train, relationship_train])
test =  pd.concat([love_test, course_test, self_esteem_test, relationship_test])

train = train.reset_index(drop=True)

train

test = test.reset_index(drop=True)

test

train.to_excel("drive/My Drive/Colab Notebooks/naver_sentiment/윌슨데이터/w_min_train_1_" + str(testset_num) + ".xlsx")
test.to_excel("drive/My Drive/Colab Notebooks/naver_sentiment/윌슨데이터/w_min_test_1_" + str(testset_num) + ".xlsx")

f = lambda x: len(x)
data_length = data['data'].astype(str).apply(f)
plt.figure(figsize=(12,5))
plt.hist(data_length, bins=200, alpha=0.5, color='orange', label='word')
plt.yscale('log', nonposy='clip')
plt.title('Log-histogram of length of data')
plt.xlabel('Length of data')
plt.ylabel('Number of data')

f = lambda x: len(x)
data_length = train['data'].astype(str).apply(f)
data_length.head()

plt.figure(figsize=(12,5))
plt.hist(data_length, bins=200, alpha=0.5, color='r', label='word')
plt.yscale('log', nonposy='clip')
plt.title('Log-histogram of length of data')
plt.xlabel('Length of data')
plt.ylabel('Number of data')

f = lambda x: len(x)
data_length = test['data'].astype(str).apply(f)
data_length.head()

plt.figure(figsize=(12,5))
plt.hist(data_length, bins=200, alpha=0.5, color="b", label='word')
plt.yscale('log', nonposy='clip')
plt.title('Log-histogram of length of data')
plt.xlabel('Length of data')
plt.ylabel('Number of data')

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


SEQ_LEN = 300

pretrained_path ="drive/My Drive/Colab Notebooks/naver_sentiment/bert"
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
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

def convert_data(data_df):
    global tokenizer
    indices, targets = [], []
    for i in tqdm(range(len(data_df))): #tqdm : for문 상태바 라이브러리
        ids, segments = tokenizer.encode(data_df[DATA_COLUMN][i], max_len=SEQ_LEN)
        indices.append(ids)
        targets.append(data_df[LABEL_COLUMN][i])
    items = list(zip(indices, targets))
    
    indices, targets = zip(*items)
    indices = np.array(indices)
    return [indices, np.zeros_like(indices)], np.array(targets)

def load_data(pandas_dataframe):
    data_df = pandas_dataframe
    data_df[DATA_COLUMN] = data_df[DATA_COLUMN].astype(str)
    data_x, data_y = convert_data(data_df)
    return data_x, data_y

train_x, train_y = load_data(train)
test_x, test_y = load_data(test)

layer_num = 12
model = load_trained_model_from_checkpoint(
    config_path,
    checkpoint_path,
    training=True,
    trainable=True,
    seq_len=SEQ_LEN,)

# bert_model = load_model(path+"/nsmc학습모델/bert_model.h5", custom_objects= get_custom_objects(), compile=False)
# # bert_model.compile(optimizer=RAdam(learning_rate=0.00001, weight_decay=0.0025),
# #       loss='binary_crossentropy',
# #       metrics=['accuracy'])


#model.summary()

def get_bert_finetuning_model(model):
  inputs = model.inputs[:2]
  dense = model.layers[-3].output

#output layer 수정
  outputs = keras.layers.Dense(4, activation='softmax',kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                              name = 'real_output')(dense)



  bert_model = keras.models.Model(inputs, outputs)
  bert_model.compile(
      optimizer=RAdam(learning_rate=0.00001, weight_decay=0.0025),
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'])
  
  return bert_model

sess = K.get_session()
uninitialized_variables = set([i.decode('ascii') for i in sess.run(tf.report_uninitialized_variables())])
init = tf.variables_initializer([v for v in tf.global_variables() if v.name.split(':')[0] in uninitialized_variables])
sess.run(init)

bert_model = get_bert_finetuning_model(model)

from IPython.display import SVG
from keras.utils import model_to_dot


SVG(model_to_dot(bert_model, dpi=65).create(prog='dot', format='svg'))

history = bert_model.fit(train_x, train_y, epochs=30, batch_size=8, verbose = 1, validation_data=(test_x, test_y), shuffle=True)

# 학습 정확성 값과 검증 정확성 값을 플롯팅 합니다. 
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 학습 손실 값과 검증 손실 값을 플롯팅 합니다.
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

bert_model.save(path+"/text_similarity.h5")
