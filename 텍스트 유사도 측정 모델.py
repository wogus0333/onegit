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

import warnings
import tensorflow as tf
warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.ERROR)

train=[]
for i in range(0,10000,500):
  a = str(i)
  b = str(i+500)
  train.append(pd.read_excel("/content/drive/My Drive/Colab Notebooks/naver_sentiment/번역된 유사도 데이터/" + a + "to" + b +".en.ko.xlsx", header=0, delimiter='\t', error_bad_lines=False))

train = pd.concat([train[0], train[1], train[2], train[3], train[4], train[5], train[6], train[7], train[8], train[9],train[10],train[11],train[12],train[13],train[14],train[15],train[16],train[17],train[18],train[19]])
train = train.reset_index(drop=True)

SEQ_LEN = 300
BATCH_SIZE = 16
EPOCHS=3
LR=1e-5

pretrained_path ="drive/My Drive/Colab Notebooks/naver_sentiment/bert"
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

DATA_COLUMN = "질문 1"
DATA_COLUMN1 = "질문 2"
LABEL_COLUMN = "is_duplicate"

token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        if "_" in token:
          token = token.replace("_","")
          token = "##" + token
        token_dict[token] = len(token_dict)

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
    indices, indices1, targets = [], [], []
    for i in tqdm(range(len(data_df))): #tqdm : for문 상태바 라이브러리
        ids, segments = tokenizer.encode(data_df[DATA_COLUMN][i], data_df[DATA_COLUMN1][i], max_len=SEQ_LEN)
        indices.append(ids)
        indices1.append(segments)
        targets.append(data_df[LABEL_COLUMN][i])
    items = list(zip(indices, indices1, targets))
    
    indices, indices1, targets = zip(*items)
    indices = np.array(indices)
    indices1 = np.array(indices1)
    return [indices, indices1], np.array(targets)

def load_data(pandas_dataframe):
    data_df = pandas_dataframe
    
    
    data_df[DATA_COLUMN] = data_df[DATA_COLUMN].astype(str)
    data_df[DATA_COLUMN1] = data_df[DATA_COLUMN1].astype(str)
    data_x, data_y = convert_data(data_df)

    return data_x, data_y


train_x, train_y = load_data(train)
list(map(tuple, np.where(np.isnan(train_y))))

layer_num = 12
model = load_trained_model_from_checkpoint(
    config_path,
    checkpoint_path,
    training=True,
    trainable=True,
    seq_len=SEQ_LEN,)

bert_model = load_model("/content/drive/My Drive/Colab Notebooks/naver_sentiment/text_similarity_quora_question_pairs.h5", custom_objects= get_custom_objects(), compile=False)
bert_model.compile(optimizer=RAdam(learning_rate=0.00001, weight_decay=0.0025),
      loss='binary_crossentropy',
      metrics=['accuracy'])
model.summary()

def get_bert_finetuning_model(model):
  inputs = model.inputs[:2]
  dense = model.layers[-3].output


  outputs = keras.layers.Dense(1, activation='sigmoid',kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                              name = 'real_output')(dense)



  bert_model = keras.models.Model(inputs, outputs)
  bert_model.compile(
      optimizer=RAdam(learning_rate=0.00001, weight_decay=0.0025),
      loss='binary_crossentropy',
      metrics=['accuracy'])
  
  return bert_model

from IPython.display import SVG
from keras.utils import model_to_dot

SVG(model_to_dot(model, dpi=65).create(prog='dot', format='svg'))

sess = K.get_session()
uninitialized_variables = set([i.decode('ascii') for i in sess.run(tf.report_uninitialized_variables())])
init = tf.variables_initializer([v for v in tf.global_variables() if v.name.split(':')[0] in uninitialized_variables])
sess.run(init)
history = bert_model.fit(train_x, train_y, epochs=3, batch_size=8, verbose = 1, shuffle=True)

bert_model.save("/content/drive/My Drive/Colab Notebooks/naver_sentiment/text_similarity_quora_question_pairs1.h5")