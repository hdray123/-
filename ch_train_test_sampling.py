import numpy as np
import pandas as pd
import re
from gensim import corpora, models, similarities
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

INPUT_DIR = "/Users/shenglong123/Documents/GitHub/tensorflow_siamese_lstm-master/ch_input"
MAX_SEQUENCE_LENGTH = 30
EMB_DIM = 300
seed = 1
train_file_path=INPUT_DIR+'/train.txt'
test_file_path=INPUT_DIR+'/test.txt'
dev_file_path=INPUT_DIR+'/dev.txt'
with open(train_file_path) as f:
    train_data=f.readlines()
with open(test_file_path) as f:
    test_data=f.readlines()
with open(dev_file_path) as f:
    dev_data=f.readlines()

#print(data)
nrof_test=len(test_data)
nrof_dev=len(dev_data)
nrof_train=len(train_data)
pre_dev_data=[]
pre_test_data=[]
pre_train_data=[]
for i in range(nrof_test):
    pre_test_data.append(re.split('\t',test_data[i]))
for i in range(nrof_dev):
    pre_dev_data.append(re.split('\t',dev_data[i]))
for i in range(nrof_train):
    pre_train_data.append(re.split('\t', train_data[i]))

test_q1=[]
test_q2=[]
test_labels=[]

dev_q1=[]
dev_q2=[]
dev_labels=[]

train_q1=[]
train_q2=[]
train_labels=[]
for i in range(nrof_train):
    train_q1.append(pre_train_data[i][1])
    train_q2.append(pre_train_data[i][2])
    train_labels.append(pre_train_data[i][3])

for i  in range(nrof_test):
    test_q1.append(pre_test_data[i][1])
    test_q2.append(pre_test_data[i][2])
    test_labels.append(pre_test_data[i][3])
for i in range(nrof_dev):
    dev_q1.append(pre_dev_data[i][1])
    dev_q2.append(pre_dev_data[i][2])
    dev_labels.append(pre_dev_data[i][3])

text_sequence=pd.Series(train_q1+train_q2+test_q1+test_q2+dev_q1+dev_q2).tolist()
w2v_input =pd.Series(train_q1+train_q2+test_q1+test_q2+dev_q1+dev_q2).apply(lambda x: x.split()).tolist()
word_set = set(" ".join(text_sequence).split())

fileObject=open('./ch_input/len_word_set.txt','w')
fileObject.write(str(len(word_set)))
fileObject.write('\n')
fileObject.close()
fileObject = open('./ch_input/text_sequence.txt', 'w')
for ip in text_sequence:
    fileObject.write(ip)
    fileObject.write('\n')
fileObject.close()
w2v_model = models.Word2Vec(w2v_input, size=EMB_DIM, window=5, min_count=1, sg=1, workers=4, seed=1234, iter=25)

tokenizer = Tokenizer(num_words=len(word_set))
tokenizer.fit_on_texts(text_sequence)

train_q1 = tokenizer.texts_to_sequences(train_q1)
train_q2 = tokenizer.texts_to_sequences(train_q2)

dev_q1  = tokenizer.texts_to_sequences(dev_q1)
dev_q2  =tokenizer.texts_to_sequences(dev_q2)

test_q1 =tokenizer.texts_to_sequences(test_q1)
test_q2 =tokenizer.texts_to_sequences(test_q2)

test_pad_q1 = pad_sequences(test_q1, maxlen=30)
test_pad_q2 = pad_sequences(test_q2, maxlen=30)

dev_pad_q1 = pad_sequences(dev_q1, maxlen=30)
dev_pad_q2 = pad_sequences(dev_q2, maxlen=30)

train_pad_q1 = pad_sequences(train_q1, maxlen=30)
train_pad_q2 = pad_sequences(train_q2, maxlen=30)
embedding_mat = np.zeros([len(tokenizer.word_index)+1, EMB_DIM])

for word, idx in tokenizer.word_index.items():
    embedding_mat[idx,:] = w2v_model.wv[word]
np.save(INPUT_DIR+"/embedding_ch.npy",embedding_mat)
np.save(INPUT_DIR+"/train_pad_q1_ch.npy",train_pad_q1)
np.save(INPUT_DIR+"/train_pad_q2_ch.npy",train_pad_q2)
np.save(INPUT_DIR+'/test_pad_q1_ch.npy',test_pad_q1)
np.save(INPUT_DIR+'/test_pad_q2_ch.npy',test_pad_q2)
np.save(INPUT_DIR+'/dev_pad_q1_ch.npy',dev_pad_q1)
np.save(INPUT_DIR+'/dev_pad_q2_ch.npy',dev_pad_q2)
np.save(INPUT_DIR+'/train_labels',train_labels)
np.save(INPUT_DIR+'/dev_labels',dev_labels)
np.save(INPUT_DIR+'/test_labels',test_labels)
