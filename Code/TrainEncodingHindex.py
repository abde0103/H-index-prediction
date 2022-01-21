import numpy as np
import pickle
import os
import tensorflow.compat.v1 as tf
import time
import pandas as pd
from tensorflow import keras
from keras import layers
with open(r"C:\Users\moham\Desktop\info\vocab_to_int.pkl", 'rb') as f:
        vocab_to_int = pickle.load(f)
with open(r"C:\Users\moham\Desktop\info\int_to_vocab.pkl", 'rb') as f:
        int_to_vocab = pickle.load(f)
n_embedding=10
n_vocab=len(vocab_to_int)


Train=pd.read_csv(r"C:\Users\moham\Desktop\info\1000_embedcsv-train.csv", index_col=0)
Test=pd.read_csv(r"C:\Users\moham\Desktop\info\1000_embedcsv-test.csv", index_col=0)


Y_Train=Train.loc[:,"hindex"].values
del Train ["hindex"]
X_Train=Train.values


Y_Test=Test.loc[:,"hindex"].values
del Test ["hindex"]
X_Test=Test.values


Max=X_Train.max(axis=1)
Min=X_Train.min(axis=1)
Max=Max.reshape((Max.shape[0],1))
Min=Min.reshape((Min.shape[0],1))
Max=np.tile(Max,X_Train.shape[1])
Min=np.tile(Min,X_Train.shape[1])
X_Train=(-X_Train+Max)/(Max-Min)
Max=X_Test.max(axis=1)
Min=X_Test.min(axis=1)
Max=Max.reshape((Max.shape[0],1))
Min=Min.reshape((Min.shape[0],1))
Max=np.tile(Max,X_Test.shape[1])
Min=np.tile(Min,X_Test.shape[1])
X_Test=(-X_Test+Max)/(Max-Min)



train_graph = tf.Graph()
with train_graph.as_default():
    inputs = tf.placeholder(tf.int32, [None], name='inputs')
    labels = tf.placeholder(tf.int32, [None, 1], name='labels')
with train_graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_vocab, n_embedding), -1, 1))
with train_graph.as_default():
    saver = tf.train.Saver()
with tf.Session(graph=train_graph) as sess:
    saver.restore(sess, os.path.join(r"C:\Users\moham\Desktop\INF552-2021-PC-code-s02",str(100)+".ckpt"))
    embed_mat = sess.run(embedding)
    
    
def myfonction(shape,dtype=float):
    return    embed_mat
x=keras.Input(shape=(69315,))
dense = layers.Dense(100, activation="relu",kernel_initializer=myfonction)
dense1=layers.Dense(10, activation="relu")
dense2=layers.Dense(1, activation="relu")
output=dense2(dense1(dense(x)))
model=keras.Model(inputs=x,outputs=output)

model.fit(X_Train,Y_Train,epochs=5)
model.evaluate(X_Test,Y_Test)
Embed=dense.get_weights()
with open(r"C:\Users\moham\Desktop\info\Hembed.pkl","rb") as f :
    pickle.dump(Embed,f)