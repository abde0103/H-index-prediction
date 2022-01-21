import numpy as np
import pickle
import os
import tensorflow.compat.v1 as tf
import time

with open(r"C:\Users\moham\Desktop\info\vocab_to_int.pkl", 'rb') as f:
        vocab_to_int = pickle.load(f)
with open(r"C:\Users\moham\Desktop\info\int_to_vocab.pkl", 'rb') as f:
        int_to_vocab = pickle.load(f)
n_embedding =  10  
n_vocab=len(int_to_vocab)   
train_graph = tf.Graph()
with train_graph.as_default():
    inputs = tf.placeholder(tf.int32, [None], name='inputs')
    labels = tf.placeholder(tf.int32, [None, 1], name='labels')
    
with train_graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_vocab, n_embedding), -1, 1))
    
with train_graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=train_graph) as sess:
    saver.restore(sess, os.path.join(r"C:\Users\moham\Desktop\INF552-2021-PC-code-s02","10.ckpt"))
    embed_mat = sess.run(embedding)



import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

viz_words = 100
tsne = TSNE()
embed_tsne = tsne.fit_transform(embed_mat[:viz_words, :])

fig, ax = plt.subplots(figsize=(30, 30))
for idx in range(viz_words):
    plt.scatter(*embed_tsne[idx, :], color='steelblue')
    plt.annotate(int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
plt.show()