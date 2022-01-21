import numpy as np
import pickle
import os
import tensorflow.compat.v1 as tf
import time
import pandas as pd

with open(r"C:\Users\moham\Desktop\info\vocab_to_int.pkl", 'rb') as f:
        vocab_to_int = pickle.load(f)
with open(r"C:\Users\moham\Desktop\info\int_to_vocab.pkl", 'rb') as f:
        int_to_vocab = pickle.load(f)
def getabstarctsembeddingandsavethem(entre):
    n_embedding =  entre  
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
        saver.restore(sess, os.path.join(r"C:\Users\moham\Desktop\INF552-2021-PC-code-s02",str(entre)+".ckpt"))
        embed_mat = sess.run(embedding)
        embed_mat=np.array(embed_mat)
    print(embed_mat.shape)
    
    df = pd.read_csv(r"C:\Users\moham\Desktop\info\train.csv")
    df=df.sort_values(by='hindex')



    f=open(r"C:\Users\moham\Desktop\info\author_papers.txt")
    os.makedirs(r"C:\Users\moham\Desktop\info\AuthordEmbed",exist_ok=True)
    os.makedirs(r"C:\Users\moham\Desktop\info\AbstractsEmbed",exist_ok=True)
    lines=f.readlines()
    S=set()
    papaerembed=dict()
    Authorembed=dict()
    with open(r"C:\Users\moham\Desktop\info\TokinizedAndTreatedTexts.pkl","rb") as f:
        Texts=pickle.load(f)
    for line in lines :
        id=line[:line.find(":")]
        line=line[line.find(":")+1:]
        T=line.split("-")
        Vect=np.array([0 for i in range(entre) ])
        s=0
        for paper in T:
         paper=paper.rstrip('\n')
         if paper in papaerembed:
             s=s+1
             Vect=Vect+papaerembed[paper]
         else :
          try : 
            VectorizedPaper=Texts[paper]
            VectorizedPaper=[vocab_to_int[word] for word in VectorizedPaper if word in vocab_to_int]
            embeds=np.array([ np.array(embed_mat[word]) for  word in VectorizedPaper ])
            embed=sum(embeds)/embeds.shape[0]
            embed=sum(embeds)
            papaerembed[paper]=embed
            Vect=Vect+embed
            s=s+1
          except :
             S.add(paper)
        if s!=0:
           Vect=Vect/s
        Authorembed[id]=Vect
    with open(os.path.join(r"C:\Users\moham\Desktop\info",str(entre)+"_PaperEmbedDict.pkl"), 'wb') as f:
        pickle.dump(papaerembed,f)
    with open(os.path.join(r"C:\Users\moham\Desktop\info",str(entre)+"_AuthorEmbedDict.pkl"), 'wb') as f:
        pickle.dump(Authorembed,f)
    
for i in [10,20,30,60,100,300]:
    getabstarctsembeddingandsavethem(i)    
  
        
        