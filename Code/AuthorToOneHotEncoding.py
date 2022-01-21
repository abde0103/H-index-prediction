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
def OnHotEncoding(word):
    T=np.array([0]*len(vocab_to_int))
    T[vocab_to_int[word]]=1
    return(T)
    
def getabstarctsembeddingandsavethem(entre):
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
        Vect=np.array([0 for i in range(len(vocab_to_int)) ])
        s=0
        for paper in T:
         paper=paper.rstrip('\n')
         if paper in papaerembed:
             s=s+1
             Vect=Vect+papaerembed[paper]
         else :
          try : 
            
            VectorizedPaper=Texts[paper]
            embeds=[OnHotEncoding(word) for word in VectorizedPaper if word in vocab_to_int]
            embed=sum(embeds)/embeds.shape[1]
            s=s+1
            papaerembed[paper]=embed
            Vect=Vect+embed
          except :
             S.add(paper)
        if s!=0:
            Vect=Vect/s
        Authorembed[id]=Vect
    with open(os.path.join(r"C:\Users\moham\Desktop\info",str(entre)+"_OneHotPaperEmbedDict.pkl"), 'wb') as f:
        pickle.dump(papaerembed,f)
    with open(os.path.join(r"C:\Users\moham\Desktop\info",str(entre)+"_OneHotAuthorEmbedDict.pkl"), 'wb') as f:
        pickle.dump(Authorembed,f)
for i in [100]:
        getabstarctsembeddingandsavethem(i)    

        
        