import pickle
import os
from bs4 import BeautifulSoup
from numpy import vectorize
import spacy
import unidecode
from word2number import w2n
import os
import pickle
#import contractions

nlp = spacy.load('en_core_web_lg')

# exclude words from spacy stopwords list
deselect_stop_words = ['no', 'not']
for w in deselect_stop_words:
    nlp.vocab[w].is_stop = False


def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text


def remove_whitespace(text):
    text = text.strip()
    return " ".join(text.split())


def remove_accented_chars(text):
    text = unidecode.unidecode(text)
    return text


def expand_contractions(text):
    text = contractions.fix(text)
    return text


def text_preprocessing(text, accented_chars=True, contractions=True, 
                       convert_num=True, extra_whitespace=True, 
                       lemmatization=True, lowercase=True, punctuations=True,
                       remove_html=True, remove_num=True, special_chars=True, 
                       stop_words=True):
    if remove_html == True: 
        text = strip_html_tags(text)
    if extra_whitespace == True:
        text = remove_whitespace(text)
    if accented_chars == True: 
        text = remove_accented_chars(text)
    if contractions == True: 
        text = expand_contractions(text)
    if lowercase == True: 
        text = text.lower()

    doc = nlp(text) 

    clean_text = []
    
    for token in doc:
        flag = True
        edit = token.text
     
        if stop_words == True and token.is_stop and token.pos_ != 'NUM': 
            flag = False
      
        if punctuations == True and token.pos_ == 'PUNCT' and flag == True: 
            flag = False
     
        if special_chars == True and token.pos_ == 'SYM' and flag == True: 
            flag = False
 
        if remove_num == True and (token.pos_ == 'NUM' or token.text.isnumeric()) \
        and flag == True:
            flag = False

        if convert_num == True and token.pos_ == 'NUM' and flag == True:
            edit = w2n.word_to_num(token.text)

        elif lemmatization == True and token.lemma_ != "-PRON-" and flag == True:
            edit = token.lemma_

        if edit != "" and flag == True:
            clean_text.append(edit)        
    return clean_text
def fix(PickleFile):
    Path=r"C:\Users\moham\Desktop\info\abstracts"
    Path_1=r"C:\Users\moham\Desktop\info\Treated_Abstracts"
    os.makedirs(Path_1,exist_ok=True)
    file=PickleFile.replace(".pkl",".txt")
    if True :
        f=open(os.path.join(Path,file),"r",encoding="UTF-8")
        lines=f.readlines()
        text=""
        for line in lines :
            text=text+" "+line
        f_1=open(os.path.join(Path_1,file[:file.find(".txt")]+".pkl"),"wb")

        pickle.dump(text_preprocessing(text), f_1)
        f_1.close()
        f.close()



Path=r"C:\Users\moham\Desktop\info\Treated_Abstracts"
vocab_to_int=dict()
max=0
for PickleFile in os.listdir(Path) :
    with open(os.path.join(Path,PickleFile), 'rb') as f:
        try:
            loaded_text = pickle.load(f)
        except :
            fix(PickleFile)
            loaded_text = pickle.load(f)
            
        for mot in loaded_text :
            if ((mot in vocab_to_int)==False):
                vocab_to_int[mot]=1
            else :
                vocab_to_int[mot]+=1

T=[]
T_1=[]
s=0
for index,value in vocab_to_int.items():  
    T_1.append([index,value])  
    if value<10:
        print (index)
        s=s+1
    else :
        T.append([index,value])
T=sorted(T,key=lambda x:x[1],reverse=True)
print(s)
vocab_to_int=dict()
for i in range(len(T)):
      vocab_to_int[T[i][0]]=i     
int_to_vocab=dict()
for index,value in vocab_to_int.items():
    int_to_vocab[value]=index
print(len(int_to_vocab),len(vocab_to_int))
with open(r"C:\Users\moham\Desktop\info\test\vocab_to_int.pkl", 'wb') as f:
    pickle.dump(vocab_to_int, f)
with open(r"C:\Users\moham\Desktop\info\test\int_to_vocab.pkl", 'wb') as f:
    pickle.dump(int_to_vocab, f)
with open(r"C:\Users\moham\Desktop\info\test\ifIchangemymind.pkl", 'wb') as f:
    pickle.dump(T_1, f)
