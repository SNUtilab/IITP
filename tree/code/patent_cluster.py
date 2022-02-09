# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import re
import pickle
import os



with open('data/patent_df.pkl','rb') as f:
    df = pickle.load(f)




#%% Doc2Vec 기반 문서 유사도
import pandas , nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import RegexpTokenizer

from nltk.stem import WordNetLemmatizer
import spacy
import nltk

n = WordNetLemmatizer()

sp = spacy.load('en_core_web_sm')
stopwords_ = sp.Defaults.stop_words

df['Abstract'] = df['Abstract'].apply(lambda x : " ".join(nltk.sent_tokenize(x)[:]))

df['Claim'] = df['Claim'].apply(lambda x : " ".join(nltk.sent_tokenize(x)[1:]))

 

#corpus: Abstract + claim
df['total'] = df.apply(lambda x : x.Title +' '+ x.Abstract +' ' + x.Claim , axis = 1)

def nltk_tokenizer(_wd):
  return RegexpTokenizer(r'\w+').tokenize(_wd.lower())

df['total'] = df['total'].apply(nltk_tokenizer)


doc_df = df[['WIPS ON key','total']].values.tolist()
tagged_data = [TaggedDocument(words=_d, tags=[str(uid)]) for uid, _d in doc_df]


#%% Training

#max_epochs = 50
max_epochs = 30

model = Doc2Vec(
    window=10,
    vector_size = 20,
    alpha=0.025, 
    min_alpha=0.025,
    min_count=2,
    dm =1,
    negative = 5,
    seed = 9999)
  
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=100)
    # decrease the learning rate
    model.alpha -= 0.002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha


#%%


with open('input/dcv_model.pkl','rb') as f:
    dcv_model = pickle.load(f)
  
#%%    
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


#number of cluster
k = 19
kmeans_model = KMeans(n_clusters=k, init= "k-means++", max_iter=100)
X = kmeans_model.fit(dcv_model.docvecs.vectors ) # document vector 전체를 가져옴. 
labels=kmeans_model.labels_.tolist()


df['do2v_clu'] = labels

df.groupby(['do2v_clu']).count()   


#%%topic 별 문서 분할

df_ao = df.iloc[:,[23,4,5,6,27]]


dfs = []
for i in range(len(set(labels))):
    condition = df_ao['do2v_clu'] == i
    ao = df_ao[condition]
    dfs.append(ao)


 #%% doc2vec cluster labeling
from scipy.cluster.vq import kmeans,vq



centroids = kmeans_model.cluster_centers_

clu_label_doc = []
for i in range(kmeans_model.n_clusters):
    

    clu_label_doc.append(dcv_model.docvecs.most_similar(positive = [centroids[i]], topn = 10))

label_doc = []
for i in range(kmeans_model.n_clusters):
    a = pd.DataFrame(columns = df_ao.columns)
    for j in range(len(clu_label_doc[0])):
        a = a.append(df_ao[df_ao['WIPS ON key']== int(clu_label_doc[i][j][0])])
    label_doc.append(a)    


#%% DF - ICF
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

n = WordNetLemmatizer()
sp = stopwords.words('english')
stopwords_ = sp
#stopwords_ = sp.Defaults.stop_words

corpus_after = []
for i in range(kmeans_model.n_clusters):
    label_doc[i]['Abstract'] = label_doc[i]['Abstract'].apply(lambda x : " ".join(nltk.sent_tokenize(x)[:]))
    
    label_doc[i]['Claim'] = label_doc[i]['Claim'].apply(lambda x : " ".join(nltk.sent_tokenize(x)[1:]))
    
    #corpus_ = df.apply(lambda x : x.Title +' '+ x.Abstract , axis = 1).tolist()    
    
    #corpus: Abstract + claim
    corpus_ = label_doc[i].apply(lambda x : x.Title +' '+ x.Abstract +' ' + x.Claim , axis = 1).tolist()  
    corpus_after.append(corpus_)

after_corpus_ = [] 
after_corpus2_ = []
for i in range(kmeans_model.n_clusters):
    after_corpus_ = [] 
    for doc in corpus_after[i] :
        
        words_ = doc.strip().split()
    
        # 소문자화
        words_ = [i.lower() for i in words_]
        
        # lemmatize
        words_ = [n.lemmatize(token, 'n') for token in words_ if len(token) >= 1] 
        
        # delete stopwords
        words_ = [token for token in words_ if token not in stopwords_] 
        
        # 특수문자 제거
        words_ = [re.sub(r'([^\s\w/-])+', '', token) for token in words_ if len(token) >= 1] 

        # 숫자문자 제거
        words_ = [re.sub(r'[0-9]+', '', token) for token in words_ if len(token) >= 1] 
        
        # 길이기반 필터링
        words_ = [token for token in words_ if len(token) >= 1] 
        
        after_corpus_.append(words_)
    
    after_corpus_ = [" ".join(i) for i in after_corpus_]
    after_corpus2_.append(after_corpus_)


my_list = []
for i in range(kmeans_model.n_clusters):
    my_list.append(" ".join(after_corpus2_[i]))

#%% df-icf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()
document_term_matrix = vect.fit_transform(my_list)       # 문서-단어 행렬 

tf = pd.DataFrame(document_term_matrix.toarray(), columns=vect.get_feature_names())  
                                             # TF (Term Frequency)
D = len(tf)
cf = tf.astype(bool).sum(axis=0)
icf = np.log((D+1) / (cf+1)) + 1             # IDF (Inverse Document Frequency)

# TF-IDF (Term Frequency-Inverse Document Frequency)
tficf = tf * icf                      
tficf = tficf / np.linalg.norm(tficf, axis=1, keepdims=True)

#%%

df_icf = []
for i in range(kmeans_model.n_clusters):
    df_icf.append(pd.DataFrame(tficf.sort_values(by = i, axis =1, ascending=False).iloc[i,0:10]))