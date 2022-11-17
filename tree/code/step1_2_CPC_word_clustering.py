# -*- coding: utf-8 -*-

#%% 1. pac & data load

import pandas as pd
import numpy as np
import re
import pickle
import random
import os
import sys


directory = 'C:/Users/tech_tree/'
sys.path.append('C:/Users/tech_tree/최종 코드 작성(module)')


os.chdir(directory)
df = pd.read_excel(directory +'furniture_data.xlsx')

df = pd.DataFrame(df)


#%%
df = df.rename(columns = {'번호': 'patent_key', '명칭(원문)': 'Title', '요약(원문)': 'Abstract', '전체 청구항': 'Claim', '발명자수': 'num_inventor',
                     '전체 청구항수': 'num_claim', '자국피인용횟수': 'num_citation', '출원일': 'application_year'})

df['year'] = pd.DataFrame(df['application_year'].str.split('.').tolist())[0]

# CPC code 없는 행 제거
df = df.dropna(subset=['공통특허분류'], inplace =True)

#%%
import re

from nltk.stem import WordNetLemmatizer
import spacy
import nltk
from nltk.corpus import stopwords


n = WordNetLemmatizer()

sp = spacy.load('en_core_web_sm')

#nltk.download("stopwords")
#stopwords_ = set(stopwords.words("english") + ["news", "new", "invention"])

stopwords_ = sp.Defaults.stop_words


def data_process(df):
    
    #df['Abstract'] = df['Abstract'].apply(lambda x : " ".join(nltk.sent_tokenize(x)[1:]))
    df['Abstract'] = df['Abstract'].apply(lambda x : " ".join(nltk.sent_tokenize(x)[:]))
    
    df['Claim'] = df['Claim'].apply(lambda x : " ".join(nltk.sent_tokenize(x)[1:]))
    
    #corpus_ = df.apply(lambda x : x.Title +' '+ x.Abstract , axis = 1).tolist()    
    
    #corpus: Abstract + claim
    corpus_ = df.apply(lambda x : x.Title +' '+ x.Abstract +' ' + x.Claim , axis = 1).tolist()  
    #corpus_ = df.apply(lambda x : x.Abstract +' ' + x.Claim , axis = 1).tolist()  
    after_corpus_ = []
    
    for doc in corpus_ :
        
        words_ = doc.strip().split()
    
        # 소문자화
        words_ = [i.lower() for i in words_]
        
        # lemmatize
        words_ = [n.lemmatize(token, 'v') for token in words_ if len(token) >= 1] 
    
        # lemmatize
        words_ = [n.lemmatize(token, 'n') for token in words_ if len(token) >= 1] 
        
        # delete stopwords
        words_ = [token for token in words_ if token not in stopwords_] 
        
        # 특수문자 제거
        words_ = [re.sub(r'([^\s\w/-])+', '', token) for token in words_ if len(token) >= 1] 
        
        # 길이기반 필터링
        words_ = [token for token in words_ if len(token) > 3] 
        
        after_corpus_.append(words_)
        
    after_corpus_ = [" ".join(i) for i in after_corpus_]
    
    return after_corpus_

corpus = data_process(df)

#%% TF-idf
docs = corpus

vocab = list(set(w for doc in docs for w in doc.split()))
vocab.sort()

N = len(docs) 

from math import log

def tf(t, d):
  return d.count(t)

def idf(t):
  df = 0
  for doc in docs:
    df += t in doc
  return log(N/(df+1))

def tfidf(t, d):
  return tf(t,d)* idf(t)


result_tf = []

# 각 문서에 대해서 아래 연산을 반복
for i in range(N):
  result_tf.append([])
  d = docs[i]
  for j in range(len(vocab)):
    t = vocab[j]
    result_tf[-1].append(tf(t, d))

tf_ = pd.DataFrame(result_tf, columns = vocab)


result_idf = []
for j in range(len(vocab)):
    t = vocab[j]
    result_idf.append(idf(t))

idf_ = pd.DataFrame(result_idf, index=vocab, columns=["IDF"])




result_tfidf = []
tfidf_ = []
for i in range(len(tf_.columns)):
    x = np.array(tf_[tf_.columns[i]]) * np.array(idf_.loc[idf_.index[i],:])

    tfidf_.append(x)


tfidf_df = pd.DataFrame(tfidf_, index = tf_.columns)

tfidf_df = tfidf_df.T


# TF-idf result cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
import math

similarity_ = cosine_similarity(np.array(tfidf_df), np.array(tfidf_df))
print(similarity_)

similarity_tfidf_df = pd.DataFrame(similarity_)


# 벡터 희소항 제거
col_list = []
for i in range(len(tfidf_df.columns)):
    if len(tfidf_df[tfidf_df.iloc[:,i] > 0]) < 30:
        col_list.append(tfidf_df.columns[i])
    
    
tfidf_vec_df_ = tfidf_df.drop(columns = col_list, axis = 1)

similarity_tfidf = cosine_similarity(np.array(tfidf_vec_df_), np.array(tfidf_vec_df_))


similarity_tfidf_df_ = pd.DataFrame(similarity_tfidf)
print(similarity_tfidf)

#%% CPC vector 생성

result_cpc_list = []
for i in df['공통특허분류'].str.split(',').tolist():
    cpc_ex = [x.strip() for x in i]
    result_cpc_list.append(cpc_ex)
    
    
# CPC sub_group 전체 리스트
sub_group_vocab = list(set(sum(result_cpc_list, [])))



df['sub_group_cpc'] = result_cpc_list

# tf_cpc 
def tf(t, d):
  return d.count(t)

def tf_cpc(docs, vocab):
    
    result = []
    
    # 각 문서에 대해서 아래 연산을 반복
    for i in range(len(docs)):
      result.append([])
      d = docs[i]
      for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tf(t, d))
    
    tf_ = pd.DataFrame(result, columns = vocab)
    
    return tf_

# module 별 CPC 분포 count
def cpc_distr(cpc_df, cpc_level):
    
    x = sum(list(cpc_df[cpc_level]), [])
    
    result_ = pd.DataFrame(x).value_counts().rename_axis('cpc').reset_index(name='counts')
    
    return result_




# CPC sub_group 전체 리스트에서 각 문서에 해당하는 sub_group count
sub_group = tf_cpc(result_cpc_list, sub_group_vocab)


#  CPC clustering cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
import math



sub_group_similarity_ = cosine_similarity(np.array(sub_group), np.array(sub_group))
print(sub_group_similarity_)


similarity_sub_group_df = pd.DataFrame(sub_group_similarity_)


#%% CPC x tfidf 결합


sub_group_mul = similarity_sub_group_df * similarity_tfidf_df

sub_group_plus = similarity_sub_group_df + similarity_tfidf_df


#%% Spherical K Means clustering
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix

from soyclustering import SphericalKMeans

x_list = [sub_group_mul, sub_group_plus]

csr_list = []

for i in x_list:
    x = np.asarray(i)
    csr_list.append(csr_matrix(x))
    
    
#%% # Inertia(군집 내 거리제곱합의 합) value (적정 군집수)

ks = range(3,15) 
inertias = [] 

for k in ks: 
    model = SphericalKMeans(
    n_clusters=k,
    max_iter=30,
    verbose=1,
    init='similar_cut',
    sparsity='minimum_df',
    minimum_df_factor=0.05
)
    model.fit_predict(csr_list[1]) 
    
    inertias.append(model.inertia_) 


# Plot ks vs inertias 
plt.figure(figsize=(4, 4)) 
plt.plot(ks, inertias, '-o') 
plt.xlabel('number of clusters, k') 
plt.ylabel('inertia') 
plt.xticks(ks) 
plt.show()

#%% 첫점과 끝점 거리 계산
class Point2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def dist(P, A, B):
    area = abs ( (A.x - P.x) * (B.y - P.y) - (A.y - P.y) * (B.x - P.x) )
    AB = ( (A.x - B.x) ** 2 + (A.y - B.y) ** 2 ) ** 0.5
    return ( area / AB )        

start = Point2D(x = 3, y = inertias[0]) # 시작점
end = Point2D(x = 15, y = inertias[len(inertias)-1]) # 끝점
 
print('start: {} {}'.format(start.x, start.y))
print('end: {} {}'.format(end.x, end.y))

#%% 거리 계산
distances = []

for i in range(0,len(inertias)):
    distances.append(dist(Point2D(x = i+1, y = inertias[i]), start, end))
print(distances) 

#%% model
spherical_kmeans = SphericalKMeans(
    n_clusters=9,
    max_iter=30,
    verbose=1,
    init='similar_cut',
    sparsity='minimum_df',
    minimum_df_factor=0.05
)


labels_subclass = spherical_kmeans.fit_predict(csr_list[1])


df_clu = df

df_clu['labels_'] = labels_subclass
#df_clu['labels_subcgroup'] = labels_subcgroup



print(df_clu.groupby(df_clu['labels_']).count())


dfs = []
for i in range(len(set(df_clu['labels_']))):
    condition = df_clu['labels_'] == i
    dfs.append(df_clu[condition])
    
cluster_list_cpc = [cpc_distr(x, 'sub_group') for x in dfs]