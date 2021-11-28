# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 02:59:32 2021

@author: tkdgu
"""

from gensim.models.ldamulticore import LdaMulticore
from gensim.models import CoherenceModel
import numpy as np
import pandas as pd


def get_topic_doc(lda_model, corpus) :
    
    topic_doc_df = pd.DataFrame(columns = range(0, lda_model.num_topics))
    
    for corp in corpus :
        
        temp = lda_model.get_document_topics(corp)
        DICT = {}
        for tup in temp :
            DICT[tup[0]] = tup[1]
        
        topic_doc_df = topic_doc_df.append(DICT, ignore_index=1)
    topic_doc_df = np.array(topic_doc_df)
    topic_doc_df = np.nan_to_num(topic_doc_df)
    
    
    return(topic_doc_df)


def get_topic_word_matrix(lda_model) :
    
    topic_word_df = pd.DataFrame()
    
    for i in range(0, lda_model.num_topics) :
        temp = lda_model.show_topic(i, 1000)
        DICT = {}
        for tup in temp :
            DICT[tup[0]] = tup[1]
            
        topic_word_df = topic_word_df.append(DICT, ignore_index =1)
        
    topic_word_df = topic_word_df.transpose()
    
    return(topic_word_df)
    
    

def cosine(u, v):
    return (np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

def get_CPC_topic_matrix(encoded_CPC, encoded_topic) :
    
    CPC_topic_matrix = pd.DataFrame(columns = range(0, encoded_topic.shape[0]), index = encoded_CPC.keys())
    
    for topic in range(0, encoded_topic.shape[0]) :
        
        for cpc in encoded_CPC.keys() :
            cpc_embedding = encoded_CPC[cpc]
            sim = cosine(encoded_topic[topic], cpc_embedding)
            
            CPC_topic_matrix[topic][cpc] =  sim
        
    
    return CPC_topic_matrix

def get_topic_novelty(CPC_topic_matrix) :
    
    result_dict = {}
    
    for topic, max_value in enumerate(CPC_topic_matrix.max()) :
        
        result_dict[topic] = 1/max_value
        
    return(result_dict)
        
    
def classifying_topic(CPC_topic_matrix, standard) :
    
    result_dict = {}
    
    for topic, max_value in enumerate(CPC_topic_matrix.max()) :
        
        if max_value <= standard :
            result_dict[topic] = 'Novel'
        else : 
            result_dict[topic] = 'Common'
            
    return(result_dict)
        
def get_topic_vol(lda_model, corpus) :

    topic_doc_df = pd.DataFrame(columns = range(0, lda_model.num_topics))
    
    for corp in corpus :
        
        temp = lda_model.get_document_topics(corp)
        DICT = {}
        for tup in temp :
            DICT[tup[0]] = tup[1]
        
        topic_doc_df = topic_doc_df.append(DICT, ignore_index=1)
    
    result = topic_doc_df.apply(np.sum).to_dict()
    
    return(result)
    
def get_topic_vol_year(lda_model, topic_doc_df, data_sample) :
    
    topic_doc_df = pd.DataFrame(topic_doc_df)
    topic_doc_df['year'] = data_sample['year']
    topic_doc_df['title'] = data_sample['title']
    
    topic_year_df = pd.DataFrame()
    for col in range(0, lda_model.num_topics) :
        grouped = topic_doc_df[col].groupby(topic_doc_df['year'])
        DICT = grouped.sum()
        topic_year_df = topic_year_df.append(DICT, ignore_index=1)
    
    topic_year_df = topic_year_df.transpose()
    
    return(topic_year_df)
#     return(result_df)
# # def get_topic_title(topic_doc_df, data_sample) :
    
    
def get_topic_CAGR(topic_year_df) :
    
    st_year = min(topic_year_df.index)
    ed_year = 2020 # 2020 fix
    
    duration = int(ed_year) - int(st_year)
    
    result = {}
    
    for col in topic_year_df :
        st_val = topic_year_df[col][0]
        ed_val = topic_year_df[col][duration]
        CAGR = (ed_val/st_val)**(1/duration) -1
        result[col] = CAGR
    
    return(result)
    

def get_topic2CPC(CPC_topic_matrix) :
    
    result_dict = {}
    
    for col in CPC_topic_matrix.columns :
        
        result_dict[col] = pd.to_numeric(CPC_topic_matrix[col]).idxmax()
    
    return(result_dict)

def get_most_similar_doc2topic(data_sample, topic_doc_df) :
    
    result_df = pd.DataFrame()
    
    for col in range(topic_doc_df.shape[1]) :
        
        DICT = {}
        idx = topic_doc_df[col].argmax()
        value = topic_doc_df[col].max()
        DICT['title'] = data_sample['title'][idx]
        DICT['year'] = data_sample['year'][idx]
        DICT['similarity'] = value
        
        result_df = result_df.append(DICT, ignore_index=1)
    
    return(result_df)