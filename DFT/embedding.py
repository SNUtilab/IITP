# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 17:31:06 2021

@author: tkdgu
"""
from collections import Counter 
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def get_CPC_Counter(df,col) :
    cpc_list = df[col].tolist()
    cpc_list = sum(cpc_list, [])
    c = Counter(cpc_list)
    # c =  {k: v for k, v in sorted(c.items(), key=lambda item: item[1], reverse=True)}
    return(c)

def generate_CPC_dict(df) :
    
    CPC_dict = {}
    
    CPC_dict['cpc_class'] = get_CPC_Counter(df, 'cpc_class')
    CPC_dict['cpc_subclass'] = get_CPC_Counter(df,'cpc_subclass')
    CPC_dict['cpc_group'] = get_CPC_Counter(df,'cpc_group')
    
    return(CPC_dict)
    
    
def filter_CPC_dict(df, CPC_dict, CPC_definition):
    
    cpc_class = CPC_dict['cpc_class']
    cpc_subclass = CPC_dict['cpc_subclass']
    cpc_group = CPC_dict['cpc_group']
    
    CPC_dict_filtered ={}
    
    class_list = [k for k,v in cpc_class.items() if v >= len(df) * 0.05]
    class_list = [i for i in class_list if i in CPC_definition.keys()]
    CPC_dict_filtered['class_list'] = class_list
    
    subclass_list = [k for k,v in cpc_subclass.items() if v >= len(df) * 0.025]
    subclass_list = [i for i in subclass_list if i[0:-1] in class_list]
    subclass_list = [i for i in subclass_list if i in CPC_definition.keys()]
    CPC_dict_filtered['subclass_list'] = subclass_list
    
    group_list = [k for k,v in cpc_group.items() if v >= len(df) * 0.0125]
    group_list = [i for i in group_list if i[0:4] in subclass_list]
    group_list = [i for i in group_list if i in CPC_definition.keys()]
    CPC_dict_filtered['group_list'] = group_list
    
    return(CPC_dict_filtered)

def CPC_embedding(CPC_definition, CPC_dict):
    
    class_list = CPC_dict['class_list']
    subclass_list = CPC_dict['subclass_list']
    group_list = CPC_dict['group_list']
    
    encoded_CPC = {}
    
    for i in subclass_list :
        encoded_CPC[i] = model.encode(CPC_definition[i].lower())
        
    for i in class_list :
        encoded_CPC[i] = model.encode(CPC_definition[i].lower())

    for i in group_list :
        encoded_CPC[i] = model.encode(CPC_definition[i].lower())

    return(encoded_CPC)

def keyword_embedding(keyword_list) :
    
    text_list = keyword_list
    
    encoded_text = {}   
        
    for text in keyword_list :
        
        if text in encoded_text.keys() :
            text_embedding = encoded_text[text]
        else :
            text_embedding = model.encode(text)
            encoded_text[text] = text_embedding

    return(encoded_text)


def docs_embedding(docs) :
    
    embedding_result = model.encode(docs)
    
    return(embedding_result)
    
    

def get_sim_dist(encoded_cpc_array, encoded_keyword) :
    sim_list = []    
    
    for k,v in encoded_keyword.items() :
        
        cpc_embedding = encoded_cpc_array
        sim = cosine(v, cpc_embedding)
        sim_list.append(sim)
    
    return(sim_list)

def get_sim_matrix(cpc_list, encoded_CPC, encoded_keyword) :
        
    sim_df = pd.DataFrame(columns = cpc_list, index = encoded_keyword.keys())
    
    for k,v in encoded_keyword.items() :
            
        for cpc in cpc_list :
            cpc_embedding = encoded_CPC[cpc]
            sim = cosine(v, cpc_embedding)
            sim_df[cpc][k] = sim
            
    return(sim_df)

def classify_keyword(sim_matirix) :
    
    DICT = {}
    for idx, row in sim_matirix.iterrows() :
        if all(word == 0 for word in row)  :
            DICT[idx] = 'Unknown'
        else :
            DICT[idx] = 'Known'
            
    return(DICT)



        # for col in sim_matirix.columns : 
            
        
    
    
# def get_standard(sim_matrix):
    
    # standard = {}
    
                # mean = np.mean(sim_list)
                # var = np.var(sim_list)
                # total_df = total_df.append([[mean,var]], ignore_index=1)
            
        # total_df.columns = ['MEAN' , 'VAR']
            
        # MEAN = np.mean(total_df['MEAN'])
        # VAR = np.mean(total_df['VAR'])
        
        # standard[level] = (MEAN, VAR)
            
        # level +=1
            
    # return(total_df)