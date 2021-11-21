# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 16:44:19 2021

@author: tmlab
"""
import pandas as pd
import numpy as np
from collections import Counter 
from spacy.cli import download

import nltk
from nltk.corpus import stopwords    
import spacy
import re

def initialize(df):
    
    df = df
    df['TAC'] = df['title'] + ' ' + df['abstract'] + ' ' + df['claims_rep']
    df['year'] = df['date'].apply(lambda x : x[0:4])
    
    df['cpc_class'] = ''
    df['cpc_subclass'] = ''
    df['cpc_group'] = ''

    for idx,row in df.iterrows() :
        
        print(idx)
        
        cpc_list = df['cpc_list'][idx]
        df['cpc_group'][idx] = [i for i in cpc_list if len(i) > 5]
        
        df['cpc_class'][idx] = [i for i in cpc_list if len(i) == 3]
        
        df['cpc_subclass'][idx] = [i for i in cpc_list if len(i) == 4]
        
    
    return (df)
    
def filter_by_year(df, col = 'year', MIN = 30) :
    
    c = Counter(df[col]) #2016 ~ 2020
    year_list = [k for k,v in c.items() if v >= MIN]
    
    df = df.loc[df[col] >= min(year_list) , :].reset_index(drop = 1)
    
    return(df)

def filter_by_textsize(df, col = 'TAC', MIN = 100) :
    
    df = df.loc[df['TAC'].str.split().str.len() >= 100 , :].reset_index(drop = 1)
    
    return(df)

def preprocess_text(df, directory) :
    
    nltk.download('stopwords')
    stopwords_nltk = set(stopwords.words('english'))
    download('en_core_web_sm')
    
    nlp = spacy.load("en_core_web_sm")
    with open(directory + '/input/stopwords_uspto.txt') as f:
        stopwords_uspto = [line.rstrip() for line in f]
    stopwords_uspto.append('-PRON-')
    
    df['TAC_nlp'] = [nlp(i) for i in df['TAC']]
    
    # get keyword
    df['TAC_keyword'] = [[token.lemma_.lower() for token in doc] for doc in df['TAC_nlp']] # lemma
    df['TAC_keyword'] = [[token for token in doc if len(token) > 2] for doc in df['TAC_keyword']] # 길이기반 제거
    df['TAC_keyword'] = [[token for token in doc if not token.isdigit() ] for doc in df['TAC_keyword']]  #숫자제거 
    df['TAC_keyword'] = [[re.sub(r"[^a-zA-Z0-9-]","",token) for token in doc ] for doc in df['TAC_keyword']] #특수문자 교체    
    df['TAC_keyword'] = [[token for token in doc if len(token) > 2] for doc in df['TAC_keyword']] # 길이기반 제거
    df['TAC_keyword'] = [[token for token in doc if token not in stopwords_uspto] for doc in df['TAC_keyword']] # 길이기반 제거
    df['TAC_keyword'] = [[token for token in doc if token not in stopwords_nltk] for doc in df['TAC_keyword']] # 길이기반 제거
    
    
    return(df)
