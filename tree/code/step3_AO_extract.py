# -*- coding: utf-8 -*-


#%% extract SA keyword relation
from spacy.symbols import nsubj, VERB, NOUN
import pandas as pd
import spacy
#pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
import en_core_web_sm
#nlp = en_core_web_sm.load()

nlp = spacy.load("en_core_web_sm")
# nlp = spacy.load("en")



def ao_extract(df,num_cluster):
    
    function_df = pd.DataFrame()

    stopwords = ['comprise', 'have', 'include',
                 'invention','say', 'relate','be',
                 'portion', 'configure', 'use', 
                 'method', 'effect', '-'
                 ,'disclosure']
    
    FROM = 0
    TO = 0
    relation = 0
    pt_id = 0
    dfs_ao = []
    for i in range(num_cluster):
    
        for idx, row in df[i].iterrows() :
            
            pt_id = row['patent_key']
            num_claim = row['num_claim']
            num_citation = row['num_citation']
            num_inventor = row['num_inventor']
            for source in ['Title', 'Abstract', 'Claim'] :
                
                doc = nlp(row[source])
                
                for possible_verb in doc:
                    
                    if possible_verb.pos == VERB:
                
                        tokens = [token for token in possible_verb.children if token.pos == NOUN]
                        if possible_verb.lemma_  in stopwords : continue
                        
                        for token in tokens :
                            if token.lemma_ in stopwords : continue
                        
                            #if token.dep_ == 'nsubj' :
                                
                                # case 1
                             #    relation = 'SA'
                             #    FROM = token.lemma_ 
                             #    TO = possible_verb.lemma_ 
                                
                            if token.dep_ == 'nsubjpass' :
                                
                                # case 2
                                relation = 'AO'
                                FROM = possible_verb.lemma_
                                TO = token.lemma_ 
                                
                            if 'obj' in token.dep_ : 
                                # case 3
                                relation = 'AO'
                                FROM = possible_verb.lemma_
                                TO = token.lemma_ 
                            
                            function_df = function_df.append({"From" : FROM,
                                                              'To' : TO,
                                                              'Realtion' : relation,
                                                              'PtNumber' : pt_id,
                                                              'num_claim' : num_claim,
                                                              'num_citation' : num_citation,
                                                              'num_inventor' : num_inventor,                                                              
                                                              'Source' : source}, ignore_index=1).reset_index(drop =1)
                            
        dfs_ao.append(function_df)
        function_df = pd.DataFrame()
        

    return dfs_ao




dfs_ao = ao_extract(dfs, len(dfs))


for i in range(len(dfs_ao)):
    dfs_ao[i] = dfs_ao[i].drop(dfs_ao[i][dfs_ao[i]['Realtion'] == 0].index)





