# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 02:59:32 2021

@author: tkdgu
"""

from gensim.models.ldamulticore import LdaMulticore
from gensim.models import CoherenceModel
import numpy as np
import pandas as pd

def compute_coherence_values(corpus, dictionary, texts, k, a, b, method = "u_mass"):
    
    result = {}
    lda_model = LdaMulticore(corpus=corpus,
                             id2word=dictionary,
                             num_topics=k, 
                             random_state=100,
                             chunksize=100,
                             passes=10,
                             alpha=a,
                             eta=b,
                             )
    
    result['perplexity'] = lda_model.log_perplexity(corpus)
    for method in ['u_mass', 'c_v', 'c_uci', 'c_npmi'] :
        coherence_model_lda = CoherenceModel(model=lda_model, 
                                             texts=texts, 
                                             dictionary=dictionary, 
                                             coherence= method)
        result[method] = coherence_model_lda.get_coherence()
        
    return result

def tunning(texts, dct, corpus) :
    
    grid = {}
    grid['Validation_Set'] = {}
    
    # Topics range  #수정
    min_topics = 5
    max_topics = 101
    step_size = 5
    topics_range = range(min_topics, max_topics, step_size)
    
    # Alpha parameter
    # alpha = list(np.linspace(0.01, 1, 0.3))
    alpha = [0.01, 0.1, 0.5, 1]
    alpha.append('symmetric')
    alpha.append('asymmetric')
    
    # Beta parameter
    # beta = list(np.arange(0.01, 1, 0.3))
    beta = [0.01, 0.1, 0.5, 1]
    beta.append('symmetric')
    
    # Validation sets
    # num_of_docs = len(corpus)
    corpus_sets = [# gensim.utils.ClippedCorpus(corpus, num_of_docs*0.25), 
                   # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.5), 
                   # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.75), 
                   corpus]
    
    corpus_title = ['100% Corpus']
    
    model_results = {'Validation_Set': [],
                     'Topics': [],
                     'Alpha': [],
                     'Beta': [],
                     'Perplexity': [],
                     'U_mass' : [],
                     'C_v' : [],
                     'C_uci' : [],
                     'C_npmi' : [],
                     
                    }
    # Can take a long time to run
    if 1 == 1:
        
        cnt = 0
        
        # iterate through validation corpuses
        for i in range(len(corpus_sets)):
            # iterate through number of topics
            for k in topics_range:
                # iterate through alpha values
                for a in alpha:
                    # iterare through beta values
                    for b in beta:
                        # get the coherence score for the given parameters
                        result = compute_coherence_values(corpus=corpus_sets[i], 
                                                      dictionary=dct,
                                                      texts = texts,
                                                      k=k, 
                                                      a=a, 
                                                      b=b)
                        
                        
                        # Save the model results
                        model_results['Validation_Set'].append(corpus_title[i])
                        model_results['Topics'].append(k)
                        model_results['Alpha'].append(a)
                        model_results['Beta'].append(b)
                        model_results['Perplexity'].append(result['perplexity'])
                        model_results['U_mass'].append(result['u_mass'])
                        model_results['C_v'].append(result['c_v'])
                        model_results['C_uci'].append(result['c_uci'])
                        model_results['C_npmi'].append(result['c_npmi'])
                        
                        
                        cnt +=1
                        print("전체 {} 중에서 {} ".format(len(alpha) *len(beta) *len(topics_range),cnt))
                        
    return(pd.DataFrame(model_results))
def lda_model(corpus, dct, Topics, Alpha, Beta) :
    
    lda_model = LdaMulticore(corpus=corpus,
                             id2word=dct,
                             num_topics= Topics, 
                             random_state=100,
                             chunksize=100,
                             passes=10,
                             alpha= Alpha,
                             eta= Beta,
                             )
    
    return(lda_model)
    
    
def model_by_tunning(tunning_results, corpus, dct) :
    
    index = tunning_results['Coherence'].idxmax()
    
    Alpha = round(float(tunning_results['Alpha'][index]),  2)
    Beta = round(float(tunning_results['Beta'][index]), 2)
    Topics = tunning_results['Topics'][index]
    
    lda_model = LdaMulticore(corpus=corpus,
                             id2word=dct,
                             num_topics= Topics, 
                             random_state=100,
                             chunksize=100,
                             passes=10,
                             alpha= Alpha,
                             eta= Beta,
                             )
    
    return(lda_model)
    

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
    
    
def get_encoded_topic(topic_doc_df, encoded_docs) :
    
    x = np.linalg.lstsq(topic_doc_df, encoded_docs, rcond = -1)
    encoded_topic = x[0]
    
    return(encoded_topic)

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