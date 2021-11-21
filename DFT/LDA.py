# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 02:59:32 2021

@author: tkdgu
"""

from gensim.models.ldamulticore import LdaMulticore
from gensim.models import CoherenceModel
import numpy as np
import pandas as pd

def compute_coherence_values(corpus, dictionary, texts, k, a, b):
    
    lda_model = LdaMulticore(corpus=corpus,
                             id2word=dictionary,
                             num_topics=k, 
                             random_state=100,
                             chunksize=100,
                             passes=10,
                             alpha=a,
                             eta=b,
                             )
    
    coherence_model_lda = CoherenceModel(model=lda_model, 
                                         texts=texts, 
                                         dictionary=dictionary, 
                                         coherence='u_mass')
    
    return coherence_model_lda.get_coherence()

def tunning(texts, dct, corpus) :
    
    grid = {}
    grid['Validation_Set'] = {}
    
    # Topics range  #수정
    min_topics = 5
    max_topics = 101
    step_size = 5
    topics_range = range(min_topics, max_topics, step_size)
    
    # Alpha parameter
    alpha = list(np.arange(0.01, 1, 0.3))
    # alpha = [0.01, 0.05, 0.1, 1]
    alpha.append('symmetric')
    alpha.append('asymmetric')
    
    # Beta parameter
    beta = list(np.arange(0.01, 1, 0.3))
    # beta = [0.01, 0.05, 0.1, 1]
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
                     'Coherence': []
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
                        cv = compute_coherence_values(corpus=corpus_sets[i], 
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
                        model_results['Coherence'].append(cv)
                        cnt +=1
                        print("전체 {} 중에서 {} ".format(len(alpha) *len(beta) *len(topics_range),cnt))
                        
    return(pd.DataFrame(model_results))
        
    
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