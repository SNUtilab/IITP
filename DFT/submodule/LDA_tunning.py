# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 02:59:32 2021

@author: tkdgu
"""

from gensim.models.ldamulticore import LdaMulticore
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary

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


class LDA_obj() :
    
    def __init__(self,  texts, n_topics, alpha, beta):
    
        # document embedding, ready to LDA
        self.texts = texts
        self.keyword_dct = Dictionary(self.texts)
        self.keyword_dct.filter_extremes(no_below = 10, no_above = 0.1)
        self.keyword_list = list(self.keyword_dct.token2id.keys())
        
        self.corpus = [self.keyword_dct.doc2bow(text) for text in self.texts]
        # encoded_keyword = embedding.keyword_embedding(keyword_list)
        
        self.texts = [[k for k in doc if k in self.keyword_list] for doc in self.texts]
        
        self.docs = [" ".join(i) for i in self.texts]
        
        self.model = lda_model(self.corpus, self.keyword_dct, n_topics, alpha, beta)    
        

    
