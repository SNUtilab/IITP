# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 16:16:58 2021

@author: tmlab
"""


if __name__ == '__main__':
    
    import os
    import sys
    import pickle
    from copy import copy
    
    directory = os.path.dirname(os.path.abspath(__file__))
    directory = directory.replace("\\", "/") # window|
    os.chdir(directory)    
    
    
    #%% phase 1. data laod
    
    import data_preprocessing
    with open( directory+ '/input/DT_211118.pkl', 'rb') as fr :
        data = pickle.load(fr)
    
    data_sample = copy(data)
    data_sample = data_preprocessing.initialize(data_sample)
    data_sample = data_preprocessing.filter_by_year(data_sample)
    data_sample = data_preprocessing.filter_by_textsize(data_sample)
    data_sample = data_preprocessing.preprocess_text(data_sample, directory)
    
    #%% phase 2. embedding
    
    import embedding
    import CPC
    from gensim.corpora import Dictionary
    
    try :
        # CPC embedding    
        with open( directory+ '/input/CPC_subclass_def.pkl', 'rb') as fr :
            CPC_definition = pickle.load(fr)
    except :
        with open( 'D:/OneDrive - 아주대학교/db/patent/CPC/CPC_subclass_def.pkl', 'rb') as fr :
            CPC_definition = pickle.load(fr)


    CPC_dict =  CPC.generate_CPC_dict(data_sample)
    CPC_dict_filtered = CPC.filter_CPC_dict(data_sample, CPC_dict,  CPC_definition)
    encoded_CPC = embedding.CPC_embedding(CPC_definition, CPC_dict_filtered)
    
    texts = data_sample['TAC_keyword']
    
    # document embedding, ready to LDA
    keyword_dct = Dictionary(texts)
    keyword_dct.filter_extremes(no_below = 10, no_above = 0.1)
    keyword_list = keyword_dct.token2id.keys()
    
    corpus = [keyword_dct.doc2bow(text) for text in texts]
    # encoded_keyword = embedding.keyword_embedding(keyword_list)
    
    texts = [[k for k in doc if k in keyword_list] for doc in texts]
    
    docs = [" ".join(i) for i in texts]
    encoded_docs = embedding.docs_embedding(docs)
    
    #%% phase 3. LDA tunning and modelling
    
    import LDA
    import pandas as pd
    
    if os.path.isfile(directory + '/lda_tuning_results.csv') :
        tunning_results = pd.read_csv(directory + '/lda_tuning_results.csv')
    else :
        tunning_results = LDA.tunning(texts, keyword_dct, corpus)
        tunning_results.to_csv(directory + '/lda_tuning_results.csv', index=False)
    
    lda_model = LDA.model_by_tunning(tunning_results, corpus, keyword_dct)
    
    #%% phase 4. Find novelty topic 
    import LDA
    
    topic_word_df = LDA.get_topic_word_matrix(lda_model)
    topic_doc_df = LDA.get_topic_doc(lda_model, corpus)
    encoded_topic = LDA.get_encoded_topic(topic_doc_df, encoded_docs)
    CPC_topic_matrix = LDA.get_CPC_topic_matrix(encoded_CPC, encoded_topic) 

    
    
    #%% test
    import LDA
    import numpy as np
    import matplotlib.pyplot as plt
    
    # CPC_topic_matrix.apply()
    standard = np.percentile(CPC_topic_matrix.min(), 80) # 거의 0.9
    standard = 0.9
    classified_topics = LDA.classifying_topic(CPC_topic_matrix, standard)

    novel_topics = [k for k,v in classified_topics.items() if v== 'Novel']
    
    temp = topic_word_df[novel_topics]
    # 전체
    # plt.hist(CPC_topic_matrix.to_numpy().flatten(), bins=100)
    
    # 최근접
    # plt.hist(CPC_topic_matrix.min().to_numpy().flatten(), bins= 10)
    
    
    #%% phase 3. genearte sim matrix
    import pandas as pd
    
    import embedding
    
    standard = {}
    standard['class'] = np.percentile(class_matrix, 95)
    standard['subclass'] = np.percentile(subclass_matrix, 95)
    standard['group'] = np.percentile(group_matrix, 95)
    
    class_matrix_ = class_matrix.applymap(lambda x : 1 if x > standard['class'] else 0)
    subclass_matrix_ = subclass_matrix.applymap(lambda x : 1 if x > standard['subclass'] else 0)
    group_matrix_ = group_matrix.applymap(lambda x : 1 if x > standard['group'] else 0)
    
    #%%
    
    import embedding
    
    word_cls_df = pd.DataFrame()
    
    for matrix in [class_matrix_, subclass_matrix_, group_matrix_] :
        DICT = embedding.classify_keyword(matrix)
        word_cls_df = word_cls_df.append(DICT, ignore_index=1)
        
    word_cls_df = word_cls_df.transpose()    
    word_cls_df.columns = ['class', 'subclass' , 'group']
    #%% phase 4. classifying keyword
    
    
    
    
    #%% test
    
    import matplotlib.pyplot as plt
    import numpy as np 
    
    temp = embedding.get_sim_dist(encoded_CPC['G05B'],encoded_keyword)
    
    plt.hist(temp, bins=50)

    plt.axvline(np.percentile(temp, 90), color = 'red')  # Q1
    plt.show()


