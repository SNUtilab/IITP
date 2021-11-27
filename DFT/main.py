# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 16:16:58 2021

@author: tmlab
"""

if __name__ == '__main__':
    
    import os
    import pickle
    from copy import copy
    from gensim.corpora import Dictionary

    directory = os.path.dirname(os.path.abspath(__file__))
    directory = directory.replace("\\", "/") # window|
    os.chdir(directory)    
    
    #%% phase 1. data laod
    
    print('phase 1. loading and preprocessing data')
    
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
    
    print('phase 2. loading CPC def and embedding')
    
    try :
        # CPC embedding    
        with open( directory+ '/input/CPC_subclass_def.pkl', 'rb') as fr :
            CPC_definition = pickle.load(fr)
    except :
        with open( 'D:/OneDrive - 아주대학교/db/patent/CPC/CPC_subclass_def.pkl', 'rb') as fr :
            CPC_definition = pickle.load(fr)
        
    model = embedding.model
    CPC_dict =  CPC.generate_CPC_dict(data_sample)
    CPC_dict_filtered = CPC.filter_CPC_dict(data_sample, CPC_dict,  CPC_definition)
    encoded_CPC = embedding.CPC_embedding(model, CPC_definition, CPC_dict_filtered)

    texts = data_sample['TAC_keyword']
    
    # document embedding, ready to LDA
    keyword_dct = Dictionary(texts)
    keyword_dct.filter_extremes(no_below = 10, no_above = 0.1)
    keyword_list = keyword_dct.token2id.keys()
    
    corpus = [keyword_dct.doc2bow(text) for text in texts]
    # encoded_keyword = embedding.keyword_embedding(keyword_list)
    
    texts = [[k for k in doc if k in keyword_list] for doc in texts]
    
    docs = [" ".join(i) for i in texts]
    
    encoded_docs = embedding.docs_embedding(model,docs)
    encoded_CPC = embedding.CPC_embedding(model, CPC_definition, CPC_dict_filtered)
    
    #%% phase 3. LDA and embedding 
    import LDA
    
    print('phase 3. LDA and embedding')
    
    try : 
        LDA_parameter = {}
        LDA_parameter['N_topics'] = input("토픽 개수를 입력하세요 : ")
        LDA_parameter['Alpha'] = float(input("파라미터 Alpha를 입력하세요 : "))
        LDA_parameter['Beta'] = float(input("파라미터 Beta를 입력하세요 : "))
        
    # 30, 0.5, 0.1
        lda_model = LDA.lda_model(corpus, keyword_dct, 
                                  LDA_parameter['N_topics'], 
                                      LDA_parameter['Alpha'], 
                                      LDA_parameter['Beta'])
    except : 
        
        lda_model = LDA.lda_model(corpus, keyword_dct, 
                                  30, 
                                  0.5, 
                                  0.1)
    
        
    topic_doc_df = LDA.get_topic_doc(lda_model, corpus)
    encoded_topic = LDA.get_encoded_topic(topic_doc_df, encoded_docs)

    print(encoded_topic)
        
    
    #%% phase 4. LDA result handling
    import LDA
    import pandas as pd
    print('phase 4. Calculate LDA2CPC ')
    
    topic_word_df = LDA.get_topic_word_matrix(lda_model)
    CPC_topic_matrix = LDA.get_CPC_topic_matrix(encoded_CPC, encoded_topic) 
    topic_year_df =  LDA.get_topic_vol_year(lda_model, topic_doc_df, data_sample)
    # standard = 0.55
    # classified_topics = LDA.classifying_topic(CPC_topic_matrix, standard)
    
    volumn_dict = LDA.get_topic_vol(lda_model, corpus)
    CAGR_dict = LDA.get_topic_CAGR(topic_year_df)
    Novelty_dict = LDA.get_topic_novelty(CPC_topic_matrix)    
    CPC_match_dict = LDA.get_topic2CPC(CPC_topic_matrix)
    
    total_df = pd.DataFrame([volumn_dict, CAGR_dict, Novelty_dict, CPC_match_dict]).transpose()
    total_df.columns = ['Volumn', 'CAGR', 'Novelty', 'CPC-match']
    
    print(total_df)
    
    topic2doc_title = LDA.get_most_similar_doc2topic(data_sample, topic_doc_df)

    import xlsxwriter
    import pandas as pd
    # directory = 'C:/Users/tmlab/Desktop/작업공간/'
    writer = pd.ExcelWriter('./output/LDA_results.xlsx', 
                            engine='xlsxwriter')
    
    topic_word_df.to_excel(writer , sheet_name = 'topic_word', index = 1)
    pd.DataFrame(topic_doc_df).to_excel(writer , sheet_name = 'topic_doc', index = 1)
    topic_year_df.to_excel(writer , sheet_name = 'topic_year', index = 1)
    topic2doc_title.to_excel(writer , sheet_name = 'topic_doc_title', index = 1)
    CPC_topic_matrix.to_excel(writer , sheet_name = 'topic2CPC', index = 1)
    total_df.to_excel(writer , sheet_name = 'topic_stats', index = 1)
    

    writer.save()
    writer.close()
    
    
    
    
    #%% phase 5. CPC visualization
    
    import Visualization
    
    Visualization.pchart_CPC_topic(CPC_topic_matrix, [0,1,2,3])
    Visualization.heatmap_CPC_topic(CPC_topic_matrix)
    Visualization.portfolio_CPC_topic(Novelty_dict, CAGR_dict, volumn_dict, CPC_topic_matrix, CPC_match_dict)
#%%
    # #%% test
    # import LDA
    # import numpy as np
    # import matplotlib.pyplot as plt
    
    # # CPC_topic_matrix.apply()
    # standard = np.percentile(CPC_topic_matrix.min(), 90) # 거의 0.9
    # standard = 0.45
    # classified_topics = LDA.classifying_topic(CPC_topic_matrix, standard)
    # novel_topics = [k for k,v in classified_topics.items() if v== 'Novel']
    # temp = topic_word_df[novel_topics]
    
    # # 전체 # 최근접
    # # plt.hist(CPC_topic_matrix.to_numpy().flatten(), bins=100)
    # # plt.hist(CPC_topic_matrix.min().to_numpy().flatten(), bins= 10)
    
    # #%% phase 3. genearte sim matrix
    # import pandas as pd
    # import embedding
    
    # standard = {}
    # # standard['class'] = np.percentile(class_matrix, 95)
    # # standard['subclass'] = np.percentile(subclass_matrix, 95)
    # # standard['group'] = np.percentile(group_matrix, 95)
    # # class_matrix_ = class_matrix.applymap(lambda x : 1 if x > standard['class'] else 0)
    # # subclass_matrix_ = subclass_matrix.applymap(lambda x : 1 if x > standard['subclass'] else 0)
    # # group_matrix_ = group_matrix.applymap(lambda x : 1 if x > standard['group'] else 0)
    
    # #%%
    
    # import embedding
    
    # word_cls_df = pd.DataFrame()
    
    # for matrix in [class_matrix_, subclass_matrix_, group_matrix_] :
    #     DICT = embedding.classify_keyword(matrix)
    #     word_cls_df = word_cls_df.append(DICT, ignore_index=1)
        
    # word_cls_df = word_cls_df.transpose()    
    # word_cls_df.columns = ['class', 'subclass' , 'group']
    #%% phase 4. classifying keyword
    
    
    
    #%% phase A-1. LDA tunning and modelling
    
    # import LDA
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # import numpy as np 
    
    # if os.path.isfile(directory + '/lda_tuning_results.csv') :
    #     tunning_results = pd.read_csv(directory + '/lda_tuning_results.csv')
    # else :
    #     tunning_results = LDA.tunning(texts, keyword_dct, corpus)
    #     tunning_results.to_csv(directory + '/lda_tuning_results.csv', index=False)
        
    # # plotting tunned    
    # temp = tunning_results.groupby('Topics').mean()
    # plt.plot(temp['U_mass'])
    
    #%% test
    
    # import matplotlib.pyplot as plt
    # import numpy as np 
    
    # temp = embedding.get_sim_dist(encoded_CPC['G05B'],encoded_keyword)
    
    # plt.hist(temp, bins=50)

    # plt.axvline(np.percentile(temp, 90), color = 'red')  # Q1
    # plt.show()


