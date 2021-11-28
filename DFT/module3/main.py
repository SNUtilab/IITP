# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 17:25:39 2021

@author: tkdgu
"""

if __name__ == '__main__':
    
    import os
    import sys
    import pickle
    from copy import copy
    from gensim.corpora import Dictionary
    import LDA_handling
    
    directory = os.path.dirname(os.path.abspath(__file__))
    directory = directory.replace("\\", "/") # window|
    os.chdir(os.path.dirname(directory))    
    sys.path.append(os.path.dirname(directory)+'/submodule/')

    with open('./input/LDA_obj.pkl', 'rb') as f :
        LDA_obj = pickle.load(f)
        
    with open('./input/encoded_CPC.pkl', 'rb') as f :
        encoded_CPC = pickle.load(f)
        
    with open('./input/encoded_topic.pkl', 'rb') as f :
        encoded_topic = pickle.load(f)
        
    with open('./input/data_prep.pkl', 'rb') as f :
        data_sample = pickle.load(f)
            
        

    #%% phase 4. LDA result handling
    
    import pandas as pd
    
    print('phase 4. Calculate LDA2CPC ')
    
    topic_doc_df = LDA_handling.get_topic_doc(LDA_obj.model, LDA_obj.corpus)
    
    topic_word_df = LDA_handling.get_topic_word_matrix(LDA_obj.model)
    CPC_topic_matrix = LDA_handling.get_CPC_topic_matrix(encoded_CPC, encoded_topic)     
    topic_year_df =  LDA_handling.get_topic_vol_year(LDA_obj.model, topic_doc_df, data_sample)
    #%%
    volumn_dict = LDA_handling.get_topic_vol(LDA_obj.model, LDA_obj.corpus)

    CAGR_dict = LDA_handling.get_topic_CAGR(topic_year_df)
    Novelty_dict = LDA_handling.get_topic_novelty(CPC_topic_matrix)    
    CPC_match_dict = LDA_handling.get_topic2CPC(CPC_topic_matrix)    


    
    #%%
    
    # standard = 0.55
    # classified_topics = LDA.classifying_topic(CPC_topic_matrix, standard)
    
    
    total_df = pd.DataFrame([volumn_dict, CAGR_dict, Novelty_dict, CPC_match_dict]).transpose()
    total_df.columns = ['Volumn', 'CAGR', 'Novelty', 'CPC-match']
    
    print(total_df)
    
    topic2doc_title = LDA_handling.get_most_similar_doc2topic(data_sample, topic_doc_df)
    #%%
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