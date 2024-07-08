# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 13:45:13 2021

@author: tkdgu
"""

## start
if __name__ == '__main__':
    
    import os
    import pickle
    from copy import copy
    from gensim.corpora import Dictionary
    import sys

    directory = os.path.dirname(os.path.abspath(__file__))
    directory = directory.replace("\\", "/") # window|
    os.chdir(os.path.dirname(directory)) 
    #%%
    sys.path.append(os.path.dirname(directory)+'/submodule/')
    
    print('phase 1. loading and preprocessing data')
#%%    
    import data_preprocessing
    with open('./input/DT_211118.pkl', 'rb') as f :
        data = pickle.load(f)
    
    data_sample = copy(data)
    data_sample = data_preprocessing.initialize(data_sample)
    data_sample = data_preprocessing.filter_by_year(data_sample)
    data_sample = data_preprocessing.filter_by_textsize(data_sample)
    data_sample = data_preprocessing.preprocess_text(data_sample, directory)
   #%% 
    with open('./output/data_prep.pkl', 'wb') as f :
        pickle.dump(data_sample, f)
    
#%%

    import LDA_tunning
    
    print('phase 2. LDA')
    
    texts = data_sample['TAC_keyword']
    
    try : 
        LDA_parameter = {}
        LDA_parameter['N_topics'] = input("토픽 개수를 입력하세요 : ")
        LDA_parameter['Alpha'] = float(input("파라미터 Alpha를 입력하세요 : "))
        LDA_parameter['Beta'] = float(input("파라미터 Beta를 입력하세요 : "))
        
    # 30, 0.5, 0.1
        LDA_obj = LDA_tunning.LDA_obj(texts,
                                  LDA_parameter['N_topics'], 
                                      LDA_parameter['Alpha'], 
                                      LDA_parameter['Beta'])
    except : 
        
        LDA_obj = LDA_tunning.LDA_obj(texts, 30, 0.5,0.1)
            
            
    
    #%%
        
    with open( directory+ '/output/LDA_obj.pkl', 'wb') as f :
        pickle.dump(LDA_obj, f)
    
    # pickle.dump(lda_model, './output/lda_model.pkl')
    # lda_model.save("./output/lda_model")
        
    
    
    
