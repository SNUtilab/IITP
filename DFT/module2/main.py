# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 13:56:02 2021

@author: tkdgu
"""

if __name__ == '__main__':
        
    import os
    import pickle
    from copy import copy
    from gensim.corpora import Dictionary
    import sys
    
    directory = os.path.dirname(os.path.abspath(__file__))
    directory = directory.replace("\\", "/") # window|
    os.chdir(directory)    
    sys.path.append(os.path.dirname(directory)+'/submodule/')
    
#%% phase 2. read CPC AND embedding

    import embedding
    import CPC
    import LDA_handling

    print('phase 2. loading CPC def and embedding')
    
    try :
        # CPC embedding    
        with open( directory+ '/input/CPC_subclass_def.pkl', 'rb') as fr :
            CPC_definition = pickle.load(fr)
    except :
        with open( 'D:/OneDrive - 아주대학교/db/patent/CPC/CPC_subclass_def.pkl', 'rb') as fr :
            CPC_definition = pickle.load(fr)
    
    with open( directory+ '/input/LDA_obj.pkl', 'rb') as f :
        LDA_obj = pickle.load(f)
        
    with open( directory+ '/input/data_sample.pkl', 'rb') as f :
        data_sample = pickle.load(f)
        
    #%%
    
    topic_doc_df = LDA_handling.get_topic_doc(LDA_obj.model, LDA_obj.corpus)
    encoded_docs = embedding.doc2vec(embedding.model, LDA_obj.docs)
    encoded_topic = embedding.topic2vec(topic_doc_df, encoded_docs)
    
    CPC_dict =  CPC.generate_CPC_dict(data_sample)
    CPC_dict_filtered = CPC.filter_CPC_dict(data_sample, CPC_dict,  CPC_definition)
    encoded_CPC = embedding.CPC2vec(embedding.model, CPC_definition, CPC_dict_filtered)
    
    #%%
        
    # with open(directory+ '/output/encoded_docs.pkl', 'wb') as f :
    #     pickle.dump(encoded_docs, f)
        
    with open(directory+ '/output/encoded_topic.pkl', 'wb') as f :
        pickle.dump(encoded_topic, f)
        
    with open(directory+ '/output/encoded_CPC.pkl', 'wb') as f :
        pickle.dump(encoded_CPC, f)
        