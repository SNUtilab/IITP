# -*- coding: utf-8 -*-

#%% AO쌍의 선/후행 관계(Edge)
def cal_x1_x2(df_ao):
    
    
    df_ao = df_ao.drop_duplicates(['From', 'PtNumber', 'To'])
    #index 초기화
    df_ao = df_ao.reset_index()
    
    df_ao['AO'] = df_ao['From']+ " "+ df_ao["To"]


    df_ao = df_ao.loc[:,['From', 'PtNumber', 'Realtion', 'Source', 'To', 'AO']]


    
    x1 = []
    patent_x1 = []
    k=0
    ao_df = pd.DataFrame(columns = ['idx', 'x1', 'idx_x2', 'x2', 'x1x2_result', 'x2x1_result'])
    
    for i in list(df_ao['AO']):
        
        condition_x1 = df_ao['AO'] == i
        x1_df = df_ao[condition_x1]
        
        set_x1 = set(x1_df['PtNumber'])
        doc_x1 = len(set_x1)
        
        patent_x1.append(set_x1)
        x1.append(doc_x1)
        
          
    for k in range(0,len(df_ao.index),1):
        
        for j in range(0,len(df_ao.index),1):
        
            doc_x1 = x1[k]
            
            #condition_x2 = ex['AO'] == ex['AO'][j]
            #x2 = ex[condition_x2]
            set_x2 = patent_x1[j]
            doc_x2 = x1[j]
            
            #c = set_x1.intersection(set_x2)
            
            c = patent_x1[k].intersection(set_x2)
            cor_c = len(c)
            
            
            x1x2_result = cor_c/doc_x1
            
            x2x1_result = cor_c/doc_x2
            
            if x1x2_result < 1 :
                if x2x1_result == 1:
                    #print("x1이 선행노드")        
                    ao_df = ao_df.append(pd.DataFrame([[df_ao['PtNumber'][k], df_ao['AO'][k], df_ao['PtNumber'][j], df_ao['AO'][j], x1x2_result, x2x1_result]],columns = ['idx', 'x1', 'idx_x2', 'x2', 'x1x2_result', 'x2x1_result']))
        print(k)    
    return ao_df
    
    

#%%
ao_df_list = []

for i in range(len(dfs_ao)):
    ao_df_list.append(cal_x1_x2(dfs_ao[i]))




#%% weight 계산함수
def weight_lift(ao_df_pickle):
    ao_df_pickle_ = ao_df_pickle.drop_duplicates(['x1', 'x2'])
    ao_df_pickle_ = ao_df_pickle_.reset_index()
    
    doc_num = len(set(ao_df_pickle['idx']))
    
    x1x2_count = []
    x1_count = []
    x2_count = []   
    for i in range(len(ao_df_pickle_.index)):
        x1x2_count.append(len(ao_df_pickle_[(ao_df_pickle_['x1'] == ao_df_pickle_['x1'][i]) & (ao_df_pickle_['x2'] == ao_df_pickle_['x2'][i])])/doc_num)
        x1_count.append(len(ao_df_pickle_[ao_df_pickle_['x1'] == ao_df_pickle_['x1'][i]])/doc_num)
        x2_count.append(len(ao_df_pickle_[ao_df_pickle_['x2'] == ao_df_pickle_['x2'][i]])/doc_num)
    
    ao_df_pickle_['weight'] = np.array(x1x2_count) / (np.array(x1_count) * np.array(x2_count))
    
    return ao_df_pickle_