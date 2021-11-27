# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 11:49:00 2021

@author: tmlab
"""

#%% 01. package and data load

if __name__ == '__main__':
    
    import pickle
    import spacy
    import re 
    from nltk.corpus import stopwords    
    import pandas as pd
    import numpy as np
    from gensim.models import CoherenceModel
    from gensim.corpora import Dictionary
    from gensim.models.ldamulticore import LdaMulticore
    import warnings    
    from sentence_transformers import SentenceTransformer
    from collections import Counter
    import xlsxwriter
    
    directory = 'D:/OneDrive - 아주대학교/db/patent/'
    
    with open(directory + 'DT_211118.pkl', 'rb') as fr :
        data = pickle.load(fr)
        
    # data_sample = data.sample(500, random_state = 12345).reset_index(drop = 1)
    data_sample = data
    data_sample['TAC'] = data_sample['title'] + ' ' + data_sample['abstract'] + ' ' + data_sample['claims_rep']
    data_sample['year'] = data_sample['date'].apply(lambda x : x[0:4])
    
    # data_sample.
    
    #%% 02. preprocessing and filtering
    
    c = Counter(data_sample['year']) #2016 ~ 2020
    year_list = [k for k,v in c.items() if v >= 30]
    stopwords_nltk = set(stopwords.words('english'))
    
    data_sample = data_sample.loc[data_sample['year'] >=min(year_list) , :].reset_index(drop = 1)
    data_sample = data_sample.loc[data_sample['TAC'].str.split().str.len() >= 100 , :].reset_index(drop = 1)
    
    nlp = spacy.load("en_core_web_sm")
    directory = 'D:/OneDrive - 아주대학교/db/dictionary/'
        
    with open(directory + 'stopwords_uspto.txt') as f:
        stopwords_uspto = [line.rstrip() for line in f]
        
    stopwords_uspto.append('-PRON-')
    
    data_sample['TAC_nlp'] = [nlp(i) for i in data_sample['TAC']]
    
    # get keyword
    data_sample['TAC_keyword'] = [[token.lemma_ for token in doc] for doc in data_sample['TAC_nlp']] # lemma
    data_sample['TAC_keyword'] = [[token for token in doc if len(token) > 2] for doc in data_sample['TAC_keyword']] # 길이기반 제거
    data_sample['TAC_keyword'] = [[token for token in doc if not token.isdigit() ] for doc in data_sample['TAC_keyword']]  #숫자제거 
    data_sample['TAC_keyword'] = [[re.sub(r"[^a-zA-Z0-9-]","",token) for token in doc ] for doc in data_sample['TAC_keyword']] #특수문자 교체    
    data_sample['TAC_keyword'] = [[token for token in doc if len(token) > 2] for doc in data_sample['TAC_keyword']] # 길이기반 제거
    data_sample['TAC_keyword'] = [[token for token in doc if token not in stopwords_uspto] for doc in data_sample['TAC_keyword']] # 길이기반 제거
    data_sample['TAC_keyword'] = [[token for token in doc if token not in stopwords_nltk] for doc in data_sample['TAC_keyword']] # 길이기반 제거
    
    data_sample['cpc_class'] = ''
    data_sample['cpc_subclass'] = ''
    data_sample['cpc_group'] = ''

    for idx,row in data_sample.iterrows() :
        
        print(idx)
        
        cpc_list = data_sample['cpc_list'][idx]
        data_sample['cpc_group'][idx] = [i for i in cpc_list if len(i) > 5]
        
        data_sample['cpc_class'][idx] = [i for i in cpc_list if len(i) == 3]
        
        data_sample['cpc_subclass'][idx] = [i for i in cpc_list if len(i) == 4]
    
    #%% 03. filtering text and cpc

    directory = 'D:/OneDrive - 아주대학교/db/patent/CPC/'
    
    with open(directory + 'CPC_definition.pkl', 'rb') as fr:
        CPC_definition = pickle.load(fr)
        
    def get_CPC_Counter(df,col) :
        cpc_list = df[col].tolist()
        cpc_list = sum(cpc_list, [])
        c = Counter(cpc_list)
        # c =  {k: v for k, v in sorted(c.items(), key=lambda item: item[1], reverse=True)}
        return(c)
    
    cpc_class = get_CPC_Counter(data_sample, 'cpc_class')
    cpc_subclass = get_CPC_Counter(data_sample,'cpc_subclass')
    cpc_group = get_CPC_Counter(data_sample,'cpc_group')
    
    
    class_list = [k for k,v in cpc_class.items() if v >= len(data_sample) * 0.05]
    class_list = [i for i in class_list if i in CPC_definition.keys()]
    
    subclass_list = [k for k,v in cpc_subclass.items() if v >= len(data_sample) * 0.025]
    subclass_list = [i for i in subclass_list if i[0:-1] in class_list]
    subclass_list = [i for i in subclass_list if i in CPC_definition.keys()]
    
    group_list = [k for k,v in cpc_group.items() if v >= len(data_sample) * 0.0125]
    group_list = [i for i in group_list if i[0:4] in subclass_list]
    group_list = [i for i in group_list if i in CPC_definition.keys()]
    
    
    #%% 04. encoding cpc and keyword
    
    # conda install -c conda-forge ipywidgets
    model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    result_df = pd.DataFrame()
    
    def cosine(u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    
    encoded_CPC = {}
    
    for i in subclass_list :
        encoded_CPC[i] = model.encode(CPC_definition[i])
        
    for i in class_list :
        encoded_CPC[i] = model.encode(CPC_definition[i])

    for i in group_list :
        encoded_CPC[i] = model.encode(CPC_definition[i])


    with open(directory + 'input/encoded_CPC.pkl', 'wb') as fw:
        pickle.dump(encoded_CPC, fw)
    
    # with open(directory + 'input/encoded_CPC.pkl', 'rb') as fr:
    #     encoded_CPC = pickle.load(fr)
    
    #%%
    texts = data_sample['TAC_keyword']
    
    dct = Dictionary(texts)
    dct.filter_extremes(no_below = 10,
                        no_above=0.1)
    keyword_list = dct.token2id.keys()
    
    text_list = keyword_list
    standard = {}
    encoded_text = {}

    level = 1
    
    for CPC in [class_list, subclass_list, group_list]  :
        
        total_df = pd.DataFrame()
        
        for text in text_list :
            
            if text in encoded_text.keys() :
                text_embedding = encoded_text[text]
            else :
                text_embedding = model.encode(text)
                encoded_text[text] = text_embedding
            
            sim_list = []
            
            for cpc_text in CPC :
                cpc_embedding = encoded_CPC[cpc_text]
                sim = cosine(text_embedding, cpc_embedding)
                sim_list.append(sim)
            
            mean = np.mean(sim_list)
            var = np.var(sim_list)
            
            total_df = total_df.append([[mean,var]], ignore_index=1)
        
        total_df.columns = ['MEAN' , 'VAR']
        
        MEAN = np.mean(total_df['MEAN'])
        VAR = np.mean(total_df['VAR'])
        
        standard[level] = (MEAN, VAR)
        level +=1
        
    #%% 05. classify keyword
    
    level = 1
    text_classification_df = pd.DataFrame()
    
    for CPC in [class_list, subclass_list, group_list]  :
        
        MEAN = standard[level][0]
        VAR = standard[level][1]
        SD = np.sqrt(VAR)
        
        result = {}
        
        for text in keyword_list :
            
            # CPC = class_list
            
            if text in result.keys() : 
                continue
        
            else :
                
                text_embedding = encoded_text[text]
                
                sim_list = []
                
                for cpc_text in CPC :
                    cpc_embedding = encoded_CPC[cpc_text]
                    sim = cosine(text_embedding, cpc_embedding)
                    sim_list.append(sim)
                
                mean = np.mean(sim_list)
                var = np.var(sim_list)
                
                if (mean >= MEAN) & (var >= VAR) : 
                    result[text] = 'DEFINED_SEPERATED'    
                elif (mean >= MEAN) & (var < VAR) : 
                    result[text] = 'DEFINED'
                else  :
                    result[text] = 'UNCLASSIFIED'
        
        level +=1
        
        text_classification_df = text_classification_df.append(result, ignore_index=1 )     
        
    text_classification_df = text_classification_df.transpose()
    
    text_classification_df.columns = ['class', 'subclass', 'group']
    
    #%% 06. LDA tunning 

    warnings.filterwarnings('ignore')
    
    # dct = Dictionary(texts)
    # dct.filter_extremes(no_below = 10,
    #                     no_above=0.1)
    
    corpus = [dct.doc2bow(text) for text in texts]
    
    
    def compute_coherence_values(corpus, dictionary, texts, k, a, b):
        
        lda_model = LdaMulticore(corpus=corpus,
                                 id2word=dictionary,
                                 num_topics=k, 
                                 random_state=100,
                                 chunksize=100,
                                 passes=10,
                                 alpha=a,
                                 eta=b,
                                  workers=15,
                                 )
        coherence_model_lda = CoherenceModel(model=lda_model, 
                                             texts=texts, 
                                             dictionary=dictionary, 
                                             coherence='u_mass')
    
        
        return coherence_model_lda.get_coherence()
    
    
    grid = {}
    grid['Validation_Set'] = {}
    
    # Topics range  #수정
    min_topics = 10
    max_topics = 51
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
    num_of_docs = len(corpus)
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
                        
        pd.DataFrame(model_results).to_csv(directory + 'lda_tuning_results.csv', index=False)
        
    #%% 07. LDA result handling  #수정
    
    lda_model = LdaMulticore(corpus=corpus,
                             id2word=dct,
                             num_topics= 50, 
                             random_state=100,
                             chunksize=100,
                             passes=10,
                             alpha=0.91,
                             eta=0.91,
                              workers=15
                             )
    
    
    #%% 08. calculate freshness 
    topic_word_df = pd.DataFrame()
    
    for i in range(0, lda_model.num_topics) :
        temp = lda_model.show_topic(i, 1000)
        DICT = {}
        for tup in temp :
            DICT[tup[0]] = tup[1]
            
        topic_word_df = topic_word_df.append(DICT, ignore_index =1)
        
    topic_word_df = topic_word_df.transpose()
    
    # freshness
    topic_fresh_dict = dict.fromkeys(range(0,50), 0)    
    
    for idx, row in topic_word_df.iterrows() :
        temp = text_classification_df.loc[[row.name]].iloc[0,:].tolist()
        if all(i== 'UNCLASSIFIED' for i in temp ) :
            for col in topic_word_df.columns :
                prob = row[col]
                if str(prob) != 'nan' :
                    topic_fresh_dict[col] += prob
    
    topic_fresh_df = pd.DataFrame(topic_fresh_dict.items())
    topic_fresh_df.columns = ['topic', 'prob']
    #%% 09. calculate volumn & cagr
    
    topic_doc_df = pd.DataFrame(columns = range(0, lda_model.num_topics))
    
    for corp in corpus :
        
        temp = lda_model.get_document_topics(corp)
        DICT = {}
        for tup in temp :
            DICT[tup[0]] = tup[1]
        
        topic_doc_df = topic_doc_df.append(DICT, ignore_index=1)
    
    volumn_df = topic_doc_df.apply(np.sum)
    #%% get CAGR
    topic_doc_df['year'] = data_sample['year']
    topic_doc_df['title'] = data_sample['title']
    
    #%%
    
    topic_year_df = pd.DataFrame()
    for col in range(0, lda_model.num_topics) :
        grouped = topic_doc_df[col].groupby(topic_doc_df['year'])
        DICT = grouped.sum()
        topic_year_df = topic_year_df.append(DICT, ignore_index=1)
    
    topic_year_df = topic_year_df.transpose()
    
    
    #%% saving result
        
    import xlsxwriter
    # directory = 'C:/Users/tmlab/Desktop/작업공간/'
    writer = pd.ExcelWriter('./output/LDA_results.xlsx', 
                            engine='xlsxwriter')
    
    topic_word_df.to_excel(writer , sheet_name = 'topic_word', index = 1)
    topic_fresh_df.to_excel(writer , sheet_name = 'topic_fresh', index = 1)
    topic_doc_df.to_excel(writer , sheet_name = 'topic_doc', index = 1)
    volumn_df.to_excel(writer , sheet_name = 'volumn', index = 1)
    topic_year_df.to_excel(writer , sheet_name = 'volumn_year', index = 1)
    topic_year_df.to_excel(writer , sheet_name = 'volumn_year', index = 1)
    
    
    
    writer.save()
    writer.close()
    
    #%% 응용 1
    
    topics = lda_model.show_topic(26)
    
    for topic in topics:
        print(topic)
       
    #%% 응용 2
    
    for idx, topic_list in enumerate(lda_model[corpus]):
        # if idx==5: break
        print(idx,'번째 문서의 topic 비율은',topic_list)
        
    #%% 응용 3 https://wikidocs.net/30708
    
    def make_topictable_per_doc(ldamodel, corpus):
        topic_table = pd.DataFrame()
    
        # 몇 번째 문서인지를 의미하는 문서 번호와 해당 문서의 토픽 비중을 한 줄씩 꺼내온다.
        for i, topic_list in enumerate(ldamodel[corpus]):
            doc = topic_list[0] if ldamodel.per_word_topics else topic_list            
            doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
            # 각 문서에 대해서 비중이 높은 토픽순으로 토픽을 정렬한다.
            # EX) 정렬 전 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (10번 토픽, 5%), (12번 토픽, 21.5%), 
            # Ex) 정렬 후 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (12번 토픽, 21.5%), (10번 토픽, 5%)
            # 48 > 25 > 21 > 5 순으로 정렬이 된 것.
    
            # 모든 문서에 대해서 각각 아래를 수행
            for j, (topic_num, prop_topic) in enumerate(doc): #  몇 번 토픽인지와 비중을 나눠서 저장한다.
                if j == 0:  # 정렬을 한 상태이므로 가장 앞에 있는 것이 가장 비중이 높은 토픽
                    topic_table = topic_table.append(pd.Series([int(topic_num), round(prop_topic,4), topic_list]), ignore_index=True)
                    # 가장 비중이 높은 토픽과, 가장 비중이 높은 토픽의 비중과, 전체 토픽의 비중을 저장한다.
                else:
                    break
        return(topic_table)
    
    topictable = make_topictable_per_doc(lda_model, corpus)
    topictable = topictable.reset_index() # 문서 번호을 의미하는 열(column)로 사용하기 위해서 인덱스 열을 하나 더 만든다.
    topictable.columns = ['문서 번호', '가장 비중이 높은 토픽', '가장 높은 토픽의 비중', '각 토픽의 비중']
    topictable[:10]
    
    topictable.to_csv(directory + 'topictable.csv', index = 0 , encoding = "euc-kr")
    
    #%% 응용 4
    
    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvis
    pyLDAvis.enable_notebook()
    
    # feed the LDA model into the pyLDAvis instance
    lda_viz = gensimvis.prepare(lda_model, corpus, dct)
    pyLDAvis.save_html(lda_viz, directory + 'lda_viz.html')