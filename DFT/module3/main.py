# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 17:25:39 2021

@author: tkdgu
"""

if __name__ == '__main__':
    
    import os
    import sys
    import pickle
    
    directory = os.path.dirname(os.path.abspath(__file__))
    directory = directory.replace("\\", "/") # window|
    os.chdir(os.path.dirname(directory))    
    sys.path.append(os.path.dirname(directory)+'/submodule/')
            
    # UI 
    import LDA_handling
    import sys
    from PyQt5.QtWidgets import *
    from PyQt5 import uic,QtCore, QtGui, QtWidgets
    from PyQt5.QtCore import QAbstractTableModel, Qt
    import xlsxwriter
    import pandas as pd
    import visualization
    
    form_class = uic.loadUiType("test.ui")[0]

    class WindowClass(QMainWindow, form_class) :
        
        def __init__(self) :
            
            super().__init__()
            
            self.setupUi(self)
            
            self.btn_execute.clicked.connect(self.btn_execute_Function)
            self.btn_directory.clicked.connect(self.btn_directory_Function)
            self.btn_load.clicked.connect(self.btn_load_Function)
            
            # 시그널
            self.listWidget_Test.itemDoubleClicked.connect(self.chkItemDoubleClicked)
            
            self.directory = directory
            self.lbl_directory.setText(directory)
            self.addComboBoxItem(['LDA-CPC 시각화 및 종합결과 출력',
                                  'LDA-topic embedding 출력',
                                  'CPC embedding 출력'])
        
        def chkItemDoubleClicked(self) :
            # print(str(self.listWidget_Test.currentRow()) + " : " + self.listWidget_Test.currentItem().text())
            path = self.directory_output +'/'+self.listWidget_Test.currentItem().text()
            os.startfile(path)
            
        #btn이 눌리면 작동할 함수
        def btn_execute_Function(self) :
            
            print(self.cmb_Test.currentIndex())
            
            topic_doc_df = LDA_handling.get_topic_doc(self.LDA_obj.model, self.LDA_obj.corpus)
            topic_word_df = LDA_handling.get_topic_word_matrix(self.LDA_obj.model)
            CPC_topic_matrix = LDA_handling.get_CPC_topic_matrix(self.encoded_CPC, self.encoded_topic)     
            topic_year_df =  LDA_handling.get_topic_vol_year(self.LDA_obj.model, topic_doc_df, self.data_sample)
            
            
            volumn_dict = LDA_handling.get_topic_vol(self.LDA_obj.model, self.LDA_obj.corpus)
            CAGR_dict = LDA_handling.get_topic_CAGR(topic_year_df)
            Novelty_dict = LDA_handling.get_topic_novelty(CPC_topic_matrix)    
            CPC_match_dict = LDA_handling.get_topic2CPC(CPC_topic_matrix)    
            
            total_df = pd.DataFrame([volumn_dict, CAGR_dict, Novelty_dict, CPC_match_dict]).transpose()
            total_df.columns = ['Volumn', 'CAGR', 'Novelty', 'CPC-match']
            topic2doc_title = LDA_handling.get_most_similar_doc2topic(self.data_sample, topic_doc_df)
            
            self.directory_output = self.directory + '/output'
            try : os.mkdir(self.directory_output)
            except : pass
            
            # directory = 'C:/Users/tmlab/Desktop/작업공간/'
            writer = pd.ExcelWriter(self.directory_output + '/LDA_results.xlsx', 
                                    engine='xlsxwriter')
            
            topic_word_df.to_excel(writer , sheet_name = 'topic_word', index = 1)
            pd.DataFrame(topic_doc_df).to_excel(writer , sheet_name = 'topic_doc', index = 1)
            topic_year_df.to_excel(writer , sheet_name = 'topic_year', index = 1)
            topic2doc_title.to_excel(writer , sheet_name = 'topic_doc_title', index = 1)
            CPC_topic_matrix.to_excel(writer , sheet_name = 'topic2CPC', index = 1)
            total_df.to_excel(writer , sheet_name = 'topic_stats', index = 1)
            
            writer.save()
            writer.close()
            
            visualization.pchart_CPC_topic(CPC_topic_matrix, [0,1,2,3], self.directory_output)
            visualization.heatmap_CPC_topic(CPC_topic_matrix, self.directory_output)
            visualization.portfolio_CPC_topic(Novelty_dict, CAGR_dict, volumn_dict, CPC_topic_matrix, CPC_match_dict
                                              , self.directory_output)
            
            file_list = os.listdir(self.directory_output)
            for file in file_list :
                self.listWidget_Test.addItem(file)
            
        def btn_directory_Function(self) :
            self.directory = QFileDialog.getExistingDirectory()
            self.lbl_directory.setText(self.directory)
            
        def btn_load_Function(self) :
            
            with open(self.directory + '/LDA_obj.pkl', 'rb') as f :
                self.LDA_obj = pickle.load(f)
                
            with open(self.directory + '/encoded_CPC.pkl', 'rb') as f :
                self.encoded_CPC = pickle.load(f)
                
            with open(self.directory + '/encoded_topic.pkl', 'rb') as f :
                self.encoded_topic = pickle.load(f)
                
            with open(self.directory+ '/data_prep.pkl', 'rb') as f :
                self.data_sample = pickle.load(f)
                
            print(self.encoded_topic)
            
            self.lbl_directory.setText('데이터 준비 완료')
            
            
        def addComboBoxItem(self, items) :
            for item in items :
                self.cmb_Test.addItem(item)
            
    app = QApplication(sys.argv)
    myWindow = WindowClass() 
    myWindow.show()
    app.exec_()
        
        #%%

with open('./input/data_prep.pkl', 'rb') as f :
    data_sample = pickle.load(f)
    #%%
    data_sample = data_sample.drop('TAC_nlp', axis =1)
    #%%
with open('./input/data_prep.pkl', 'wb') as f :
    pickle.dump(data_sample,f)

