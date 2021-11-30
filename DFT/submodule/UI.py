# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 14:16:25 2021

@author: tmlab
"""
import sys
import os
from PyQt5.QtWidgets import *
from PyQt5 import uic

form_class = uic.loadUiType("test.ui")[0]

class WindowClass(QMainWindow, form_class) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        
        self.btn_select.clicked.connect(self.btn_select_Function)
        self.btn_directory.clicked.connect(self.btn_directory_Function)
        
        self.lbl_directory.setText(os.path.dirname(os.path.abspath(__file__)))
        
        self.addComboBoxItem(['LDA-CPC 종합 결과 저장',
                                  'LDA-CPC 유사도 히트맵 출력',
                                  'LDA-CPC 포트폴리오 맵 출력'])
        
    #btn이 눌리면 작동할 함수
    def btn_select_Function(self) :
        print(self.cmb_Test.currentIndex())
        
    def btn_directory_Function(self) :
        self.directory = QFileDialog.getExistingDirectory()
        self.lbl_directory.setText(self.directory)
        
        
    def clearComboBoxItem(self) :
        self.cmb_Test.clear()

    def addComboBoxItem(self, items) :
        
        for item in items :
            self.cmb_Test.addItem(item)

    def deleteComboBoxItem(self) :
        self.delidx = self.cmb_second.currentIndex()
        self.cmb_Test.removeItem(self.delidx)
        self.cmb_second.removeItem(self.delidx)
        print("Item Deleted")
        
        
