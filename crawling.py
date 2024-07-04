# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 20:43:28 2022

@author: jinny
"""


# package
from bs4 import BeautifulSoup
from urllib.request import urlopen
import argparse
import pandas as pd
from selenium import webdriver
from urllib.request import urlopen
import time
import os, sys
from PIL import Image
import numpy as np
from selenium.webdriver.common.keys import Keys

# Crawling
# Crawling
os.chdir('C:/Users/jinny/Documents/test') 
os.getcwd()

driver = webdriver.Chrome(r"D:/chromedriver/chromedriver.exe")
driver.implicitly_wait(3)

data = pd.read_csv('pmid_test.csv', encoding='UTF-8')


for i in range(len(data)):

    a = int(data['PMID'][i])
    
    
    
    driver.get('https://pubmed.ncbi.nlm.nih.gov/'+str(a)+'/?from_single_result='+str(a)+'&expanded_search_query=' + str(a))
    try:
        print(i, '   ', a)
        
        abstract = driver.find_element_by_id('en-abstract')
        
        
        data['abstract'][i] = str(abstract.text)
    except: 
        print(i, '   ', a, '   없음')
        data['abstract'][i] = 'no_text'

    if i % 5000 == 0:
        data.to_csv('pmid.csv')
    driver.implicitly_wait(3)
    
data.to_csv('pmid.csv')
