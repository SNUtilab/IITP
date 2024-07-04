# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 19:41:11 2021

@author: Woojin
"""
# settings
import pandas as pd
import numpy as np
import statistics
import math
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:\\Windows\\Fonts\\malgun.ttf").get_name()
rc('font', family=font_name)


data = pd.read_csv("C:\\Users\\tmlab\\Desktop\\TotalData.csv", engine='python')

citation = data['Bibliographic']


key = type(citation[0])

final_data = []
nan_data = []

for i in range(len(citation)):
    
    if type(citation[i]) == key:
        nan_data.append(np.nan)
    
    else:
        final_data.append(citation[i].split('|'))
    

find_us = "US"


list1 = []
list2 = []
list3 = []
for i in range(len(final_data)):
    patent_citation = [j for j in range(len(final_data[i])) if find_us in final_data[i][j]]
    list1.append(len(patent_citation))

    for k in patent_citation:
        list2.append(final_data[i][k])
        # if final_data[i][j].startswith('US'):
        #     pass
        # else:
        #     pass
            
np.median(list1)
np.mean(list1)

for v in list2:
    if v not in list3:
        list3.append(v)
        
        
list4 = []
# for i in range(len(final_data)):
#     patent_citation = [j for j in range(len(final_data[i])) if find_us in final_data[i][j]]
    
#     for k in patent_citation:
#         if i+1 < len(final_data):
#             if k in final_data[i+1]:
#                 list4.append(k)
        
#         else:
#             break

################################################################################################################

## 결과
result_index = []
result = []     
for i in range(len(final_data)):
    for j in range(len(final_data)):
        p = 0
        for s1 in final_data[i]:
            if s1 in final_data[j]:
                p = p+1         
        
        result_index.append(str(i)+","+str(j))
        result.append(p / math.sqrt(len(final_data[i]) * len(final_data[j])))    
        
## 리스트 행렬로 변환
def list_matrix(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

result_list = list_matrix(result, 59)   ##total222, part59
result_matrix = list(map(list, zip(*result_list)))

## CSV

import csv
# with open('filename.csv', 'w') as f:
#    writer = csv.writer(f)
#    for i, row in enumerate(result_matrix):
#        writer.writerow([str(i), 'name{}'.format(i), row])
with open('data1.csv', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',',  quotechar='"',
                                 quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerows(result_matrix)


        
def isBiggerThan(x):
    return x > 0.9                
        
newlist = list(filter(isBiggerThan, result))        

