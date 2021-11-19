# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 16:16:58 2021

@author: tmlab
"""


if __name__ == '__main__':
    
    import os
    import sys
    import pickle
    from copy import copy
    import data_preprocessing
    
    directory = os.path.dirname(os.path.abspath(__file__))
    directory = directory.replace("\\", "/") # window
    
    
    with open( directory+ '/input/DT_211118.pkl', 'rb') as fr :
        data = pickle.load(fr)
        
    data_sample = copy(data)
    data_sample = data_preprocessing.initialize(data_sample)