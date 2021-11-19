# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 16:44:19 2021

@author: tmlab
"""
import pandas as pd
import numpy as np


def initialize(df):
    
    df = df
    df['TAC'] = df['title'] + ' ' + df['abstract'] + ' ' + df['claims_rep']
    df['year'] = df['date'].apply(lambda x : x[0:4])
    
    return (df)
    
def filtering_by_year(df) :
    
    
    return(df)