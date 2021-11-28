# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 22:33:03 2021

@author: tmlab
"""


import pandas as pd
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import plotly
import seaborn as sns
from sklearn.preprocessing import minmax_scale
import pandas as pd

pio.renderers.default = "browser"    

def pchart_CPC_topic(CPC_topic_matrix, topic_list) :
    
    df = pd.DataFrame()
    for idx, row in CPC_topic_matrix.iterrows() :
        DICT = {}
        for col in CPC_topic_matrix.columns :
            if col in topic_list :
                DICT['Similarity'] = row[col]
                DICT['CPC'] = idx
                DICT['Topic'] = str(col)
                df = df.append(DICT, ignore_index=1)
            else : pass
    
    fig = px.line_polar(df, 
                        r="Similarity", 
                        theta="CPC", 
                        color="Topic", 
                        line_close=True,
                        color_discrete_sequence=px.colors.sequential.Plasma_r,
                        template="seaborn",
                        )
           # ['ggplot2', 'seaborn', 'simple_white', 'plotly',
         # 'plotly_white', 'plotly_dark', 'presentation', 'xgridoff',
         # 'ygridoff', 'gridon', 'none']
    fig.show()

def heatmap_CPC_topic(CPC_topic_matrix) :
    
    fig = go.Figure(data=go.Heatmap(
                        z=CPC_topic_matrix,
                        x=CPC_topic_matrix.columns,
                        y=CPC_topic_matrix.index,
                        hoverongaps = False,
                        colorscale='RdBu',
                        colorbar={"title": "<b>Similarity<b>"}
                      ))
    fig.update_layout(
    
        xaxis_title="<b>Topic<b>",
        yaxis_title="<b>CPC-subclass<b>",
        xaxis_nticks= CPC_topic_matrix.shape[1],
        font=dict(
            family='Times New Roman',
            size=18,
        # color="#7f7f7f",
        ))
    
    fig.show()
    plotly.offline.plot(fig, filename='./output/heatmap_C2T.html')
    
    
def portfolio_CPC_topic(Novelty_dict, CAGR_dict, volumn_dict, CPC_topic_matrix, CPC_match_dict) :
    
    temp = minmax_scale(list(Novelty_dict.values()), axis=0, copy=True)
    temp_ = minmax_scale(list(CAGR_dict.values()), axis=0, copy=True)
    # temp__ = minmax_scale(list(volumn_dict.values()), axis=0, copy=True)
    temp__ = list(volumn_dict.values())
    
    df = pd.DataFrame.from_dict([temp, temp_, temp__]).transpose()
    df.columns = ['Novelty', 'CAGR', 'Volumn']    
    df['Most_similar'] = list(CPC_match_dict.values())
    
    color_list = sns.color_palette('hls', len(set(df['Most_similar']))).as_hex()
    colorsIdx = dict(zip(list(set(df['Most_similar'])), color_list))    
    fig = go.Figure()
    
    for group in set(df['Most_similar']) :
        temp_df = df.loc[df['Most_similar'] == group,:] 
        fig.add_trace(go.Scatter(
            x= temp_df['Novelty'], y=temp_df['CAGR'],
            name = group,
            mode ='markers',
            marker=dict(
                color = colorsIdx[group],
                size= temp_df['Volumn'],
                sizemode='area',
                sizeref=2*max(temp_df['Volumn'])/(60.**2),
                sizemin=2,
            )        
    ))
    
    fig.add_vline(0.5)
    fig.add_hline(0.5)
    fig.update_layout(
        width=1000,
        height=1000,

        xaxis_title="<b>Novelty<b>",
        yaxis_title="<b>Growth-rate<b>",
        xaxis_nticks= CPC_topic_matrix.shape[1],
        font=dict(
            family='Times New Roman',
            size=24,
        # color="#7f7f7f",
        ))
    
    fig.show()
    plotly.offline.plot(fig, filename='./output/portfolio_topic.html')