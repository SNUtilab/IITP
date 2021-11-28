# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 16:23:51 2021

@author: tmlab
"""

from collections import Counter 

def get_CPC_Counter(df,col) :
    cpc_list = df[col].tolist()
    cpc_list = sum(cpc_list, [])
    c = Counter(cpc_list)
    # c =  {k: v for k, v in sorted(c.items(), key=lambda item: item[1], reverse=True)}
    return(c)


def generate_CPC_dict(df) :
    
    CPC_dict = {}
    
    CPC_dict['cpc_class'] = get_CPC_Counter(df, 'cpc_class')
    CPC_dict['cpc_subclass'] = get_CPC_Counter(df,'cpc_subclass')
    CPC_dict['cpc_group'] = get_CPC_Counter(df,'cpc_group')
    
    return(CPC_dict)


def filter_CPC_dict(df, CPC_dict, CPC_definition):
    
    # cpc_class = CPC_dict['cpc_class']
    cpc_subclass = CPC_dict['cpc_subclass']
    m_c = CPC_dict['cpc_subclass'].most_common(20)
    subclass_list = [i[0] for i in m_c]
    CPC_dict_filtered ={}
    
    # class_list = [k for k,v in cpc_class.items() if v >= len(df) * 0.05]
    # class_list = [i for i in class_list if i in CPC_definition.keys()]
    # CPC_dict_filtered['class_list'] = class_list
    
    # subclass_list = [k for k,v in cpc_subclass.items() if v >= len(df) * 0.025]
    # subclass_list = [k for k,v in cpc_subclass.items() if v >= len(df) * 0.025]
    # subclass_list = [i for i in subclass_list if i[0:-1] in class_list]
    # subclass_list = [i for i in subclass_list if i in CPC_definition.keys()]
    
    CPC_dict_filtered['subclass_list'] = subclass_list
    
    # group_list = [k for k,v in cpc_group.items() if v >= len(df) * 0.0125]
    # group_list = [i for i in group_list if i[0:4] in subclass_list]
    # group_list = [i for i in group_list if i in CPC_definition.keys()]
    # CPC_dict_filtered['group_list'] = group_list
    
    return (CPC_dict_filtered)