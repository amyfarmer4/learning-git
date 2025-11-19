# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 15:36:48 2025

@author: af1226
"""

def get_fname():
    from os import name
    if name == 'posix':
        pass
    elif name == 'nt':
        fname_left = 'A:\Amy\Protein Library\LIFEtime library\LEFT detector\*.csv'
        fname_right = 'A:\Amy\Protein Library\LIFEtime library\RIGHT detector\*.csv'
    else:
        print ('Rogue OS! Stopping now.')
    return fname_left, fname_right
