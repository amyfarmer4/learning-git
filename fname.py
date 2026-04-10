# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 15:36:48 2025

@author: af1226
"""

def get_fname():
    from os import name
    if name == 'posix':
        fname_left = '/home/amy/data/LEFT detector/*.csv'
        fname_right = '/home/amy/data/RIGHT detector/*.csv'
        fname_summer = '/home/amy/data/summer/*.csv'
        fname_winter = '/home/amy/data/winter/*.csv'
    elif name == 'nt':
        fname_left = 'A:\Amy\Protein Library\LIFEtime library\LEFT detector\*.csv'
        fname_right = 'A:\Amy\Protein Library\LIFEtime library\RIGHT detector\*.csv'
        fname_summer = 'A:\Amy\Protein Library\ATHENA\data_for_ML\summer\interpolated\*.csv'
        fname_winter = 'A:\Amy\Protein Library\ATHENA\data_for_ML\winter\interpolated\*.csv'
    else:
        print ('Rogue OS! Stopping now.')
    return fname_left, fname_right, fname_summer, fname_winter

