# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 10:08:28 2025

@author: af1226
"""
# import things you always need
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import copy

# import things needed for pipeline building and GridSearching
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GroupKFold
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression

# import models
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import VotingRegressor
from sklearn.multioutput import RegressorChain

# import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error

# load in data

proteins = ['Myoglobin', 'HSA', 'Calmodulin', 'cytochrome c', 'phosphorylase b', 'Lysozyme', 'Creatine', 'DT diaphorase', 'Lipoxidase', 
            'Lactoferrin bovin', 'Protease', 'apo transferrin', 'Lactoferrin human', 'Catalase', 'Conalbumin', 'alpha amylase', 
            'Choline oxidase', 'Gly 3 phos', 'Ovalbumin', 'b glucuronidase', 'Ribonuclease A', 'Ubiquitin', 'alpha 2 macroglobulin',
            'Chymotrypsinogen A', 'Thaumatin', 'TCI', 'Elastase', 'beta-Lactoglobulin', 'Pepsin', 'Tryp', 'SD', 'IgG', 'Concanavalin A',
            'Peroxidase', 'prealbumin']

def ss(x):
    return [x for _ in range(99)]

def ss_peroxidase(x):
    return [x for _ in range(66)]

def ss_prealbumin(x):
    return [x for _ in range(33)]

def protein_label(x):
    return [x for _ in range(99)]

def protein_label_peroxidase(x):
    return [x for _ in range(66)]

def protein_label_prealbumin(x):
    return [x for _ in range(33)]

class_labels = {'Myoglobin': ss(0),
                'HSA': ss(0),
                'Calmodulin': ss(0),
                'cytochrome c': ss(0),
                'phosphorylase b': ss(0),
                'Lysozyme': ss(0),
                'Creatine': ss(0),
                'DT diaphorase': ss(1),
                'Lipoxidase': ss(1),
                'Lactoferrin bovin': ss(1),
                'Protease': ss(1),
                'apo transferrin': ss(1),
                'Lactoferrin human': ss(1),
                'Catalase': ss(1),
                'Conalbumin': ss(1),
                'alpha amylase': ss(1),
                'Choline oxidase': ss(1),
                'Gly 3 phos': ss(1),
                'Ovalbumin': ss(1),
                'b glucuronidase': ss(1),
                'Ribonuclease A': ss(1),
                'Ubiquitin': ss(1),
                'alpha 2 macroglobulin': ss(1),
                'Chymotrypsinogen A': ss(2),
                'Thaumatin': ss(2),
                'TCI': ss(2),
                'Elastase': ss(2),
                'beta-Lactoglobulin': ss(2),
                'Pepsin': ss(2),
                'Tryp': ss(2),
                'SD': ss(2),
                'IgG': ss(2),
                'Concanavalin A': ss(2),
                'Peroxidase': ss_peroxidase(0),
                'prealbumin': ss_prealbumin(2)}

alpha_labels = {'Myoglobin': ss(0.7124),
                'HSA': ss(0.6861),
                'Calmodulin': ss(0.5114),
                'cytochrome c': ss(0.4135),
                'phosphorylase b': ss(0.4327),
                'Lysozyme': ss(0.3101),
                'Creatine': ss(0.3526),
                'DT diaphorase': ss(0.2894),
                'Lipoxidase': ss(0.3048),
                'Lactoferrin bovin': ss(0.2920),
                'Protease': ss(0.2956),
                'apo transferrin': ss(0.2960),
                'Lactoferrin human': ss(0.2967),
                'Catalase': ss(0.2725),
                'Conalbumin': ss(0.2656),
                'alpha amylase': ss(0.2337),
                'Choline oxidase': ss(0.2086),
                'Gly 3 phos': ss(0.2530),
                'Ovalbumin': ss(0.2902),
                'b glucuronidase': ss(0.1758),
                'Ribonuclease A': ss(0.1774),
                'Ubiquitin': ss(0.1579),
                'alpha 2 macroglobulin': ss(0.1245),
                'Chymotrypsinogen A': ss(0.0735),
                'Thaumatin': ss(0.1063),
                'TCI': ss(0),
                'Elastase': ss(0.0583),
                'beta-Lactoglobulin': ss(0.0988),
                'Pepsin': ss(0.1104),
                'Tryp': ss(0),
                'SD': ss(0.0265),
                'IgG': ss(0.0323),
                'Concanavalin A': ss(0),
                'Peroxidase': ss_peroxidase(0.4444),
                'prealbumin': ss_prealbumin(0.0472)}

beta_labels = {'Myoglobin': ss(0),
                'HSA': ss(0),
                'Calmodulin': ss(0.0238),
                'cytochrome c': ss(0),
                'phosphorylase b': ss(0.1419),
                'Lysozyme': ss(0.0620),
                'Creatine': ss(0.1421),
                'DT diaphorase': ss(0.1136),
                'Lipoxidase': ss(0.1305),
                'Lactoferrin bovin': ss(0.1748),
                'Protease': ss(0.1788),
                'apo transferrin': ss(0.1811),
                'Lactoferrin human': ss(0.1881),
                'Catalase': ss(0.1663),
                'Conalbumin': ss(0.1756),
                'alpha amylase': ss(0.1986),
                'Choline oxidase': ss(0.2068),
                'Gly 3 phos': ss(0.2651),
                'Ovalbumin': ss(0.3131),
                'b glucuronidase': ss(0.2968),
                'Ribonuclease A': ss(0.3306),
                'Ubiquitin': ss(0.3158),
                'alpha 2 macroglobulin': ss(0.2997),
                'Chymotrypsinogen A': ss(0.3224),
                'Thaumatin': ss(0.3575),
                'TCI': ss(0.2702),
                'Elastase': ss(0.3417),
                'beta-Lactoglobulin': ss(0.4074),
                'Pepsin': ss(0.4356),
                'Tryp': ss(0.3465),
                'SD': ss(0.3907),
                'IgG': ss(0.4373),
                'Concanavalin A': ss(0.4336),
                'Peroxidase': ss_peroxidase(0.0196),
                'prealbumin': ss_prealbumin(0.4803)}

group_name = {'Myoglobin': protein_label(0),
                'HSA': protein_label(1),
                'Calmodulin': protein_label(2),
                'cytochrome c': protein_label(3),
                'phosphorylase b': protein_label(4),
                'Lysozyme': protein_label(5),
                'Creatine': protein_label(6),
                'DT diaphorase': protein_label(7),
                'Lipoxidase': protein_label(8),
                'Lactoferrin bovin': protein_label(9),
                'Protease': protein_label(10),
                'apo transferrin': protein_label(11),
                'Lactoferrin human': protein_label(12),
                'Catalase': protein_label(13),
                'Conalbumin': protein_label(14),
                'alpha amylase': protein_label(15),
                'Choline oxidase': protein_label(16),
                'Gly 3 phos': protein_label(17),
                'Ovalbumin': protein_label(18),
                'b glucuronidase': protein_label(19),
                'Ribonuclease A': protein_label(20),
                'Ubiquitin': protein_label(21),
                'alpha 2 macroglobulin': protein_label(22),
                'Chymotrypsinogen A': protein_label(23),
                'Thaumatin': protein_label(24),
                'TCI': protein_label(25),
                'Elastase': protein_label(26),
                'beta-Lactoglobulin': protein_label(27),
                'Pepsin': protein_label(28),
                'Tryp': protein_label(29),
                'SD': protein_label(30),
                'IgG': protein_label(31),
                'Concanavalin A': protein_label(32),
                'Peroxidase': protein_label_peroxidase(33),
                'prealbumin': protein_label_prealbumin(34)}

protein_name = {'Myoglobin': ss('Myoglobin'),
                'HSA': ss('HSA'),
                'Calmodulin': ss('Calmodulin'),
                'cytochrome c': ss('cytochrome c'),
                'phosphorylase b': ss('phosphorylase b'),
                'Lysozyme': ss('Lysozyme'),
                'Creatine': ss('Creatine'),
                'DT diaphorase': ss('DT diaphorase'),
                'Lipoxidase': ss('Lipoxidase'),
                'Lactoferrin bovin': ss('Lactoferrin bovin'),
                'Protease': ss('Protease'),
                'apo transferrin': ss('apo transferrin'),
                'Lactoferrin human': ss('Lactoferrin human'),
                'Catalase': ss('Catalase'),
                'Conalbumin': ss('Conalbumin'),
                'alpha amylase': ss('alpha amylase'),
                'Choline oxidase': ss('Choline oxidase'),
                'Gly 3 phos': ss('Gly 3 phos'),
                'Ovalbumin': ss('Ovalbumin'),
                'b glucuronidase': ss('b glucuronidase'),
                'Ribonuclease A': ss('Ribonuclease A'),
                'Ubiquitin': ss('Ubiquitin'),
                'alpha 2 macroglobulin': ss('alpha 2 macroglobulin'),
                'Chymotrypsinogen A': ss('Chymotrypsinogen A'),
                'Thaumatin': ss('Thaumatin'),
                'TCI': ss('TCI'),
                'Elastase': ss('Elastase'),
                'beta-Lactoglobulin': ss('beta-Lactoglobulin'),
                'Pepsin': ss('Pepsin'),
                'Tryp': ss('Tryp'),
                'SD': ss('SD'),
                'IgG': ss('IgG'),
                'Concanavalin A': ss('Concanavalin A'),
                'Peroxidase': ss_peroxidase('Peroxidase'),
                'prealbumin': ss_prealbumin('prealbumin')}


left_csv_files = glob.glob('A:\Amy\Protein Library\LIFEtime library\LEFT detector\*.csv')
right_csv_files = glob.glob('A:\Amy\Protein Library\LIFEtime library\RIGHT detector\*.csv')

# for whole 2D

left_proteins_2D = pd.DataFrame()
right_proteins_2D = pd.DataFrame()

for protein in proteins:
    left_protein_2D = pd.DataFrame()
    for left_csv_file in left_csv_files:
        if protein in left_csv_file:
            df_2D = pd.read_csv(left_csv_file, delimiter = '\t')
            stacked_left_df_2D = (pd.DataFrame(df_2D.loc[43:127, '1549.79349786178': '1737.31299582371'])).reset_index(drop = True).T.stack().to_frame().T
            left_protein_2D = (pd.concat([left_protein_2D, stacked_left_df_2D], axis = 0)).reset_index(drop = True)
    left_protein_2D['label'] = pd.DataFrame(class_labels.get(protein))
    left_protein_2D['alpha'] = pd.DataFrame(alpha_labels.get(protein))
    left_protein_2D['beta'] = pd.DataFrame(beta_labels.get(protein))
    left_protein_2D['group'] = pd.DataFrame(group_name.get(protein))
    left_protein_2D['protein'] = pd.DataFrame(protein_name.get(protein))
    left_proteins_2D = (pd.concat([left_proteins_2D, left_protein_2D], axis = 0)).reset_index(drop = True)   
    right_protein_2D = pd.DataFrame()
    for right_csv_file in right_csv_files:
        if protein in right_csv_file:
            df_2D = pd.read_csv(right_csv_file, delimiter = '\t')
            stacked_right_df_2D = (pd.DataFrame(df_2D.loc[16:100, '1549.79349786178': '1737.31299582371'])).reset_index(drop = True).T.stack().to_frame().T
            right_protein_2D = (pd.concat([right_protein_2D, stacked_right_df_2D], axis = 0)).reset_index(drop = True)
    right_protein_2D['label'] = pd.DataFrame(class_labels.get(protein))
    right_protein_2D['alpha'] = pd.DataFrame(alpha_labels.get(protein))
    right_protein_2D['beta'] = pd.DataFrame(beta_labels.get(protein))
    right_protein_2D['group'] = pd.DataFrame(group_name.get(protein))
    right_protein_2D['protein'] = pd.DataFrame(protein_name.get(protein))
    right_proteins_2D = (pd.concat([right_proteins_2D, right_protein_2D], axis = 0)).reset_index(drop = True)   

all_proteins_2D = (pd.concat([left_proteins_2D, right_proteins_2D], axis = 0)).reset_index(drop=True)
print(all_proteins_2D)

# for diagonals

left_proteins_diag = pd.DataFrame()
right_proteins_diag = pd.DataFrame()

for protein in proteins:
    left_protein_diag = pd.DataFrame()
    for left_csv_file in left_csv_files:
        if protein in left_csv_file:
            df_diag = pd.read_csv(left_csv_file, delimiter = '\t')        
            ints = pd.DataFrame(df_diag.loc[43:127, '1549.79349786178': '1737.31299582371'])       
            probe_freqs = pd.DataFrame(df_diag.loc[43:127, '0'])                                   
            spectrum = pd.concat([probe_freqs, ints], axis = 1).reset_index(drop = True)      
            pump_freq = []             
            diff_df = pd.DataFrame()    
            for (colname, colval) in spectrum.items():        
                pump = float(colname)                         
                pump_freq.append(pump)                        
                probe = spectrum.iloc[:, 0].to_list()         
                diff = pd.DataFrame([i - pump for i in probe])   
                diff_df = pd.concat([diff_df, diff], axis = 1)   
            diff_df.columns = [pump_freq]                        
            select = pd.DataFrame()                              
            for (colname, colval) in diff_df.items():            
                column_values = (colval.values).tolist()         
                selection = pd.DataFrame([(i*1 if (min(column_values, key = abs) == i) else i*0) for i in column_values])   
                select = pd.concat([select, selection], axis = 1)   
            select.columns = [pump_freq]                            
            select_df = select.drop(select.columns[0], axis = 1)    
            diag_row_index = []
            for (colname, colval) in select_df.items():
                column_values = (colval.values)
                diag_row_index.append([i for i in range(len(column_values)) if column_values[i] != 0])   
            row_index = []
            for i in diag_row_index:
                row_index.append(int(''.join(map(str, i))))   
            spec = spectrum[spectrum.columns[1:]]
            spec.columns = [list(range(0, 35))]               
            diag_column_index = list(range(0, 35))
            diagonal = []
            for a, b in zip(row_index, diag_column_index):
                diagonal.append(spec.iat[a, b])
            print(diagonal)
            diagonal_df = (pd.DataFrame(diagonal)).T
            left_protein_diag = (pd.concat([left_protein_diag, diagonal_df], axis = 0)).reset_index(drop = True)
    left_protein_diag['label'] = pd.DataFrame(class_labels.get(protein))
    left_protein_diag['alpha'] = pd.DataFrame(alpha_labels.get(protein))
    left_protein_diag['beta'] = pd.DataFrame(beta_labels.get(protein))
    left_protein_diag['group'] = pd.DataFrame(group_name.get(protein))
    left_protein_diag['protein'] = pd.DataFrame(protein_name.get(protein))
    left_proteins_diag = pd.concat([left_proteins_diag, left_protein_diag], axis = 0).reset_index(drop = True)
    right_protein_diag = pd.DataFrame()
    for right_csv_file in right_csv_files:
        if protein in right_csv_file:
            df_diag = pd.read_csv(right_csv_file, delimiter = '\t')        
            ints = pd.DataFrame(df_diag.loc[16:100, '1549.79349786178': '1737.31299582371'])
            probe_freqs = pd.DataFrame(df_diag.loc[16:100, '0'])                                   
            spectrum = pd.concat([probe_freqs, ints], axis = 1).reset_index(drop = True)      
            pump_freq = []             
            diff_df = pd.DataFrame()    
            for (colname, colval) in spectrum.items():        
                pump = float(colname)                         
                pump_freq.append(pump)                        
                probe = spectrum.iloc[:, 0].to_list()         
                diff = pd.DataFrame([i - pump for i in probe])   
                diff_df = pd.concat([diff_df, diff], axis = 1)   
            diff_df.columns = [pump_freq]                        
            select = pd.DataFrame()                              
            for (colname, colval) in diff_df.items():            
                column_values = (colval.values).tolist()         
                selection = pd.DataFrame([(i*1 if (min(column_values, key = abs) == i) else i*0) for i in column_values])   
                select = pd.concat([select, selection], axis = 1)   
            select.columns = [pump_freq]                            
            select_df = select.drop(select.columns[0], axis = 1)    
            diag_row_index = []
            for (colname, colval) in select_df.items():
                column_values = (colval.values)
                diag_row_index.append([i for i in range(len(column_values)) if column_values[i] != 0])   
            row_index = []
            for i in diag_row_index:
                row_index.append(int(''.join(map(str, i))))   
            spec = spectrum[spectrum.columns[1:]]
            spec.columns = [list(range(0, 35))]               
            diag_column_index = list(range(0, 35))
            diagonal = []
            for a, b in zip(row_index, diag_column_index):
                diagonal.append(spec.iat[a, b])
            print(diagonal)
            diagonal_df = (pd.DataFrame(diagonal)).T
            right_protein_diag = (pd.concat([right_protein_diag, diagonal_df], axis = 0)).reset_index(drop = True)
    right_protein_diag['label'] = pd.DataFrame(class_labels.get(protein))
    right_protein_diag['alpha'] = pd.DataFrame(alpha_labels.get(protein))
    right_protein_diag['beta'] = pd.DataFrame(beta_labels.get(protein))
    right_protein_diag['group'] = pd.DataFrame(group_name.get(protein))
    right_protein_diag['protein'] = pd.DataFrame(protein_name.get(protein))
    right_proteins_diag = pd.concat([right_proteins_diag, right_protein_diag], axis = 0).reset_index(drop = True)

print(left_proteins_diag)
print(right_proteins_diag)

all_proteins_diag = (pd.concat([left_proteins_diag, right_proteins_diag], axis = 0)).reset_index(drop=True)
print(all_proteins_diag)

## looping through input data type (2D or diagonals), feature selection method, and model

# define feature selection method
pca = PCA(whiten = True)
Agglom = FeatureAgglomeration()
AF_class = SelectKBest(score_func = f_classif)
MIR_class = SelectKBest(score_func = mutual_info_classif)
AF_reg = SelectKBest(score_func = f_regression)
MIR_reg = SelectKBest(score_func = mutual_info_regression)

# define models - classification

svc = SVC(decision_function_shape = 'ovr', class_weight = 'balanced')
dt = DecisionTreeClassifier()
knn = KNeighborsClassifier()
rf = RandomForestClassifier()
ert = ExtraTreesClassifier()
HGBC = HistGradientBoostingClassifier()
adaboost = AdaBoostClassifier(algorithm = 'SAMME')

# define models - regression

svr = SVR()
dt_reg = DecisionTreeRegressor()
knn_reg = KNeighborsRegressor()
rf_reg = RandomForestRegressor()
ert_reg = ExtraTreesRegressor()
HGBC_reg = HistGradientBoostingRegressor()
adaboost_reg = AdaBoostRegressor()

# voting models

base_models = [
    ('dt', DecisionTreeClassifier()),
    ('svm', SVC(kernel = 'rbf', decision_function_shape = 'ovr', class_weight = 'balanced')),
    ('knn', KNeighborsClassifier())
]
voting = VotingClassifier(base_models)

# and their GridSearch parameters - classification

svm_params = {'svc__C': [0.01, 0.1, 1, 10, 100],
              'svc__gamma': [0.01, 0.1, 1, 10, 100]}

dt_params = {'decisiontreeclassifier__max_depth': [None, 1, 2, 3],
             'decisiontreeclassifier__criterion': ['gini', 'entropy']}

knn_params = {'kneighborsclassifier__n_neighbors': [1, 2, 3, 4, 5, 10], 
              'kneighborsclassifier__weights': ['uniform', 'distance'], 
              'kneighborsclassifier__metric': ['euclidean', 'manhattan', 'minkowski']}

rf_params = {'randomforestclassifier__n_estimators': [50, 100, 200, 500],
             'randomforestclassifier__max_depth': [3, 5, 10, None],
             'randomforestclassifier__criterion' : ['gini', 'entropy', 'log_loss']}

ert_params = {'extratreesclassifier__n_estimators': [50, 100, 200, 500],
              'extratreesclassifier__max_depth': [3, 5, 10, None],
              'extratreesclassifier__criterion' : ['gini', 'entropy', 'log_loss']}

HGBC_params = {'histgradientboostingclassifier__learning_rate': [0.01, 0.1, 0.2, 1],
               'histgradientboostingclassifier__max_depth': [3, 5, 10, None]}

adaboost_params = {'adaboostclassifier__n_estimators': [5, 10, 25, 50],
                   'adaboostclassifier__learning_rate': [0.001, 0.01, 0.1, 1]}

# and their GridSearch parameters - regression

svr_params = {'svr__C': [0.01, 0.1, 1, 10, 100],
              'svr__gamma': [0.01, 0.1, 1, 10, 100]}

dt_reg_params = {'decisiontreeregressor__max_depth': [None, 1, 2, 3],
                 'decisiontreeregressor__criterion': ['gini', 'entropy']}

knn_reg_params = {'kneighborsregressor__n_neighbors': [1, 2, 3, 4, 5, 10], 
                  'kneighborsregressor__weights': ['uniform', 'distance'], 
                  'kneighborsregressor__metric': ['euclidean', 'manhattan', 'minkowski']}

rf_reg_params = {'randomforestregressor__n_estimators': [50, 100, 200, 500],
                 'randomforestregressor__max_depth': [3, 5, 10, None],
                 'randomforestregressor__criterion' : ['gini', 'entropy', 'log_loss']}

ert_reg_params = {'extratreesregressor__n_estimators': [50, 100, 200, 500],
                  'extratreesregressor__max_depth': [3, 5, 10, None],
                  'extratreesregressor__criterion' : ['gini', 'entropy', 'log_loss']}

HGBC_reg_params = {'histgradientboostingregressor__learning_rate': [0.01, 0.1, 0.2, 1],
                   'histgradientboostingregressorr__max_depth': [3, 5, 10, None]}

adaboost_reg_params = {'adaboostregressor__n_estimators': [5, 10, 25, 50],
                       'adaboostregressor__learning_rate': [0.001, 0.01, 0.1, 1]}

# voting model parameters

voting_params = {
    'votingclassifier__dt__max_depth': [None, 1, 2, 3],
    'votingclassifier__dt__criterion': ['gini', 'entropy'],
    'votingclassifier__svm__C': [0.01, 0.1, 1, 10, 100],
    'votingclassifier__svm__gamma': [0.01, 0.1, 1, 10],
    'votingclassifier__knn__n_neighbors': [2, 3, 5, 10],
    'votingclassifier__knn__weights': ['uniform', 'distance'],
    'votingclassifier__knn__metric': ['euclidean', 'manhattan', 'minkowski']
}

# parameter dictionary

params_dict = {svc : svm_params,
               dt : dt_params,
               knn : knn_params,
               rf : rf_params,
               ert : ert_params,
               HGBC : HGBC_params,
               adaboost : adaboost_params,
               svr : svr_params,
               dt_reg : dt_reg_params,
               knn_reg : knn_reg_params,
               rf_reg : rf_reg_params,
               ert_reg : ert_reg_params,
               HGBC_reg : HGBC_reg_params,
               adaboost_reg : adaboost_reg_params}

# input data type list

input_types = [all_proteins_2D]

# feature method and model lists - classification

class_feature_methods = [pca, Agglom, AF_class, MIR_class]
class_models = [svc, dt, knn, rf, ert, HGBC, adaboost]

# feature method and model lists - regression

reg_feature_methods = [pca, Agglom, AF_reg, MIR_reg]
reg_models = [svr, dt_reg, knn_reg, rf_reg, ert_reg, HGBC_reg, adaboost_reg]

# big loop - classification

diag_PCA_results_df = pd.DataFrame()
diag_Agglom_results_df = pd.DataFrame()
diag_AF_results_df = pd.DataFrame()
diag_MIR_results_df = pd.DataFrame()
PCA_2D_results_df = pd.DataFrame()
Agglom_2D_results_df = pd.DataFrame()
AF_2D_results_df = pd.DataFrame()
MIR_2D_results_df = pd.DataFrame()

for input_type in input_types:
    protein_7 = []
    for i, val in enumerate(proteins):
        test_protein = val
        train_proteins = copy.deepcopy(proteins)
        train_proteins.pop(i)
        print(f'test protein: {test_protein}')
        test_set = input_type[input_type['protein'].str.contains(test_protein)]
        train_set = pd.DataFrame()
        for train_protein in train_proteins:
            train_set_1 = input_type[input_type['protein'].str.contains(train_protein)]
            train_set = pd.concat([train_set, train_set_1], axis = 0).reset_index(drop = True)
        test_labels = pd.DataFrame(test_set.label)
        train_labels = pd.DataFrame(train_set.label)
        y_train = train_labels.values.ravel()
        y_test = test_labels.values.ravel()
        group_test = (pd.DataFrame(test_set.group)).values.ravel()
        group_train = (pd.DataFrame(train_set.group)).values.ravel()
        if input_type is all_proteins_diag:
            x_train = train_set[train_set.columns[:35]]
            x_test = test_set[test_set.columns[:35]]
        else:
            x_train = train_set[train_set.columns[:2975]]
            x_test = test_set[test_set.columns[:2975]]
        protein_7 = [test_protein] * 7
        gkf = GroupKFold(n_splits = 5)
        cv = GroupKFold(n_splits = 10)
        hyperparameters_PCA = []
        training_accuracies_PCA = []
        training_acc_stddev_PCA = []
        testing_accuracy_PCA = []
        hyperparameters_Agglom = []
        training_accuracies_Agglom = []
        training_acc_stddev_Agglom = []
        testing_accuracy_Agglom = []
        hyperparameters_AF = []
        training_accuracies_AF = []
        training_acc_stddev_AF = []
        testing_accuracy_AF = []
        hyperparameters_MIR = []
        training_accuracies_MIR = []
        training_acc_stddev_MIR = []
        testing_accuracy_MIR = []
        for class_model in class_models:
            for class_feature_method in class_feature_methods:
                if class_feature_method is pca:
                    print('feature selection method: PCA')
                    pipeline_pca = make_pipeline(StandardScaler(), pca, class_model)
                    grid_pca = params_dict.get(class_model)
                    grid_pca = {**grid_pca, 'pca__n_components': [2, 3, 4, 5, 10]}
                    print(f'GridSearch: {grid_pca}')
                    grid_search_pca = GridSearchCV(estimator = pipeline_pca, param_grid = grid_pca, cv = gkf)
                    grid_search_pca.fit(X = x_train, y = y_train, groups = group_train)
                    parameters_pca = grid_search_pca.best_params_
                    final_pipe_pca = grid_search_pca.best_estimator_
                    hyperparameters_PCA.append(parameters_pca)
                    cross_val_scores_PCA = cross_val_score(final_pipe_pca, X = x_train, y = y_train, cv = cv, groups = group_train)
                    training_accuracies_PCA.append(np.average(cross_val_scores_PCA))
                    training_acc_stddev_PCA.append(np.std(cross_val_scores_PCA))
                    final_pipe_pca.fit(x_train, y_train)
                    y_predict_PCA = final_pipe_pca.predict(x_test)
                    accuracy_PCA = accuracy_score(y_test, y_predict_PCA)
                    print(f'testing accuracy = {accuracy_PCA}')
                    testing_accuracy_PCA.append(accuracy_PCA)
                elif class_feature_method is Agglom:
                    print('feature selection method: feature agglomeration')
                    pipeline_Agglom = make_pipeline(StandardScaler(), Agglom, class_model)
                    grid_Agglom = params_dict.get(class_model)
                    grid_Agglom = {**grid_Agglom, 'featureagglomeration__n_clusters': [2, 3, 4, 5, 10]}
                    print(f'GridSearch: {grid_Agglom}')
                    grid_search_Agglom = GridSearchCV(estimator = pipeline_Agglom, param_grid = grid_Agglom, cv = gkf)
                    grid_search_Agglom.fit(X = x_train, y = y_train, groups = group_train)
                    parameters_Agglom = grid_search_Agglom.best_params_
                    final_pipe_Agglom = grid_search_Agglom.best_estimator_
                    hyperparameters_Agglom.append(parameters_Agglom)
                    cross_val_scores_Agglom = cross_val_score(final_pipe_Agglom, X = x_train, y = y_train, cv = cv, groups = group_train)
                    training_accuracies_Agglom.append(np.average(cross_val_scores_Agglom))
                    training_acc_stddev_Agglom.append(np.std(cross_val_scores_Agglom))
                    final_pipe_Agglom.fit(x_train, y_train)
                    y_predict_Agglom = final_pipe_Agglom.predict(x_test)
                    accuracy_Agglom = accuracy_score(y_test, y_predict_Agglom)
                    print(f'testing accuracy = {accuracy_Agglom}')
                    testing_accuracy_Agglom.append(accuracy_Agglom)
                elif class_feature_method is AF_class:
                    print('feature selection method: f_classif')
                    pipeline_AF = make_pipeline(StandardScaler(), AF_class, class_model)
                    grid_AF = params_dict.get(class_model)
                    if input_type is all_proteins_diag:
                        grid_AF = {**grid_AF, 'selectkbest__k': [5, 10, 15, 20, 25, 30]}
                    else:
                        grid_AF = {**grid_AF, 'selectkbest__k': [10, 20, 30, 40, 50]}
                    print(f'GridSearch: {grid_AF}')
                    grid_search_AF = GridSearchCV(estimator = pipeline_AF, param_grid = grid_AF, cv = gkf)
                    grid_search_AF.fit(X = x_train, y = y_train, groups = group_train)
                    parameters_AF = grid_search_AF.best_params_
                    final_pipe_AF = grid_search_AF.best_estimator_
                    hyperparameters_AF.append(parameters_AF)
                    cross_val_scores_AF = cross_val_score(final_pipe_AF, X = x_train, y = y_train, cv = cv, groups = group_train)
                    training_accuracies_AF.append(np.average(cross_val_scores_AF))
                    training_acc_stddev_AF.append(np.std(cross_val_scores_AF))
                    final_pipe_AF.fit(x_train, y_train)
                    y_predict_AF = final_pipe_AF.predict(x_test)
                    accuracy_AF = accuracy_score(y_test, y_predict_AF)
                    print(f'testing accuracy = {accuracy_AF}')
                    testing_accuracy_AF.append(accuracy_AF)
                else:
                    print('feature selection method: mutual info classif')
                    pipeline_MIR = make_pipeline(StandardScaler(), MIR_class, class_model)
                    grid_MIR = params_dict.get(class_model)
                    if input_type is all_proteins_diag:
                        grid_MIR = {**grid_MIR, 'selectkbest__k': [5, 10, 15, 20, 25, 30]}
                    else:
                        grid_MIR = {**grid_MIR, 'selectkbest__k': [10, 20, 30, 40, 50]}
                    print(f'GridSearch: {grid_MIR}')
                    grid_search_MIR = GridSearchCV(estimator = pipeline_MIR, param_grid = grid_MIR, cv = gkf)
                    grid_search_MIR.fit(X = x_train, y = y_train, groups = group_train)
                    parameters_MIR = grid_search_MIR.best_params_
                    final_pipe_MIR = grid_search_MIR.best_estimator_
                    hyperparameters_MIR.append(parameters_MIR)
                    cross_val_scores_MIR = cross_val_score(final_pipe_MIR, X = x_train, y = y_train, cv = cv, groups = group_train)
                    training_accuracies_MIR.append(np.average(cross_val_scores_MIR))
                    training_acc_stddev_MIR.append(np.std(cross_val_scores_MIR))
                    final_pipe_MIR.fit(x_train, y_train)
                    y_predict_MIR = final_pipe_MIR.predict(x_test)
                    accuracy_MIR = accuracy_score(y_test, y_predict_MIR)
                    print(f'testing accuracy = {accuracy_MIR}')
                    testing_accuracy_MIR.append(accuracy_MIR)
        results_PCA = pd.DataFrame({'protein': protein_7, 'model': class_models, 'hyperparameters': hyperparameters_PCA, 
                                    'training accuracy': training_accuracies_PCA, '+-': training_acc_stddev_PCA, 
                                    'testing accuracy': testing_accuracy_PCA})
        print(results_PCA)
        results_Agglom = pd.DataFrame({'protein': protein_7, 'model': class_models, 'hyperparameters': hyperparameters_Agglom, 
                                       'training accuracy': training_accuracies_Agglom, '+-': training_acc_stddev_Agglom, 
                                       'testing accuracy': testing_accuracy_Agglom})
        print(results_Agglom)
        results_AF = pd.DataFrame({'protein': protein_7, 'model': class_models, 'hyperparameters': hyperparameters_AF, 
                                    'training accuracy': training_accuracies_AF, '+-': training_acc_stddev_AF, 
                                    'testing accuracy': testing_accuracy_AF})
        print(results_AF)
        results_MIR = pd.DataFrame({'protein': protein_7, 'model': class_models, 'hyperparameters': hyperparameters_MIR, 
                                    'training accuracy': training_accuracies_MIR, '+-': training_acc_stddev_MIR, 
                                    'testing accuracy': testing_accuracy_MIR})
        print(results_MIR)
        if input_type is all_proteins_diag:
            diag_PCA_results_df = pd.concat([diag_PCA_results_df, results_PCA], axis = 0).reset_index(drop = True)
            diag_Agglom_results_df = pd.concat([diag_Agglom_results_df, results_Agglom], axis = 0).reset_index(drop = True)
            diag_AF_results_df = pd.concat([diag_AF_results_df, results_AF], axis = 0).reset_index(drop = True)
            diag_MIR_results_df = pd.concat([diag_MIR_results_df, results_MIR], axis = 0).reset_index(drop = True)
        else:
            PCA_2D_results_df = pd.concat([PCA_2D_results_df, results_PCA], axis = 0).reset_index(drop = True)
            Agglom_2D_results_df = pd.concat([Agglom_2D_results_df, results_Agglom], axis = 0).reset_index(drop = True)
            AF_2D_results_df = pd.concat([AF_2D_results_df, results_AF], axis = 0).reset_index(drop = True)
            MIR_2D_results_df = pd.concat([MIR_2D_results_df, results_MIR], axis = 0).reset_index(drop = True)
        print(diag_PCA_results_df)
        print(diag_Agglom_results_df)
        print(diag_AF_results_df)
        print(diag_MIR_results_df)
        print(PCA_2D_results_df)
        print(Agglom_2D_results_df)
        print(AF_2D_results_df)
        print(MIR_2D_results_df)

print(diag_PCA_results_df)
print(diag_Agglom_results_df)
print(diag_AF_results_df)
print(diag_MIR_results_df)
print(PCA_2D_results_df)
print(Agglom_2D_results_df)
print(AF_2D_results_df)
print(MIR_2D_results_df)



