# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 16:30:46 2026

@author: af1226
"""
# import things you always need
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

# import load in functions
from fname import get_fname
from load2DfuncTest import load_in_2D, generateDiags, load_in_diag

# import things needed for pipeline building and GridSearching
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, GroupKFold
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif, mutual_info_regression, SelectKBest

# import models
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

# import metrics
from sklearn.metrics import accuracy_score, cohen_kappa_score

# GridSearch + Prediction function

def GridSearchPredict(x_train, y_train, x_test, y_test, group_train, set_name, pipeline, grid):
    gkf = GroupKFold(n_splits = 5)
    cv = GroupKFold(n_splits = 10)
    grid_search = GridSearchCV(estimator = pipeline, param_grid = grid, cv = gkf, n_jobs = 8)
    grid_search.fit(X = x_train, y = y_train, groups = group_train)
    parameters = grid_search.best_params_
    hyperparameters = [parameters]
    final_pipe = grid_search.best_estimator_
    cross_val_scores = cross_val_score(final_pipe, X = x_train, y = y_train, cv = cv, groups = group_train)
    training_acc = np.average(cross_val_scores)
    training_acc_std = np.std(cross_val_scores)
    final_pipe.fit(x_train, y_train)
    y_predict = final_pipe.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    kappa = cohen_kappa_score(y_test, y_predict)
    prediction = pd.DataFrame({'Test set': set_name, 'Hyperparameters': hyperparameters, 'Training accuracy': training_acc, 
                               '+-': training_acc_std,'Testing accuracy': accuracy, 'Kappa': kappa})
    return prediction

# list proteins and label dictionaries

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

# extract appropriate files 

fname_left, fname_right = get_fname()
left_csv_files = glob.glob(fname_left)
right_csv_files = glob.glob(fname_right)

# load in whole 2D

left_2D, right_2D, all_2D = load_in_2D(proteins, left_csv_files, right_csv_files, class_labels, alpha_labels, beta_labels, group_name, protein_name)
print(all_2D)

# building separate left and right models

test_proteins_1 = ['Creatine', 'Lipoxidase', 'Protease', 'alpha amylase', 'Ovalbumin', 'b glucuronidase', 'TCI']
test_proteins_2 = ['Calmodulin', 'cytochrome c', 'DT diaphorase', 'Lactoferrin bovin', 'Conalbumin', 'Ovalbumin', 'Elastase']
test_proteins_3 = ['HSA', 'Protease', 'Choline oxidase', 'Gly 3 phos', 'Ubiquitin', 'Thaumatin', 'Concanavalin A']

test_proteins = [test_proteins_1, test_proteins_2, test_proteins_3]
test_set_names = ['test_1', 'test_2', 'test_3']

svc = SVC(decision_function_shape = 'ovr', class_weight = 'balanced')

svm_params = {'svc__C': [0.01, 0.1, 1, 10, 100],
              'svc__gamma': [0.01, 0.1, 1, 10, 100]}

params_dict = {svc : svm_params}

classification_80_20 = pd.DataFrame()
for lst, set_name in zip(test_proteins, test_set_names):
    print(f'test proteins: {lst}')
    train_proteins = [i for i in proteins if i not in lst]
    print(f'train proteins: {train_proteins}')
    test_set = pd.DataFrame()
    for i in lst:
        test_set_1 = all_2D[all_2D['protein'].str.contains(i)]
        test_set = pd.concat([test_set, test_set_1], axis = 0).reset_index(drop=True)
    train_set = pd.DataFrame()
    for train_protein in train_proteins:
        train_set_1 = all_2D[all_2D['protein'].str.contains(train_protein)]
        train_set = pd.concat([train_set, train_set_1], axis = 0).reset_index(drop=True)
    test_labels = pd.DataFrame(test_set.label)
    train_labels = pd.DataFrame(train_set.label)
    y_train = train_labels.values.ravel()
    y_test = test_labels.values.ravel()
    group_test = (pd.DataFrame(test_set.group)).values.ravel()
    group_train = (pd.DataFrame(train_set.group)).values.ravel()
    grid = params_dict.get(svc)
    x_train = train_set[train_set.columns[:2975]]
    x_test = test_set[test_set.columns[:2975]]
    pipeline = make_pipeline(StandardScaler(), svc)
    print(f'GridSearch: {grid}')
    prediction = GridSearchPredict(x_train, y_train, x_test, y_test, group_train, set_name, pipeline, grid)
    classification_80_20 = pd.concat([classification_80_20, prediction], axis = 0).reset_index(drop = True)
    print(classification_80_20)

print(classification_80_20)  

