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
from sklearn.svm import SVC, SVR
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor 
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.multioutput import RegressorChain

# import metrics
from sklearn.metrics import accuracy_score, cohen_kappa_score, r2_score, mean_squared_error, root_mean_squared_error

# putting function here test

def GridSearchPredict(x_train, y_train, x_test, y_test, group_train, test_protein, class_model, class_feature_method, pipeline, grid):
    gkf = GroupKFold(n_splits = 5)
    cv = GroupKFold(n_splits = 10)
    grid_search = GridSearchCV(estimator = pipeline, param_grid = grid, cv = gkf)
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
    prediction = pd.DataFrame({'Protein': test_protein, 'Model': class_model, 'Feature Selection': class_feature_method,
                               'Hyperparameters': hyperparameters, 'Training accuracy': training_acc, '+-': training_acc_std,
                               'Testing accuracy': accuracy})
    return prediction

# Protein names and labels

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

all_proteins_2D = load_in_2D(proteins, left_csv_files, right_csv_files, class_labels, alpha_labels, beta_labels, group_name, protein_name)
print(all_proteins_2D)

# load in diagonals

all_proteins_diag = load_in_diag(proteins, left_csv_files, right_csv_files, class_labels, alpha_labels, beta_labels, group_name, protein_name)
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

class_all_pred = pd.DataFrame()

for input_type in input_types:
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
        for class_model in class_models:
            for class_feature_method in class_feature_methods:
                pipeline = make_pipeline(StandardScaler(), class_feature_method, class_model)
                grid = params_dict.get(class_model)
                if class_feature_method is pca:
                    print(f'feature selection method: PCA')
                    grid = {**grid, 'pca__n_components': [2, 3, 4, 5, 10]}
                    print(f'GridSearch: {grid}')
                elif class_feature_method is Agglom:
                    print(f'feature selection method: Feature Agglomeration')
                    grid = {**grid, 'featureagglomeration__n_clusters': [2, 3, 4, 5, 10]}
                    print(f'GridSearch: {grid}')
                elif class_feature_method is AF_class:
                    print(f'feature selection method: ANOVA-F')
                    if input_type is all_proteins_diag:
                        grid = {**grid, 'selectkbest__k': [5, 10, 15, 20, 25, 30]}
                    else:
                        grid = {**grid, 'selectkbest__k': [10, 20, 30, 40, 50]}
                    print(f'GridSearch: {grid}')
                else:
                    print('feature selection method: Mutual Info')
                    if input_type is all_proteins_diag:
                        grid = {**grid, 'selectkbest__k': [5, 10, 15, 20, 25, 30]}
                    else:
                        grid = {**grid, 'selectkbest__k': [10, 20, 30, 40, 50]}
                    print(f'GridSearch: {grid}')
                prediction = GridSearchPredict(x_train, y_train, x_test, y_test, group_train, test_protein, 
                                               class_model, class_feature_method, pipeline, grid)
                class_all_pred = pd.concat([class_all_pred, prediction], axis = 0).reset_index(drop = True)
                print(class_all_pred)
        
print(class_all_pred)  