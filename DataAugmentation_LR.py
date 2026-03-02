# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 10:30:28 2026

@author: af1226
"""

# import things you always need
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import copy
import random

# import load in functions
from fname import get_fname
from load2DfuncTest import load_in_2D

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
from sklearn.ensemble import VotingClassifier
from sklearn.multioutput import RegressorChain

# import metrics
from sklearn.metrics import accuracy_score, cohen_kappa_score

# GridSearch + Prediction function

def GridSearchPredictReg(x_train, y_train, group_train, pipeline, grid, test_protein, k_value):
    gkf = GroupKFold(n_splits = 5)
    cv = GroupKFold(n_splits = 10)
    grid_search = GridSearchCV(estimator = pipeline, param_grid = grid, cv = gkf)
    grid_search.fit(X = x_train, y = y_train, groups = group_train)
    parameters = grid_search.best_params_
    C = [parameters.values()][0]
    final_pipe = grid_search.best_estimator_
    cross_val_scores = cross_val_score(final_pipe, X = x_train, y = y_train, cv = cv, groups = group_train)
    training_rmse = np.average(cross_val_scores)
    training_rmse_std = np.std(cross_val_scores)
    training_performance = pd.DataFrame({'Test protein': test_protein, 'C': C, 'k_value': k_value,
                                         'rmse': training_rmse, '+-': training_rmse_std}, index = [0])
    return training_performance

def TrainTestReg(test_protein, train_proteins, dataset):
    test_set = pd.DataFrame()
    test_set = dataset[dataset['protein'].str.contains(test_protein)]
    train_set = pd.DataFrame()
    for train_protein in train_proteins:
        train_set_1 = dataset[dataset['protein'].str.contains(train_protein)]
        train_set = pd.concat([train_set, train_set_1], axis = 0).reset_index(drop=True)
    x_train = train_set[train_set.columns[:2975]]
    y1_train = (pd.DataFrame(train_set.alpha)).values.ravel()
    y2_train = (pd.DataFrame(train_set.beta)).values.ravel()
    y_train = np.array(pd.concat([pd.DataFrame(train_set.alpha), pd.DataFrame(train_set.beta)], axis = 1))
    x_test = test_set[test_set.columns[:2975]]
    y1_test = (pd.DataFrame(test_set.alpha)).values.ravel()
    y2_test = (pd.DataFrame(test_set.beta)).values.ravel()
    y_test = np.array(pd.concat([pd.DataFrame(test_set.alpha), pd.DataFrame(test_set.beta)], axis = 1))
    group_train = (pd.DataFrame(train_set.group)).values.ravel()
    return x_train, y1_train, y2_train, y_train, x_test, y1_test, y2_test, y_test, group_train

def CombinedFtest(x_train, x_test, y1_train, y2_train, k_value):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import f_regression
    scaler = StandardScaler()
    scaled = scaler.fit_transform(x_train)
    feature = pd.DataFrame(list(range(0, len(x_train.columns) + 1)))
    label1 = pd.DataFrame((pd.DataFrame(f_regression(scaled, y1_train))).iloc[0])
    label1['feature'] = feature
    label1 = label1.sort_values(by = [0])
    label2 = pd.DataFrame((pd.DataFrame(f_regression(scaled, y2_train))).iloc[0])
    label2['feature'] = feature
    label2 = label2.sort_values(by = [0])
    features_1 = label1.iloc[(len(x_train.columns) - k_value):]
    features_2 = label2.iloc[(len(x_train.columns) - k_value):]
    combined_features = (((pd.concat([features_1, features_2], axis = 0)).reset_index(drop = True).loc[:,'feature']).drop_duplicates()).values.tolist()
    x_train_AF = pd.DataFrame()
    for i in combined_features:
        feature_values = pd.DataFrame(x_train.iloc[:,i])
        x_train_AF = pd.concat([x_train_AF, feature_values], axis = 1)
    x_test_AF = pd.DataFrame()
    for i in combined_features:
        feature_values = pd.DataFrame(x_test.iloc[:,i])
        x_test_AF = pd.concat([x_test_AF, feature_values], axis = 1)
    return x_train_AF, x_test_AF

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
print(left_2D, right_2D, all_2D)

# data augmentation LOO reg

svr = SVR()
reg_chain = RegressorChain(SVR(kernel = 'rbf'), order = [1, 0])
svm_params = {'regressorchain__base_estimator__C': [0.01, 0.1, 1, 10, 100]}
k_values = [20, 30, 40, 50]
params_dict = {svr : svm_params}

scaling = np.linspace(0.7, 1.3, num = 1000).tolist()
noises = np.linspace(0.000025, 0.00001, num = 1000).tolist()

training_performance_left = pd.DataFrame()
training_performance_right = pd.DataFrame()
prediction_left_dfs = []
prediction_right_dfs = []
for i, val in enumerate(proteins):
    test_protein = val
    train_proteins = copy.deepcopy(proteins)
    train_proteins.pop(i)
    print(f'test protein: {test_protein}')
    pipeline = make_pipeline(StandardScaler(), reg_chain)
    grid = params_dict.get(svr)
    # left
    x_train_left_all, y1_train_left_half, y2_train_left_half, y_train_left_half, x_test_left, y1_test_left, y2_test_left, y_test_left, group_train_left_half = TrainTestReg(test_protein, train_proteins, dataset = left_2D)
    x_test_left = pd.DataFrame(np.array(x_test_left))
    y1_train_left = np.concatenate((y1_train_left_half, y1_train_left_half))
    y2_train_left = np.concatenate((y2_train_left_half, y2_train_left_half))
    y_train_left = np.concatenate((y_train_left_half, y_train_left_half))
    group_train_left = np.concatenate((group_train_left_half, group_train_left_half))
    # augment data
    print('augmenting data...')
    augmented = pd.DataFrame()
    augment = copy.deepcopy(x_train_left_all)
    for i, row in augment.iterrows():
        scaled = np.array(row)*random.choice(scaling)
        noise = np.random.normal(0, random.choice(noises), 2975)
        scaled_noisy = pd.DataFrame(scaled + noise).T
        augmented = pd.concat([augmented, scaled_noisy], axis = 0).reset_index(drop=True)
    x_train_left = pd.DataFrame(np.concatenate((np.array(x_train_left_all), augmented), axis = 0))
    # train
    print('training left...')
    find_k_left = pd.DataFrame()
    for k_value in k_values:
        x_train_AF_left, x_test_AF_left = CombinedFtest(x_train = x_train_left, x_test = x_test_left, y1_train = y1_train_left, 
                                                        y2_train = y2_train_left, k_value = k_value)
        per_k = GridSearchPredictReg(x_train = x_train_AF_left, y_train = y_train_left, group_train = group_train_left, 
                                     pipeline = pipeline, grid = grid, test_protein = test_protein, k_value = k_value)
        find_k_left = pd.concat([find_k_left, per_k], axis = 0).reset_index(drop=True)
    min_index_left = find_k_left['rmse'].idxmax()
    C_final_left = find_k_left['C'].iloc[min_index_left]
    k_final_left = find_k_left['k_value'].iloc[min_index_left]
    to_save = pd.DataFrame(find_k_left.iloc[min_index_left]).T
    training_performance_left = pd.concat([training_performance_left, to_save], axis = 0).reset_index(drop=True)
    # right
    x_train_right_all, y1_train_right_half, y2_train_right_half, y_train_right_half, x_test_right, y1_test_right, y2_test_right, y_test_right, group_train_right_half = TrainTestReg(test_protein, train_proteins, dataset = right_2D)
    x_test_right = pd.DataFrame(np.array(x_test_right))
    y1_train_right = np.concatenate((y1_train_right_half, y1_train_right_half))
    y2_train_right = np.concatenate((y2_train_right_half, y2_train_right_half))
    y_train_right = np.concatenate((y_train_right_half, y_train_right_half))
    group_train_right = np.concatenate((group_train_right_half, group_train_right_half))
    # augment data
    print('augmenting data...')
    augmented = pd.DataFrame()
    augment = copy.deepcopy(x_train_right_all)
    for i, row in augment.iterrows():
        scaled = np.array(row)*random.choice(scaling)
        noise = np.random.normal(0, random.choice(noises), 2975)
        scaled_noisy = pd.DataFrame(scaled + noise).T
        augmented = pd.concat([augmented, scaled_noisy], axis = 0).reset_index(drop=True)
    x_train_right = pd.DataFrame(np.concatenate((np.array(x_train_right_all), augmented), axis = 0))
    # train
    print('training right...')
    find_k_right = pd.DataFrame()
    for k_value in k_values:
        x_train_AF_right, x_test_AF_right = CombinedFtest(x_train = x_train_right, x_test = x_test_right, y1_train = y1_train_right, 
                                                          y2_train = y2_train_right, k_value = k_value)
        per_k = GridSearchPredictReg(x_train = x_train_AF_right, y_train = y_train_right, group_train = group_train_right, 
                                     pipeline = pipeline, grid = grid, test_protein = test_protein, k_value = k_value)
        find_k_right = pd.concat([find_k_right, per_k], axis = 0).reset_index(drop=True)
    min_index_right = find_k_right['rmse'].idxmax()
    C_final_right = find_k_right['C'].iloc[min_index_right]
    k_final_right = find_k_right['k_value'].iloc[min_index_right]
    to_save = pd.DataFrame(find_k_right.iloc[min_index_right]).T
    training_performance_right = pd.concat([training_performance_right, to_save], axis = 0).reset_index(drop=True)
    # x_train and test according to k_value_final
    x_train_left_fin, x_test_left_fin = CombinedFtest(x_train = x_train_left, x_test = x_test_left, y1_train = y1_train_left, 
                                                      y2_train = y2_train_left, k_value = k_final_left)
    x_train_right_fin, x_test_right_fin = CombinedFtest(x_train = x_train_right, x_test = x_test_right, y1_train = y1_train_right, 
                                                       y2_train = y2_train_right, k_value = k_final_right)
    # final pipelines
    pipe_left = make_pipeline(StandardScaler(), RegressorChain(SVR(kernel = 'rbf', C = C_final_left), order = [1, 0]))
    pipe_right = make_pipeline(StandardScaler(), RegressorChain(SVR(kernel = 'rbf', C = C_final_right), order = [1, 0]))
    pipe_left.fit(x_train_left_fin, y_train_left)
    pipe_right.fit(x_train_right_fin, y_train_right)
    # testing
    y_predict_left = pd.DataFrame((pipe_left.predict(x_test_left_fin)), columns = ['alpha_prediction', 'beta_prediction'])
    y_predict_right = pd.DataFrame((pipe_right.predict(x_test_right_fin)), columns = ['alpha_prediction', 'beta_prediction'])
    y_predict_left['alpha'] = y1_test_left.tolist()
    y_predict_left['beta'] = y2_test_left.tolist()
    y_predict_right['alpha'] = y1_test_right.tolist()
    y_predict_right['beta'] = y2_test_right.tolist()
    print(f'left prediction: {y_predict_left}')
    print(f'right prediction: {y_predict_right}')
    prediction_left_dfs.append(y_predict_left)
    prediction_right_dfs.append(y_predict_right)

predictions_left = pd.DataFrame()
for i in range(0, len(prediction_left_dfs)):
    globals()[f'prediction_left_dfs{i+1}'] = prediction_left_dfs[i]
    prediction_left_dfs[i] = pd.DataFrame(prediction_left_dfs[i])
    predictions_left = (pd.concat([predictions_left, prediction_left_dfs[i]], axis = 0)).reset_index(drop = True)

predictions_right = pd.DataFrame()
for i in range(0, len(prediction_right_dfs)):
    globals()[f'prediction_right_dfs{i+1}'] = prediction_right_dfs[i]
    prediction_right_dfs[i] = pd.DataFrame(prediction_right_dfs[i])
    predictions_right = (pd.concat([predictions_right, prediction_right_dfs[i]], axis = 0)).reset_index(drop = True)
    
training_performance_left.to_csv('/home/Amy/dataAugmentation/training LOO left regression performance DATA AUGMENTATION.csv')
training_performance_right.to_csv('/home/Amy/dataAugmentation/training LOO right regression performance DATA AUGMENTATION.csv')
    
predictions_left.to_csv('/home/Amy/dataAugmentation/independent LOO left regression DATA AUGMENTATION.csv')
predictions_right.to_csv('/home/Amy/dataAugmentation/indepdendent LOO right regression DATA AUGMENTATION.csv')