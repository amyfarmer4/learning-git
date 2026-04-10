# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:55:45 2026

@author: af1226
"""

# import things you always need
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import copy
import random
import sys
sys.path.append('A:\Amy\Protein Library\py scripts')

# import load in functions
from fname import get_fname
from load2DfuncTest import load_in_2Dv3

# import things needed for pipeline building and GridSearching
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, GroupKFold
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.decomposition import PCA

# import models
from sklearn.svm import SVR
from sklearn.multioutput import RegressorChain

# homemade functions

def GridSearchPredictReg(x_train, y_train, group_train, pipeline, grid, test_protein, k_value):
    gkf = GroupKFold(n_splits = 5)
    cv = GroupKFold(n_splits = 5)
    grid_search = GridSearchCV(estimator = pipeline, param_grid = grid, cv = gkf, n_jobs = 12, scoring = 'neg_root_mean_squared_error')
    grid_search.fit(X = x_train, y = y_train, groups = group_train)
    parameters = grid_search.best_params_
    C = list(parameters.values())[0]
    final_pipe = grid_search.best_estimator_
    cross_val_scores = cross_val_score(final_pipe, X = x_train, y = y_train, cv = cv, groups = group_train, scoring = 'neg_root_mean_squared_error')
    training_rmse = np.average(cross_val_scores)
    training_rmse_std = np.std(cross_val_scores)
    training_performance = pd.DataFrame({'Test protein': test_protein, 'C': C, 'k_value': k_value,
                                         'rmse': training_rmse, '+-': training_rmse_std}, index = [0])
    return training_performance

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

def Mahalanobis(point, dataset, operation):
    distribution = pd.concat([point, dataset], axis = 0).reset_index(drop=True)
    cov = np.cov(distribution.values.T)
    inv_cov = np.linalg.pinv(cov)
    if operation == 'distribution':
        y = np.array(point) - np.array(dataset.mean(axis = 0))
        left = np.dot(y, inv_cov)
        mahal = np.dot(left, y.T)
        return mahal
    if operation == 'point':
        mahal_lst = []
        for i, row in dataset.iterrows():
            row_df = pd.DataFrame(row).T
            y = np.array(point) - np.array(row_df)
            left = np.dot(y, inv_cov)
            mahal_lst.append(np.dot(left, y.T)[0][0])
        return mahal_lst

def TrainReg(train_proteins, dataset):
    train_set = pd.DataFrame()
    for train_protein in train_proteins:
        train_set_1 = dataset[dataset['protein'].str.contains(train_protein)]
        train_set = pd.concat([train_set, train_set_1], axis = 0).reset_index(drop=True)
    x_train = train_set[train_set.columns[:2765]]
    y1_train = (pd.DataFrame(train_set.alpha)).values.ravel()
    y2_train = (pd.DataFrame(train_set.beta)).values.ravel()
    y_train = np.array(pd.concat([pd.DataFrame(train_set.alpha), pd.DataFrame(train_set.beta)], axis = 1))
    group_train = (pd.DataFrame(train_set.group)).values.ravel()
    return train_set, x_train, y1_train, y2_train, y_train, group_train

def TestReg(test_protein, dataset):
    test_set = pd.DataFrame()
    test_set = dataset[dataset['protein'].str.contains(test_protein)]
    x_test = test_set[test_set.columns[:2765]]
    y1_test = (pd.DataFrame(test_set.alpha)).values.ravel()
    y2_test = (pd.DataFrame(test_set.beta)).values.ravel()
    y_test = np.array(pd.concat([pd.DataFrame(test_set.alpha), pd.DataFrame(test_set.beta)], axis = 1))
    return x_test, y1_test, y2_test, y_test

def FtestNoTest(x_train, y1_train, y2_train, k_value):
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
    return x_train_AF

def hypers_no(x_train, y1_train, y2_train, y_train, group_train, test_protein, k_values):
    find_k = pd.DataFrame()
    for k_value in k_values:
        x_train_AF = FtestNoTest(x_train, y1_train, y2_train, k_value)
        per_k = GridSearchPredictReg(x_train = x_train_AF, y_train = y_train, group_train = group_train, 
                                     pipeline = pipeline, grid = grid, test_protein = test_protein, k_value = k_value)
        find_k = pd.concat([find_k, per_k], axis = 0).reset_index(drop=True)
    min_index = find_k['rmse'].idxmax()
    C_final = find_k['C'].iloc[min_index]
    k_final = find_k['k_value'].iloc[min_index]
    to_save = pd.DataFrame(find_k.iloc[min_index]).T
    return C_final, k_final, to_save

# list proteins and label dictionaries

proteins = ['Myoglobin', 'HSA', 'Calmodulin', 'cytochrome c', 'phosphorylase b', 'Lysozyme', 'Creatine', 'DT diaphorase', 'Lipoxidase', 
            'Lactoferrin bovin', 'Protease', 'apo transferrin', 'Lactoferrin human', 'Catalase', 'Conalbumin', 'alpha amylase', 
            'Choline oxidase', 'Gly 3 phos', 'Ovalbumin', 'b glucuronidase', 'Ribonuclease A', 'Ubiquitin', 'alpha 2 macroglobulin',
            'Chymotrypsinogen A', 'Thaumatin', 'TCI', 'Elastase', 'beta-Lactoglobulin', 'Pepsin', 'Tryp', 'SD', 'IgG', 'Concanavalin A',
            'Peroxidase', 'prealbumin', 'Aldolase', 'Enolase', 'Glucose Oxidase', 'Hexokinase', 'Lactoperoxidase']

def ss(x):
    return x

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
                'Peroxidase': ss(0),
                'prealbumin': ss(2),
                'Aldolase': ss(0),
                'Enolase': ss(0),
                'Glucose Oxidase': ss(1),
                'Hexokinase': ss(0),
                'Lactoperoxidase': ss(0)}

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
                'Peroxidase': ss(0.4444),
                'prealbumin': ss(0.0472),
                'Aldolase': ss(0.4187),
                'Enolase': ss(0.3739),
                'Glucose Oxidase': ss(0.2633),
                'Hexokinase': ss(0.3827),
                'Lactoperoxidase': ss(0.3244)}

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
                'Peroxidase': ss(0.0196),
                'prealbumin': ss(0.4803),
                'Aldolase': ss(0.1377),
                'Enolase': ss(0.1697),
                'Glucose Oxidase': ss(0.1928),
                'Hexokinase': ss(0.1613),
                'Lactoperoxidase': ss(0.0571)}

group_name = {'Myoglobin': ss(0),
                'HSA': ss(1),
                'Calmodulin': ss(2),
                'cytochrome c': ss(3),
                'phosphorylase b': ss(4),
                'Lysozyme': ss(5),
                'Creatine': ss(6),
                'DT diaphorase': ss(7),
                'Lipoxidase': ss(8),
                'Lactoferrin bovin': ss(9),
                'Protease': ss(10),
                'apo transferrin': ss(11),
                'Lactoferrin human': ss(12),
                'Catalase': ss(13),
                'Conalbumin': ss(14),
                'alpha amylase': ss(15),
                'Choline oxidase': ss(16),
                'Gly 3 phos': ss(17),
                'Ovalbumin': ss(18),
                'b glucuronidase': ss(19),
                'Ribonuclease A': ss(20),
                'Ubiquitin': ss(21),
                'alpha 2 macroglobulin': ss(22),
                'Chymotrypsinogen A': ss(23),
                'Thaumatin': ss(24),
                'TCI': ss(25),
                'Elastase': ss(26),
                'beta-Lactoglobulin': ss(27),
                'Pepsin': ss(28),
                'Tryp': ss(29),
                'SD': ss(30),
                'IgG': ss(31),
                'Concanavalin A': ss(32),
                'Peroxidase': ss(33),
                'prealbumin': ss(34),
                'Aldolase': ss(35),
                'Enolase': ss(36),
                'Glucose Oxidase': ss(37),
                'Hexokinase': ss(38),
                'Lactoperoxidase': ss(39)}

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
                'Peroxidase': ss('Peroxidase'),
                'prealbumin': ss('prealbumin'),
                'Aldolase': ss('Aldolase'),
                'Enolase': ss('Enolase'),
                'Glucose Oxidase': ss('Glucose Oxidase'),
                'Hexokinase': ss('Hexokinase'),
                'Lactoperoxidase': ss('Lactoperoxidase')}

# extract appropriate files 

fname_left, fname_right, fname_summer, fname_winter = get_fname()
left_csv_files = glob.glob(fname_left)
right_csv_files = glob.glob(fname_right)
ATH_sum_files = glob.glob(fname_summer)
ATH_win_files = glob.glob(fname_winter)

# load in whole 2D

left_LIFE = load_in_2Dv3(proteins = proteins, files = left_csv_files, delimiter = '\t', pr1 = 48, pr2 = 127, pu1 = 282, pu2 = 317, 
                         class_labels = class_labels, alpha_labels = alpha_labels, beta_labels = beta_labels, group_name = group_name, 
                         protein_name = protein_name)
right_LIFE = load_in_2Dv3(proteins = proteins, files = right_csv_files, delimiter = '\t', pr1 = 21, pr2 = 100, pu1 = 282, pu2 = 317, 
                          class_labels = class_labels, alpha_labels = alpha_labels, beta_labels = beta_labels, group_name = group_name, 
                          protein_name = protein_name)
winter_ATHENA = load_in_2Dv3(proteins = proteins, files = ATH_win_files, delimiter = ',', pr1 = 5, pr2 = 84, pu1 = 1, pu2 = 36, 
                             class_labels = class_labels, alpha_labels = alpha_labels, beta_labels = beta_labels, group_name = group_name, 
                             protein_name = protein_name)
summer_ATHENA = load_in_2Dv3(proteins = proteins, files = ATH_sum_files, delimiter = ',', pr1 = 5, pr2 = 84, pu1 = 1, pu2 = 36, 
                             class_labels = class_labels, alpha_labels = alpha_labels, beta_labels = beta_labels, group_name = group_name, 
                             protein_name = protein_name)

LIFEtime = pd.concat([left_LIFE, right_LIFE], axis = 0).reset_index(drop=True)
ATHENA = pd.concat([summer_ATHENA, winter_ATHENA], axis = 0).reset_index(drop=True)

left_LIFE['dataset'] = ['left'] * len(left_LIFE)
right_LIFE['dataset'] = ['right'] * len(right_LIFE)
ATHENA['dataset'] = ['athena'] * len(ATHENA)

LIFEtime_proteins = list(dict.fromkeys(left_LIFE[left_LIFE.columns[2769]].tolist()))
ATHENA_summer_proteins = list(dict.fromkeys(summer_ATHENA[summer_ATHENA.columns[2769]].tolist()))
ATHENA_winter_proteins = list(dict.fromkeys(winter_ATHENA[winter_ATHENA.columns[2769]].tolist()))
ATHENA_proteins = list(dict.fromkeys(ATHENA[ATHENA.columns[2769]].tolist()))

# Model + Hyperparameters

AF_reg = SelectKBest(score_func = f_regression)
svr = SVR()
reg_chain = RegressorChain(SVR(kernel = 'rbf'), order = [1, 0])
svm_params = {'regressorchain__base_estimator__C': [0.01, 0.1, 1, 10, 100]}
k_values = [20, 30, 40, 50]
params_dict = {svr : svm_params}

# PCA_Mahalanobis_JIT point-to-point

training_dfs = []
training_performance_LIFE_left = pd.DataFrame()
training_performance_LIFE_right = pd.DataFrame()
training_performance_ATHENA = pd.DataFrame()
test_predictions = pd.DataFrame()
pipeline = make_pipeline(StandardScaler(), reg_chain)
grid = params_dict.get(svr)
for i, val in enumerate(LIFEtime_proteins):
    test_protein = val
    test_for_rest = 'left_'+val+''
    train_proteins = copy.deepcopy(LIFEtime_proteins)
    train_proteins.pop(i)
    print(f'LIFEtime left test protein: {test_protein}')
    x_test, y1_test, y2_test, y_test = TestReg(test_protein, dataset = left_LIFE)
    train_l, x_train_l, _, _, _, _ = TrainReg(train_proteins = train_proteins, dataset = left_LIFE)
    train_r, x_train_r, _, _, _, _ = TrainReg(train_proteins = LIFEtime_proteins, dataset = right_LIFE)
    train_a, x_train_a, _, _, _, _ = TrainReg(train_proteins = ATHENA_proteins, dataset = ATHENA)
    all_train = pd.concat([train_l, train_r, train_a], axis = 0).reset_index(drop=True)
    # identifying local data
    print('measuring mahalanobis distances...')
    scaler = StandardScaler()
    pca = PCA(n_components = 10)
    scaled_x_train_l = scaler.fit_transform(x_train_l)
    reduced_x_train_l = pd.DataFrame(pca.fit_transform(scaled_x_train_l))
    scaled_x_train_r = scaler.fit_transform(x_train_r)
    reduced_x_train_r = pd.DataFrame(pca.fit_transform(scaled_x_train_r))
    scaled_x_train_a = scaler.fit_transform(x_train_a)
    reduced_x_train_a = pd.DataFrame(pca.fit_transform(scaled_x_train_a))
    test_pred = pd.DataFrame()
    for i, row in x_test.iterrows():
        row_df = pd.DataFrame(row).T
        scaled = scaler.transform(row_df)
        reduced = pd.DataFrame(pca.transform(scaled))
        mahal_l = Mahalanobis(point = reduced, dataset = reduced_x_train_l, operation = 'point')
        mahal_r = Mahalanobis(point = reduced, dataset = reduced_x_train_r, operation = 'point')
        mahal_a = Mahalanobis(point = reduced, dataset = reduced_x_train_a, operation = 'point')
        mahals = mahal_l + mahal_r + mahal_a
        idx = sorted(range(len(mahals)), key = lambda k: mahals[k])[:1000]
        vals = [mahals[i] for i in idx]
        train = all_train.iloc[idx]
        training_dfs.append(train[train.columns[len(train.columns)-2:]])
        # training!
        x_train = train[train.columns[:2765]]
        y1_train = (pd.DataFrame(train.alpha)).values.ravel()
        y2_train = (pd.DataFrame(train.beta)).values.ravel()
        y_train = np.array(pd.concat([pd.DataFrame(train.alpha), pd.DataFrame(train.beta)], axis = 1))
        group_train = (pd.DataFrame(train.group)).values.ravel()
        C_final, k_final, to_save = hypers_no(x_train = x_train, y1_train = y1_train, y2_train = y2_train, y_train = y_train, 
                                              group_train = group_train, test_protein = test_for_rest, k_values = k_values)
        training_performance_LIFE_left = pd.concat([training_performance_LIFE_left, to_save], axis = 0).reset_index(drop=True)
        # testing!
        x_train_fin = FtestNoTest(x_train = x_train, y1_train = y1_train, y2_train = y2_train, k_value = k_final)
        pipe = make_pipeline(StandardScaler(), RegressorChain(SVR(kernel = 'rbf', C = C_final), order = [1, 0]))
        pipe.fit(x_train_fin, y_train)
        _ , x1_test = CombinedFtest(x_train = x_train, x_test = row_df, y1_train = y1_train, y2_train = y2_train, k_value = k_final)
        x1_arr = np.array(x1_test).reshape(1, -1)
        y_predict = pd.DataFrame((pipe.predict(x1_arr)), columns = ['alpha_prediction', 'beta_prediction'])
        test_pred = pd.concat([test_pred, y_predict], axis = 0).reset_index(drop=True)
    test_pred['alpha'] = y1_test
    test_pred['beta'] = y2_test
    test_pred['spectral set'] = [test_for_rest] * len(y1_test)
    print(test_pred)
    test_predictions = pd.concat([test_predictions, test_pred], axis = 0).reset_index(drop=True) 

training_sets = pd.DataFrame()
for i in range(0, len(training_dfs)):
    globals()[f'training_dfs{i+1}'] = training_dfs[i]
    training_dfs[i] = pd.DataFrame(training_dfs[i])
    training_sets = (pd.concat([training_sets, training_dfs[i]], axis = 0)).reset_index(drop = True)

print(test_predictions)
