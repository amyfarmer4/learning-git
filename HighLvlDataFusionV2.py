# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 10:50:15 2026

@author: af1226
"""

# import things you always need
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import copy
import random
from itertools import product

# import load in functions
from fname import get_fname
from load2DfuncTest import load_in_2D

# import things needed for pipeline building and GridSearching
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, GroupKFold
from sklearn.feature_selection import f_classif, SelectKBest

# import models
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

# import metrics
from sklearn.metrics import accuracy_score

# GridSearch + Prediction function

def GridSearchPredictV3(x_train, y_train, group_train, pipeline, grid, iteration, test_protein):
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
    training_performance = pd.DataFrame({'Train_Val Iteration': iteration, 'Test protein': test_protein, 'Hyperparameters': hyperparameters,
                                         'Training accuracy': training_acc, '+-': training_acc_std})
    return parameters

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

# High Level Data Fusion (Voting ensemble)

def TrainTest(test_protein, val_proteins, train_proteins, dataset):
    test_set = pd.DataFrame()
    test_set = dataset[dataset['protein'].str.contains(test_protein)]
    val_set = pd.DataFrame()
    for val_protein in val_proteins:
        val_set_1 = dataset[dataset['protein'].str.contains(val_protein)]
        val_set = pd.concat([val_set, val_set_1], axis = 0).reset_index(drop=True)
    train_set = pd.DataFrame()
    for train_protein in train_proteins:
        train_set_1 = dataset[dataset['protein'].str.contains(train_protein)]
        train_set = pd.concat([train_set, train_set_1], axis = 0).reset_index(drop=True)
    x_train = train_set[train_set.columns[:2975]]
    y_train = (pd.DataFrame(train_set.label)).values.ravel()
    group_train = (pd.DataFrame(train_set.group)).values.ravel()
    return test_set, val_set, train_set, x_train, y_train, group_train

def ProbaPred(pipeline, x_train, y_train, x_test):
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)
    class_0_prob = []
    class_1_prob = []
    class_2_prob = []
    for (pred, prob) in enumerate(probabilities):
        class_0_prob.append(prob[0])
        class_1_prob.append(prob[1])
        class_2_prob.append(prob[2])
    probabilities_df = pd.DataFrame({'class_0_prob': class_0_prob, 'class_1_prob': class_1_prob, 'class_2_prob': class_2_prob}) 
    return probabilities_df  

AF_class = SelectKBest(score_func = f_classif)
svc = SVC(decision_function_shape = 'ovr', class_weight = 'balanced', probability = True)

k_values = [20, 30, 40, 50]
C_values = [0.01, 0.1, 1, 10, 100]
w1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
w2 = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

H_L = list(product(k_values, C_values))
H_R = list(product(k_values, C_values))

training_accuracy = []
training_stddev = []
kL_testing = []
CL_testing = []
kR_testing = []
CR_testing = []
wL_testing = []
wR_testing = []
testing_accuracy = []
weighted_probs = []
for i, val in enumerate(proteins):
    test_protein = val
    train_val_proteins = copy.deepcopy(proteins)
    train_val_proteins.pop(i)
    print(f'test protein: {test_protein}')
    pipeline = make_pipeline(StandardScaler(), AF_class, svc)
    H = pd.DataFrame()
    for hL in H_L:
        for hR in H_R:
                for (a, b) in zip(w1, w2):
                    kL = hL[0]
                    CL = hL[1]
                    kR = hR[0]
                    CR = hR[1]
                    hyper = pd.DataFrame({'k_left': kL, 'C_left': CL, 'k_right': kR, 'C_right' : CR, 'w1': a, 'w2': b}, index = [0])
                    H = pd.concat([H, hyper], axis = 0).reset_index(drop=True)
    iter = 1
    while iter <= 5:
        val_proteins = random.sample(train_val_proteins, 4)
        train_proteins = [i for i in train_val_proteins if i not in val_proteins]
        print(f'validation proteins: {val_proteins}')
        # left
        test_left, val_left, train_left, x_train_left, y_train_left, group_train_left = TrainTest(test_protein, val_proteins, 
                                                                                                  train_proteins, dataset = left_2D)
        # right
        test_right, val_right, train_right, x_train_right, y_train_right, group_train_right = TrainTest(test_protein, val_proteins, 
                                                                                                        train_proteins, dataset = right_2D)
        # validation set
        x_val_left = val_left[val_left.columns[:2975]]
        y_val_left = (pd.DataFrame(val_left.label)).values.ravel()
        x_val_right = val_right[val_right.columns[:2975]]
        y_val_right = (pd.DataFrame(val_right.label)).values.ravel()
        # search through hyperparameters
        accuracy_list = []
        for hL in H_L:
            for hR in H_R:
                    for (a, b) in zip(w1, w2):
                        kL = hL[0]
                        CL = hL[1]
                        kR = hR[0]
                        CR = hR[1]
                        print(f'k_left = {kL}, C_left = {CL}, k_right = {kR}, C_right = {CR}, (w1, w2) = {a, b}')
                        pipeline_left = make_pipeline(StandardScaler(), SelectKBest(score_func = f_classif, k = kL), 
                                                      SVC(decision_function_shape = 'ovr', class_weight = 'balanced', 
                                                      probability = True, C = CL))
                        pipeline_right = make_pipeline(StandardScaler(), SelectKBest(score_func = f_classif, k = kR), 
                                                      SVC(decision_function_shape = 'ovr', class_weight = 'balanced', 
                                                      probability = True, C = CR))
                        left_probs = ProbaPred(pipeline = pipeline_left, x_train = x_train_left, 
                                               y_train = y_train_left, x_test = x_val_left)
                        right_probs = ProbaPred(pipeline = pipeline_right, x_train = x_train_right, 
                                                y_train = y_train_right, x_test = x_val_right)
                        prediction = []
                        for (row_left, row_right) in zip(left_probs.itertuples(index=False), right_probs.itertuples(index=False)):
                            class_0 = ((a*row_left.class_0_prob) + (b*row_right.class_0_prob))/2
                            class_1 = ((a*row_left.class_1_prob) + (b*row_right.class_1_prob))/2
                            class_2 = ((a*row_left.class_2_prob) + (b*row_right.class_2_prob))/2
                            prediction.append(np.argmax([class_0, class_1, class_2]))
                        accuracy = accuracy_score(y_val_left, prediction)
                        accuracy_list.append(accuracy)
                        print(f'accuracy: {accuracy}')
        fold = str(iter)
        H[f'fold_{fold}'] = accuracy_list
        print(H)
        iter += 1
    # training performance
    H['average'] = (H[H.columns[6:]]).mean(axis = 1)
    H['+-'] = (H[H.columns[6:]]).std(axis = 1)
    index = H['average'].idxmax()
    training_accuracy.append(H['average'].max())
    training_stddev.append(H['+-'].loc[index])
    # extract hyperparameters and save
    kL_final = H['k_left'].loc[index]
    CL_final = H['C_left'].loc[index]
    kR_final = H['k_right'].loc[index]
    CR_final = H['C_right'].loc[index]
    wL = H['w1'].loc[index]
    wR = H['w2'].loc[index]
    kL_testing.append(kL_final)
    CL_testing.append(CL_final)
    kR_testing.append(kR_final)
    CR_testing.append(CR_final)
    wL_testing.append(wL)
    wR_testing.append(wR)
    # setting up for testing - left
    training_left = pd.concat([train_left, val_left], axis = 0).reset_index(drop=True)
    x_train_left_fin = training_left[training_left.columns[:2975]]
    y_train_left_fin = (pd.DataFrame(training_left.label)).values.ravel()
    # setting up for testing - right
    training_right = pd.concat([train_right, val_right], axis = 0).reset_index(drop=True)
    x_train_right_fin = training_right[training_right.columns[:2975]]
    y_train_right_fin = (pd.DataFrame(training_right.label)).values.ravel()
    # testing
    x_test_left = test_left[test_left.columns[:2975]]
    y_test_left = (pd.DataFrame(test_left.label)).values.ravel()
    x_test_right = test_right[test_right.columns[:2975]]
    pipe_left = make_pipeline(StandardScaler(), SelectKBest(score_func = f_classif, k = kL_final), 
                              SVC(decision_function_shape = 'ovr', class_weight = 'balanced', probability = True, C = CL_final))
    pipe_right = make_pipeline(StandardScaler(), SelectKBest(score_func = f_classif, k = kR_final), 
                               SVC(decision_function_shape = 'ovr', class_weight = 'balanced', probability = True, C = CR_final))
    left_probs = ProbaPred(pipeline = pipe_left, x_train = x_train_left_fin, 
                           y_train = y_train_left_fin, x_test = x_test_left)
    right_probs = ProbaPred(pipeline = pipe_right, x_train = x_train_right_fin, 
                            y_train = y_train_right_fin, x_test = x_test_right)
    prediction = []
    weighted_test_probs = pd.DataFrame()
    for (row_left, row_right) in zip(left_probs.itertuples(index=False), right_probs.itertuples(index=False)):
        class_0 = ((wL*row_left.class_0_prob) + (wR*row_right.class_0_prob))/2
        class_1 = ((wL*row_left.class_1_prob) + (wR*row_right.class_1_prob))/2
        class_2 = ((wL*row_left.class_2_prob) + (wR*row_right.class_2_prob))/2
        prediction.append(np.argmax([class_0, class_1, class_2]))
        test_probs = pd.DataFrame({'class_0': class_0, 'class_1': class_1, 'class_2': class_2}, index = [0])
        weighted_test_probs = pd.concat([weighted_test_probs, test_probs], axis = 0).reset_index(drop=True)
    testing_accuracy.append(accuracy_score(y_test_left, prediction))
    print(f'Testing accuracy: {accuracy_score(y_test_left, prediction)}')
    print(f'with hyperparameters kL = {kL_final}, CL = {CL_final}, kR = {kR_final}, CR = {CR_final}, w1 = {wL}, w2 = {wR}')
    weighted_probs.append(weighted_test_probs)
    
LOO_analysis = pd.DataFrame({'Protein': proteins, 'k_left': kL_testing, 'C_left': CL_testing, 'k_right': kR_testing, 'C_right': CR_testing,
                             'w1': wL_testing, 'w2': wR_testing, 'Accuracy': testing_accuracy})
print(LOO_analysis)

LOO_analysis.to_csv('/home/amy/HighLevelDataFusion/attempt3')

for i in range(0, len(weighted_probs)):
    globals()[f'weighted_probs{i+1}'] = weighted_probs[i]
    print(weighted_probs[i])
    weighted_probs[i].to_csv('/home/amy/HighLevelDataFusion/attempt3/probabilities/probabilities '+str([i])+'.csv')
    