# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 17:13:56 2026

@author: af1226
"""

# import things you always need
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import copy
import random
from collections import Counter

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
from sklearn.ensemble import VotingClassifier

# import metrics
from sklearn.metrics import accuracy_score, cohen_kappa_score

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
    return final_pipe, training_performance

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

left_2D_x = left_2D[left_2D.columns[:2975]]
right_2D_x = right_2D[right_2D.columns[:2975]]

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
    return test_set, val_set, train_set

def ProbaPred(final_pipe, x_train, y_train, x_test):
    final_pipe.fit(x_train, y_train)
    probabilities = final_pipe.predict_proba(x_test)
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

svm_params = {'svc__C': [0.01, 0.1, 1, 10, 100],
              'svc__gamma': [0.01, 0.1, 1, 10, 100]}

params_dict = {svc : svm_params}

left_training_performance = pd.DataFrame()
right_training_performance = pd.DataFrame()
weights_LOO = []
accuracy = []
for i, val in enumerate(proteins):
    test_protein = val
    train_val_proteins = copy.deepcopy(proteins)
    train_val_proteins.pop(i)
    print(f'test protein: {test_protein}')
    grid = params_dict.get(svc)
    pipeline = make_pipeline(StandardScaler(), AF_class, svc)
    grid = {**grid, 'selectkbest__k': [10, 20, 30, 40, 50]}
    opt_weights = []
    iter = 1
    while iter <= 10:
        val_proteins = random.sample(train_val_proteins, 4)
        train_proteins = [i for i in train_val_proteins if i not in val_proteins]
        print(f'validation proteins: {val_proteins}')
        # left
        test_left, val_left, train_left = TrainTest(test_protein, val_proteins, train_proteins, dataset = left_2D)
        x_train_left = train_left[train_left.columns[:2975]]
        y_train_left = (pd.DataFrame(train_left.label)).values.ravel()
        group_train_left = (pd.DataFrame(train_left.group)).values.ravel()
        print('Training independent left model...')
        final_pipe_left, training_performance_left = GridSearchPredictV3(x_train = x_train_left, y_train = y_train_left,
                                                                         group_train = group_train_left, pipeline = pipeline,
                                                                         grid = grid, iteration = iter, test_protein = test_protein)
        left_training_performance = pd.concat([left_training_performance, training_performance_left], axis = 0).reset_index(drop=True)
        # right
        test_right, val_right, train_right = TrainTest(test_protein, val_proteins, train_proteins, dataset = right_2D)
        x_train_right = train_right[train_right.columns[:2975]]
        y_train_right = (pd.DataFrame(train_right.label)).values.ravel()
        group_train_right = (pd.DataFrame(train_right.group)).values.ravel()
        print('Training independent right model...')
        final_pipe_right, training_performance_right = GridSearchPredictV3(x_train = x_train_right, y_train = y_train_right,
                                                                           group_train = group_train_right, pipeline = pipeline,
                                                                           grid = grid, iteration = iter, test_protein = test_protein)
        right_training_performance = pd.concat([right_training_performance, training_performance_right], axis = 0).reset_index(drop=True)
        # optmisation of weights
        validation = pd.concat([val_left, val_right], axis = 0).reset_index(drop=True)
        x_val = validation[validation.columns[:2975]]
        y_val = (pd.DataFrame(validation.label)).values.ravel()
        left_probs = ProbaPred(final_pipe = final_pipe_left, x_train = x_train_left, y_train = y_train_left, x_test = x_val)
        right_probs = ProbaPred(final_pipe = final_pipe_right, x_train = x_train_right, y_train = y_train_right, x_test = x_val)
        print('Optimising weights...')
        w1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        w2 = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        correct_weights = []
        for (i, row_left, row_right) in zip(y_val, left_probs.itertuples(index=False), right_probs.itertuples(index=False)):
            for (a, b) in zip(w1, w2):
                class_0 = ((a*row_left.class_0_prob) + (b*row_right.class_0_prob))/2
                class_1 = ((a*row_left.class_1_prob) + (b*row_right.class_1_prob))/2
                class_2 = ((a*row_left.class_2_prob) + (b*row_right.class_2_prob))/2
                pred_class = np.argmax([class_0, class_1, class_2])
                if pred_class == i:
                    tup = (a, b)
                    correct_weights.append(tup)
                else:
                    pass
        count = Counter(correct_weights)
        count_weights = list(count.keys())
        count_values = list(count.values())
        find_opt = pd.DataFrame({'Weights': count_weights, 'Correct_instances': count_values})
        index = find_opt['Correct_instances'].idxmax()
        opt_weights.append(find_opt['Weights'].iloc[index])
        iter += 1
    print(opt_weights)
    count_2 = Counter(opt_weights)
    count_weights_2 = list(count_2.keys())
    count_values_2 = list(count_2.values())
    find_opt_2 = pd.DataFrame({'Weights': count_weights_2, 'Correct_instances': count_values_2})
    index = find_opt_2['Correct_instances'].idxmax()
    final_weights = find_opt_2['Weights'].iloc[index]
    print(f'Optimised weights: {final_weights}')
    weights_LOO.append(final_weights)
    # set up for testing
    testing = pd.concat([test_left, test_right], axis = 0).reset_index(drop=True)
    training = pd.concat([train_left, train_right, validation], axis = 0).reset_index(drop=True)
    x_train_final = training[training.columns[:2975]]
    y_train_final = (pd.DataFrame(training.label)).values.ravel()
    group_train_final = (pd.DataFrame(training.group)).values.ravel()
    x_test = testing[testing.columns[:2975]]
    y_test = (pd.DataFrame(testing.label)).values.ravel()
    wL = final_weights[0]
    wR = final_weights[1]
    # testing
    print('Predicting...')
    left_test_probs = ProbaPred(final_pipe = final_pipe_left, x_train = x_train_final, y_train = y_train_final, x_test = x_test)
    right_test_probs = ProbaPred(final_pipe = final_pipe_right, x_train = x_train_final, y_train = y_train_final, x_test = x_test)
    predicted_class = []
    for (row_left, row_right) in zip(left_test_probs.itertuples(index=False), right_test_probs.itertuples(index=False)):
        class_0 = ((wL*row_left.class_0_prob) + (wR*row_right.class_0_prob))/2
        class_1 = ((wL*row_left.class_1_prob) + (wR*row_right.class_1_prob))/2
        class_2 = ((wL*row_left.class_2_prob) + (wR*row_right.class_2_prob))/2
        predicted_class.append(np.argmax([class_0, class_1, class_2]))
    print(predicted_class)
    accuracy.append(accuracy_score(y_test, predicted_class))

LOO_analysis = pd.DataFrame({'Protein': proteins, 'Weights': weights_LOO, 'Accuracy': accuracy})

