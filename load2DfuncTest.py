# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 12:34:43 2025

@author: af1226
"""
# function to load in 2D dataframe

def load_in_2D(proteins, left_csv_files, right_csv_files, class_labels, alpha_labels, beta_labels, group_name, protein_name):
    import pandas as pd
    import numpy as np
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
    return all_proteins_2D