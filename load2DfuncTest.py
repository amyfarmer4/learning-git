# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 12:34:43 2025

@author: af1226
"""

def load_in_2D(proteins, left_csv_files, right_csv_files, class_labels, alpha_labels, beta_labels, group_name, protein_name):
    import pandas as pd
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
                stacked_right_df_2D = (pd.DataFrame(df_2D.loc[13:97, '1549.79349786178': '1737.31299582371'])).reset_index(drop = True).T.stack().to_frame().T
                right_protein_2D = (pd.concat([right_protein_2D, stacked_right_df_2D], axis = 0)).reset_index(drop = True)
        right_protein_2D['label'] = pd.DataFrame(class_labels.get(protein))
        right_protein_2D['alpha'] = pd.DataFrame(alpha_labels.get(protein))
        right_protein_2D['beta'] = pd.DataFrame(beta_labels.get(protein))
        right_protein_2D['group'] = pd.DataFrame(group_name.get(protein))
        right_protein_2D['protein'] = pd.DataFrame(protein_name.get(protein))
        right_proteins_2D = (pd.concat([right_proteins_2D, right_protein_2D], axis = 0)).reset_index(drop = True)
    all_proteins_2D = (pd.concat([left_proteins_2D, right_proteins_2D], axis = 0)).reset_index(drop=True)
    return left_proteins_2D, right_proteins_2D, all_proteins_2D

def generateDiags(spectrum):
    import pandas as pd
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
    diagonal_df = (pd.DataFrame(diagonal)).T
    return diagonal_df

def load_in_diag(proteins, left_csv_files, right_csv_files, class_labels, alpha_labels, beta_labels, group_name, protein_name):
    import pandas as pd
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
                diagonal_df = generateDiags(spectrum)
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
                diagonal_df = generateDiags(spectrum)
                right_protein_diag = (pd.concat([right_protein_diag, diagonal_df], axis = 0)).reset_index(drop = True)
        right_protein_diag['label'] = pd.DataFrame(class_labels.get(protein))
        right_protein_diag['alpha'] = pd.DataFrame(alpha_labels.get(protein))
        right_protein_diag['beta'] = pd.DataFrame(beta_labels.get(protein))
        right_protein_diag['group'] = pd.DataFrame(group_name.get(protein))
        right_protein_diag['protein'] = pd.DataFrame(protein_name.get(protein))
        right_proteins_diag = pd.concat([right_proteins_diag, right_protein_diag], axis = 0).reset_index(drop = True)
    all_proteins_diag = (pd.concat([left_proteins_diag, right_proteins_diag], axis = 0)).reset_index(drop=True)
    return left_proteins_diag, right_proteins_diag, all_proteins_diag







    
