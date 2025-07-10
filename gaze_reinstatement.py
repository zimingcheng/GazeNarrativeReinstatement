# -*- coding: utf-8 -*-
"""
Created on Sun May 14 22:22:14 2023

@author: czm19
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import narrative_reinstatement


def excel_to_3darray(excel_array, n_cols, ptps_YA, ptps_OA):
    """
    Convert the 3-d array in R that is stored in excel format back to its 
    proper 3-d array format. in the resulted array, the first index indicate
    the number of participants, and the second and third index indicate the 
    size of the correlation matrix
    """
    n_ptp = len(ptps_YA) + len(ptps_OA)
    array_3d = np.zeros((n_ptp,n_cols,n_cols))
    for i in range(excel_array.shape[1] // n_cols):
        array_3d[i,:,:] = excel_array[:,i*n_cols:(i+1)*n_cols]
    # <array_3d> is organized such that all YAs are at the front and all OAs are at the back
    # get the first <len(ptps_YA)> items of the 3d array to represent younger adults
    array_3d_YA = array_3d[:len(ptps_YA),:,:]
    # get the items from <len(ptps_YA)> to the last items of the 3d array to represent older adults
    array_3d_OA = array_3d[len(ptps_YA):,:,:]
    return {'YA':array_3d_YA, 'OA':array_3d_OA}

def excel_to_3darray_manual(excel_array, n_cols, ptps_YA, ptps_OA):
    """
    Convert the 3-d array in R that is stored in excel format back to its 
    proper 3-d array format. in the resulted array, the first index indicate
    the number of participants, and the second and third index indicate the 
    size of the correlation matrix
    """
    n_ptp = len(ptps_YA) + len(ptps_OA)
    array_3d = np.zeros((380*50,n_cols,n_cols))
    for i in range(excel_array.shape[1] // n_cols):
        array_3d[i,:,:] = excel_array[:,i*n_cols:(i+1)*n_cols]
    # <array_3d> is organized such that all YAs are at the front and all OAs are at the back
    # get the first <len(ptps_YA)> items of the 3d array to represent younger adults
    array_3d_YA = array_3d[:len(ptps_YA),:,:]
    # get the items from <len(ptps_YA)> to the last items of the 3d array to represent older adults
    array_3d_OA = array_3d[len(ptps_YA):,:,:]
    return {'YA':array_3d_YA, 'OA':array_3d_OA}

def diagonal_similarity(result):
    age_output,offset_output,similarity_output = [],[],[]
    for age in ['YA','OA']:
        array_3d = result[age]
        max_offset = array_3d.shape[1]-1
        for offset in range(-max_offset, max_offset+1):
            diag = np.diagonal(array_3d, offset=offset, axis1=1, axis2=2)
            values = list(diag.flatten())
            similarity_output += values
            age_output += [age]*len(values)
            offset_output += [offset]*len(values)
    diagonal_result = pd.DataFrame({'age':age_output,'offset':offset_output,
                                    'similarity':similarity_output}) 
    return diagonal_result   

def plot_diagonal_difference(diag_df, n):
    diag_df['diagonal'] = np.where((diag_df['offset'] >= -n) & (diag_df['offset'] <= n), 1, 0)
    sns.catplot(x='diagonal',y='similarity',col='age',data=diag_df,kind='bar')
    plt.show()
        

if __name__ == '__main__':
    df_trial = pd.read_excel('output/df_trial.xlsx')
    gaze_reinstatement_windowed = pd.read_excel('output/gaze_reinstatement_windowed.xlsx').to_numpy()
    baseline_reinstatement_windowed = pd.read_excel('output/subject_baseline_reinstatement_windowed.xlsx').to_numpy()
    baseline_manual_reinstatement_windowed = pd.read_csv('output/manual_reinstatement_windowed.csv').to_numpy()
    
    df_YA = df_trial[df_trial['age']=='YA']
    ptps_YA = list(df_YA['ptp'].unique())
    df_OA = df_trial[df_trial['age']=='OA']
    ptps_OA = list(df_OA['ptp'].unique())
    n_cols = 17
    gaze_reinstatement_result = excel_to_3darray(gaze_reinstatement_windowed, n_cols, ptps_YA, ptps_OA)
    baseline_reinstatement_result = excel_to_3darray(baseline_reinstatement_windowed, n_cols, ptps_YA, ptps_OA)
    baseline_manual_reinstatement_result = excel_to_3darray_manual(baseline_manual_reinstatement_windowed, n_cols, ptps_YA, ptps_OA)
    
    diff_reinstatement_result = {}
    diff_reinstatement_result['YA'] = gaze_reinstatement_result['YA'] - baseline_reinstatement_result['YA']
    diff_reinstatement_result['OA'] = gaze_reinstatement_result['OA'] - baseline_reinstatement_result['OA']
    
    narrative_reinstatement.plot_USE_time_matrix(diff_reinstatement_result, 0.1, 0)
    narrative_reinstatement.plot_USE_line(diff_reinstatement_result, 0.1, -0.1)
    
    diag_df = diagonal_similarity(diff_reinstatement_result)
    plot_diagonal_difference(diag_df, 0)
    plot_diagonal_difference(diag_df, 5)
