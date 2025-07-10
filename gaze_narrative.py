# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 12:53:13 2023

@author: zcheng
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import narrative_reinstatement # for plotting similarity matrices
import gaze_reinstatement # for reading excel to 3d arrays
from scipy.stats import pearsonr
from scipy.spatial import distance

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

def excel_to_3darray_age(excel_array_YA, excel_array_OA, n_cols, ptps_YA, ptps_OA):
    n_ptp_YA = len(ptps_YA)
    n_ptp_OA = len(ptps_OA)
    array_3d_YA = np.zeros((n_ptp_YA,n_cols,n_cols))
    array_3d_OA = np.zeros((n_ptp_OA,n_cols,n_cols))
    for i in range(excel_array_YA.shape[1] // n_cols):
        array_3d_YA[i,:,:] = excel_array_YA[:,i*n_cols:(i+1)*n_cols]
    for i in range(excel_array_OA.shape[1] // n_cols):
        array_3d_OA[i,:,:] = excel_array_OA[:,i*n_cols:(i+1)*n_cols]
    return {'YA':array_3d_YA, 'OA':array_3d_OA}

def narrative_substract_baseline(narrative_reinstatement_df):
    narrative_reinstatement_short = narrative_reinstatement_df[
        narrative_reinstatement_df['reins_type'] == 'narrative reinstatement']
    narrative_reinstatement_short['baseline'] = np.nan
    for index, row in narrative_reinstatement_short.iterrows():
        ptp, image = row['ptp'], row['images']
        baseline_df = narrative_reinstatement_df[
            (narrative_reinstatement_df['reins_type'] == 'baseline reinstatement') &
            (narrative_reinstatement_df['ptp'] == ptp) &
            (narrative_reinstatement_df['images'] == image)]
        baseline_reinstatement = np.mean(baseline_df['similarity'])
        narrative_reinstatement_short.at[index, 'baseline'] = baseline_reinstatement
    narrative_reinstatement_short['corrected_similarity'] = \
        narrative_reinstatement_short['similarity'] - \
            narrative_reinstatement_short['baseline'] 
    return narrative_reinstatement_short

def combine_gaze_narrative_whole(gaze_reinstatement_df, narrative_reinstatement_df):
    gaze_reinstatement_df = gaze_reinstatement_df[['RECORDING_SESSION_LABEL','IMAGE','AGE','eye_sim_diff']]
    gaze_reinstatement_df.columns = ['ptp','images','age','gaze_reinstatement']
    narrative_reinstatement_df = narrative_reinstatement_df[['ptp','images','age','corrected_similarity']]
    narrative_reinstatement_df.columns = ['ptp','images','age','narrative_reinstatement']
    df_narrative_gaze = pd.merge(gaze_reinstatement_df, narrative_reinstatement_df,
                                 on=['ptp','images','age'], how='inner')
    df_narrative_gaze_ptp_mean = df_narrative_gaze.groupby(['ptp']).mean().reset_index()
    df_narrative_gaze_ptp_mean.columns = ['ptp','gaze_reinstatement_mean', 'narrative_reinstatement_mean']
    df_narrative_gaze = pd.merge(df_narrative_gaze, df_narrative_gaze_ptp_mean, on=['ptp'], how='inner')
    df_narrative_gaze['gaze_reinstatement_center'] = df_narrative_gaze['gaze_reinstatement'] - df_narrative_gaze['gaze_reinstatement_mean']
    df_narrative_gaze['narrative_reinstatement_center'] = df_narrative_gaze['narrative_reinstatement'] - df_narrative_gaze['narrative_reinstatement_mean']
    return df_narrative_gaze

def narrative_gaze_time(gaze_reinstatement_windowed, narrative_reinstatement_windowed):
    output = {}
    for age in ['YA','OA']:
        gaze_matrix = gaze_reinstatement_windowed[age]
        narrative_matrix = narrative_reinstatement_windowed[age]
        n_col = gaze_matrix.shape[1]
        result = np.zeros((n_col, n_col))
        for m in range(n_col):
            for n in range(n_col):
                gaze_array = gaze_matrix[:,m,n]
                narrative_array = narrative_matrix[:,m,n]
                sim_tile, _ = pearsonr(gaze_array, narrative_array)
                result[m,n] = sim_tile
        output[age] = result
        plt.matshow(result, cmap='RdBu_r', vmax=1, vmin=-1)
        plt.colorbar()
        plt.title(f'narrative gaze similarity: {age}, {how}')
        plt.show()
    age_output = ['YA']*n_col + ['OA']*n_col
    time_output = list(range(n_col))*2
    cor_output = list(np.diagonal(output['YA'])) + list(np.diagonal(output['OA']))
    temp_df = pd.DataFrame({'age':age_output,'time_window':time_output,'correlation':cor_output})
    sns.lineplot(data=temp_df,x='time_window',y='correlation',hue='age')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('narrative gaze correlation')
    plt.ylim((-1,1))
    plt.show()
    return output



if __name__ == '__main__':
    # read trial information
    df_trial = pd.read_excel('output/df_trial.xlsx')
    df_YA = df_trial[df_trial['age']=='YA']
    ptps_YA = list(df_YA['ptp'].unique())
    df_OA = df_trial[df_trial['age']=='OA']
    ptps_OA = list(df_OA['ptp'].unique())
    ptps = ptps_YA + ptps_OA
     
    # read time-independent reinstatements
    gaze_reinstatement_df = pd.read_excel('output/gaze_reinstatement.xlsx')
    narrative_reinstatement_df = pd.read_excel('output/narrative_reinstatement.xlsx')
    
    
    
    # read in gaze reinstatement. The excel file of gaze reinstatement doesn't
    # separate YA and OA, but the <excel_to_3darray> function takes in the 
    # list of YA and OA participants, and does the separation here. This results
    # in a dictionary with 'YA' and 'OA' as the keys, and their respective 3d 
    # array as the values. 
    raw_gaze_reinstatement_windowed = pd.read_excel(
        'output/gaze_reinstatement_windowed.xlsx').to_numpy()
    baseline_gaze_reinstatement_windowed = pd.read_excel(
        'output/baseline_reinstatement_windowed.xlsx').to_numpy()
    n_cols = raw_gaze_reinstatement_windowed.shape[0]
    raw_gaze_reinstatement_windowed = excel_to_3darray(
        raw_gaze_reinstatement_windowed, n_cols, ptps_YA, ptps_OA)
    baseline_gaze_reinstatement_windowed = gaze_reinstatement.excel_to_3darray(
        baseline_gaze_reinstatement_windowed, n_cols, ptps_YA, ptps_OA)
    gaze_reinstatement_windowed = {}
    gaze_reinstatement_windowed['YA'] = raw_gaze_reinstatement_windowed['YA'] - baseline_gaze_reinstatement_windowed['YA']
    gaze_reinstatement_windowed['OA'] = raw_gaze_reinstatement_windowed['OA'] - baseline_gaze_reinstatement_windowed['OA']
    
    # read in narrative reinstatement. There are separate excel files of narrative
    # reinstatement for YA and OA. The <excel_to_3darray_age> function takes 
    # in two separate numpy arrays, and combine them to a dictionary with 'YA' 
    # and 'OA' as the keys, and their respective 3d array as the values. 
    narrative_reinstatement_windowed_YA = pd.read_excel(
        'output/narrative_reinstatement_time_YA.xlsx').to_numpy()
    narrative_reinstatement_windowed_OA = pd.read_excel(
        'output/narrative_reinstatement_time_OA.xlsx').to_numpy()
    narrative_reinstatement_windowed = excel_to_3darray_age(
        narrative_reinstatement_windowed_YA, narrative_reinstatement_windowed_OA, 
        n_cols, ptps_YA, ptps_OA)
    
    # read in the diagonal narrative and gaze reinstatement with 0.5s step size
    gaze_reinstatement_diagonal_df = pd.read_excel('output/gaze_reinstatement_corrected_windowed_diagonal.xlsx')
    gaze_reinstatement_diagonal_array = np.asarray(gaze_reinstatement_diagonal_df).T
    gaze_reinstatement_diagonal_array_YA = gaze_reinstatement_diagonal_array[:32,:]
    gaze_reinstatement_diagonal_array_OA = gaze_reinstatement_diagonal_array[32:,:]
    narrative_reinstatement_diagonal_df_YA = pd.read_excel('output/narrative_reinstatement_time_diagonal_YA.xlsx')
    narrative_reinstatement_diagonal_df_OA = pd.read_excel('output/narrative_reinstatement_time_diagonal_OA.xlsx')
    narrative_reinstatement_diagonal_array_YA = np.asarray(narrative_reinstatement_diagonal_df_YA)
    narrative_reinstatement_diagonal_array_OA = np.asarray(narrative_reinstatement_diagonal_df_OA)

    gaze_diagonal_df_fine, gaze_YA_error, gaze_OA_error = narrative_reinstatement.lineplot_2d_array(
        gaze_reinstatement_diagonal_array_YA, gaze_reinstatement_diagonal_array_OA, 
        'gaze reinstatement 0.5s step', 0.05, -0.05, add_aste_idx=[], ylabel='sim')
    narrative_diagonal_df_fine, _, _ = narrative_reinstatement.lineplot_2d_array(
        narrative_reinstatement_diagonal_array_YA, narrative_reinstatement_diagonal_array_OA, 
        'narrative reinstatement 0.5s step', 0.35, 0, add_aste_idx=[], ylabel='sim')
    
    # plot the time matrix and USE line for gaze and narrative reinstatement
    # separately 
    narrative_reinstatement.plot_USE_time_matrix(gaze_reinstatement_windowed, 0.03, 0, analysis='gaze_heatmap')
    gaze_diagonal_df_course = narrative_reinstatement.plot_USE_line(gaze_reinstatement_windowed, 0.05, -0.05, [], analysis='gaze')
    narrative_reinstatement.plot_USE_time_matrix(narrative_reinstatement_windowed, 0.3, 0, analysis='narrative_heatmap')
    narrative_diagonal_df_course = narrative_reinstatement.plot_USE_line(narrative_reinstatement_windowed, 0.35, 0, [], analysis='narrative')
    
    narrative_reinstatement_df = narrative_substract_baseline(narrative_reinstatement_df)
    
    sns.barplot(data=narrative_reinstatement_df, x='age', y='corrected_similarity')
    
    df_narrative_gaze = combine_gaze_narrative_whole(gaze_reinstatement_df, narrative_reinstatement_df)
    
    
    # add participants to the diagonal dataframes to run multilevel models in R
    ptp_col = []
    for ptp in ptps:
        for i in range(n_cols):
            ptp_col.append(ptp)
    gaze_diagonal_df_fine.to_excel('output/gaze_diagonal_0.5s_toR.xlsx', index=None)
    narrative_diagonal_df_fine.to_excel('output/narrative_diagonal_0.5s_toR.xlsx', index=None)
    
    
    df_narrative_gaze.to_excel('output/df_narrative_gaze.xlsx', index=None)