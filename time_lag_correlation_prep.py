# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 11:56:05 2023

@author: zcheng
"""

import pandas as pd
import numpy as np

df_trial = pd.read_excel('output/df_trial.xlsx')
df_gaze_diagonal = pd.read_excel('output/gaze_reinstatement_corrected_windowed_diagonal.xlsx')
df_narrative_diagonal_YA = pd.read_excel('output/narrative_reinstatement_time_diagonal_YA.xlsx')
df_narrative_diagonal_OA = pd.read_excel('output/narrative_reinstatement_time_diagonal_OA.xlsx')

YA_ptps = df_trial[df_trial['age']=='YA']['ptp'].unique()
OA_ptps = df_trial[df_trial['age']=='OA']['ptp'].unique()

df_gaze_diagonal_YA = df_gaze_diagonal.iloc[:,:len(YA_ptps)]
df_gaze_diagonal_OA = df_gaze_diagonal.iloc[:,len(YA_ptps):]

for i in range(df_gaze_diagonal_YA.shape[1]):
    # note that each participant is one column in gaze, but one row in narrative
    gaze_diagonal_YA = df_gaze_diagonal_YA.to_numpy()[:,i]
    narrative_diagonal_YA = df_narrative_diagonal_YA.to_numpy()[i,:]
    df_correlation_map = pd.DataFrame({'gaze':gaze_diagonal_YA, 
                                       'narrative':narrative_diagonal_YA})
    df_correlation_map.to_excel(f'output/correlation_map_analysis_data/YA/{i}.xlsx', index=None)
    
for i in range(df_gaze_diagonal_OA.shape[1]):
    # note that each participant is one column in gaze, but one row in narrative
    gaze_diagonal_OA = df_gaze_diagonal_OA.to_numpy()[:,i]
    narrative_diagonal_OA = df_narrative_diagonal_OA.to_numpy()[i,:]
    df_correlation_map = pd.DataFrame({'gaze':gaze_diagonal_OA, 
                                       'narrative':narrative_diagonal_OA})
    df_correlation_map.to_excel(f'output/correlation_map_analysis_data/OA/{i}.xlsx', index=None)