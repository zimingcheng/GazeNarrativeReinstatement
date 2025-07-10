# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:49:12 2023

@author: zcheng
"""
import os 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from scipy.stats import ttest_ind
import matplotlib.patches as patches 
from scipy.ndimage import label, find_objects
from skimage.measure import find_contours
from matplotlib.collections import LineCollection

# get the list of participants and their age group, and link it with the cma id (which just go from 0 to 31)
df_trial = pd.read_excel('output/df_trial.xlsx')
df_ptp = df_trial.groupby(['ptp','age']).count().reset_index()[['ptp','age']]
df_ptp['cma_id'] = list(range(0,64))
id_2_ptp = dict(zip(df_ptp['cma_id'], df_ptp['ptp']))
id_2_age = dict(zip(df_ptp['cma_id'], df_ptp['age']))
window_size = 10

# extract all cma results 
cma_result={}
for age in ['YA','OA']:
    folder = f'output/correlation_map/{age}'
    files = [f for f in os.listdir(folder)]
    files.remove('desktop.ini')
    cma_result[age] = np.zeros((17,321,len(files)))
    for i in range(len(files)):
        file = files[i]
        file_path = os.path.join(folder, file)
        df = pd.read_excel(file_path, header=None)
        array = df.to_numpy()
        cma_result[age][:,:,i] = array
cma_result_array = np.concatenate((cma_result['YA'], cma_result['OA']), axis=2)

def plot_reinstatement(ptps_to_plot, age, title='temp'):
    if age =='OA':
        ptps_to_plot = [(ptp - 32) for ptp in ptps_to_plot]
    folder = f'output/correlation_map_analysis_data/{age}'
    n_plots = len(ptps_to_plot)
    fig, axes = plt.subplots(n_plots, 1, figsize=(8, n_plots*3))
    for ptp in ptps_to_plot:
        df = pd.read_excel(f'{folder}/{ptp}.xlsx')
        df['time'] = list(np.arange(0,90.5-window_size,0.5))
        df = df.melt(id_vars=['time'], value_vars=['gaze','narrative'], var_name='type',value_name='reinstatement')
        sns.lineplot(df, x='time', y='reinstatement', hue='type', ax=axes[ptp], vmin=-0.5, vmax=1)
    fig.savefig(f'figures/reinstatement/{title}.svg')
    
def plot_reinstatement_avg_age(ptps_to_plot, age, title='temp'):
    if age =='OA':
        ptps_to_plot = [(ptp - 32) for ptp in ptps_to_plot]
    folder = f'output/correlation_map_analysis_data/{age}'
    fig, axes = plt.subplots(1, 1, figsize=(8, 3))
    df_all_ptp = pd.DataFrame({'time':[],'type':[],'reinstatement':[]})
    for ptp in ptps_to_plot:
        df = pd.read_excel(f'{folder}/{ptp}.xlsx')
        df['time'] = list(np.arange(0,90.5-window_size,0.5))
        df = df.melt(id_vars=['time'], value_vars=['gaze','narrative'], var_name='type',value_name='reinstatement')
        df_all_ptp = pd.concat([df_all_ptp,df])
    df_all_ptp = df_all_ptp.reset_index(drop=True)
    sns.lineplot(data=df_all_ptp, x='time', y='reinstatement', hue='type')
    axes.set_ylim(-0.05, 0.35)
    fig.savefig(f'figures/{title}_avg.svg')
    
def plot_reinstatement_avg_kind(ptps_to_plot, kind, title='temp'):
    for age in ['OA','YA']:
        folder = f'output/correlation_map_analysis_data/{age}'
        fig, axes = plt.subplots(1, 1, figsize=(9, 3))
        df_all_ptp = pd.DataFrame({'time':[],'age':[],'reinstatement':[]})
        for ptp in ptps_to_plot:
            df = pd.read_excel(f'{folder}/{ptp}.xlsx')
            df['time'] = list(np.arange(0,90.5-window_size,0.5))
            df = df.melt(id_vars=['time'], value_vars=[kind], var_name='type',value_name='reinstatement')
            df['age'] = age
            df_all_ptp = pd.concat([df_all_ptp,df])
    df_all_ptp = df_all_ptp.reset_index(drop=True)
    sns.lineplot(data=df_all_ptp, x='time', y='reinstatement', hue='age')
    if kind == 'gaze':
        axes.set_ylim(-0.05, 0.05)
    else:
        axes.set_ylim(-0.02, 0.14)
    fig.savefig(f'figures/{title}_avg.svg')
    
def plot_reinstatement_by_type_avg(ptps_to_plot, kind):
    fig, axes = plt.subplots(1, 1, figsize=(9, 3))
    folder = 'output/correlation_map_analysis_data/YA'
    df_YA = pd.DataFrame({'age':[],'ptp-fake':[],'time':[],'type':[],'reinstatement':[]})
    for ptp in ptps_to_plot:
        df = pd.read_excel(f'{folder}/{ptp}.xlsx')
        df['time'] = list(np.arange(0,90.5-window_size,0.5))
        df = df.melt(id_vars=['time'], value_vars=['gaze','narrative'], var_name='type',value_name='reinstatement')
        df['age'] = 'younger adult'
        df['ptp-fake'] = ptp
        df_YA = pd.concat([df_YA,df])
    folder = 'output/correlation_map_analysis_data/OA'
    df_OA = pd.DataFrame({'age':[],'ptp-fake':[],'time':[],'type':[],'reinstatement':[]})
    for ptp in ptps_to_plot:
        df = pd.read_excel(f'{folder}/{ptp}.xlsx')
        df['time'] = list(np.arange(0,90.5-window_size,0.5))
        df = df.melt(id_vars=['time'], value_vars=['gaze','narrative'], var_name='type',value_name='reinstatement')
        df['age'] = 'older adult'
        # this ptp-fake is not the real participant id, just a index to let the
        # code know which data belong to the same participant. 
        df['ptp-fake'] = ptp + 32
        df_OA = pd.concat([df_OA,df])
    df_all_ptp = pd.concat([df_YA, df_OA]).reset_index(drop=True)
    df_all_ptp = df_all_ptp[df_all_ptp['type']==kind]
    sns.lineplot(data=df_all_ptp, x='time', y='reinstatement', hue='age')
    if kind == 'narrative':
        axes.set_ylim(0, 0.35)
    elif kind == 'gaze':
        axes.set_ylim(-0.05, 0.05)
        plt.axhline(0, color='red', linestyle='--')
    fig.savefig(f'figures/reinstatement/{kind}_avg.svg')
    return df_all_ptp
          
    
def plot_cma(ptps_to_plot, title='temp', cma_result=cma_result_array):
    n_plots = len(ptps_to_plot)
    fig, axes = plt.subplots(n_plots, 1, figsize=(15, n_plots*5))
    for i in range(n_plots):
        cma_id = ptps_to_plot[i]
        sns.heatmap(cma_result[:,:,cma_id], cmap='RdBu_r', vmin=-1, vmax=1, ax=axes[i])
        axes[i].set_title(f'{id_2_ptp[cma_id]}: {id_2_age[cma_id]}')
        axes[i].set_xticks([0,40,80,120,160,200,240,280,320])
        axes[i].set_xticklabels([0,10,20,30,40,50,60,70,80])
        axes[i].set_yticks([0.5,2.5,4.5,6.5,8.5,10.5,12.5,14.5,16.5])
        axes[i].set_yticklabels([-4,-3,-2,-1,0,1,2,3,4])
    fig.savefig(f'figures/{title}.svg')
    
def plot_cma_avg(ptps_to_plot, title='temp', cma_result=cma_result_array):
    selected_array = cma_result[:,:,ptps_to_plot]
    avg_array = np.mean(selected_array, axis=2)
    fig, axes = plt.subplots(figsize=(9, 3))
    sns.heatmap(avg_array, cmap='RdBu_r', vmin=-0.3, vmax=0.3, ax=axes)
    axes.set_xticks([0,40,80,120,160,200,240,280,320])
    axes.set_xticklabels([0,10,20,30,40,50,60,70,80])
    axes.set_yticks([0.5,2.5,4.5,6.5,8.5,10.5,12.5,14.5,16.5])
    axes.set_yticklabels([-4,-3,-2,-1,0,1,2,3,4])
    
    # 1. Create a binary mask for significant cells
    threshold = 0.1
    significant_mask = (avg_array > threshold) | (avg_array < -threshold)

    # 2. Label connected components (blobs)
    #    Default connectivity for label is 4-connectivity. For 8-connectivity, use:
    #    structure = np.ones((3,3), dtype=bool)
    #    labeled_array, num_features = label(significant_mask, structure=structure)
    labeled_array, num_features = label(significant_mask) 
    
    rows, cols = avg_array.shape
    all_border_segments = [] # To store all [(x1,y1),(x2,y2)] line segments

    # 3. Identify exterior edges for each blob
    for blob_idx in range(1, num_features + 1):
        # Get coordinates of all cells belonging to the current blob
        blob_cells_coords = np.argwhere(labeled_array == blob_idx) # Array of [row, col]

        for r, c in blob_cells_coords:
            # For cell (r,c), check its 4 neighbors.
            # If a neighbor is not part of the same blob, the shared edge is a border.

            # Check top neighbor (cell r-1, c)
            if r == 0 or labeled_array[r - 1, c] != blob_idx:
                # Add top edge of cell (r,c): from (c, r) to (c+1, r)
                all_border_segments.append([(c, r), (c + 1, r)])
            
            # Check bottom neighbor (cell r+1, c)
            if r == rows - 1 or labeled_array[r + 1, c] != blob_idx:
                # Add bottom edge of cell (r,c): from (c, r+1) to (c+1, r+1)
                all_border_segments.append([(c, r + 1), (c + 1, r + 1)])
                
            # Check left neighbor (cell r, c-1)
            if c == 0 or labeled_array[r, c - 1] != blob_idx:
                # Add left edge of cell (r,c): from (c, r) to (c, r+1)
                all_border_segments.append([(c, r), (c, r + 1)])
                
            # Check right neighbor (cell r, c+1)
            if c == cols - 1 or labeled_array[r, c + 1] != blob_idx:
                # Add right edge of cell (r,c): from (c+1, r) to (c+1, r+1)
                all_border_segments.append([(c + 1, r), (c + 1, r + 1)])

    # 4. Draw all collected border segments
    if all_border_segments:
        line_col = LineCollection(all_border_segments,
                                  colors='black',      # Border color
                                  linewidths=1.5,    # Border thickness
                                  linestyle='solid') # Border style
        axes.add_collection(line_col)
        
    fig.savefig(f'figures/{title}.svg')

    
    
def plot_cma_1davg(ptps_to_plot, cma_result, title='temp'):
    selected_array = cma_result[:,ptps_to_plot]
    avg_array = np.mean(selected_array, axis=1)
    fig, axes = plt.subplots(figsize=(9, 3))
    sns.lineplot(x=np.arange(len(avg_array)), y=avg_array, ax=axes)
    axes.set_xticks([0,40,80,120,160,200,240,280,320])
    axes.set_xticklabels([0,10,20,30,40,50,60,70,80])
    axes.set_ylim(-0.1, 0.1)
    fig.savefig(f'figures/{title}.svg')
    
def cluster_similarity_data(cma_result_array):
    ks,types,ptp1s,ptp2s,sims = [],[],[],[],[]
    for k in range(2, 11):
        clusters_df = pd.read_csv(f'2D_clustering/cma_2dEM_output_{k}groups.csv',header=None)
        clusters = np.asarray(clusters_df)[0,:]
        cma_by_cluster = {}
        # loop through each cluster number
        # i = cluster index, k = total number of clusters 
        for i in range(1,k+1):
            ptp_in_cluster = list(np.where(clusters==i)[0])
            cma_this_cluster = cma_result_array[:,:,ptp_in_cluster]
            cma_by_cluster[i] = cma_this_cluster
            # within cluster similarity 
            for ptp_id2 in range(len(ptp_in_cluster)):
                for ptp_id1 in range(ptp_id2):
                    ptp1 = ptp_in_cluster[ptp_id1]
                    ptp2 = ptp_in_cluster[ptp_id2]
                    sim = kendalltau(cma_result_array[:,:,ptp1].flatten(), cma_result_array[:,:,ptp2].flatten())[0]
                    ks.append(k)
                    types.append('within')
                    ptp1s.append(ptp1)
                    ptp2s.append(ptp2)
                    sims.append(sim)
            # between cluster similarity 
            ptps_not_in_cluster = list(set(range(64)) - set(ptp_in_cluster))
            for ptp2 in ptps_not_in_cluster:
                for ptp1 in ptp_in_cluster:
                    sim = kendalltau(cma_result_array[:,:,ptp1].flatten(), cma_result_array[:,:,ptp2].flatten())[0]
                    ks.append(k)
                    types.append('between')
                    ptp1s.append(ptp1)
                    ptp2s.append(ptp2)
                    sims.append(sim)
                    
        
    output = pd.DataFrame({'k':ks, 'type':types, 'ptp1':ptp1s, 'ptp2':ptp2s, 'sim':sims})
    return output

def cma_to_array(df, window_size):
    array_len = len(list(np.arange(0,90.5-window_size,0.5)))
    YA_array = np.zeros((32, array_len))
    OA_array = np.zeros((32, array_len))
    for ptp in list(range(0,32)):
        df_sub = df[df['ptp-fake'] == ptp]
        reinstatement = np.asarray(df_sub['reinstatement'])
        YA_array[ptp,:] = reinstatement
    for ptp in list(range(32,64)):
        df_sub = df[df['ptp-fake'] == ptp]
        reinstatement = np.asarray(df_sub['reinstatement'])
        OA_array[ptp-32,:] = reinstatement
    return YA_array, OA_array

def error_to_asterisks(errors):
    output = []
    for error in errors:
        if error[0]>0 or error[1]<0:
            output.append('.')
        else:
            output.append('')
    return output
        
def lineplot_2d_array(array_YA, array_OA, window_size, title, ymax, ymin, add_aste_idx=[], ylabel='sim'):
    """
    axis 0 indicates participant. axis 1 indicates time window
    """
    array_len = len(list(np.arange(0,90.5-window_size,0.5)))
    ptp_len_YA = array_YA.shape[0]
    ptp_len_OA = array_OA.shape[0]
    timeline_len = array_YA.shape[1]
    timeline_YA = np.tile(np.arange(timeline_len), (ptp_len_YA, 1))
    timeline_OA = np.tile(np.arange(timeline_len), (ptp_len_OA, 1))
    array_YA_flat = array_YA.flatten()
    array_OA_flat = array_OA.flatten()
    timeline_flat_YA = timeline_YA.flatten()
    timeline_flat_OA = timeline_OA.flatten()
    df_YA = pd.DataFrame({'time (s)':timeline_flat_YA, ylabel:array_YA_flat, 'age':'YA'})
    df_OA = pd.DataFrame({'time (s)':timeline_flat_OA, ylabel:array_OA_flat, 'age':'OA'})
    df = pd.concat([df_YA, df_OA]).reset_index(drop=True)
    g_YA = sns.barplot(data=df_YA,x='time (s)',y=ylabel,n_boot=1000,seed=2023)
    plt.show()
    g_OA = sns.barplot(data=df_OA,x='time (s)',y=ylabel,n_boot=1000,seed=2023)
    plt.show()
    df['age'] = df['age'].map({'YA':'younger','OA':'older'})
    ax = sns.lineplot(data=df,x='time (s)',y=ylabel,hue='age',n_boot=1000,seed=2023)
    ax.legend(loc='lower left')
    # ax = sns.lineplot(data=df[df['age']=='younger'],x='time (s)',y=ylabel,n_boot=1000,seed=2023)
    YA_error = []
    OA_error = []
    for i in range(5, 5+array_len):
        l = g_YA.lines[i]
        xy = l.get_xydata()
        upper = xy[1,1]
        lower = xy[0,1]
        YA_error.append((lower,upper))
    for i in range(len(g_OA.lines)):
        l = g_OA.lines[i]
        xy = l.get_xydata()
        upper = xy[1,1]
        lower = xy[0,1]
        OA_error.append((lower,upper))
    if ymax!=0 and ymin!=0:
        plt.axhline(y=0, color='black', linestyle='-')
    aste_YA = error_to_asterisks(YA_error)
    aste_OA = error_to_asterisks(OA_error)
    if len(add_aste_idx) > 0:
        aste_added = [''] * timeline_len
        for index in add_aste_idx:
            aste_added[index] = '*'
        for idx, pval in enumerate(aste_added):
            plt.text(x=idx-0.15, y=ymax*0.9, s=pval, color='black')
    for idx, pval in enumerate(aste_YA):
        plt.text(x=idx-0.15, y=ymax*0.8, s=pval, color='blue')
    for idx, pval in enumerate(aste_OA):
        plt.text(x=idx-0.15, y=ymax*0.85, s=pval, color='orange')
    plt.title(title)
    plt.ylim((ymin,ymax))
    plt.xticks([0,20,40,60,80,100,120,140,160], [0,10,20,30,40,50,60,70,80])
    plt.savefig('figures/line.svg', format='svg')
    plt.show()
    return YA_error, OA_error
                                  

plot_cma_avg(list(range(32)), 'YA_CMA')
plot_cma_avg(list(range(32,64)), 'OA_CMA')

