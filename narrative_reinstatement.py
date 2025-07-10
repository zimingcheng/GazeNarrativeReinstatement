# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:07:44 2023

@author: ziming cheng
"""
# Import necessary libraries
import pandas as pd
import glob
import numpy as np
import tensorflow_hub as hub
USE = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
from tqdm import tqdm
import random
import nltk


def get_all_transcripts(df_trial):
    """Reads and combines individual transcript files into a single DataFrame."""
    df_transcript = pd.DataFrame()
    # read transcripts from the transcription folder based on the trials in <df_trial>
    for index, row in df_trial.iterrows():
        ptp = str(row['ptp']).zfill(3)
        task = row['task'].lower()
        image = row['image']
        files = glob.glob(f'../transcription/sub-{ptp}/Transcript_auto/sub-{ptp}_*_task-{task}_image-{image}*.csv')
        if len(files) == 1:
            file = files[0]
            df_transcript_sub = pd.read_csv(file)
        # concat the current transcript to the master dataframe
        # this results in one word per line
            df_transcript = pd.concat([df_transcript, df_transcript_sub])
        else:
            print(f'sub-{ptp}_{task}_{image} does not exist')
    return df_transcript

def clean_transcripts(df_transcript):
    """Parses the filename column to extract metadata like participant, task, and image."""
    df_transcript['ptp'] = df_transcript['filename'].apply(lambda x: x.split('_')[0].split('-')[1])
    df_transcript['trial'] = df_transcript['filename'].apply(lambda x: x.split('_')[1].split('-')[1])
    df_transcript['task'] = df_transcript['filename'].apply(lambda x: x.split('_')[2].split('-')[1])
    df_transcript['image'] = df_transcript['filename'].apply(lambda x: x.split('_')[3].split('-')[1])
    return df_transcript

def USE_similarity(USE, sentence1, sentence2):
    """Calculates the semantic similarity between two sentences using the pre-loaded USE model."""
    USE_output = np.array(USE([sentence1, sentence2]))
    similarity = np.inner(USE_output[0], USE_output[1])
    return similarity

def combine_transcripts(df_transcript):
    """Converts a DataFrame of individual words into a DataFrame of full trial narratives."""
    # this turns df with one word per line into one trial per line
    ptp_out, image_out, task_out, transcript_out = [],[],[],[]
    for index, row in df_transcript.iterrows():
        # if this is a new trial
        if index==0 or (
                row['ptp'] != last_ptp or 
                row['image'] != last_image or 
                row['task'] != last_task):
            # start a new entry
            ptp_out.append(row['ptp'])
            image_out.append(row['image'])
            task_out.append(row['task'])
            transcript_out.append(row['word'])
            last_ptp = row['ptp']
            last_image = row['image']
            last_task = row['task']
        # if this is an old trial
        else:
            # add the word to the current entry
            transcript_out[-1] = transcript_out[-1] + ' ' + row['word']
    df = pd.DataFrame({'ptp':ptp_out,'image':image_out,'task':task_out,
                   'transcript': transcript_out})
    return df

def add_age(df):
    """Adds an age column based on the participant ID."""
    df['age'] = ''
    for index, row in df.iterrows():
        ptp = row['ptp']
        if ptp[0] == '0':
            df.at[index, 'age'] = 'YA'
        elif ptp[0] == '1':
            df.at[index, 'age'] = 'OA'
    return df

def USE_permutation(df, USE):
    """Calculates overall narrative reinstatement and a permuted baseline."""
    print('==========')
    print('calculating overall time-independent narrative reinstatement')
    print('==========')
    ptps, ages, reins_types, similarities, images, narrative1s, narrative2s = [],[],[],[],[],[],[]
    for age in ['YA','OA']:
        print(f'===={age}====')
        df_age = df[df['age'] == age]
        all_ptps = list(df_age['ptp'].unique())
        # 1. Calculate true "narrative reinstatement"
        print('narrative reinstatement')
        for i in tqdm(range(len(all_ptps))):
            df_this_ptp = df_age[df_age['ptp']==all_ptps[i]]
            images_this_ptp = list(df_this_ptp['image'].unique())
            for j in range(len(images_this_ptp)):
                narrative1 = df_age[(df_age['ptp']==all_ptps[i])&(df_age['image']==images_this_ptp[j])&
                                (df_age['task']=='desc')].reset_index()\
                                ['transcript'][0]
                narrative2 = df_age[(df_age['ptp']==all_ptps[i])&(df_age['image']==images_this_ptp[j])&
                                (df_age['task']=='rec')].reset_index()\
                                ['transcript'][0]
                # Calculate similarity and store the results.
                sim = USE_similarity(USE, narrative1, narrative2)
                ptps.append(all_ptps[i])
                ages.append(age)
                reins_types.append('narrative reinstatement')
                similarities.append(sim)
                images.append(images_this_ptp[j])
                narrative1s.append(narrative1)
                narrative2s.append(narrative2)
        # 2. Calculate "baseline reinstatement" (permutation).
        print('baseline reinstatement')
        for i in tqdm(range(len(all_ptps))):
            df_this_ptp = df_age[df_age['ptp']==all_ptps[i]]
            images_this_ptp = list(df_this_ptp['image'].unique())
            for j in range(len(images_this_ptp)):
                # Get a list of all OTHER images seen by this participant.
                other_images = images_this_ptp[:]
                other_images.remove(images_this_ptp[j])
                baseline_descs = list(df_age[(df_age['ptp']==all_ptps[i])&\
                                    (df_age['image'].isin(other_images))&\
                                    (df_age['task']=='desc')]['transcript'])
                narrative2 = df_age[(df_age['ptp']==all_ptps[i])&(df_age['image']==images_this_ptp[j])&
                                (df_age['task']=='rec')].reset_index()\
                                ['transcript'][0]
                # Compare the recall to each of the "wrong" descriptions and store results.
                for narrative1 in baseline_descs:
                    sim = USE_similarity(USE, narrative1, narrative2)
                    ptps.append(all_ptps[i])
                    ages.append(age)
                    reins_types.append('baseline reinstatement')
                    similarities.append(sim)
                    images.append(images_this_ptp[j])
                    narrative1s.append(narrative1)
                    narrative2s.append(narrative2)
    df_USE = pd.DataFrame({'ptp':ptps, 'age':ages, 'reins_type':reins_types,'similarity':similarities,
                           'images':images,'narrative1':narrative1s,
                           'narrative2':narrative2s})
    return df_USE

def combine_transcripts_time(df_transcript, frame_dur, step, overlap=True):
    """Chunks the word-by-word transcripts into time windows."""
    ptp_out, image_out, task_out, time_id_out, time_start_out, time_end_out, transcript_out = [],[],[],[],[],[],[]
    all_ptps = df_transcript['ptp'].unique()
    for i in tqdm(range(len(all_ptps))):
        ptp = all_ptps[i]
        df_this_ptp = df_transcript[df_transcript['ptp']==ptp]
        images_this_ptp = list(df_this_ptp['image'].unique())
        for image in images_this_ptp:
            for task in df_transcript['task'].unique():
                last_end_time = 5
                i = 0
                # Create sliding time windows (e.g., 5-15s, 10-20s, etc.).
                for start_time in np.arange(5, 95-frame_dur+1, step):
                    overlap_switch = 1
                    end_time = start_time + frame_dur
                    if not overlap:
                        if start_time < last_end_time:
                            overlap_switch = 0
                    if overlap_switch == 1:
                        i += 1
                        df_this_transcript = df_transcript[
                            (df_transcript['ptp']==ptp) &
                            (df_transcript['image']==image) &
                            (df_transcript['task']==task) &
                            (df_transcript['start']>=start_time) &
                            (df_transcript['start']<=end_time)]
                        this_transcript = ' '.join(df_this_transcript['word'])
                        last_end_time = end_time
                        ptp_out.append(ptp)
                        image_out.append(image)
                        task_out.append(task)
                        time_id_out.append(i)
                        time_start_out.append(start_time)
                        time_end_out.append(end_time)
                        transcript_out.append(this_transcript)
    df = pd.DataFrame({'ptp':ptp_out,'image':image_out,'task':task_out,
                   'time_id':time_id_out,'time_start':time_start_out,
                   'time_end':time_end_out,'transcript': transcript_out})
    return df

def USE_direct_comparison_time(df_time, m_max, n_max):
    """Computes the temporal cross-similarity matrix for each participant."""
    result = {}
    for age in ['YA','OA']:
        df_age = df_time[df_time['age'] == age]
        all_ptps = list(df_age['ptp'].unique())
        # all_ptps = ['001']
        sim_result = np.zeros((len(all_ptps),m_max,n_max))
        for i in range(len(all_ptps)):
            print(all_ptps[i])
            for m in tqdm(range(1,m_max+1), desc='desc (m)'):
                for n in range(1,n_max+1):
                    sim_time_dependent_all_images = []
                    df_this_ptp = df_age[df_age['ptp']==all_ptps[i]]
                    images_this_ptp = list(df_this_ptp['image'].unique())
                    for j in range(len(images_this_ptp)):
                        narrative1 = df_age[(df_age['ptp']==all_ptps[i])&(df_age['image']==images_this_ptp[j])&
                                        (df_age['time_id']==m)&(df_age['task']=='desc')].reset_index()\
                                        ['transcript'][0]
                        narrative2 = df_age[(df_age['ptp']==all_ptps[i])&(df_age['image']==images_this_ptp[j])&
                                        (df_age['time_id']==n)&(df_age['task']=='rec')].reset_index()\
                                        ['transcript'][0]
                        if narrative1 != '' and narrative2 != '':
                            sim = USE_similarity(USE, narrative1, narrative2)
                            sim_time_dependent_all_images.append(sim)
                        else:
                            sim_time_dependent_all_images.append(0)
                    sim_time_dependent = np.mean(sim_time_dependent_all_images)
                    sim_result[i,m-1,n-1] = sim_time_dependent
        result[age] = sim_result
    return result

def USE_direct_comparison_diagonal(df_time, m_max):
    """Computes similarity only for matching time windows (the matrix diagonal)."""
    result = {}
    for age in ['YA','OA']:
        df_age = df_time[df_time['age'] == age]
        df_age = df_age.fillna('')
        all_ptps = list(df_age['ptp'].unique())
        sim_result = np.zeros((len(all_ptps),m_max))
        for i in range(len(all_ptps)):
            print(all_ptps[i])
            for m in tqdm(np.arange(1,m_max+1), desc='desc (m)'):
                sim_time_dependent_all_images = []
                df_this_ptp = df_age[df_age['ptp']==all_ptps[i]]
                images_this_ptp = list(df_this_ptp['image'].unique())
                for j in range(len(images_this_ptp)):
                    narrative1 = df_age[(df_age['ptp']==all_ptps[i])&(df_age['image']==images_this_ptp[j])&
                                    (df_age['time_id']==m)&(df_age['task']=='desc')].reset_index()\
                                    ['transcript'][0]
                    narrative2 = df_age[(df_age['ptp']==all_ptps[i])&(df_age['image']==images_this_ptp[j])&
                                    (df_age['time_id']==m)&(df_age['task']=='rec')].reset_index()\
                                    ['transcript'][0]
                    if narrative1 != '' and narrative2 != '':
                        sim = USE_similarity(USE, narrative1, narrative2)
                        sim_time_dependent_all_images.append(sim)
                    else:
                        sim_time_dependent_all_images.append(0)
                sim_time_dependent = np.mean(sim_time_dependent_all_images)
                sim_result[i,m-1] = sim_time_dependent
        result[age] = sim_result
    return result

# The following two functions (`USE_permutation_time`, `USE_permutation_diagonal`) are the
# baseline/control versions of the time-dependent analyses. Instead of comparing a recall
# window to its corresponding description window, they compare it to description windows
# from OTHER images by the same participant. This measures baseline similarity due to a
# person's general narrative style, rather than specific memory reinstatement.

def USE_permutation_time(df_time, m_max, n_max, sample=50):
    """Computes the permuted (baseline) temporal cross-similarity matrix."""
    result = {}
    for age in ['YA','OA']:
        df_age = df_time[df_time['age'] == age]
        all_ptps = list(df_age['ptp'].unique())
        # all_ptps = ['001']
        sim_result = np.zeros((len(all_ptps),m_max,n_max))
        for i in range(len(all_ptps)):
            print(all_ptps[i])
            for m in tqdm(range(1,m_max+1), desc='desc (m)'):
                for n in range(1,n_max+1):
                    sim_time_dependent_all_images = []
                    df_this_ptp = df_age[df_age['ptp']==all_ptps[i]]
                    images_this_ptp = list(df_this_ptp['image'].unique())
                    for j in range(len(images_this_ptp)):
                        # find all the other participants in the same age group
                        other_images = images_this_ptp[:]
                        other_images.remove(images_this_ptp[j])
                        if len(other_images) > sample:
                            other_images = random.choices(other_images, k=sample)
                        for k in range(len(other_images)):
                            # finding the similarity between the target recall trial and 
                            # all the descriptions of other images of the same person
                            narrative1 = df_age[(df_age['ptp']==all_ptps[i])&(df_age['image']==other_images[k])&
                                            (df_age['time_id']==m)&(df_age['task']=='desc')].reset_index()\
                                            ['transcript'][0]
                            narrative2 = df_age[(df_age['ptp']==all_ptps[i])&(df_age['image']==images_this_ptp[j])&
                                            (df_age['time_id']==n)&(df_age['task']=='rec')].reset_index()\
                                            ['transcript'][0]
                            if narrative1 != '' and narrative2 != '':
                                sim = USE_similarity(USE, narrative1, narrative2)
                                sim_time_dependent_all_images.append(sim)
                            else:
                                sim_time_dependent_all_images.append(0)
                    sim_time_dependent = np.mean(sim_time_dependent_all_images)
                    sim_result[i,m-1,n-1] = sim_time_dependent
        result[age] = sim_result
    return result

def USE_permutation_diagonal(df_time, m_max, sample=50):
    """Computes the permuted (baseline) similarity for the diagonal."""
    result = {}
    for age in ['YA','OA']:
        df_age = df_time[df_time['age'] == age]
        df_age = df_age.fillna('')
        all_ptps = list(df_age['ptp'].unique())
        sim_result = np.zeros((len(all_ptps),m_max))
        for i in range(len(all_ptps)):
            print(all_ptps[i])
            for m in tqdm(np.arange(1,m_max+1), desc='desc (m)'):
                sim_time_dependent_all_images = []
                df_this_ptp = df_age[df_age['ptp']==all_ptps[i]]
                images_this_ptp = list(df_this_ptp['image'].unique())
                for j in range(len(images_this_ptp)):
                    # find all the other participants in the same age group
                    other_images = images_this_ptp[:]
                    other_images.remove(images_this_ptp[j])
                    if len(other_images) > sample:
                        other_images = random.choices(other_images, k=sample)
                    for k in range(len(other_images)):
                        # finding the similarity between the target recall trial and 
                        # all the descriptions of other images of the same person
                        narrative1 = df_age[(df_age['ptp']==all_ptps[i])&(df_age['image']==other_images[k])&
                                        (df_age['time_id']==m)&(df_age['task']=='desc')].reset_index()\
                                        ['transcript'][0]
                        narrative2 = df_age[(df_age['ptp']==all_ptps[i])&(df_age['image']==images_this_ptp[j])&
                                        (df_age['time_id']==m)&(df_age['task']=='rec')].reset_index()\
                                        ['transcript'][0]
                        if narrative1 != '' and narrative2 != '':
                            sim = USE_similarity(USE, narrative1, narrative2)
                            sim_time_dependent_all_images.append(sim)
                        else:
                            sim_time_dependent_all_images.append(0)
                sim_time_dependent = np.mean(sim_time_dependent_all_images)
                sim_result[i,m-1,] = sim_time_dependent
        result[age] = sim_result
    return result

def plot_USE_time_matrix(df_time_result, vmax, vmin, analysis=''):
    """Visualizes the temporal similarity matrix as a heatmap."""
    for age in ['YA','OA']:
        USE_matrix = np.nanmean(df_time_result[age], axis=0)
        fig, ax = plt.subplots()
        im = ax.matshow(USE_matrix, vmin=vmin, vmax=vmax, cmap='plasma')
        fig.colorbar(im, ax=ax)
        if age=='YA':
            ax.set_title('younger adults')
        else:
            ax.set_title('older adults')
        ticklabel = [0,0,10,20,30,40,50,60,70,80]
        ax.set_xticklabels(ticklabel)
        ax.set_yticklabels(ticklabel)
        ax.set_xlabel('recall')
        ax.set_ylabel('description')
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labeltop=False)
        plt.savefig(f'figures/{age}_{analysis}.svg', format='svg')
        plt.show()
        
def error_to_asterisks(errors):
    """Draw asterisks to where the 95% CI is above 0"""
    output = []
    for error in errors:
        if error[0]>0 or error[1]<0:
            output.append('*')
        else:
            output.append('')
    return output
        
def lineplot_2d_array(array_YA, array_OA, title, ymax, ymin, add_aste_idx=[], ylabel='sim', analysis=''):
    """
    Creates a line plot for the diagonal data with confidence intervals. 
    This function reshapes the data, plots it using seaborn,
    calculates confidence intervals, and adds significance markers
    """
    ptp_len_YA = array_YA.shape[0]
    ptp_len_OA = array_OA.shape[0]
    timeline_len = array_YA.shape[1]
    timeline_YA = np.tile(np.arange(timeline_len), (ptp_len_YA, 1))
    timeline_OA = np.tile(np.arange(timeline_len), (ptp_len_OA, 1))
    array_YA_flat = array_YA.flatten()
    array_OA_flat = array_OA.flatten()
    timeline_flat_YA = timeline_YA.flatten()
    timeline_flat_OA = timeline_OA.flatten()
    # note that ptp_fake is a generated participant list to tell apart different participants, not their actual ID
    ptp_fake_YA = list(range(ptp_len_YA))*timeline_len
    ptp_fake_OA = list(range(ptp_len_OA))*timeline_len
    ptp_fake_YA.sort()
    ptp_fake_OA.sort()
    df_YA = pd.DataFrame({'time (s)':timeline_flat_YA, ylabel:array_YA_flat, 'age':'YA', 'ptp_fake':ptp_fake_YA})
    df_OA = pd.DataFrame({'time (s)':timeline_flat_OA, ylabel:array_OA_flat, 'age':'OA', 'ptp_fake':ptp_fake_OA})
    df = pd.concat([df_YA, df_OA]).reset_index(drop=True)
    g_YA = sns.barplot(data=df_YA,x='time (s)',y=ylabel,n_boot=1000,seed=2023)
    plt.show()
    g_OA = sns.barplot(data=df_OA,x='time (s)',y=ylabel,n_boot=1000,seed=2023)
    plt.show()
    df['age'] = df['age'].map({'YA':'younger','OA':'older'})
    ax = sns.lineplot(data=df,x='time (s)',y=ylabel,hue='age',n_boot=1000,seed=2023,marker='o',markersize=2,markeredgewidth=0,linewidth=1)
    ax.legend(loc='lower left')
    # ax = sns.lineplot(data=df[df['age']=='younger'],x='time (s)',y=ylabel,n_boot=1000,seed=2023)
    YA_error = []
    OA_error = []
    for i in range(len(g_YA.lines)):
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
    # if len(add_aste_idx) > 0:
    #     aste_added = [''] * timeline_len
    #     for index in add_aste_idx:
    #         aste_added[index] = '*'
    #     for idx, pval in enumerate(aste_added):
    #         plt.text(x=idx-0.15, y=ymax*0.9, s=pval, color='black')
    # for idx, pval in enumerate(aste_YA):
    #     plt.text(x=idx-0.15, y=ymax*0.8, s=pval, color='blue')
    # for idx, pval in enumerate(aste_OA):
    #     plt.text(x=idx-0.15, y=ymax*0.85, s=pval, color='orange')
    plt.title(title)
    plt.ylim((ymin,ymax))
    if array_YA.shape[1] > 30:
        plt.xticks([0,20,40,60,80,100,120,140,160], [0,10,20,30,40,50,60,70,80])
    else:
        plt.xticks([0,2,4,6,8,10,12,14,16], [0,10,20,30,40,50,60,70,80])
    plt.savefig(f'figures/line_{analysis}.svg', format='svg')
    plt.show()
    return df, aste_YA, aste_OA
        
def plot_USE_line(df_time_result, ymax, ymin, add_aste=[], ylabel='sim', analysis=''):
    """Extracts the diagonal from a result matrix and plots it as a line graph."""
    diagonal_YA = np.diagonal(df_time_result['YA'], axis1=1, axis2=2)
    diagonal_OA = np.diagonal(df_time_result['OA'], axis1=1, axis2=2)
    diagonal_df = lineplot_2d_array(diagonal_YA, diagonal_OA, '', ymax, ymin, add_aste, ylabel=ylabel, analysis=analysis)
    return diagonal_df

    
def save_3d_matrix(result):
    """Reshapes a 3D matrix into a 2D format and saves it to an Excel file."""
    for age in ['YA','OA']:
        matrix_3d = result[age]
        n_ptps = matrix_3d.shape[0]
        side_len = matrix_3d.shape[1]
        matrix_2d = np.zeros((side_len, n_ptps*side_len))
        for i in range(n_ptps):
            slice_ = matrix_3d[i,:,:]
            matrix_2d[:,side_len*i:side_len*(i+1)] = slice_
        df = pd.DataFrame(matrix_2d)
        df.to_excel(f'output/narrative_reinstatement_time_{age}.xlsx', index=None)
        
def word_count(string):
    tokens = nltk.word_tokenize(string)
    words = [word for word in tokens if word.isalpha()]
    return len(words)
    
# This is the main block of the script that executes the entire workflow.
if __name__ == '__main__':
    # 1. Load, clean, and combine the transcript data.
    df_trial = pd.read_excel('output/df_trial.xlsx')
    df_transcript = get_all_transcripts(df_trial)
    df_transcript = clean_transcripts(df_transcript)
    df_transcript.to_excel('output/df_transcript.xlsx', index=False)
    df = combine_transcripts(df_transcript)
    df = add_age(df)
    df['word_count'] = df['transcript'].apply(word_count)
    df.to_excel('output/df_narratives.xlsx', index=False)
      
    # 2. Run the overall, time-independent reinstatement analysis.
    df_USE = USE_permutation(df, USE)
    df_USE.to_excel('output/narrative_reinstatement.xlsx')
    sns.catplot(data=df_USE, x='reins_type', y='similarity', col='age')
    plt.show()
    
    # 3. Run the coarse-grained time-dependent analysis (5s steps).
    df_transcript = pd.read_excel('output/df_transcript.xlsx') 
    df = pd.read_excel('output/df_narratives.xlsx')
    frame_duration = 10
    step = 5

    df_time = combine_transcripts_time(df_transcript, frame_duration, step, overlap=True)
    df_time['ptp'] = df_time['ptp'].apply(lambda x: str(x).zfill(3))
    df_time = add_age(df_time)
    df_time = df_time.fillna('')
    df_time.to_excel('output/df_narratives_time_course.xlsx')
    # calculate how big the USE time-dependent similarity matrix should be according
    # to the current frame_duration and step
    result_size = (95-5-frame_duration)//step+1
    df_time_result = USE_direct_comparison_time(df_time, result_size, result_size)
    plot_USE_time_matrix(df_time_result, 0.5, 0)
    plot_USE_line(df_time_result, 0.5, 0)
    
    df_time_result_perm = USE_permutation_time(df_time, result_size, result_size)
    plot_USE_time_matrix(df_time_result_perm, 0.5, 0)
    plot_USE_line(df_time_result_perm, 0.5, 0)
    
    df_time_result_diff = {}
    df_time_result_diff['YA'] = df_time_result['YA'] - df_time_result_perm['YA']
    df_time_result_diff['OA'] = df_time_result['OA'] - df_time_result_perm['OA']
    plot_USE_time_matrix(df_time_result_diff, 0.3, 0)
    plot_USE_line(df_time_result_diff, 0.3, 0)
    save_3d_matrix(df_time_result_diff)
    
    
    # 4. Run the high-resolution diagonal-only analysis (0.5s steps).
    frame_duration = 10
    step = 0.5
    df_time = combine_transcripts_time(df_transcript, frame_duration, step, overlap=True)
    df_time['ptp'] = df_time['ptp'].apply(lambda x: str(x).zfill(3))
    df_time = add_age(df_time)
    df_time.to_excel('output/df_narratives_time_fine.xlsx')
    df_time = pd.read_excel('output/df_narratives_time_fine.xlsx')
    # calculate how big the USE time-dependent similarity matrix should be according
    # to the current frame_duration and step
    result_size = int((95-5-frame_duration)/step+1)
    df_time_result_diagonal = USE_direct_comparison_diagonal(df_time, result_size)
    
    df_time_result_perm_diagonal = USE_permutation_diagonal(df_time, result_size)

    df_time_result_diff_diagonal = {}
    df_time_result_diff_diagonal['YA'] = df_time_result_diagonal['YA'] - df_time_result_perm_diagonal['YA']
    df_time_result_diff_diagonal['OA'] = df_time_result_diagonal['OA'] - df_time_result_perm_diagonal['OA']
    
    pd.DataFrame(df_time_result_diff_diagonal['YA']).to_excel('output/narrative_reinstatement_time_diagonal_YA.xlsx', index=None)
    pd.DataFrame(df_time_result_diff_diagonal['OA']).to_excel('output/narrative_reinstatement_time_diagonal_OA.xlsx', index=None)
