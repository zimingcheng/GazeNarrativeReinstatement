# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 17:17:56 2023

@author: czm19
"""
import pandas as pd
import numpy as np
import seaborn as sns
import nltk
import matplotlib.pyplot as plt
from tqdm import tqdm
import narrative_reinstatement

IMAGES = ['boat','camping','circus','construction','crocodile','dancing',
          'farming','golf','icecream','museum']
TASKS = ['DESC','REC']


def clean_images(df_fixation, PTPS):
    """
    this also works for saccades
    """
    df_fixation['AGE'] = ''
    df_fixation['IMAGE'] = ''
    df_fixation = df_fixation[
        (df_fixation['CONDITION']=='DESC')|(df_fixation['CONDITION']=='REC')]
    for index, row in df_fixation.iterrows():
        condition = row['CONDITION']
        ptp = row['RECORDING_SESSION_LABEL'][4:7]
        df_fixation.at[index, 'RECORDING_SESSION_LABEL'] = ptp
        if condition == 'DESC':
            df_fixation.at[index, 'IMAGE'] = row['imagename_desc']
        elif condition == 'REC':
            df_fixation.at[index, 'IMAGE'] = row['imagename_rec']
        if ptp[0] == '0':
            df_fixation.at[index, 'AGE'] = 'YA'
        elif ptp[0] == '1':
            df_fixation.at[index, 'AGE'] = 'OA'
    df_fixation = df_fixation[df_fixation['IP_INDEX']==2]
    df_fixation = df_fixation.drop(columns=['IP_INDEX','IP_LABEL',
                                            'DIFF_DESC','DIFF_REC','VIVIDNESS_REC',
                                            'imagename_desc','imagename_rec'])
    df_fixation = df_fixation[df_fixation['RECORDING_SESSION_LABEL'].isin(PTPS)]
    return df_fixation.reset_index(drop=True)
            
def segment_fixation_till_first_word(df_fixation, df_transcript, PTPS):
    df_transcript['ptp'] = df_transcript['ptp'].apply(lambda x: str(x).zfill(3))
    df_fixation_seg = pd.DataFrame()
    for ptp in tqdm(PTPS):
        df_this_ptp = df_transcript[df_transcript['ptp']==ptp]
        images_this_ptp = list(df_this_ptp['image'].unique())
        for image in images_this_ptp:
            for task in ['desc','rec']:
                this_df_transcript = df_transcript[
                    (df_transcript['ptp']==ptp) & 
                    (df_transcript['image']==image) & 
                    (df_transcript['task']==task)]
                first_word_start_time = 0
                row = 0
                while first_word_start_time < 5000:
                    first_word_start_time = this_df_transcript.iloc[row]['start'] * 1000
                    row += 1
                this_df_fixation = df_fixation[
                    (df_fixation['RECORDING_SESSION_LABEL']==ptp) & 
                    (df_fixation['IMAGE']==image) & 
                    (df_fixation['CONDITION']==task.upper())]
                this_df_fixation = this_df_fixation[
                    this_df_fixation['CURRENT_FIX_START'] <= first_word_start_time]
                # this_df_fixation.loc[this_df_fixation['CURRENT_FIX_START'] > last_word_end_time,
                #                      'CURRENT_FIX_X'] = -1000
                this_df_fixation['FIRST_WORD_START_TIME'] = first_word_start_time
                df_fixation_seg = pd.concat([df_fixation_seg, this_df_fixation])
    return df_fixation_seg

def segment_fixation_till_last_word(df_fixation, df_transcript, PTPS):
    df_transcript['ptp'] = df_transcript['ptp'].apply(lambda x: str(x).zfill(3))
    df_fixation_seg = pd.DataFrame()
    for ptp in tqdm(PTPS):
        df_this_ptp = df_transcript[df_transcript['ptp']==ptp]
        images_this_ptp = list(df_this_ptp['image'].unique())
        for image in images_this_ptp:
            for task in ['desc','rec']:
                this_df_transcript = df_transcript[
                    (df_transcript['ptp']==ptp) & 
                    (df_transcript['image']==image) & 
                    (df_transcript['task']==task)]
                last_word_end_time = this_df_transcript.iloc[-1]['end'] * 1000
                this_df_fixation = df_fixation[
                    (df_fixation['RECORDING_SESSION_LABEL']==ptp) & 
                    (df_fixation['IMAGE']==image) & 
                    (df_fixation['CONDITION']==task.upper())]
                this_df_fixation = this_df_fixation[
                    this_df_fixation['CURRENT_FIX_START'] <= last_word_end_time]
                # this_df_fixation.loc[this_df_fixation['CURRENT_FIX_START'] > last_word_end_time,
                #                      'CURRENT_FIX_X'] = -1000
                this_df_fixation['LAST_WORD_END_TIME'] = last_word_end_time
                df_fixation_seg = pd.concat([df_fixation_seg, this_df_fixation])
    return df_fixation_seg

def segment_saccade_till_first_word(df_saccade, df_transcript, PTPS):
    df_transcript['ptp'] = df_transcript['ptp'].apply(lambda x: str(x).zfill(3))
    df_saccade_seg = pd.DataFrame()
    for ptp in tqdm(PTPS):
        df_this_ptp = df_transcript[df_transcript['ptp']==ptp]
        images_this_ptp = list(df_this_ptp['image'].unique())
        for image in images_this_ptp:
            for task in ['desc','rec']:
                this_df_transcript = df_transcript[
                    (df_transcript['ptp']==ptp) & 
                    (df_transcript['image']==image) & 
                    (df_transcript['task']==task)]
                first_word_start_time = 0
                row = 0
                while first_word_start_time < 5000:
                    first_word_start_time = this_df_transcript.iloc[row]['start'] * 1000
                    row += 1
                this_df_saccade = df_saccade[
                    (df_saccade['RECORDING_SESSION_LABEL']==ptp) & 
                    (df_saccade['IMAGE']==image) & 
                    (df_saccade['CONDITION']==task.upper())]
                this_df_saccade = this_df_saccade[
                    this_df_saccade['CURRENT_SAC_START_TIME'] <= first_word_start_time]
                this_df_saccade['FIRST_WORD_START_TIME'] = first_word_start_time
                df_saccade_seg = pd.concat([df_saccade_seg, this_df_saccade])
    return df_saccade_seg

def segment_saccade_till_last_word(df_saccade, df_transcript, PTPS):
    df_transcript['ptp'] = df_transcript['ptp'].apply(lambda x: str(x).zfill(3))
    df_saccade_seg = pd.DataFrame()
    for ptp in tqdm(PTPS):
        df_this_ptp = df_transcript[df_transcript['ptp']==ptp]
        images_this_ptp = list(df_this_ptp['image'].unique())
        for image in images_this_ptp:
            for task in ['desc','rec']:
                this_df_transcript = df_transcript[
                    (df_transcript['ptp']==ptp) & 
                    (df_transcript['image']==image) & 
                    (df_transcript['task']==task)]
                last_word_end_time = this_df_transcript.iloc[-1]['end'] * 1000
                this_df_saccade = df_saccade[
                    (df_saccade['RECORDING_SESSION_LABEL']==ptp) & 
                    (df_saccade['IMAGE']==image) & 
                    (df_saccade['CONDITION']==task.upper())]
                this_df_saccade = this_df_saccade[
                    this_df_saccade['CURRENT_SAC_START_TIME'] <= last_word_end_time]
                this_df_saccade['LAST_WORD_END_TIME'] = last_word_end_time
                df_saccade_seg = pd.concat([df_saccade_seg, this_df_saccade])
    return df_saccade_seg

def within_screen(x, y):
    if x >= 0 and x <= 1920 and y >= 0 and y <= 1080:
        return 1
    else:
        return 0

def find_remove_trials(df_fixation):
    df_fixation['within_screen'] = df_fixation.apply(lambda x: within_screen(
        x.CURRENT_FIX_X, x.CURRENT_FIX_Y), axis=1)
    df_fixation_trial = df_fixation.groupby([
        'AGE','RECORDING_SESSION_LABEL','IMAGE','CONDITION','within_screen']).sum()['CURRENT_FIX_DURATION'].reset_index()
    age_out,ptp_out,image_out,task_out,total_duration_out,fix_percent_out,in_screen_percent_out = [],[],[],[],[],[],[]
    for ptp in PTPS:
        for image in IMAGES:
            for task in ['DESC','REC']:
                df_this_trial = df_fixation_trial[
                    (df_fixation_trial['RECORDING_SESSION_LABEL']==ptp) &
                    (df_fixation_trial['IMAGE']==image) &
                    (df_fixation_trial['CONDITION']==task)].reset_index(drop=True)
                if len(df_this_trial)==2:
                    in_screen_duration = df_this_trial[df_this_trial['within_screen']==1]['CURRENT_FIX_DURATION'].reset_index(drop=True)[0]
                    off_screen_duration = df_this_trial[df_this_trial['within_screen']==0]['CURRENT_FIX_DURATION'].reset_index(drop=True)[0]
                    total_duration = in_screen_duration + off_screen_duration
                    fix_percent = total_duration / 90000
                    in_screen_percent = in_screen_duration / total_duration
                elif len(df_this_trial)==1:
                    if df_this_trial['within_screen'][0] == 1:
                        in_screen_duration = df_this_trial['CURRENT_FIX_DURATION'][0]
                        off_screen_duration = 0
                    elif df_this_trial['within_screen'][0] == 0:
                        in_screen_duration = 0
                        off_screen_duration = df_this_trial['CURRENT_FIX_DURATION'][0]
                    total_duration = in_screen_duration + off_screen_duration
                    fix_percent = total_duration / 90000
                    in_screen_percent = in_screen_duration / total_duration
                else:
                    total_duration = -1
                    fix_percent = -1
                    in_screen_percent = -1
                if ptp[0]=='0':
                    age = 'YA'
                elif ptp[0]=='1':
                    age = 'OA'
                age_out.append(age)
                ptp_out.append(ptp)
                image_out.append(image)
                task_out.append(task)
                total_duration_out.append(total_duration)
                fix_percent_out.append(fix_percent)
                in_screen_percent_out.append(in_screen_percent)
    df_to_remove_info = pd.DataFrame({'age':age_out,'ptp':ptp_out,'image':image_out,'task':task_out,
                                'total_duration':total_duration_out,
                                'fix_percent':fix_percent_out,
                                'in_screen_percent':in_screen_percent_out})
    df_to_remove_info = df_to_remove_info[(df_to_remove_info['fix_percent']<0.5)|
                                     (df_to_remove_info['in_screen_percent']<0.5)]
    df_to_remove = df_to_remove_info.groupby(['age','ptp','image']).count().reset_index()[['age','ptp','image']]
    df_to_remove_ptp = df_to_remove.groupby(['age','ptp']).count().reset_index()
    for ptp in PTPS:
        if ptp not in df_to_remove_ptp['ptp'].values:
            if ptp[0]=='0':
                age='YA'
            else:
                age='OA'
            df_to_remove_ptp = df_to_remove_ptp.append({'age':age,'ptp':ptp, 'image':0}, ignore_index=True)
    df_to_remove_ptp = df_to_remove_ptp.sort_values(by='ptp')
    discard_n_mean_YA = np.mean(df_to_remove_ptp[df_to_remove_ptp['age']=='YA']['image'])
    discard_n_std_YA = np.std(df_to_remove_ptp[df_to_remove_ptp['age']=='YA']['image'])
    discard_n_upper_YA = discard_n_mean_YA + 2.5*discard_n_std_YA
    discard_n_mean_OA = np.mean(df_to_remove_ptp[df_to_remove_ptp['age']=='OA']['image'])
    discard_n_std_OA = np.std(df_to_remove_ptp[df_to_remove_ptp['age']=='OA']['image'])
    discard_n_upper_OA = discard_n_mean_OA + 2.5*discard_n_std_OA
    df_to_remove_ptp_YA = df_to_remove_ptp[
        (df_to_remove_ptp['age'] == 'YA') &
        (df_to_remove_ptp['image'] > discard_n_upper_YA)].reset_index(drop=True)
    df_to_remove_ptp_OA = df_to_remove_ptp[
        (df_to_remove_ptp['age'] == 'OA') &
        (df_to_remove_ptp['image'] > discard_n_upper_OA)].reset_index(drop=True)
    print(f'YA remove average = {discard_n_mean_YA}, std = {discard_n_std_YA}')
    print(f'OA remove average = {discard_n_mean_OA}, std = {discard_n_std_OA}')
    print(f'YA largest missing trials = {discard_n_upper_YA}')
    print(f'OA largest missing trials = {discard_n_upper_OA}')
    ptps_to_remove = list(df_to_remove_ptp_YA['ptp']) + list(df_to_remove_ptp_OA['ptp'])
    for ptp in ptps_to_remove:
        df_to_remove_this_ptp = df_to_remove[df_to_remove['ptp'] == ptp]
        if ptp[0]=='0':
            age='YA'
        else:
            age='OA'
        for image in IMAGES:
            if image not in df_to_remove_this_ptp['image'].values:
                df_to_remove = df_to_remove.append({'age':age,'ptp':ptp,'image':image}, ignore_index=True)
    df_to_remove = df_to_remove.sort_values(by=['ptp','image'])
    return df_to_remove

def remove_trials(df_fixation,df_saccade,df_narrative,df_behavioral,df_to_remove):
    # iterrate each row in <df_to_remove>
    for _, row in df_to_remove.iterrows():
        ptp = row['ptp']
        image = row['image']
        df_fixation = df_fixation.drop(
            df_fixation[(df_fixation['RECORDING_SESSION_LABEL'] == ptp) &
                        (df_fixation['IMAGE'] == image)].index)
        df_saccade = df_saccade.drop(
            df_saccade[(df_saccade['RECORDING_SESSION_LABEL'] == ptp) &
                        (df_saccade['IMAGE'] == image)].index)
        df_narrative = df_narrative.drop(
            df_narrative[(df_narrative['ptp'] == ptp) &
                        (df_narrative['image'] == image)].index)
        df_behavioral = df_behavioral.drop(
            df_behavioral[(df_behavioral['ptp'] == ptp) &
                        (df_behavioral['image'] == image)].index)
    return df_fixation.reset_index(drop=True), df_saccade.reset_index(drop=True), df_narrative.reset_index(drop=True), df_behavioral.reset_index(drop=True)
    

def trial_summary(df_fixation_seg, df_saccade_seg):
    df_saccade_seg = df_saccade_seg[df_saccade_seg['CURRENT_SAC_CONTAINS_BLINK']==False]
    df_saccade_seg = df_saccade_seg[df_saccade_seg['CURRENT_SAC_AMPLITUDE']!='.']
    df_saccade_seg['CURRENT_SAC_AMPLITUDE'] = df_saccade_seg['CURRENT_SAC_AMPLITUDE'].astype(float)
    
    df_fixation_seg_trial_mean = df_fixation_seg.groupby(['RECORDING_SESSION_LABEL','CONDITION','AGE','IMAGE']).mean().reset_index()
    df_fixation_seg_trial_sum = df_fixation_seg.groupby(['RECORDING_SESSION_LABEL','CONDITION','AGE','IMAGE']).sum().reset_index()
    df_fixation_seg_trial_max = df_fixation_seg.groupby(['RECORDING_SESSION_LABEL','CONDITION','AGE','IMAGE']).max().reset_index()
    df_saccade_seg_trial_mean = df_saccade_seg.groupby(['RECORDING_SESSION_LABEL','CONDITION','AGE','IMAGE']).mean().reset_index()
    df_saccade_seg_trial_sum = df_saccade_seg.groupby(['RECORDING_SESSION_LABEL','CONDITION','AGE','IMAGE']).sum().reset_index()
    df_saccade_seg_trial_max = df_saccade_seg.groupby(['RECORDING_SESSION_LABEL','CONDITION','AGE','IMAGE']).max().reset_index()

    df_trial = pd.DataFrame()
    df_trial['ptp'] = df_fixation_seg_trial_mean['RECORDING_SESSION_LABEL']
    df_trial['age'] = df_fixation_seg_trial_mean['AGE']
    df_trial['trial'] = df_fixation_seg_trial_mean['TRIAL_INDEX']
    df_trial['image'] = df_fixation_seg_trial_mean['IMAGE']
    df_trial['task'] = df_fixation_seg_trial_mean['CONDITION']
    # df_trial['narrative_duration'] = df_fixation_seg_trial_max['LAST_WORD_END_TIME']
    df_trial['n_fixation'] = df_fixation_seg_trial_max['CURRENT_FIX_INDEX']
    df_trial['avg_fixation_duration'] = df_fixation_seg_trial_mean['CURRENT_FIX_DURATION']
    df_trial['total_fixation_duration'] = df_fixation_seg_trial_sum['CURRENT_FIX_DURATION']
    df_trial['n_saccade'] = df_saccade_seg_trial_max['CURRENT_SAC_INDEX']
    df_trial['avg_saccade_duration'] = df_saccade_seg_trial_mean['CURRENT_SAC_DURATION']
    df_trial['total_saccade_duration'] = df_saccade_seg_trial_sum['CURRENT_SAC_DURATION']
    df_trial['avg_saccade_amplitude'] = df_saccade_seg_trial_mean['CURRENT_SAC_AMPLITUDE']
    # df_trial['total_blink_duration'] = df_trial['narrative_duration'] - \
        # df_trial['total_fixation_duration'] - df_trial['total_saccade_duration']
    df_trial = df_trial.sort_values(['ptp','trial']).reset_index(drop=True)
    return df_trial


def plot_summary(df_fixation_seg, df_saccade_seg):
    df_saccade_seg = df_saccade_seg[df_saccade_seg['CURRENT_SAC_CONTAINS_BLINK']==False]
    sns.displot(data=df_fixation_seg, x='CURRENT_FIX_DURATION', col='CONDITION', kind='hist')
    plt.xlim(0, 2000)
    plt.show()
    sns.displot(data=df_saccade_seg, x='CURRENT_SAC_AMPLITUDE', col='CONDITION', kind='hist')
    sns.displot(data=df_saccade_seg, x='CURRENT_SAC_DURATION', col='CONDITION', kind='hist', binwidth=4)
    plt.xlim(0, 500)
    plt.show()
    sns.relplot(data=df_saccade_seg, x='CURRENT_SAC_DURATION', y='CURRENT_SAC_AMPLITUDE', col='CONDITION', alpha=0.1)
    plt.xlim(0, 500)
    plt.show()
    sns.relplot(data=df_saccade_seg, x='CURRENT_SAC_DURATION', y='CURRENT_SAC_INDEX', col='CONDITION', alpha=0.1)
    plt.xlim(0, 500)
    plt.show()
    sns.relplot(data=df_saccade_seg, x='CURRENT_SAC_DURATION', y='CURRENT_SAC_AVG_VELOCITY', col='CONDITION', alpha=0.1)
    plt.xlim(0, 500)
    plt.show()
    df_trial = trial_summary(df_fixation_seg, df_saccade_seg)
    df_ptp = df_trial.groupby(['age', 'task']).mean()
    return df_trial, df_ptp

def saccade_duration_by_amplitude(df_saccade_seg):
    for amplitude in [2,3.5,5,6.5,8,9.5,11,12.5,1000]:
        df_saccade_seg_amp = df_saccade_seg[df_saccade_seg['CURRENT_SAC_AMPLITUDE'] < amplitude]
        p = sns.displot(data=df_saccade_seg_amp, x='CURRENT_SAC_DURATION', 
                    col='RECORDING_SESSION_LABEL', row='CONDITION', kind='hist', binwidth=4)
        plt.xlim(0, 500)
        plt.ylim(0, 400)
        p.fig.suptitle(f'saccade amplitude < {amplitude}')
        plt.show()
        
def word_count(string):
    tokens = nltk.word_tokenize(string)
    words = [word for word in tokens if word.isalpha()]
    return len(words)
    
    
df_trial = pd.read_excel('output/df_trial_b4exclusion.xlsx')
# note: it is normal that all the training transcripts (training_grocery &
# training_pharmacy does not exist)
df_transcript = narrative_reinstatement.get_all_transcripts(df_trial)
df_transcript = narrative_reinstatement.clean_transcripts(df_transcript)
df_transcript.to_excel('output/df_transcript.xlsx', index=False)
df_narrative = narrative_reinstatement.combine_transcripts(df_transcript)
df_narrative = narrative_reinstatement.add_age(df_narrative)
df_narrative['word_count'] = df_narrative['transcript'].apply(word_count)
df_narrative.to_excel('output/df_narratives.xlsx', index=False)

PTPS = df_trial['ptp'].unique()
PTPS = [str(ptp).zfill(3) for ptp in PTPS]
df_fixation = pd.read_excel('../eyetracking_fixation/fixation_report_master.xlsx')
df_fixation = clean_images(df_fixation, PTPS)
df_saccade = pd.read_excel('../eyetracking_saccade/saccade_report_master.xlsx')
df_saccade = clean_images(df_saccade, PTPS)
df_to_remove = find_remove_trials(df_fixation)
df_behavioral = pd.read_excel('output/df_behavioral.xlsx')
df_behavioral['ptp'] = df_behavioral['ptp'].apply(lambda x: str(x).zfill(3))
df_fixation_out, df_saccade_out, df_narrative_out, df_behavioral_out = remove_trials(df_fixation,df_saccade,df_narrative,df_behavioral,df_to_remove)
df_fixation_out.to_excel('output/df_fixation.xlsx', index=None)
df_saccade_out.to_excel('output/df_saccade.xlsx', index=None)
df_narrative_out.to_excel('output/df_narratives.xlsx', index=None)
df_to_remove.to_excel('output/df_removed_trials.xlsx', index=None)
df_trial, df_ptp = plot_summary(df_fixation_out, df_saccade_out)
df_behavioral_out.to_excel('output/df_behavioral.xlsx', index=None)
df_trial.to_excel('output/df_trial.xlsx', index=None)


df_trial = pd.read_excel('output/df_trial.xlsx')
PTPS = df_trial['ptp'].unique()
PTPS = [str(ptp).zfill(3) for ptp in PTPS]
df_fixation = pd.read_excel('output/df_fixation.xlsx')
df_fixation['RECORDING_SESSION_LABEL'] = df_fixation['RECORDING_SESSION_LABEL'].apply(lambda x: str(x).zfill(3))
df_saccade = pd.read_excel('output/df_saccade.xlsx')
df_transcript = pd.read_excel('output/df_transcript.xlsx')
df_fixation_seg = segment_fixation_till_first_word(df_fixation, df_transcript, PTPS)
df_saccade_seg = segment_saccade_till_first_word(df_saccade, df_transcript, PTPS)

# uncomment if you would like to generate fixation and saccade reports until people
# stop talking. not used in my analyses
# df_fixation_seg.to_excel('output/df_fixation_seg.xlsx', index=None)
# df_saccade_seg.to_excel('output/df_saccade_seg.xlsx', index=None)

g = sns.displot(data=df_fixation_out, x='CURRENT_FIX_X', col='CONDITION',binwidth=20)
g.map(plt.axvline, x=0, color='red')
g.map(plt.axvline, x=1920, color='red')
g.set(xlim=(-400,1920+400))
plt.show()

g = sns.displot(data=df_fixation_out, x='CURRENT_FIX_Y', col='CONDITION',binwidth=20)
g.map(plt.axvline, x=0, color='red')
g.map(plt.axvline, x=1080, color='red')
g.set(xlim=(-400,1080+400))
plt.show()