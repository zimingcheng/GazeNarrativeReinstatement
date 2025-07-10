# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 09:39:39 2023

@author: czm19
"""

# Import necessary libraries for data handling, visualization, and statistical analysis
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt

# Set constants for participant IDs, image names, and task types
PTPS = ['001','002','003','006','007','008','009','011','012','013','014','015',
        '017','018','019','020','021','022','023','024','025','026','027',
        '028','030','031','032','033','034','035','036','037','040','041',
        '101','102','104','105','106','107','108','109','110','111','112',
        '113','114','115','116','117','118','119','120','121','123','124','125',
        '126','127','128','129','130','131','132','133','134','135']
IMAGES = ['boat','camping','circus','construction','crocodile','dancing',
          'farming','golf','icecream','museum']
TASKS = ['DESC','REC']

# ===================== TRIAL DATA EXTRACTION =====================

# Create an empty DataFrame to store trial-level data
df_trial = pd.DataFrame()

# Loop through each participant
for PTP in PTPS:
    # Initialize lists to collect data for each participant
    ages, ptps, trials, images, tasks = [],[],[],[],[]
    
    # Load participant data from expected file path
    if os.path.exists(f"../../raw_data/{PTP}/RESULTS_FILE.txt"):
        df_ptp = pd.read_csv(f"../../raw_data/{PTP}/RESULTS_FILE.txt", sep="\t")
    else:
        df_ptp = pd.read_csv(f"../../raw_data/sub-{PTP}/sub-{PTP}_experimental_raw.txt", sep="\t")
    
    # Extract relevant trial information
    for index, row in df_ptp.iterrows():
        ptps.append(PTP)
        
        # Assign age group based on participant ID prefix
        if PTP[0] == '0':
            age = 'YA'  # Young Adults
        elif PTP[0] == '1':
            age = 'OA'  # Older Adults
        ages.append(age)
        
        trials.append(row['Trial_Index_'])  # Trial number
        task = row['CONDITION']
        tasks.append(task)
        
        # Get corresponding image name based on the task type
        if task == 'DESC_training':
            images.append(row['imagename_desc_training'])
        elif task == 'REC_training':
            images.append(row['imagename_rec_training'])
        elif task == 'DESC':
            images.append(row['imagename_desc'])
        elif task == 'REC':
            images.append(row['imagename_rec'])
    
    # Create a DataFrame for this participant’s trials and append to master DataFrame
    df_ptp_trial = pd.DataFrame({'age':ages,'ptp':ptps,'trial':trials,
                                 'image':images,'task':tasks})
    df_trial = pd.concat([df_trial, df_ptp_trial], ignore_index=True)

# Save the combined trial data before exclusion
df_trial.to_excel('output/df_trial_b4exclusion.xlsx', index=None)

# ===================== BEHAVIORAL RATINGS EXTRACTION =====================

# Create an empty DataFrame to store behavioral data
df_beh = pd.DataFrame()

# Loop through each participant
for PTP in PTPS:
    # Initialize lists to collect behavioral ratings
    ages, ptps, images, diff_descs, diff_recs, vivid_recs = [],[],[],[],[],[]
    
    # Load participant data from expected file path
    if os.path.exists(f"../../raw_data/{PTP}/RESULTS_FILE.txt"):
        df_ptp = pd.read_csv(f"../../raw_data/{PTP}/RESULTS_FILE.txt", sep="\t")
    else:
        df_ptp = pd.read_csv(f"../../raw_data/sub-{PTP}/sub-{PTP}_experimental_raw.txt", sep="\t")
    
    # Keep only relevant conditions for behavioral analysis
    df_ptp = df_ptp[df_ptp['CONDITION'].isin(['DESC','REC'])]
    
    # Loop through each image to collect its ratings
    for IMAGE in IMAGES:
        ptps.append(PTP)
        
        # Assign age group
        if PTP[0] == '0':
            age = 'YA'
        elif PTP[0] == '1':
            age = 'OA'
        ages.append(age)
        images.append(IMAGE)
        
        # Extract difficulty and vividness ratings for the image in each task
        diff_descs.append(int(df_ptp[(df_ptp['imagename_desc']==IMAGE)&
                                 (df_ptp['CONDITION']=='DESC')].\
                          reset_index()['DIFF_DESC'][0]))
        diff_recs.append(int(df_ptp[(df_ptp['imagename_rec']==IMAGE)&
                                 (df_ptp['CONDITION']=='REC')].\
                          reset_index()['DIFF_REC'][0]))
        vivid_recs.append(int(df_ptp[(df_ptp['imagename_rec']==IMAGE)&
                                 (df_ptp['CONDITION']=='REC')].\
                          reset_index()['VIVIDNESS_REC'][0]))
    
    # Create a DataFrame for this participant’s behavioral data and append
    df_ptp_beh = pd.DataFrame({'age':ages,'ptp':ptps,'image':images,'description_difficulty':diff_descs,
                              'recall_difficulty':diff_recs,'recall_vividness':vivid_recs})
    df_beh = pd.concat([df_beh, df_ptp_beh], ignore_index=True)

# ===================== DATA VISUALIZATION =====================

# Plot description difficulty per image by age group
sns.barplot(data=df_beh, x='image',y='description_difficulty',hue='age')
plt.xticks(rotation=45)
plt.show()

# Plot recall difficulty per image by age group
sns.barplot(data=df_beh, x='image',y='recall_difficulty',hue='age')
plt.xticks(rotation=45)
plt.show()

# Plot recall vividness per image by age group
sns.barplot(data=df_beh, x='image',y='recall_vividness',hue='age')
plt.xticks(rotation=45)
plt.show()

# ===================== SUMMARY AND EXPORT =====================

# Compute mean behavioral ratings by age group
df_beh_summary = df_beh.groupby('age').mean()

# Save the full behavioral dataset
df_beh.to_excel('output/df_behavioral.xlsx', index=None)
