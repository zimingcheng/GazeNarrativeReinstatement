**Abstract**

Reinstating encoding-related eye movements benefits memory recall, but the dynamic relationship between eye movements and recall across time remains to be investigated. To address this question, younger and older adults described the contents of photos while their eye movements were recorded during initial viewing and subsequent recall in the absence of visual input. Significant narrative reinstatement of scene descriptions (i.e., repeating details in the original temporal order) occurred throughout the entire period of recall, with younger adults showing higher reinstatement early during recall. Significant gaze reinstatement occurred only during parts of the trial, emerging earlier in younger adults and later in older adults. Narrative and gaze reinstatement formed a reciprocal, dynamic, and iterative feedback loop across time, such that each preceded and followed the other, but aging changed the timing and directionality   of this relationship. 

**Analysis Workflow**

This README provides a comprehensive guide to the analysis scripts, detailing their purpose, dependencies, inputs, and outputs. To ensure proper execution, please run the scripts in the specified order, as many rely on outputs from previous steps. If you encounter a "directory does not exist" error, manually create the indicated folder. All outputed excel sheets are in the <output> folder.

1. <beh_analysis.py>:
This script processes behavioral data from a memory experiment involving young and older adults. It extracts trial-level information and subjective ratings (description difficulty, recall difficulty, recall vividness) for each image.
Outputs:
output/df_trial_b4exclusion.xlsx: Trial-level data.
output/df_behavioral.xlsx: Behavioral ratings.

2. <BIDS_format.py>:
This script adjusts filenames to be consistent with the BIDS (Brain Imaging Data Structure) format. DO NOT RUN if your data is already in BIDS format. It changes the filenames directly without any outputs.

3. Eye-tracking Fixation Report (Manual Step):
This step requires using the SR Research Data Viewer proprietary GUI to generate a fixation report with the X coordinate, Y coordinate, starting time, and ending time of each fixation. Manually convert the output from .xls to xlsx format. You can refer to the data viewer file for exact settings, though a paid subscription is required.
Outputs:
../eyetracking_fixation/fixation_report_master.xlsx

4. <Audio_Transcriber_Addis-PDT-reins_word.py>:
This script transcribes audio, with each word on a new line, including its start and end times.
Important: This script requires your own Google Cloud Computing account and bucket name. Just hitting run will return an error. Alternatively, you can use any other transcription service or transcribe manually.
Outputs:
../transcription/sub-{sub-id}/{filename_base}_auto-transcript.csv

-- start from here if transcription already done --

-- raw data is not included due to participant consent --

5. <eye_tracking_segment.py>:
This script processes and analyzes eye-tracking data in conjunction with narrative transcripts. Its primary goal is to clean and segment eye-tracking data based on speech onset or offset, identify and remove problematic trials, and generate summary statistics and visualizations of eye-movement patterns. It also integrates with a separate narrative reinstatement module for handling transcript data.
Outputs:
output/df_transcript.xlsx: Cleaned and processed word-by-word transcripts.
output/df_narratives.xlsx: Full narrative transcripts for each trial.
output/df_fixation.xlsx: Cleaned eye-tracking fixation data.
output/df_saccade.xlsx: Cleaned eye-tracking saccade data.
output/df_removed_trials.xlsx: List of removed trials.
output/df_behavioral.xlsx: Updated behavioral data.
output/df_trial.xlsx: Summary of eye-tracking metrics per trial.

6. <narrative_reinstatement.py>:
This script performs an in-depth analysis of narrative reinstatement, a measure of how similar a recalled narrative is to its initial description. It investigates both overall (time-independent) and time-dependent reinstatement by comparing the semantic similarity of transcribed narratives, differentiating between younger and older adults.
Outputs:
output/df_transcript.xlsx: Combined and cleaned word-by-word transcripts.
output/df_narratives.xlsx: Full narrative transcripts for each trial.
output/narrative_reinstatement.xlsx: Overall (time-independent) narrative reinstatement results.
output/df_narratives_time_course.xlsx: Time-chunked narratives (coarse-grained).
output/narrative_reinstatement_time_(YA/OA).xlsx: 3D matrices of time-dependent similarity results (reshaped to 2D).
output/df_narratives_time_fine.xlsx: High-resolution time-chunked narratives.
output/narrative_reinstatement_time_diagonal_(YA/OA).xlsx: High-resolution time-dependent similarity diagonal results.

7. <gaze_reinstatement.R>:
This R script analyzes eye-tracking data to measure gaze reinstatement, which quantifies the similarity between eye movements during encoding and recall. The analysis is performed at three levels: overall trial comparison, detailed time-windowed analysis, and high-resolution analysis focusing on matching time windows.
Outputs:
output/gaze_reinstatement.xlsx: Overall, trial-level gaze reinstatement scores.
output/gaze_reinstatement_windowed.xlsx: Matrix of raw gaze similarity scores from the time-windowed analysis.
output/baseline_reinstatement_windowed.xlsx: Matrix of baseline (permuted) similarity scores from the time-windowed analysis.
output/gaze_reinstatement_corrected_windowed_diagonal.xlsx: Corrected (raw minus baseline) gaze reinstatement scores from the high-resolution diagonal analysis.

8. <gaze_narrative.py>:
This script processes raw data from various Excel files, converting them into a structured 3D format. It calculates corrected reinstatement similarities by subtracting baseline activity and merges different data streams for comprehensive analysis. The script generates several plots to illustrate reinstatement patterns over time and across age groups.
Outputs:
output/gaze_diagonal_0.5s_toR.xlsx: Processed gaze reinstatement data for multilevel model analysis in R.
output/narrative_diagonal_0.5s_toR.xlsx: Processed narrative reinstatement data for analysis in R.
output/df_narrative_gaze.xlsx: Combined, whole-trial gaze and narrative reinstatement data.
figures/line_(gaze/narrative): Line plot showing the change of gaze/narrative reinstatement over the trial from both younger and older adults. 
figures/(YA/OA)_(gaze/narrative)_heatmap: Heatmap showing time-matching and non-time-matching gaze/narrative reinstatement from younger/older adults.

9. <time_lag_correlation_prep.py>:
This script processes gaze and narrative reinstatement data, separating it by age group (Younger Adults - YA and Older Adults - OA). It extracts and combines diagonal reinstatement values for gaze and narrative into individual Excel files for each participant.
Outputs:
Individual Excel files (e.g., 0.xlsx, 1.xlsx, etc.) within output/correlation_map_analysis_data/YA/ and output/correlation_map_analysis_data/OA/.

10. <cma/cma_code.m>: (mirrored from https://github.com/avspeech/cma-matlab)
This MATLAB script analyzes the moment-by-moment relationship between a participant's gaze patterns and their narrative reinstatement data. It computes an instantaneous correlation map for each participant, referencing a methodology by Adriano Vilela Barbosa et al. (2012).
Outputs:
Correlation map for each participant, saved in output/correlation_map/YA/ and output/correlation_map/OA/.

11. <cma_plot.py>:
This Python script analyzes and visualizes correlation maps (CMA) derived from gaze and narrative reinstatement data. It calculates and plots the average correlation map for each age group as heatmaps, highlighting regions where correlation values exceed a significance threshold (±0.1).
Outputs:
figures/(YA/OA)_CMA.svg: Averaged correlation map for Younger/Older Adults.

12. <stats.R>:
This comprehensive R script performs a detailed statistical analysis of narrative and gaze reinstatement data, comparing Younger and Older Adults. It conducts t-tests, linear regression models, ANOVAs, post-hoc tests, and linear mixed-effects models to explore various aspects of the data.
Analyses Performed:
One-sample t-tests (reinstatement scores vs. zero).
Two-sample t-tests (age group comparisons).
Linear regression models (correlation between narrative and gaze reinstatement).
Descriptive and inferential analyses on word count, eye-tracking metrics, and behavioral ratings.
Linear mixed-effects models for time-series data.
Outputs:
None. Results displayed as R outputs instead of xlsx. 

This results in figures detailing the change of gaze and narrative reinstatement over the trial. Significant narrative reinstatement of scene descriptions (i.e., repeating details in the original temporal order) occurred throughout the entire period of recall, with younger adults showing higher reinstatement early during recall. 
<img width="3430" height="2540" alt="reinstatement_result_course" src="https://github.com/user-attachments/assets/f39c86e3-5ea3-450c-bdc9-b21b1a1975af" />

Narrative and gaze reinstatement formed a reciprocal, dynamic, and iterative feedback loop across time, such that each preceded and followed the other, but aging changed the timing and directionality of this relationship. 
<img width="3301" height="3137" alt="CMA" src="https://github.com/user-attachments/assets/df5951d6-2543-43a8-8466-448651083541" />


