# Load necessary libraries for the analysis.
# eyesim: core package for eye movement similarity analysis
# patchwork: for combining multiple ggplot plots
# dplyr & tidyverse: for general data manipulation and wrangling
# readxl & writexl: for reading and writing Excel files
# ggplot2: for creating plots
# rstudioapi: for dynamically getting the script's path
library(eyesim)
library(patchwork)
library(dplyr)
library(tidyverse)
library(readxl)
library(writexl)
library(ggplot2)
library(rstudioapi)

#### set the working directory ####
set.seed(2023)
path = dirname(getSourceEditorContext()$path)
setwd(path)

df <- read_excel('output/df_fixation.xlsx')

#### Overall gaze reinstatement ####
# This first section computes a single gaze reinstatement score for each trial.
# The syntax of eyesim is the same for the following windowed analysis, so I only
# included detailed comments here for simplicity's sake.

# Convert the standard dataframe 'df' into an 'eye_table' object, which is required by the eyesim package.
# It maps dataframe columns to eye-tracking variables.

# convert df to eye table format for eyesim
eyetab <- eye_table("CURRENT_FIX_X", "CURRENT_FIX_Y", "CURRENT_FIX_DURATION",
                    "CURRENT_FIX_START",
                    groupvar=c("RECORDING_SESSION_LABEL", "AGE", "CONDITION", "IMAGE"),
                    clip_bounds=c(0, 1920, 1080, 0),
                    data=df)

# Create spatial density maps for the encoding ("DESC") phase.
# These maps serve as the "templates" to which recall gaze patterns will be compared.
# The density is calculated for each unique combination of Image and Participant.
template_subj_enc_dens <- eyetab %>%
  filter(CONDITION == "DESC") %>%
  density_by(groups=c("IMAGE", "RECORDING_SESSION_LABEL"),
             sigma=200, xbounds=c(0, 1920), ybounds=c(0, 1080))

# Create a unique identifier for each subject-image trial to match encoding and recall.
template_subj_enc_dens <- template_subj_enc_dens %>%
  mutate(Subject_Image = interaction(IMAGE, RECORDING_SESSION_LABEL))

# Create spatial density maps for the recall ("REC") phase.
# These are the "test" maps that will be compared against the encoding templates.
ret_dens <- eyetab %>%
  filter(CONDITION == "REC") %>%
  density_by(groups=c("IMAGE", "RECORDING_SESSION_LABEL"),
             sigma=200, xbounds=c(0, 1920), ybounds=c(0, 1080))
# Create the same unique identifier for matching.
ret_dens_subj <- ret_dens %>%
  mutate(Subject_Image = interaction(IMAGE, RECORDING_SESSION_LABEL))

# Calculate the similarity between encoding and recall density maps.
# here, we are moving in order of the recall sessions (<ret_dens_subj>),
# and for each recall session, finding up to 50 description sessions (only 9 in
# this case because we only have 10 images) from the same participant but not the
# original image to do the permutation. The same for all template_similarity
# call following this
simres_subj <- template_similarity(template_subj_enc_dens, ret_dens_subj,
                                   match_on="Subject_Image", method="fisherz",
                                   permute_on='RECORDING_SESSION_LABEL',permutations=50)

# Add an AGE column based on the first character of the participant ID.
simres_subj <- simres_subj %>%
  mutate(AGE_CODE=substr(simres_subj$RECORDING_SESSION_LABEL,1,1)) %>%
  mutate(AGE=case_when(AGE_CODE=='0'~'YA', AGE_CODE=='1'~'OA'))

# Visualize data
simres_subj_YA <- simres_subj%>%filter(AGE=='YA')
simres_subj_OA <- simres_subj%>%filter(AGE=='OA')
par(mfrow=c(1,3))
hist(simres_subj_YA$eye_sim, main="YA raw eye movement similarity")
hist(simres_subj_YA$perm_sim, main="YA image-permuted eye movement similarity")
hist(simres_subj_YA$eye_sim_diff, main="YA corrected eye movement similarity")
t.test(simres_subj_YA$eye_sim_diff)
par(mfrow=c(1,3))
hist(simres_subj_OA$eye_sim, main="OA raw eye movement similarity")
hist(simres_subj_OA$perm_sim, main="OA image-permuted eye movement similarity")
hist(simres_subj_OA$eye_sim_diff, main="OA corrected eye movement similarity")
write_xlsx(simres_subj, "output/gaze_reinstatement.xlsx")


# ################################################################################
# #### time-dependent reinstatement ####
# ################################################################################
# Define window parameters: 10000ms (10s) window size, moving in 5000ms (5s) steps.
window_size <- 10000
step <- 5000
# Calculate the number of windows/steps that fit within the trial time.
row <- (95000 - 5000 - window_size) / step + 1
n_ptp <- length(unique(df$RECORDING_SESSION_LABEL))

# Initialize arrays to store the results.
# _avg arrays will hold the mean similarity across all participants.
gaze_reinstatement_windowed_avg <- array(0, dim=c(row, row))
baseline_reinstatement_windowed_avg <- array(0, dim=c(row, row))
# These 3D arrays will hold the similarity scores for each participant individually.
gaze_reinstatement_windowed <- array(0, dim=c(row, row, n_ptp))
baseline_reinstatement_windowed <- array(0, dim=c(row, row, n_ptp))

# Outer loop: iterates through start times of the encoding window. starts at 5 s
# because the cue to start talking is at 5 s
for (i in seq(5000,95000-window_size, by=step)){
  # Inner loop: iterates through start times of the recall window.
  for (j in seq(5000, 95000-window_size, by=step)){
    print(i)
    print(j)
    # Define the time boundaries for the current encoding and recall windows.
    desc_start_t <- i
    desc_end_t <- i + window_size
    rec_start_t <- j
    rec_end_t <- j + window_size
    # Filter the main dataframe to include only fixations within the current time windows.
    df_windowed <- df %>%
      filter((CONDITION=='DESC' & CURRENT_FIX_START>=desc_start_t & CURRENT_FIX_START<desc_end_t) |
               (CONDITION=='REC' & CURRENT_FIX_START>=rec_start_t & CURRENT_FIX_START<rec_end_t))
    # The following steps are identical to the overall analysis but performed on the windowed data.
    eyetab_windowed <- eye_table("CURRENT_FIX_X", "CURRENT_FIX_Y", "CURRENT_FIX_DURATION",
                                 "CURRENT_FIX_START",
                                 groupvar=c("RECORDING_SESSION_LABEL", "AGE", "CONDITION", "IMAGE"),
                                 clip_bounds=c(0, 1920, 1080, 0),
                                 data=df_windowed)
    template_subj_enc_dens_windowed <- eyetab_windowed %>% filter(CONDITION == "DESC") %>% density_by(groups=c("IMAGE", "RECORDING_SESSION_LABEL"), sigma=200,
                                                                                                      xbounds=c(0, 1920),
                                                                                                      ybounds=c(0, 1080))
    template_subj_enc_dens_windowed <- template_subj_enc_dens_windowed %>% mutate(Subject_Image = interaction(IMAGE, RECORDING_SESSION_LABEL))
    subj_ret_dens_windowed <- eyetab_windowed %>% filter(CONDITION == "REC") %>% density_by(groups=c("IMAGE", "RECORDING_SESSION_LABEL"), sigma=200,
                                                                                            xbounds=c(0, 1920),
                                                                                            ybounds=c(0, 1080))
    subj_ret_dens_windowed <- subj_ret_dens_windowed %>% mutate(Subject_Image = interaction(IMAGE, RECORDING_SESSION_LABEL))
    
    simres_subj_windowed <- template_similarity(template_subj_enc_dens_windowed, subj_ret_dens_windowed, match_on="Subject_Image", method="fisherz", permute_on="RECORDING_SESSION_LABEL", permutations=50)
    
    # Group results by participant and calculate their mean raw and permuted similarity for this time window pair.
    simres_subj_windowed_byptp <- simres_subj_windowed %>%
      group_by(RECORDING_SESSION_LABEL) %>%
      summarize(mean_eye_sim=mean(eye_sim), mean_perm_sim=mean(perm_sim))
     
    # Calculate the group average raw (eye_sim) and baseline (perm_sim) reinstatement for this specific time-window pair.
    raw_reinstatement <- mean(simres_subj_windowed$eye_sim)
    baseline_reinstatement <- mean(simres_subj_windowed$perm_sim)
    # 
    # Calculate the indices (m, n) for storing the results in the output matrices.
    m <- (i - 5000) / step + 1
    n <- (j - 5000) / step + 1
    all_ptp <- data.frame(RECORDING_SESSION_LABEL=c("001","002","003","006","007","008","009","011","012","013","014","015","017","018","019","020","021","022","023","025","026","027","028",
                                  "030","031","032","033","035","036","037","040","041","101","102","104","105","106","107","108","109","110","112","113","114","115","116",
                                  "117","118","119","120","121","123","124","125","126","127","128","129","130","131","132","133","134","135"))
    simres_subj_windowed_byptp <- all_ptp %>%
      left_join(simres_subj_windowed_byptp, by = "RECORDING_SESSION_LABEL")
    baseline_reinstatement_windowed[m,n,] <- simres_subj_windowed_byptp$mean_perm_sim
    gaze_reinstatement_windowed[m,n,] <- simres_subj_windowed_byptp$mean_eye_sim
    baseline_reinstatement_windowed_avg[m,n] <- baseline_reinstatement
    gaze_reinstatement_windowed_avg[m,n] <- raw_reinstatement
  }
}

write_xlsx(data.frame(gaze_reinstatement_windowed), 
           'output/gaze_reinstatement_windowed.xlsx')
write_xlsx(data.frame(baseline_reinstatement_windowed), 
           'output/baseline_reinstatement_windowed.xlsx')


# ==== high resolution diagonal similarity ====
# This section calculates reinstatement only for matching time windows between encoding and recall
# (e.g., 5-15s encoding vs. 5-15s recall), but with a smaller step size for higher temporal resolution.
# Only performed on the diagonal, but otherwise the same idea as above. 
window_size <- 10000
step <- 500
row <- (95000 - 5000 - window_size) / step + 1
n_ptp <- length(unique(df$RECORDING_SESSION_LABEL))
gaze_reinstatement_corrected_windowed <- array(0, dim=c(row, n_ptp))
for (i in seq(5000,95000-window_size, by=step)){
  print(i)
  desc_start_t <- i
  desc_end_t <- i + window_size
  rec_start_t <- i
  rec_end_t <- i + window_size
  df_windowed <- df %>%
    filter((CONDITION=='DESC' & CURRENT_FIX_START>=desc_start_t & CURRENT_FIX_START<desc_end_t) |
             (CONDITION=='REC' & CURRENT_FIX_START>=rec_start_t & CURRENT_FIX_START<rec_end_t))
  eyetab_windowed <- eye_table("CURRENT_FIX_X", "CURRENT_FIX_Y", "CURRENT_FIX_DURATION",
                               "CURRENT_FIX_START",
                               groupvar=c("RECORDING_SESSION_LABEL", "AGE", "CONDITION", "IMAGE"),
                               clip_bounds=c(0, 1920, 1080, 0),
                               data=df_windowed)
  template_subj_enc_dens_windowed <- eyetab_windowed %>% filter(CONDITION == "DESC") %>% density_by(groups=c("IMAGE", "RECORDING_SESSION_LABEL"), sigma=200,
                                                                                                    xbounds=c(0, 1920),
                                                                                                    ybounds=c(0, 1080))
  template_subj_enc_dens_windowed <- template_subj_enc_dens_windowed %>% mutate(Subject_Image = interaction(IMAGE, RECORDING_SESSION_LABEL))
  subj_ret_dens_windowed <- eyetab_windowed %>% filter(CONDITION == "REC") %>% density_by(groups=c("IMAGE", "RECORDING_SESSION_LABEL"), sigma=200,
                                                                                          xbounds=c(0, 1920),
                                                                                          ybounds=c(0, 1080))
  subj_ret_dens_windowed <- subj_ret_dens_windowed %>% mutate(Subject_Image = interaction(IMAGE, RECORDING_SESSION_LABEL))
  
  simres_subj_windowed <- template_similarity(template_subj_enc_dens_windowed, subj_ret_dens_windowed, match_on="Subject_Image", method="fisherz", permute_on="RECORDING_SESSION_LABEL", permutations=50)
  
  simres_subj_windowed_byptp <- simres_subj_windowed %>%
    group_by(RECORDING_SESSION_LABEL) %>%
    summarize(mean_eye_sim_diff=mean(eye_sim_diff))
  
  corrected_reinstatement <- mean(simres_subj_windowed$eye_sim_diff)
  # 
  # calculate the indices to store the reinstatement values
  m <- (i - 5000) / step + 1
  all_ptp <- data.frame(RECORDING_SESSION_LABEL=c("001","002","003","006","007","008","009","011","012","013","014","015","017","018","019","020","021","022","023","025","026","027","028",
                                                  "030","031","032","033","035","036","037","040","041","101","102","104","105","106","107","108","109","110","112","113","114","115","116",
                                                  "117","118","119","120","121","123","124","125","126","127","128","129","130","131","132","133","134","135"))
  simres_subj_windowed_byptp <- all_ptp %>%
    left_join(simres_subj_windowed_byptp, by = "RECORDING_SESSION_LABEL")
  gaze_reinstatement_corrected_windowed[m,] <- simres_subj_windowed_byptp$mean_eye_sim_diff
}

write_xlsx(data.frame(gaze_reinstatement_corrected_windowed), 
           'output/gaze_reinstatement_corrected_windowed_diagonal.xlsx')

