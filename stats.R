#### input packages ####
# data management packages
library(tidyverse)
library(rstudioapi)
library(readxl)
# stats
library(lme4)
library(lmerTest)
library(emmeans)
library(car)
library(cluscmc)
library(tidyr)
library(dplyr)
library(broom)
# plots
library(ggplot2)


#### set the working directory ####
path = dirname(getSourceEditorContext()$path)
setwd(path)


#### one-sample t-test ####
# read in data
df_narrative_gaze <- read_excel('output/df_narrative_gaze.xlsx')
# calculate average by participant
df_narrative_gaze_ptp <- df_narrative_gaze %>%
  group_by(ptp,age) %>%
  summarise(gaze_reinstatement_mean = mean(gaze_reinstatement), 
            narrative_reinstatement_mean = mean(narrative_reinstatement)) %>%
  ungroup()
df_narrative_gaze_ptp_YA <- df_narrative_gaze_ptp %>%
  filter(age=='YA')
df_narrative_gaze_ptp_OA <- df_narrative_gaze_ptp %>%
  filter(age=='OA')
# one sample t-test
YA_narrative <- t.test(df_narrative_gaze_ptp_YA$narrative_reinstatement_mean)
YA_gaze <- t.test(df_narrative_gaze_ptp_YA$gaze_reinstatement_mean)
OA_narrative <- t.test(df_narrative_gaze_ptp_OA$narrative_reinstatement_mean)
OA_gaze <- t.test(df_narrative_gaze_ptp_OA$gaze_reinstatement_mean)
print('==== one-sample t-test on YA narrative ====')
print(YA_narrative)
print('sd of x')
print(sd(df_narrative_gaze_ptp_YA$narrative_reinstatement_mean))
print('==== one-sample t-test on YA gaze ====')
print(YA_gaze)
print('sd of x')
print(sd(df_narrative_gaze_ptp_YA$gaze_reinstatement_mean))
print('====  one-sample t-test on OA narrative ====')
print(OA_narrative)
print('sd of x')
print(sd(df_narrative_gaze_ptp_OA$narrative_reinstatement_mean))
print('==== one-sample t-test on OA gaze ====')
print(OA_gaze)
print('sd of x')
print(sd(df_narrative_gaze_ptp_OA$gaze_reinstatement_mean))

# ==== two sample t-test ====
YA_OA_narrative <- t.test(df_narrative_gaze_ptp_YA$narrative_reinstatement_mean,
                          df_narrative_gaze_ptp_OA$narrative_reinstatement_mean,
                          var.equal = TRUE)
YA_OA_gaze <- t.test(df_narrative_gaze_ptp_YA$gaze_reinstatement_mean,
                     df_narrative_gaze_ptp_OA$gaze_reinstatement_mean,
                     var.equal = TRUE)
YA_OA_narrative
YA_OA_gaze

# ==== correlation between narrative and gaze ====
# YA
YA_cor <- lm(narrative_reinstatement_mean~gaze_reinstatement_mean, data=df_narrative_gaze_ptp_YA)
print(summary(YA_cor))
# OA
OA_cor <- lm(narrative_reinstatement_mean~gaze_reinstatement_mean, data=df_narrative_gaze_ptp_OA)
print(summary(OA_cor))
# both 
YA_OA_cor <- lm(narrative_reinstatement_mean~gaze_reinstatement_mean*age, data=df_narrative_gaze_ptp)
summary(YA_OA_cor)
# ==== narrative and gaze basic description analyses ====
df_narrative_basic <- read_excel('output/df_narratives.xlsx')
df_fixation_basic <- read_excel('output/df_fixation.xlsx')
df_saccade_basic <- read_excel('output/df_saccade.xlsx') %>%
  filter(CURRENT_SAC_CONTAINS_BLINK==FALSE)
df_fixation_basic_seg <- read_excel('output/df_fixation_seg.xlsx')
df_behavior <- read_excel('output/df_behavioral.xlsx')
# narrative word length
df_narrative_basic_ptp <- df_narrative_basic %>%
  group_by(ptp,age,task) %>%
  summarise(word_count_mean = mean(word_count)) %>% 
  ungroup()
df_narrative_basic_age <- df_narrative_basic_ptp %>%
  group_by(age,task) %>%
  summarise(word_count_mean_mean = mean(word_count_mean),
            word_count_mean_sd = sd(word_count_mean)) %>%
  ungroup()
view(df_narrative_basic_age)
fit <- aov(word_count_mean~age*task, data=df_narrative_basic_ptp)
summary(fit)
TukeyHSD(fit, "age")
TukeyHSD(fit, "task")
ggplot(data=df_narrative_basic_ptp, aes(x=task, y=word_count_mean, colour=age))+
  geom_boxplot()
# fixation metrics
df_fixation_basic_trial <- df_fixation_basic %>%
  group_by(RECORDING_SESSION_LABEL,AGE,CONDITION,TRIAL_INDEX) %>%
  summarise(fixation_number = n(),
            fixation_duration = mean(CURRENT_FIX_DURATION),
            total_fixation_duration = sum(CURRENT_FIX_DURATION / 1000)) %>%
  ungroup()
df_fixation_basic_ptp <- df_fixation_basic_trial %>%
  group_by(RECORDING_SESSION_LABEL,AGE,CONDITION) %>%
  summarise(fixation_number_mean = mean(fixation_number),
            fixation_duration_mean = mean(fixation_duration),
            total_fixation_duration_mean = mean(total_fixation_duration)) %>% 
  ungroup()
df_fixation_basic_age <- df_fixation_basic_ptp %>%
  group_by(AGE,CONDITION) %>%
  summarise(fixation_number_mean_mean = mean(fixation_number_mean),
            fixation_number_mean_sd = sd(fixation_number_mean),
            fixation_duration_mean_mean = mean(fixation_duration_mean),
            fixation_duration_mean_sd = sd(fixation_duration_mean),
            total_fixation_duration_mean_mean = mean(total_fixation_duration_mean),
            total_fixation_duration_mean_sd = sd(total_fixation_duration_mean)) %>%
  ungroup()
view(df_fixation_basic_age)
fit <- aov(fixation_number_mean~AGE*CONDITION, data=df_fixation_basic_ptp)
summary(fit)
TukeyHSD(fit, "AGE")
TukeyHSD(fit, "CONDITION")
ggplot(data=df_fixation_basic_ptp, aes(x=CONDITION, y=fixation_number_mean, colour=AGE))+
  geom_boxplot()
fit <- aov(fixation_duration_mean~AGE*CONDITION, data=df_fixation_basic_ptp)
summary(fit)
TukeyHSD(fit, "AGE")
TukeyHSD(fit, "CONDITION")
ggplot(data=df_fixation_basic_ptp, aes(x=CONDITION, y=fixation_duration_mean, colour=AGE))+
  geom_boxplot()
fit <- aov(total_fixation_duration_mean~AGE*CONDITION, data=df_fixation_basic_ptp)
summary(fit)
TukeyHSD(fit, "AGE")
TukeyHSD(fit, "CONDITION")
ggplot(data=df_fixation_basic_ptp, aes(x=CONDITION, y=total_fixation_duration_mean, colour=AGE))+
  geom_boxplot()
# behavioral results
df_behavior_ptp <- df_behavior %>%
  group_by(ptp,age) %>%
  summarise(description_difficulty_mean = mean(description_difficulty),
            recall_difficulty_mean = mean(recall_difficulty),
            recall_vividness_mean = mean(recall_vividness)) %>% 
  ungroup()
df_behavior_age <- df_behavior_ptp %>%
  group_by(age) %>%
  summarise(description_difficulty_mean_mean = mean(description_difficulty_mean),
            description_difficulty_mean_sd = sd(description_difficulty_mean),
            recall_difficulty_mean_mean = mean(recall_difficulty_mean),
            recall_difficulty_mean_sd = sd(recall_difficulty_mean),
            recall_vividness_mean_mean = mean(recall_vividness_mean),
            recall_vividness_mean_sd = sd(recall_vividness_mean)) %>%
  ungroup()
view(df_behavior_age)
fit <- aov(description_difficulty~image, data=filter(df_behavior, age=='YA'))
summary(fit)
fit <- aov(recall_difficulty~image, data=filter(df_behavior, age=='YA'))
summary(fit)
fit <- aov(recall_vividness~image, data=filter(df_behavior, age=='YA'))
summary(fit)
fit <- aov(description_difficulty~age*image, data=df_behavior)
summary(fit)
TukeyHSD(fit, "age")
fit <- aov(recall_difficulty~age*image, data=df_behavior)
summary(fit)
TukeyHSD(fit, "age")
fit <- aov(recall_vividness~age*image, data=df_behavior)
summary(fit)
TukeyHSD(fit, "age")


#==== diagonal ====
df_diagonal <- read_excel('output/df_diagonal.xlsx')

t.test(df_diagonal$narrative_YA_on,df_diagonal$narrative_YA_off,var.equal = FALSE)
t.test(df_diagonal$narrative_OA_on,df_diagonal$narrative_OA_off,var.equal = FALSE)
t.test(df_diagonal$gaze_YA_on,df_diagonal$gaze_YA_off,paired = TRUE,var.equal = FALSE)
t.test(df_diagonal$gaze_OA_on,df_diagonal$gaze_OA_off,paired = TRUE,var.equal = FALSE)

YA_OA_narrative
YA_OA_gaze

df_narrative <- read_excel('output/narrative_diagonal_5s_toR.xlsx')
df_gaze <- read_excel('output/gaze_diagonal_5s_toR.xlsx')
df_narrative$time <- factor(df_narrative$`time (s)`)
df_narrative$ptp <- factor(df_narrative$ptp)
df_gaze$time <- factor(df_gaze$`time (s)`)
df_gaze$ptp <- factor(df_gaze$`ptp_fake`)

narrative_t_test <- df_narrative %>%
  filter(age=='younger') %>%
  group_by(time) %>%
  summarize(tidy(t.test(sim, mu=0, alternative = "greater"))) %>%
  mutate(p_fdr = p.adjust(p.value, method='fdr'))

narrative_t_test <- df_narrative %>%
  filter(age=='older') %>%
  group_by(time) %>%
  summarize(tidy(t.test(sim, mu=0, alternative = "greater"))) %>%
  mutate(p_fdr = p.adjust(p.value, method='fdr'))

gaze_t_test <- df_gaze %>%
  filter(age=='younger') %>%
  group_by(time) %>%
  summarize(tidy(t.test(sim, mu=0, alternative = "greater"))) %>%
  mutate(p_fdr = p.adjust(p.value, method='fdr'))

gaze_t_test <- df_gaze %>%
  filter(age=='older') %>%
  group_by(time) %>%
  summarize(tidy(t.test(sim, mu=0, alternative = "greater"))) %>%
  mutate(p_fdr = p.adjust(p.value, method='fdr'))

model_narrative <- lmer(sim ~ time * age + (1|ptp), data = df_narrative)
model_gaze <- lmer(sim ~ time * age + (1|ptp), data = df_gaze)

anova(model_narrative)
anova(model_gaze)

marginal_narrative = emmeans(model_narrative, ~ age | time)
marginal_gaze = emmeans(model_gaze, ~ age | time)
pairs_narrative <- pairs(marginal_narrative, adjust="none")
pairs_gaze <- pairs(marginal_gaze, adjust="none")
summary(pairs_narrative, infer = TRUE, by = NULL, adjust = "fdr")
summary(pairs_gaze, infer = TRUE, by = NULL, adjust = "fdr")
