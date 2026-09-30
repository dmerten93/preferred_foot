###############################################################################
# DOES PLAYER PREFERRED FOOT MATTER?
#
# Purpose:
# Build and evaluate two expected-goals (xG) models:
#   1) Baseline model without preferred-foot information
#   2) Extended model including preferred-foot information
#
# The script covers:
#   - data acquisition from StatsBomb
#   - merge with preferred-foot data
#   - feature engineering
#   - exploratory data analysis
#   - preprocessing and model fitting
#   - repeated cross-validation
#   - hold-out test evaluation
#   - calibration and lift analysis
#   - coefficient-based and SHAP-based interpretation
#
###############################################################################


##############################
# 1) Packages and settings
##############################

# Core data wrangling and plotting
library(tidyverse)

# StatsBomb open-data access and cleaning
library(StatsBombR)

# Helper functions for cleaner variable names
library(janitor)

# Modelling framework, resampling, preprocessing, evaluation
library(tidymodels)

# Permutation variable importance
library(vip)

# Model tidying and coefficient extraction
library(broom)

# Marginal effects and model-based comparisons
library(marginaleffects)

# Marginal prediction plots
library(ggeffects)

# Plot composition
library(patchwork)

# SHAP values for local explanations
library(fastshap)

# Correlation heatmap
library(ggcorrplot)


# Set working directory
setwd("C:/Users/d.merten/Downloads/football")


##############################
# 2) General helper functions
##############################

# House style used across publication-style plots
theme_thesis <- function(base_size = 13) {
  theme_minimal(base_size = base_size) +
    theme(legend.position = "top",
          legend.title = element_text(face = "bold", size = 12),
          legend.text = element_text(face = "bold", size = 11),
          plot.title = element_text(face = "bold"),
          axis.title = element_text(face = "bold", size = 12),
          axis.text = element_text(face = "bold", size = 11),
          panel.grid.major = element_line(color = "grey70", size = 0.4),
          panel.grid.minor = element_blank())
}

# Convenience wrapper for saving plots
save_plot <- function(filename, plot, width = 8, height = 5, dpi = 300) {
  ggsave(filename, plot = plot, width = width, height = height, dpi = dpi)
}

# Generic density plot comparing goals vs non-goals for one numeric feature
density_plot_by_target <- function(data, x_var, x_label, filename) {
  p <- ggplot(data, aes(x = .data[[x_var]], fill = factor(target))) +
    geom_density(alpha = 0.4, color = "black", size = 0.4) +
    scale_fill_manual(name = "Shot outcome",
                      values = c("0" = "#d7191c", "1" = "#1a9641"),
                      labels = c("No goal", "Goal")) +
    labs(x = x_label, y = "Density") +
    theme_thesis()
  
  save_plot(filename, p)
  p
}

# Manual log-loss helper used in cross-validation and test evaluation
log_loss_vec <- function(y_true, y_prob, eps = 1e-15) {
  y_prob <- pmin(pmax(y_prob, eps), 1 - eps)
  y_true_num <- as.integer(as.character(y_true))
  mean(-(y_true_num * log(y_prob) + (1 - y_true_num) * log(1 - y_prob)))
}

# Manual Brier score helper for probability accuracy
brier_vec <- function(y_true, y_prob) {
  y_true_num <- as.integer(as.character(y_true))
  mean((y_true_num - y_prob)^2)
}

# Build calibration table for plotting predicted vs observed probabilities
make_calibration_tbl <- function(df, model_name, n_bins = 10) {
  df %>%
    mutate(y = as.integer(target == "1"), bin = ntile(.pred_1, n_bins)) %>%
    group_by(bin) %>%
    summarise(n = n(), mean_pred = mean(.pred_1), obs_rate = mean(y), .groups = "drop") %>%
    mutate(model = model_name)
}

# Compute Expected Calibration Error (ECE)
compute_ece <- function(df, n_bins = 20) {
  df %>%
    mutate(y = as.integer(target == "1"), bin = ntile(.pred_1, n_bins)) %>%
    group_by(bin) %>%
    summarise(n = n(), p_hat = mean(.pred_1), p_obs = mean(y), .groups = "drop") %>%
    summarise(ece = sum((n / sum(n)) * abs(p_obs - p_hat))) %>%
    pull(ece)
}

# Evaluate threshold-dependent classification metrics on predicted probabilities
compute_threshold_metrics <- function(pred_df, thresholds) {
  map_dfr(thresholds, function(t) {
    tmp <- pred_df %>%
      mutate(.pred_class = factor(if_else(.pred_1 >= t, "1", "0"), levels = c("0", "1")))
    
    tibble(
      threshold = t,
      accuracy = accuracy(tmp, truth = target, estimate = .pred_class)$.estimate,
      sens = sens(tmp, truth = target, estimate = .pred_class, event_level = "second")$.estimate,
      spec = spec(tmp, truth = target, estimate = .pred_class, event_level = "second")$.estimate,
      precision = precision(tmp, truth = target, estimate = .pred_class, event_level = "second")$.estimate
    )
  }) %>%
    mutate(f1 = 2 * (precision * sens) / (precision + sens))
}

# Plot marginal effects / predicted probabilities for numeric terms
plot_numeric_term <- function(model, term) {
  p <- ggpredict(model, terms = term)
  
  ggplot(p, aes(x = x, y = predicted, group = group)) +
    geom_line() +
    geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.2) +
    labs(title = term, x = term, y = "Predicted P(goal)") +
    theme_minimal(base_size = 11) +
    theme(plot.title = element_text(size = 10))
}

# Plot marginal effects / predicted probabilities for binary or categorical terms
plot_categorical_term <- function(model, term) {
  p <- ggpredict(model, terms = term)
  
  x_num <- suppressWarnings(as.numeric(as.character(p$x)))
  is_binary01 <- !anyNA(x_num) && length(unique(x_num)) <= 2
  
  if (is_binary01) {
    ggplot(p, aes(x = factor(x), y = predicted)) +
      geom_point(size = 2.5) +
      geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.15) +
      labs(title = term, x = term, y = "Predicted P(goal)") +
      theme_minimal(base_size = 11) +
      theme(plot.title = element_text(size = 10))
  } else {
    ggplot(p, aes(x = x, y = predicted)) +
      geom_point(size = 2.5) +
      geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.15) +
      labs(title = term, x = term, y = "Predicted P(goal)") +
      theme_minimal(base_size = 11) +
      theme(plot.title = element_text(size = 10),
            axis.text.x = element_text(angle = 30, hjust = 1))
  }
}


##############################
# 3) Data acquisition
##############################

# Define competition and seasons:
# La Liga across six seasons from 2015/16 to 2020/21.
comp_id_val <- 11
season_id_val <- c(90, 42, 4, 1, 2, 27)

# Pull competition metadata and keep only the relevant rows
comps <- FreeCompetitions()
laliga_comp <- comps %>%
  filter(competition_id == comp_id_val, season_id %in% season_id_val)

# Pull match-level metadata
matches <- FreeMatches(laliga_comp)
saveRDS(matches, "laliga_2015_2021_matches.rds")
glimpse(matches)

# Pull all event-level data and apply StatsBombR cleaning
events <- free_allevents(MatchesDF = matches, Parallel = FALSE)
events_clean <- allclean(events)
saveRDS(events_clean, "laliga_2015_2021_events.rds")
glimpse(events_clean)


##############################
# 4) Shot extraction and merge
##############################

# Restrict to open-play shots to ensure comparability of scoring situations
shots <- events_clean %>%
  filter(type.name == "Shot", shot.type.name == "Open Play")

# Load preferred-foot data and keep only players with known dominant foot
preferred_foot <- read_csv("laliga_2015_2021_players_preferred_foot_export_final_altered.csv") %>%
  filter(!is.na(preferred_foot_final))

# Merge player-level preferred-foot information with shot-level event data
shots_preferred_foot <- left_join(shots, preferred_foot, by = c("player.name" = "player_name"))

# Construct preferred-foot indicator:
# "preferred" if the shot foot matches the player's dominant foot,
# "weak" if it does not.
shots_preferred_foot <- shots_preferred_foot %>%
  mutate(preferred_foot_indicator = case_when(
    (preferred_foot_final == "right" & shot.body_part.name == "Right Foot") |
      (preferred_foot_final == "left" & shot.body_part.name == "Left Foot") |
      (preferred_foot_final == "both") ~ "preferred",
    (preferred_foot_final == "right" & shot.body_part.name == "Left Foot") |
      (preferred_foot_final == "left" & shot.body_part.name == "Right Foot") ~ "weak",
    shot.body_part.name == "Head" ~ "head",
    shot.body_part.name == "Other" ~ "other"
  ))

saveRDS(shots_preferred_foot, "laliga_2015_2021_shots_preferred_foot.rds")


##############################
# 5) Filtering and target construction
##############################

# Restrict to footed shots with an unambiguous preferred-foot classification
# This excludes headers, other body parts, and players labelled as "both".
shots_model <- shots_preferred_foot %>%
  filter(preferred_foot_indicator %in% c("preferred", "weak"),
         shot.body_part.name %in% c("Left Foot", "Right Foot"),
         preferred_foot_final %in% c("left", "right"))

# Binary target variable: 1 = goal, 0 = no goal
shots_model <- shots_model %>%
  mutate(target_variable = if_else(shot.outcome.name == "Goal", 1, 0))


##############################
# 6) Modelling dataset
##############################

# Build compact modelling dataset with the final feature set
# Includes geometric, defensive, goalkeeper, and player-specific variables.
df <- shots_model %>%
  transmute(
    target = target_variable,
    dist_shot_to_goal = DistToGoal,
    dist_goal_to_keeper = DistToKeeper,
    angle_deviation = AngleDeviation,
    distance_to_defender_1 = distance.ToD1.360,
    distance_to_defender_2 = distance.ToD2.360,
    density_radius = density,
    density_in_cone = density.incone,
    in_cone_gk = InCone.GK,
    defenders_behind_ball = DefendersBehindBall,
    preferred_foot_indicator = preferred_foot_indicator,
    body_part = shot.body_part.name
  ) %>%
  clean_names()

glimpse(df)
summary(df)

saveRDS(df, "laliga_2015_2021_shots_training_test_data.rds")


##############################
# 7) Exploratory data analysis
##############################

# Quick grouped summary for the key player-specific variables
df %>%
  group_by(preferred_foot_indicator, body_part) %>%
  summarise(shots = n(), goal_rate = mean(target, na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(goal_rate))

# Density plots for core numeric predictors
p_density_dist_shot_to_goal <- density_plot_by_target(
  df, "dist_shot_to_goal", "Distance to goal (meters)", "p_density_dist_shot_to_goal.png")

p_density_dist_goal_to_keeper <- density_plot_by_target(
  df, "dist_goal_to_keeper", "Distance from goal center to goalkeeper (meters)",
  "p_density_dist_goal_to_keeper.png")

p_density_angle_deviation <- density_plot_by_target(
  df, "angle_deviation", "Shot angle deviation (degrees)", "p_density_angle_deviation.png")

p_density_distance_to_defender_1 <- density_plot_by_target(
  df, "distance_to_defender_1", "Distance to closest defender (meters)",
  "p_density_distance_to_defender_1.png")

p_density_distance_to_defender_2 <- density_plot_by_target(
  df, "distance_to_defender_2", "Distance to second-closest defender (meters)",
  "p_density_distance_to_defender_2.png")

p_density_density_radius <- density_plot_by_target(
  df, "density_radius", "Defensive density (radius-based)", "p_density_density_radius.png")

p_density_density_in_cone <- density_plot_by_target(
  df, "density_in_cone", "Defensive density within shooting cone",
  "p_density_density_in_cone.png")

# Grouped summaries for discrete defensive / goalkeeper features
df %>%
  group_by(in_cone_gk) %>%
  summarise(shots = n(), goal_rate = mean(target), .groups = "drop") %>%
  arrange(desc(goal_rate))

df %>%
  group_by(defenders_behind_ball) %>%
  summarise(shots = n(), goal_rate = mean(target), .groups = "drop") %>%
  arrange(desc(goal_rate))

# Correlation analysis:
# used to assess redundancy and motivate dropping weak / overlapping predictors
num_df <- df %>%
  select(where(is.numeric))

cor_mat <- cor(num_df, use = "pairwise.complete.obs")

numerical_predictors_correlation <- ggcorrplot(
  cor_mat, type = "upper", lab = TRUE, lab_size = 3,
  colors = c("#d7191c", "white", "#1a9641"),
  outline.color = "black", ggtheme = theme_minimal()) +
  scale_fill_gradient2(name = "Correlation", low = "#d7191c", mid = "white",
                       high = "#1a9641", midpoint = 0) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),
        axis.text.y = element_text(face = "bold"),
        legend.title = element_text(face = "bold", size = 13, margin = margin(b = 10)),
        legend.text = element_text(face = "bold", size = 11))

save_plot("numerical_predictors_correlation.png", numerical_predictors_correlation)


##############################
# 8) Final modelling dataset
##############################

# Convert target to factor for classification and remove the two defender-distance
# variables that were excluded later based on weak target relationship and overlap.
df2 <- df %>%
  mutate(target = factor(target, levels = c(0, 1))) %>%
  select(-distance_to_defender_1, -distance_to_defender_2)

# Stratified split keeps goal / no-goal ratio similar across train and test
set.seed(1)
split <- initial_split(df2, strata = target)
train <- training(split)
test <- testing(split)


##############################
# 9) Preprocessing recipes
##############################

# Full recipe:
# includes preferred-foot information
rec <- recipe(target ~ ., data = train) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors(), new_level = "(Missing)") %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())

# Baseline recipe:
# removes preferred-foot information to isolate its incremental contribution
rec_no_pref <- recipe(target ~ ., data = train) %>%
  step_rm(preferred_foot_indicator) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors(), new_level = "(Missing)") %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())


##############################
# 10) Model specification
##############################

# Logistic regression is chosen because:
# - it directly estimates probabilities
# - it is standard in xG modelling
# - coefficients remain interpretable
log_reg_spec <- logistic_reg(mode = "classification") %>%
  set_engine("glm")

wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(log_reg_spec)

wf_base <- workflow() %>%
  add_recipe(rec_no_pref) %>%
  add_model(log_reg_spec)


##############################
# 11) Cross-validation
##############################

# Repeated 10-fold CV to obtain stable estimates of model performance
set.seed(123)
folds <- vfold_cv(train, v = 10, repeats = 10, strata = target)

# Fit resamples and keep validation predictions for custom metrics
set.seed(123)
cv_res <- fit_resamples(wf, resamples = folds, control = control_resamples(save_pred = TRUE))
cv_base <- fit_resamples(wf_base, resamples = folds, control = control_resamples(save_pred = TRUE))

# Fold-wise CV metrics for the extended model
cv_metrics_full <- cv_res %>%
  collect_predictions() %>%
  mutate(y = factor(as.integer(target == "1"), levels = c(0, 1)),
         .pred_1 = as.numeric(.pred_1)) %>%
  group_by(id, id2) %>%
  summarise(
    pr_auc = yardstick::pr_auc_vec(y, .pred_1, event_level = "second"),
    roc_auc = yardstick::roc_auc_vec(y, .pred_1, event_level = "second"),
    log_loss = log_loss_vec(y, .pred_1),
    brier = brier_vec(y, .pred_1),
    .groups = "drop"
  )

# Fold-wise CV metrics for the baseline model
cv_metrics_base <- cv_base %>%
  collect_predictions() %>%
  mutate(y = factor(as.integer(target == "1"), levels = c(0, 1)),
         .pred_1 = as.numeric(.pred_1)) %>%
  group_by(id, id2) %>%
  summarise(
    pr_auc = yardstick::pr_auc_vec(y, .pred_1, event_level = "second"),
    roc_auc = yardstick::roc_auc_vec(y, .pred_1, event_level = "second"),
    log_loss = log_loss_vec(y, .pred_1),
    brier = brier_vec(y, .pred_1),
    .groups = "drop"
  )

# Summary table across resamples
cv_summary <- tibble(
  Metric = c("PR AUC", "ROC AUC", "Log Loss", "Brier Score"),
  `Mean Base Model` = c(mean(cv_metrics_base$pr_auc),
                        mean(cv_metrics_base$roc_auc),
                        mean(cv_metrics_base$log_loss),
                        mean(cv_metrics_base$brier)),
  `Standard Error Base Model` = c(sd(cv_metrics_base$pr_auc) / sqrt(nrow(cv_metrics_base)),
                                  sd(cv_metrics_base$roc_auc) / sqrt(nrow(cv_metrics_base)),
                                  sd(cv_metrics_base$log_loss) / sqrt(nrow(cv_metrics_base)),
                                  sd(cv_metrics_base$brier) / sqrt(nrow(cv_metrics_base))),
  `Mean Full Model` = c(mean(cv_metrics_full$pr_auc),
                        mean(cv_metrics_full$roc_auc),
                        mean(cv_metrics_full$log_loss),
                        mean(cv_metrics_full$brier)),
  `Standard Error Full Model` = c(sd(cv_metrics_full$pr_auc) / sqrt(nrow(cv_metrics_full)),
                                  sd(cv_metrics_full$roc_auc) / sqrt(nrow(cv_metrics_full)),
                                  sd(cv_metrics_full$log_loss) / sqrt(nrow(cv_metrics_full)),
                                  sd(cv_metrics_full$brier) / sqrt(nrow(cv_metrics_full)))
)

cv_summary


##############################
# 12) Fit final models on full training data
##############################

# Prepare baked training / test data for the extended model
prep_rec <- prep(rec, training = train, retain = TRUE)
train_baked <- bake(prep_rec, new_data = train)
test_baked <- bake(prep_rec, new_data = test)

# Prepare baked training / test data for the baseline model
prep_rec_base <- prep(rec_no_pref, training = train, retain = TRUE)
train_baked_base <- bake(prep_rec_base, new_data = train)
test_baked_base <- bake(prep_rec_base, new_data = test)

# Fit logistic regression on baked datasets
glm_fit <- glm(target ~ ., data = train_baked, family = binomial())
glm_base <- glm(target ~ ., data = train_baked_base, family = binomial())


##############################
# 13) Hold-out test evaluation
##############################

# Yardstick metrics that directly evaluate probability predictions
metrics_tbl <- metric_set(roc_auc, pr_auc, mn_log_loss)

# Extended model predictions on test set
pred_test <- tibble(
  target = test_baked$target,
  .pred_1 = predict(glm_fit, newdata = test_baked, type = "response")
) %>%
  mutate(target = factor(target, levels = c("0", "1")),
         .pred_1 = as.numeric(.pred_1),
         .pred_0 = 1 - .pred_1)

# Baseline model predictions on test set
pred_test_base <- tibble(
  target = test_baked_base$target,
  .pred_1 = predict(glm_base, newdata = test_baked_base, type = "response")
) %>%
  mutate(target = factor(target, levels = c("0", "1")),
         .pred_1 = as.numeric(.pred_1),
         .pred_0 = 1 - .pred_1)

# Core metrics for extended model
yard_metrics <- metrics_tbl(pred_test, truth = target, .pred_1, event_level = "second")
brier_manual <- mean((as.integer(pred_test$target == "1") - pred_test$.pred_1)^2)
p_base <- mean(pred_test$target == "1")
brier_base <- mean((as.integer(pred_test$target == "1") - p_base)^2)
brier_skill_score <- 1 - (brier_manual / brier_base)

core_metrics <- bind_rows(
  yard_metrics,
  tibble(.metric = "brier", .estimator = "binary", .estimate = brier_manual),
  tibble(.metric = "brier_skill_score", .estimator = "binary", .estimate = brier_skill_score)
)

core_metrics

# Core metrics for baseline model
yard_metrics_base <- metrics_tbl(pred_test_base, truth = target, .pred_1, event_level = "second")
brier_manual_base <- mean((as.integer(pred_test_base$target == "1") - pred_test_base$.pred_1)^2)
p_base_base <- mean(pred_test_base$target == "1")
brier_base_base <- mean((as.integer(pred_test_base$target == "1") - p_base_base)^2)
brier_skill_score_base <- 1 - (brier_manual_base / brier_base_base)

core_metrics_base <- bind_rows(
  yard_metrics_base,
  tibble(.metric = "brier", .estimator = "binary", .estimate = brier_manual_base),
  tibble(.metric = "brier_skill_score", .estimator = "binary", .estimate = brier_skill_score_base)
)

core_metrics_base

# Side-by-side comparison of baseline vs extended model
test_model_comparison <- bind_rows(
  core_metrics_base %>% mutate(model = "no_preferred_foot"),
  core_metrics %>% mutate(model = "with_preferred_foot")
) %>%
  select(model, .metric, .estimate) %>%
  pivot_wider(names_from = model, values_from = .estimate) %>%
  mutate(improvement = with_preferred_foot - no_preferred_foot)

test_model_comparison


##############################
# 14) Calibration analysis
##############################

# Expected Calibration Error (ECE)
ece <- compute_ece(pred_test, n_bins = 20)
ece_base <- compute_ece(pred_test_base, n_bins = 20)

ece
ece_base

# Calibration intercept and slope:
# assess systematic under- / over-confidence in probability estimates
eps <- 1e-15

df_cal <- pred_test %>%
  mutate(y = as.integer(target == "1"),
         logit_p = qlogis(pmin(pmax(.pred_1, eps), 1 - eps)))

cal_fit <- glm(y ~ logit_p, data = df_cal, family = binomial())
coef(cal_fit)

df_cal_base <- pred_test_base %>%
  mutate(y = as.integer(target == "1"),
         logit_p = qlogis(pmin(pmax(.pred_1, eps), 1 - eps)))

cal_fit_base <- glm(y ~ logit_p, data = df_cal_base, family = binomial())
coef(cal_fit_base)

# Calibration plot:
# compares average predicted probabilities with observed goal rates
cal_plot_df <- bind_rows(
  make_calibration_tbl(pred_test_base, "No preferred foot", n_bins = 10),
  make_calibration_tbl(pred_test, "With preferred foot", n_bins = 10)
)

calibration_plot <- ggplot(cal_plot_df, aes(x = mean_pred, y = obs_rate, color = model)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black") +
  annotate("text", x = 0.35, y = 0.38, label = "Perfect calibration", angle = 39,
           size = 4, color = "black", fontface = "bold") +
  geom_line(size = 1.2) +
  geom_point(size = 2.8) +
  labs(x = "Mean predicted probability", y = "Observed goal rate", color = "Model") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "right",
        legend.title = element_text(face = "bold", size = 13),
        legend.text = element_text(face = "bold", size = 11),
        plot.title = element_text(face = "bold"),
        axis.title = element_text(face = "bold", size = 13),
        axis.text = element_text(face = "bold", size = 11),
        panel.grid.major = element_line(color = "grey70", size = 0.4),
        panel.grid.minor = element_blank())

save_plot("calibration_plot.png", calibration_plot)


##############################
# 15) Lift analysis
##############################

# Lift analysis assesses ranking quality:
# do the highest-xG shots truly contain more goals than average?
lift_tbl <- pred_test %>%
  mutate(y = as.integer(target == "1"), decile = ntile(.pred_1, 5)) %>%
  group_by(decile) %>%
  summarise(n = n(),
            mean_xg = mean(.pred_1),
            goal_rate = mean(y),
            lift_vs_base = goal_rate / mean(pred_test$target == "1"),
            .groups = "drop") %>%
  arrange(desc(decile))

lift_tbl

lift_tbl_base <- pred_test_base %>%
  mutate(y = as.integer(target == "1"), decile = ntile(.pred_1, 5)) %>%
  group_by(decile) %>%
  summarise(n = n(),
            mean_xg = mean(.pred_1),
            goal_rate = mean(y),
            lift_vs_base = goal_rate / mean(pred_test_base$target == "1"),
            .groups = "drop") %>%
  arrange(desc(decile))

lift_tbl_base


##############################
# 16) Threshold-based metrics
##############################

# These metrics are less central for xG than probability quality,
# but they help illustrate classification behaviour at plausible thresholds.
thresholds <- c(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35)

thr_metrics <- compute_threshold_metrics(pred_test, thresholds)
thr_metrics_base <- compute_threshold_metrics(pred_test_base, thresholds)

thr_metrics
thr_metrics_base


##############################
# 17) Permutation importance
##############################

# Permutation importance:
# useful to compare how much predictive performance drops when a variable is shuffled
set.seed(10)
vip_plot_auc <- vip::vip(
  object = glm_fit,
  method = "permute",
  target = "target",
  metric = yardstick::roc_auc_vec,
  nsim = 50,
  train = train_baked,
  pred_wrapper = function(object, newdata) predict(object, newdata = newdata, type = "response"),
  smaller_is_better = TRUE,
  event_level = "second",
  num_features = 31
)

vip_plot_auc


##############################
# 18) Coefficients and odds ratios
##############################

# Tidy coefficient table and transform into odds-ratio scale
or_tbl <- broom::tidy(glm_fit, conf.int = TRUE, exponentiate = FALSE) %>%
  mutate(odds_ratio = exp(estimate),
         conf.low_or = exp(conf.low),
         conf.high_or = exp(conf.high)) %>%
  arrange(desc(abs(estimate)))

or_tbl

# Coefficient plot on odds-ratio scale
coef_plot <- or_tbl %>%
  filter(term != "(Intercept)") %>%
  mutate(term = reorder(term, abs(estimate))) %>%
  ggplot(aes(x = odds_ratio, y = term)) +
  geom_point(size = 2.5, color = "#1a9641") +
  geom_errorbarh(aes(xmin = conf.low_or, xmax = conf.high_or), height = 0.2) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  scale_x_log10() +
  labs(x = "Odds ratio (log scale)", y = NULL) +
  theme_minimal(base_size = 13) +
  theme(legend.title = element_text(face = "bold", size = 13),
        legend.text = element_text(face = "bold", size = 11),
        plot.title = element_text(face = "bold"),
        axis.title = element_text(face = "bold", size = 13),
        axis.text = element_text(face = "bold", size = 11),
        panel.grid.major = element_line(color = "grey70", size = 0.4),
        panel.grid.minor = element_blank())

ggsave("coef_plot.png", coef_plot, width = 8, height = 6, dpi = 300)

save_plot("coef_plot.png", coef_plot, width = 8, height = 6)


##############################
# 19) Marginal effects
##############################

# Average marginal effects:
# provide a more intuitive interpretation on the probability scale
avg_slopes(glm_fit) %>%
  arrange(desc(abs(estimate)))

# Average comparisons for factor variables
avg_comparisons(glm_fit)


##############################
# 20) Preferred-foot effect plot
##############################

# Isolate the average effect of shooting with the preferred vs weak foot
pred_pref <- predict(glm_fit,
                     newdata = mutate(test_baked, preferred_foot_indicator_weak = 0),
                     type = "link", se.fit = TRUE)

pred_weak <- predict(glm_fit,
                     newdata = mutate(test_baked, preferred_foot_indicator_weak = 1),
                     type = "link", se.fit = TRUE)

inv_logit <- function(z) 1 / (1 + exp(-z))

pref_plot_df <- tibble(
  x = factor(c("Preferred foot", "Weak foot"), levels = c("Preferred foot", "Weak foot")),
  fit_link = c(mean(pred_pref$fit), mean(pred_weak$fit)),
  se_link = c(mean(pred_pref$se.fit), mean(pred_weak$se.fit))
) %>%
  mutate(predicted = inv_logit(fit_link),
         conf.low = inv_logit(fit_link - 1.96 * se_link),
         conf.high = inv_logit(fit_link + 1.96 * se_link))

pref_effect_plot <- ggplot(pref_plot_df, aes(x = x, y = predicted)) +
  geom_point(size = 3, color = "#1a9641") +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.15) +
  labs(x = "Foot used for shot", y = "Average predicted probability of scoring") +
  theme_minimal(base_size = 13) +
  theme(axis.title = element_text(face = "bold", size = 13),
        axis.text = element_text(face = "bold", size = 11),
        panel.grid.major = element_line(color = "grey70", size = 0.4),
        panel.grid.minor = element_blank())

save_plot("preferred_foot_effect_plot.png", pref_effect_plot, width = 6, height = 4)


##############################
# 21) Marginal prediction plots
##############################

# Inspect model-implied relationships for each fitted term
terms_in_model <- attr(terms(glm_fit), "term.labels")

# Identify numeric vs binary / categorical terms in the baked training data
is_num <- sapply(terms_in_model, function(t) {
  x <- train_baked[[t]]
  is.numeric(x) && length(unique(x[!is.na(x)])) > 2
})

num_terms <- terms_in_model[is_num]
cat_terms <- setdiff(terms_in_model, num_terms)

plots_num <- lapply(num_terms, function(term) plot_numeric_term(glm_fit, term))
plots_cat <- lapply(cat_terms, function(term) plot_categorical_term(glm_fit, term))

grid_num <- if (length(plots_num) > 0) {
  wrap_plots(plots_num, ncol = 3) + plot_annotation(title = "Marginal effects — numeric terms")
} else {
  NULL
}

grid_cat <- if (length(plots_cat) > 0) {
  wrap_plots(plots_cat, ncol = 3) + plot_annotation(title = "Marginal effects — categorical / binary terms")
} else {
  NULL
}

# Print both grids stacked
grid_num / grid_cat


##############################
# 22) SHAP analysis
##############################

# SHAP values provide local explanations:
# how much each feature contributes to individual predictions
set.seed(1)

# Use the baked test set without the target variable
X_test <- test_baked %>%
  select(-target)

# Optional sampling for speed on larger test sets
X_test_s <- X_test %>%
  slice_sample(n = min(1000, nrow(X_test)))

# Prediction wrapper required by fastshap
pred_fun <- function(object, newdata) {
  predict(object, newdata = newdata, type = "response")
}

# Compute SHAP values
sh <- fastshap::explain(
  object = glm_fit,
  X = X_test_s,
  pred_wrapper = pred_fun,
  nsim = 30
)

# Reshape SHAP values to long format
sh_long <- as_tibble(sh) %>%
  mutate(row_id = row_number()) %>%
  pivot_longer(-row_id, names_to = "feature", values_to = "shap")

# Reshape original feature values to long format
x_long <- as_tibble(X_test_s) %>%
  mutate(row_id = row_number()) %>%
  pivot_longer(-row_id, names_to = "feature", values_to = "value")

# Combine SHAP values with underlying feature values
# Feature values are clipped to 5th–95th percentile to stabilise colour scaling.
swarm_df <- sh_long %>%
  left_join(x_long, by = c("row_id", "feature")) %>%
  group_by(feature) %>%
  mutate(lo = quantile(value, 0.05, na.rm = TRUE),
         hi = quantile(value, 0.95, na.rm = TRUE),
         value_clip = pmin(pmax(value, lo), hi),
         value_scaled = ifelse(hi > lo, (value_clip - lo) / (hi - lo), 0.5)) %>%
  ungroup() %>%
  select(-lo, -hi, -value_clip)

# Order features by average absolute SHAP contribution
feat_order <- swarm_df %>%
  group_by(feature) %>%
  summarise(mean_abs = mean(abs(shap), na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(mean_abs)) %>%
  pull(feature)

swarm_df <- swarm_df %>%
  mutate(feature = factor(feature, levels = rev(feat_order)))

# SHAP summary plot
shap_summary_plot <- ggplot(swarm_df, aes(x = shap, y = feature, color = value_scaled)) +
  geom_point(alpha = 0.4, size = 1, position = position_jitter(height = 0.25, width = 0)) +
  scale_color_gradient2(low = "#2c7bb6", mid = "grey95", high = "#d7191c",
                        midpoint = 0.5, name = "Feature\nvalue") +
  labs(x = "SHAP value (impact on prediction)", y = NULL) +
  theme_minimal(base_size = 13) +
  theme(legend.title = element_text(face = "bold", size = 12),
        legend.text = element_text(face = "bold", size = 11),
        axis.title = element_text(face = "bold", size = 13),
        axis.text = element_text(face = "bold", size = 11),
        panel.grid.major = element_line(color = "grey70", size = 0.4),
        panel.grid.minor = element_blank())

save_plot("shap_summary_plot.png", shap_summary_plot, width = 8, height = 5.5)


##############################
# 23) Nested model comparison
##############################

# Likelihood-ratio test:
# formal test of whether preferred-foot information improves model fit
glm_red <- update(glm_fit, . ~ . - preferred_foot_indicator_weak)
anova(glm_red, glm_fit, test = "Chisq")