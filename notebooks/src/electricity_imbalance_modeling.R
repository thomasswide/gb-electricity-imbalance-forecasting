# Great Britain Electricity Imbalance Forecasting
# ------------------------------------------------------------
# Public-facing analysis script for GitHub
#
# Project:
#   Forecast one-hour-ahead Great Britain electricity system imbalance
#   conditions using Elexon settlement and market index data.
#
# Author:
#   Thomas Swide
#
# Notes:
#   - Raw monthly SSD and MD1 CSV files are expected in data/raw/.
#   - Outputs are written to outputs/tables/ and outputs/figures/.
#   - Large raw data files are intentionally not included in the GitHub repository.
# ------------------------------------------------------------

# -----------------------------
# 1. Packages and configuration
# -----------------------------

required_packages <- c(
  "tidyverse",
  "lubridate",
  "slider",
  "caret",
  "pROC",
  "randomForest",
  "nnet",
  "broom",
  "knitr"
)

missing_packages <- required_packages[
  !(required_packages %in% rownames(installed.packages()))
]

if (length(missing_packages) > 0) {
  stop(
    "Please install the following packages before running this script: ",
    paste(missing_packages, collapse = ", ")
  )
}

library(tidyverse)
library(lubridate)
library(slider)
library(caret)
library(pROC)
library(randomForest)
library(nnet)
library(broom)
library(knitr)

set.seed(123)

data_dir <- "data/raw"
processed_dir <- "data/processed"
table_dir <- "outputs/tables"
figure_dir <- "outputs/figures"

dir.create(processed_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

start_date <- as.Date("2024-01-01")
end_date <- as.Date("2026-04-30")
target_horizon_periods <- 2      # two half-hour settlement periods = one hour
threshold_quantile <- 0.25       # removes near-zero/ambiguous imbalance periods


# -----------------------------
# 2. Helper functions
# -----------------------------

read_monthly_csvs <- function(path, pattern) {
  files <- list.files(path = path, pattern = pattern, full.names = TRUE)

  if (length(files) == 0) {
    stop(
      "No files found in ", path, " matching pattern: ", pattern,
      "\nExpected monthly raw files such as SSD_202401.csv and MD1_202401.csv."
    )
  }

  purrr::map_dfr(files, ~ readr::read_csv(.x, show_col_types = FALSE))
}


load_market_data <- function(data_dir, start_date, end_date) {
  message("Reading raw SSD files...")
  ssd_raw <- read_monthly_csvs(data_dir, "^SSD_\\d{6}\\.csv$")

  message("Reading raw MD1 files...")
  md1_raw <- read_monthly_csvs(data_dir, "^MD1_\\d{6}\\.csv$")

  ssd_clean <- ssd_raw %>%
    transmute(
      flow_run_date = ymd(Flow_Run_Date),
      settlement_date = ymd(Settlement_Date),
      settlement_period = Settlement_Period,
      system_buy_price = System_Buy_Price,
      system_sell_price = System_Sell_Price,
      net_imbalance_volume = Net_Imbalance_Volume,
      price_derivation_code = Price_Derivation_Code,
      loss_of_load_probability = Loss_Of_Load_Probability,
      de_rated_margin = `De-rated_Margin`,
      reserve_scarcity_price = Reserve_Scarcity_Price
    )

  md1_clean <- md1_raw %>%
    transmute(
      flow_run_date = ymd(Flow_Run_Date),
      settlement_date = ymd(Settlement_Date),
      settlement_period = Settlement_Period,
      market_index_data_provider_id = Market_Index_Data_Provider_ID,
      market_index_price = Market_Index_Price,
      market_index_volume = Market_Index_Volume
    ) %>%
    filter(market_index_data_provider_id == "APXMIDP")

  # Keep the most recent run for each settlement date/period.
  ssd_latest <- ssd_clean %>%
    group_by(settlement_date, settlement_period) %>%
    slice_max(order_by = flow_run_date, n = 1, with_ties = FALSE) %>%
    ungroup()

  md1_latest <- md1_clean %>%
    group_by(settlement_date, settlement_period) %>%
    slice_max(order_by = flow_run_date, n = 1, with_ties = FALSE) %>%
    ungroup()

  market_df <- ssd_latest %>%
    inner_join(
      md1_latest %>%
        select(
          settlement_date,
          settlement_period,
          market_index_price,
          market_index_volume
        ),
      by = c("settlement_date", "settlement_period")
    ) %>%
    filter(
      settlement_date >= start_date,
      settlement_date <= end_date
    ) %>%
    arrange(settlement_date, settlement_period) %>%
    mutate(
      month = month(settlement_date),
      day_of_week = wday(settlement_date, label = TRUE),
      hour = (settlement_period - 1) %/% 2,
      halfhour = if_else(settlement_period %% 2 == 1, 0, 30)
    )

  message("Merged market data rows: ", nrow(market_df))
  market_df
}


build_modeling_dataset <- function(
    market_df,
    target_horizon_periods = 2,
    threshold_quantile = 0.25
) {
  feature_df <- market_df %>%
    arrange(settlement_date, settlement_period) %>%
    mutate(
      # Lagged system imbalance features
      niv_lag1 = lag(net_imbalance_volume, 1),
      niv_lag2 = lag(net_imbalance_volume, 2),
      niv_lag3 = lag(net_imbalance_volume, 3),
      niv_lag4 = lag(net_imbalance_volume, 4),
      niv_lag5 = lag(net_imbalance_volume, 5),
      niv_lag6 = lag(net_imbalance_volume, 6),

      # Lagged market price and volume features
      market_price_lag1 = lag(market_index_price, 1),
      market_price_lag2 = lag(market_index_price, 2),
      market_price_lag3 = lag(market_index_price, 3),
      market_volume_lag1 = lag(market_index_volume, 1),

      buy_price_lag1 = lag(system_buy_price, 1),
      buy_price_lag2 = lag(system_buy_price, 2),
      sell_price_lag1 = lag(system_sell_price, 1),
      sell_price_lag2 = lag(system_sell_price, 2),

      # Change features
      niv_change_1 = niv_lag1 - niv_lag2,
      niv_change_2 = niv_lag2 - niv_lag3,
      market_price_change_1 = market_price_lag1 - market_price_lag2,

      # Rolling features using lagged values only to avoid look-ahead bias
      niv_rollmean_4 = slide_dbl(
        lag(net_imbalance_volume, 1),
        mean,
        .before = 3,
        .complete = TRUE
      ),
      niv_rollsd_4 = slide_dbl(
        lag(net_imbalance_volume, 1),
        sd,
        .before = 3,
        .complete = TRUE
      ),
      market_price_rollmean_4 = slide_dbl(
        lag(market_index_price, 1),
        mean,
        .before = 3,
        .complete = TRUE
      ),
      market_price_rollsd_4 = slide_dbl(
        lag(market_index_price, 1),
        sd,
        .before = 3,
        .complete = TRUE
      ),

      # Cyclical settlement-period features
      settlement_period_sin = sin(2 * pi * settlement_period / 48),
      settlement_period_cos = cos(2 * pi * settlement_period / 48),

      # One-hour-ahead target value
      niv_tplus_horizon = lead(net_imbalance_volume, target_horizon_periods)
    )

  threshold_value <- quantile(
    abs(feature_df$niv_tplus_horizon),
    threshold_quantile,
    na.rm = TRUE
  )

  modeling_df <- feature_df %>%
    mutate(
      target = case_when(
        niv_tplus_horizon > threshold_value ~ "Short",
        niv_tplus_horizon < -threshold_value ~ "Long",
        TRUE ~ NA_character_
      )
    ) %>%
    filter(!is.na(target)) %>%
    transmute(
      settlement_date,
      settlement_period,
      month,
      day_of_week = factor(day_of_week),
      hour,
      halfhour,
      de_rated_margin,
      settlement_period_sin,
      settlement_period_cos,
      niv_lag1,
      niv_lag2,
      niv_lag3,
      niv_lag4,
      niv_lag5,
      niv_lag6,
      market_price_lag1,
      market_price_lag2,
      market_price_lag3,
      market_volume_lag1,
      buy_price_lag1,
      buy_price_lag2,
      sell_price_lag1,
      sell_price_lag2,
      niv_change_1,
      niv_change_2,
      market_price_change_1,
      niv_rollmean_4,
      niv_rollsd_4,
      market_price_rollmean_4,
      market_price_rollsd_4,
      niv_tplus_horizon,
      target = factor(target, levels = c("Short", "Long"))
    ) %>%
    drop_na() %>%
    arrange(settlement_date, settlement_period)

  list(
    data = modeling_df,
    threshold_value = threshold_value
  )
}


chronological_split <- function(df, train_share = 0.80) {
  n <- nrow(df)
  train_n <- floor(train_share * n)

  list(
    train = df[1:train_n, ],
    test = df[(train_n + 1):n, ]
  )
}


get_best_cv_row <- function(model_object) {
  model_object$results %>%
    filter(ROC == max(ROC)) %>%
    slice(1)
}


save_table <- function(df, filename) {
  readr::write_csv(df, file.path(table_dir, filename))
}


# -----------------------------
# 3. Load and prepare data
# -----------------------------

market_df <- load_market_data(
  data_dir = data_dir,
  start_date = start_date,
  end_date = end_date
)

model_build <- build_modeling_dataset(
  market_df = market_df,
  target_horizon_periods = target_horizon_periods,
  threshold_quantile = threshold_quantile
)

model_df <- model_build$data
threshold_value <- model_build$threshold_value

readr::write_csv(
  model_df,
  file.path(processed_dir, "gb_power_modeling_dataset.csv")
)

message("Final modeling rows: ", nrow(model_df))
message("Threshold value: ", round(as.numeric(threshold_value), 4))

class_balance <- model_df %>%
  count(target) %>%
  mutate(share = round(n / sum(n), 4))

message("Class balance:")
print(class_balance)

save_table(class_balance, "class_balance.csv")


# -----------------------------
# 4. Summary statistics
# -----------------------------

summary_stats <- market_df %>%
  select(
    net_imbalance_volume,
    system_buy_price,
    system_sell_price,
    market_index_price,
    market_index_volume
  ) %>%
  summarise(
    across(
      everything(),
      list(
        mean = ~ mean(.x, na.rm = TRUE),
        sd = ~ sd(.x, na.rm = TRUE),
        min = ~ min(.x, na.rm = TRUE),
        max = ~ max(.x, na.rm = TRUE)
      )
    )
  ) %>%
  pivot_longer(
    cols = everything(),
    names_to = c("variable", ".value"),
    names_sep = "_(?=[^_]+$)"
  ) %>%
  mutate(across(where(is.numeric), ~ round(.x, 3)))

save_table(summary_stats, "summary_statistics.csv")

print(kable(summary_stats, caption = "Summary Statistics for Core Market Variables"))


# -----------------------------
# 5. Train/test split
# -----------------------------

split <- chronological_split(model_df, train_share = 0.80)
train_df <- split$train
test_df <- split$test

message("Training date range:")
print(range(train_df$settlement_date))

message("Test date range:")
print(range(test_df$settlement_date))

full_predictors <- c(
  "month",
  "day_of_week",
  "hour",
  "halfhour",
  "de_rated_margin",
  "settlement_period_sin",
  "settlement_period_cos",
  "niv_lag1",
  "niv_lag2",
  "niv_lag3",
  "niv_lag4",
  "niv_lag5",
  "niv_lag6",
  "market_price_lag1",
  "market_price_lag2",
  "market_price_lag3",
  "market_volume_lag1",
  "buy_price_lag1",
  "buy_price_lag2",
  "sell_price_lag1",
  "sell_price_lag2",
  "niv_change_1",
  "niv_change_2",
  "market_price_change_1",
  "niv_rollmean_4",
  "niv_rollsd_4",
  "market_price_rollmean_4",
  "market_price_rollsd_4"
)

logit_predictors <- c(
  "month",
  "day_of_week",
  "hour",
  "halfhour",
  "de_rated_margin",
  "settlement_period_sin",
  "settlement_period_cos",
  "niv_lag1",
  "niv_lag2",
  "niv_lag3",
  "market_price_lag1",
  "market_price_lag2",
  "market_volume_lag1",
  "buy_price_lag1",
  "sell_price_lag1",
  "niv_change_1",
  "market_price_change_1",
  "niv_rollmean_4",
  "niv_rollsd_4",
  "market_price_rollmean_4",
  "market_price_rollsd_4"
)

train_x <- train_df %>% select(all_of(full_predictors))
test_x <- test_df %>% select(all_of(full_predictors))

train_y <- train_df$target
test_y <- test_df$target


# -----------------------------
# 6. Cross-validated model training
# -----------------------------

cv_folds <- createFolds(train_y, k = 5, returnTrain = TRUE)

cv_control <- trainControl(
  method = "cv",
  number = 5,
  index = cv_folds,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# Logistic regression
set.seed(123)
logit_cv <- train(
  x = train_x %>% select(all_of(logit_predictors)),
  y = train_y,
  method = "glm",
  family = binomial(),
  metric = "ROC",
  trControl = cv_control
)

# Random forest
set.seed(123)
rf_grid <- expand.grid(mtry = c(3, 5, 7, 9))

rf_cv <- train(
  x = train_x,
  y = train_y,
  method = "rf",
  metric = "ROC",
  trControl = cv_control,
  tuneGrid = rf_grid,
  ntree = 400,
  importance = TRUE
)

# Neural network
set.seed(123)
nn_grid <- expand.grid(
  size = c(5, 10, 15),
  decay = c(0, 0.001, 0.01)
)

nn_cv <- train(
  x = train_x,
  y = train_y,
  method = "nnet",
  metric = "ROC",
  trControl = cv_control,
  tuneGrid = nn_grid,
  preProcess = c("center", "scale"),
  trace = FALSE,
  maxit = 300
)


# -----------------------------
# 7. Model comparison
# -----------------------------

logit_best <- get_best_cv_row(logit_cv)
rf_best <- get_best_cv_row(rf_cv)
nn_best <- get_best_cv_row(nn_cv)

cv_comparison <- tibble(
  model = c("Logistic Regression", "Random Forest", "Neural Network"),
  mean_cv_auc = c(logit_best$ROC, rf_best$ROC, nn_best$ROC),
  mean_cv_sensitivity = c(logit_best$Sens, rf_best$Sens, nn_best$Sens),
  mean_cv_specificity = c(logit_best$Spec, rf_best$Spec, nn_best$Spec)
) %>%
  mutate(across(where(is.numeric), ~ round(.x, 4))) %>%
  arrange(desc(mean_cv_auc))

message("Cross-validated model comparison:")
print(cv_comparison)

save_table(cv_comparison, "cv_model_comparison.csv")


# -----------------------------
# 8. Test-set evaluation
# -----------------------------

winner_name <- cv_comparison$model[1]

winner_model <- switch(
  winner_name,
  "Logistic Regression" = logit_cv,
  "Random Forest" = rf_cv,
  "Neural Network" = nn_cv
)

winner_test_x <- if (winner_name == "Logistic Regression") {
  test_x %>% select(all_of(logit_predictors))
} else {
  test_x
}

winner_probs <- predict(
  winner_model,
  newdata = winner_test_x,
  type = "prob"
)[, "Short"]

winner_pred <- predict(
  winner_model,
  newdata = winner_test_x
)

confusion_matrix <- confusionMatrix(
  data = winner_pred,
  reference = test_y,
  positive = "Short"
)

roc_winner <- roc(
  response = test_y,
  predictor = winner_probs,
  levels = c("Long", "Short")
)

test_metrics <- tibble(
  model = winner_name,
  threshold_quantile = threshold_quantile,
  threshold_value = round(as.numeric(threshold_value), 4),
  test_accuracy = round(as.numeric(confusion_matrix$overall["Accuracy"]), 4),
  test_sensitivity = round(as.numeric(confusion_matrix$byClass["Sensitivity"]), 4),
  test_specificity = round(as.numeric(confusion_matrix$byClass["Specificity"]), 4),
  test_balanced_accuracy = round(as.numeric(confusion_matrix$byClass["Balanced Accuracy"]), 4),
  test_auc = round(as.numeric(auc(roc_winner)), 4)
)

message("Winning model:")
print(winner_name)

message("Winning model test-set metrics:")
print(test_metrics)

save_table(test_metrics, "winning_model_test_metrics.csv")


# -----------------------------
# 9. Figures
# -----------------------------

# Figure 1: Distribution of one-hour-ahead net imbalance volume
fig_niv_distribution <- ggplot(model_df, aes(x = niv_tplus_horizon)) +
  geom_histogram(bins = 60) +
  labs(
    title = "Distribution of One-Hour-Ahead Net Imbalance Volume",
    x = "Net imbalance volume at t+2",
    y = "Count"
  ) +
  theme_minimal()

ggsave(
  filename = file.path(figure_dir, "niv_tplus2_distribution.png"),
  plot = fig_niv_distribution,
  width = 8,
  height = 5,
  dpi = 300
)

# Figure 2: Cross-validated model comparison
fig_model_comparison <- ggplot(
  cv_comparison,
  aes(x = reorder(model, mean_cv_auc), y = mean_cv_auc)
) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Cross-Validated Model Comparison",
    x = "",
    y = "Mean CV ROC-AUC"
  ) +
  theme_minimal()

ggsave(
  filename = file.path(figure_dir, "cv_model_comparison.png"),
  plot = fig_model_comparison,
  width = 8,
  height = 5,
  dpi = 300
)

# Figure 3: ROC curve for the winning model
roc_df <- tibble(
  specificity = roc_winner$specificities,
  sensitivity = roc_winner$sensitivities
) %>%
  mutate(false_positive_rate = 1 - specificity)

fig_roc <- ggplot(roc_df, aes(x = false_positive_rate, y = sensitivity)) +
  geom_line() +
  geom_abline(linetype = "dashed") +
  labs(
    title = paste("ROC Curve - Winning Model:", winner_name),
    subtitle = paste("Test AUC =", round(as.numeric(auc(roc_winner)), 4)),
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  theme_minimal()

ggsave(
  filename = file.path(figure_dir, "winning_model_roc_curve.png"),
  plot = fig_roc,
  width = 8,
  height = 5,
  dpi = 300
)

# Figure 4: Random forest variable importance
rf_importance <- randomForest::importance(rf_cv$finalModel) %>%
  as.data.frame() %>%
  rownames_to_column("feature")

if ("MeanDecreaseGini" %in% names(rf_importance)) {
  rf_importance <- rf_importance %>%
    arrange(desc(MeanDecreaseGini)) %>%
    slice_head(n = 15) %>%
    mutate(feature = reorder(feature, MeanDecreaseGini))

  fig_rf_importance <- ggplot(
    rf_importance,
    aes(x = MeanDecreaseGini, y = feature)
  ) +
    geom_col() +
    labs(
      title = "Random Forest Variable Importance",
      x = "Mean decrease in Gini",
      y = ""
    ) +
    theme_minimal()

  ggsave(
    filename = file.path(figure_dir, "random_forest_variable_importance.png"),
    plot = fig_rf_importance,
    width = 8,
    height = 5,
    dpi = 300
  )
}

# Figure 5: Confusion matrix heatmap
confusion_df <- as.data.frame(confusion_matrix$table)

fig_confusion <- ggplot(
  confusion_df,
  aes(x = Reference, y = Prediction, fill = Freq)
) +
  geom_tile() +
  geom_text(aes(label = Freq)) +
  labs(
    title = paste("Confusion Matrix -", winner_name),
    x = "Actual Class",
    y = "Predicted Class"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

ggsave(
  filename = file.path(figure_dir, "winning_model_confusion_matrix.png"),
  plot = fig_confusion,
  width = 6,
  height = 5,
  dpi = 300
)


# -----------------------------
# 10. Console summary
# -----------------------------

message("\nAnalysis complete.")
message("Processed data saved to: ", processed_dir)
message("Tables saved to: ", table_dir)
message("Figures saved to: ", figure_dir)

print(kable(cv_comparison, caption = "Cross-Validated Model Comparison"))
print(kable(test_metrics, caption = "Winning Model Test-Set Performance"))

