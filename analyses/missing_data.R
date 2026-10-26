# Missing-data sensitivity analysis.
# Sourced from run_all.R; uses `d`, `train`, `test` and the helpers in R/.
#
# Scenarios (pooled models, sex as a feature):
#   Baseline        median/mode imputation for Elastic Net, native NA handling for XGBoost
#   Median/mode     single imputation for both models
#   Median + flags  single imputation plus a 0/1 "was missing" column per variable
#   Complete cases  drop every participant with any missing predictor (train and test)
#   MICE (m = 5)    multiple imputation by chained equations; a model is fitted to
#                   each imputed training set and test predictions are averaged

suppressPackageStartupMessages(library(mice))

cat("\n==== Missing-data sensitivity analysis ====\n")
pred_vars <- c(numeric_vars, factor_vars)

# ---- Missingness by sex, and whether it is related to the outcome ----------
miss_tab <- bind_rows(lapply(pred_vars, function(v) {
  m <- is.na(d[[v]])
  data.frame(
    variable = v,
    pct_missing_women = 100 * mean(m[d$Sex == "Women"]),
    pct_missing_men   = 100 * mean(m[d$Sex == "Men"]),
    diabetes_pct_if_missing  = if (any(m)) 100 * mean(d$diabetes[m]) else NA,
    diabetes_pct_if_observed = 100 * mean(d$diabetes[!m])
  )
})) |> filter(pct_missing_women + pct_missing_men > 0) |>
  arrange(desc(pct_missing_women + pct_missing_men))
write.csv(miss_tab, "outputs/tables/missingness_by_sex.csv", row.names = FALSE)
cat("Missingness (%):\n"); print(miss_tab, digits = 3)
cat("Participants with >= 1 missing predictor:",
    round(100 * mean(!complete.cases(d[pred_vars])), 1), "%\n")

# ---- Helpers -----------------------------------------------------------------
impute_simple <- function(x, pp) {
  for (v in numeric_vars) x[[v]][is.na(x[[v]])] <- pp$medians[[v]]
  for (v in factor_vars)  x[[v]][is.na(x[[v]])] <- pp$modes[[v]]
  x
}

# Missingness indicators for the variables that have missing values in training
flag_vars <- numeric_vars[vapply(numeric_vars, function(v) anyNA(train[[v]]), logical(1))]
add_flags <- function(m, x) {
  f <- sapply(flag_vars, function(v) as.numeric(is.na(x[[v]])))
  colnames(f) <- paste0("miss_", flag_vars)
  cbind(m, f)
}

# Fit both models on `tr`, return test-set predictions for `te`.
# `xform(x, pp, scaled)` turns a data frame into a design matrix.
fit_predict <- function(tr, te, xform) {
  pp_s <- fit_preprocessor(tr)
  en <- fit_enet(xform(tr, pp_s, TRUE), tr$diabetes)
  xg <- fit_xgb(xform(tr, pp_s, FALSE), tr$diabetes)
  list("Elastic Net" = predict_enet(en, xform(te, pp_s, TRUE)),
       "XGBoost"     = predict_xgb(xg, xform(te, pp_s, FALSE)))
}

# AUC plus sensitivity/specificity at 10% risk, overall and by sex
summarise_scenario <- function(scenario, te, preds) {
  bind_rows(lapply(names(preds), function(m) bind_rows(lapply(c("All", "Women", "Men"), function(s) {
    idx <- if (s == "All") rep(TRUE, nrow(te)) else te$Sex == s
    y <- te$diabetes[idx]; p <- preds[[m]][idx]
    tm <- threshold_metrics(y, p, 0.10)
    cbind(scenario = scenario, model = m, sex = s, evaluate(y, p),
          sens_10 = tm$value[tm$metric == "sensitivity"],
          spec_10 = tm$value[tm$metric == "specificity"])
  }))))
}

# ---- Scenarios ---------------------------------------------------------------
mi_res <- list()

# Baseline = the main analysis
mi_res$baseline <- summarise_scenario("Baseline", test, pooled$pred)

cat("Median/mode...\n")
p_med <- fit_predict(train, test, function(x, pp, scaled)
  make_matrix(impute_simple(x, pp), pp, impute_scale = scaled))
mi_res$median <- summarise_scenario("Median/mode", test, p_med)

cat("Median + missingness flags...\n")
p_flag <- fit_predict(train, test, function(x, pp, scaled)
  add_flags(make_matrix(impute_simple(x, pp), pp, impute_scale = scaled), x))
mi_res$flags <- summarise_scenario("Median + flags", test, p_flag)

cat("Complete cases...\n")
cc_tr <- train[complete.cases(train[pred_vars]), ]
cc_te <- test[complete.cases(test[pred_vars]), ]
p_cc <- fit_predict(cc_tr, cc_te, function(x, pp, scaled) make_matrix(x, pp, impute_scale = scaled))
mi_res$cc <- summarise_scenario("Complete cases", cc_te, p_cc)
# Baseline predictions on the same complete-case test subset, for a like-for-like comparison
cc_idx <- complete.cases(test[pred_vars])
mi_res$baseline_cc <- summarise_scenario("Baseline (complete-case test subset)", cc_te,
                                         lapply(pooled$pred, `[`, cc_idx))

cat("MICE (m = 5)...\n")
# The imputation model is estimated on training rows only (`ignore` = test rows),
# and the outcome is left out so test labels never inform test imputations.
all_x   <- bind_rows(train, test)[c(pred_vars, "Sex")]
is_test <- c(rep(FALSE, nrow(train)), rep(TRUE, nrow(test)))
imp <- mice(all_x, m = 5, maxit = 10, ignore = is_test, seed = 2026, printFlag = FALSE)
p_mi <- lapply(seq_len(imp$m), function(k) {
  comp <- complete(imp, k)
  tr_k <- train; tr_k[pred_vars] <- comp[!is_test, pred_vars]
  te_k <- test;  te_k[pred_vars] <- comp[is_test, pred_vars]
  fit_predict(tr_k, te_k, function(x, pp, scaled) make_matrix(x, pp, impute_scale = scaled))
})
# Average predicted probabilities across the imputations
p_mi <- lapply(c("Elastic Net" = "Elastic Net", "XGBoost" = "XGBoost"), function(m)
  rowMeans(sapply(p_mi, `[[`, m)))
mi_res$mice <- summarise_scenario("MICE (m = 5)", test, p_mi)

mi_res <- bind_rows(mi_res)
write.csv(mi_res, "outputs/tables/missing_data_sensitivity.csv", row.names = FALSE)
cat("\nMissing-data scenarios:\n")
print(mi_res |> select(scenario, model, sex, n, auc, auc_lo, auc_hi, sens_10, spec_10), digits = 3)

# Fig 9: AUC by scenario and sex
scen_levels <- c("Baseline", "Median/mode", "Median + flags", "MICE (m = 5)",
                 "Complete cases", "Baseline (complete-case test subset)")
p9 <- mi_res |>
  filter(sex != "All") |>
  mutate(scenario = factor(scenario, levels = rev(scen_levels)),
         sex = factor(sex, levels = c("Women", "Men"))) |>
  ggplot(aes(auc, scenario, colour = sex)) +
  geom_pointrange(aes(xmin = auc_lo, xmax = auc_hi),
                  position = position_dodge(width = 0.5), size = 0.4, linewidth = 0.7) +
  facet_wrap(~model) +
  scale_colour_manual(values = sex_cols, name = NULL) +
  labs(title = "Missing-data sensitivity: test-set AUC by sex",
       subtitle = "Pooled models; 95% DeLong CI. Complete-case rows use a smaller test set.",
       x = "AUC", y = NULL)
save_fig(p9, "fig9_missing_data_sensitivity", 10, 5)
