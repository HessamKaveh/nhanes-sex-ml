#!/usr/bin/env Rscript
# Predicting diabetes in NHANES adults with Elastic Net and XGBoost,
# explained with SHAP, and compared between women and men.
#
# Usage: Rscript run_all.R   (run from the project root)

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
})
for (f in sort(list.files("R", full.names = TRUE, pattern = "\\.R$"))) source(f)
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/tables",  recursive = TRUE, showWarnings = FALSE)

# ---- 1. Data ----------------------------------------------------------------
d  <- load_nhanes()
sp <- split_data(d)
pp <- fit_preprocessor(sp$train)
train <- sp$train; test <- sp$test

cohort <- d |>
  group_by(Sex) |>
  summarise(n = n(), diabetes_pct = 100 * mean(diabetes),
            age_mean = mean(Age), bmi_mean = mean(BMI, na.rm = TRUE),
            sbp_mean = mean(BPSysAve, na.rm = TRUE),
            hdl_mean = mean(DirectChol, na.rm = TRUE))
write.csv(cohort, "outputs/tables/cohort_by_sex.csv", row.names = FALSE)
cat("Cohort:\n"); print(cohort)

# ---- 2. Models ----------------------------------------------------------------
# (a) Pooled models: trained on everyone, Sex as a feature.
# (b) Sex-specific models: trained separately on women and on men.
fit_both <- function(tr, te, include_sex) {
  xtr_en <- make_matrix(tr, pp, include_sex, impute_scale = TRUE)
  xte_en <- make_matrix(te, pp, include_sex, impute_scale = TRUE)
  xtr_xg <- make_matrix(tr, pp, include_sex, impute_scale = FALSE)
  xte_xg <- make_matrix(te, pp, include_sex, impute_scale = FALSE)
  en <- fit_enet(xtr_en, tr$diabetes)
  xg <- fit_xgb(xtr_xg, tr$diabetes)
  list(
    en = en, xg = xg, test = te,
    pred = list("Elastic Net" = predict_enet(en, xte_en),
                "XGBoost"     = predict_xgb(xg, xte_xg)),
    shap = list("Elastic Net" = shap_enet(en, xte_en, xtr_en),
                "XGBoost"     = shap_xgb(xg, xte_xg))
  )
}

cat("\nFitting pooled models...\n")
pooled <- fit_both(train, test, include_sex = TRUE)
cat("Fitting sex-specific models...\n")
by_sex <- lapply(c(Women = "Women", Men = "Men"), function(s)
  fit_both(filter(train, Sex == s), filter(test, Sex == s), include_sex = FALSE))

hyper <- bind_rows(
  data.frame(scheme = "Pooled", sex = "All",
             enet_alpha = pooled$en$alpha, enet_lambda = pooled$en$lambda.1se,
             xgb_nrounds = attr(pooled$xg, "best_nrounds")),
  lapply(names(by_sex), function(s) data.frame(
    scheme = "Sex-specific", sex = s,
    enet_alpha = by_sex[[s]]$en$alpha, enet_lambda = by_sex[[s]]$en$lambda.1se,
    xgb_nrounds = attr(by_sex[[s]]$xg, "best_nrounds")))
)
write.csv(hyper, "outputs/tables/hyperparameters.csv", row.names = FALSE)

# ---- 3. Performance by sex --------------------------------------------------
perf <- list()
for (m in names(pooled$pred)) {
  p <- pooled$pred[[m]]
  for (s in c("All", "Women", "Men")) {
    idx <- if (s == "All") TRUE else test$Sex == s
    perf[[length(perf) + 1]] <- cbind(model = m, scheme = "Pooled", sex = s,
                                      evaluate(test$diabetes[idx], p[idx]))
  }
  for (s in names(by_sex)) {
    perf[[length(perf) + 1]] <- cbind(model = m, scheme = "Sex-specific", sex = s,
                                      evaluate(by_sex[[s]]$test$diabetes, by_sex[[s]]$pred[[m]]))
  }
}
perf <- bind_rows(perf)
write.csv(perf, "outputs/tables/performance.csv", row.names = FALSE)
cat("\nTest-set performance:\n"); print(perf, digits = 3)

# Is discrimination different between women and men? (DeLong test, unpaired)
sex_tests <- bind_rows(lapply(names(pooled$pred), function(m) {
  p <- pooled$pred[[m]]
  rw <- roc(test$diabetes[test$Sex == "Women"], p[test$Sex == "Women"], quiet = TRUE, direction = "<")
  rm <- roc(test$diabetes[test$Sex == "Men"],   p[test$Sex == "Men"],   quiet = TRUE, direction = "<")
  t <- roc.test(rw, rm, method = "delong")
  data.frame(model = m, auc_women = as.numeric(auc(rw)), auc_men = as.numeric(auc(rm)),
             z = unname(t$statistic), p_value = t$p.value)
}))
write.csv(sex_tests, "outputs/tables/auc_women_vs_men.csv", row.names = FALSE)
cat("\nAUC women vs men (pooled models):\n"); print(sex_tests, digits = 3)

# Fig 1: AUC with 95% CI
p1 <- perf |>
  filter(sex != "All") |>
  mutate(sex = factor(sex, levels = c("Women", "Men"))) |>
  ggplot(aes(auc, model, colour = sex)) +
  geom_pointrange(aes(xmin = auc_lo, xmax = auc_hi),
                  position = position_dodge(width = 0.5), linewidth = 0.8, size = 0.5) +
  facet_wrap(~scheme) +
  scale_colour_manual(values = sex_cols, name = NULL) +
  labs(title = "Test-set AUC by sex (95% DeLong CI)",
       subtitle = "Pooled = one model for everyone (sex as a feature); Sex-specific = separate models",
       x = "AUC", y = NULL)
save_fig(p1, "fig1_auc_by_sex", 9, 4)

# Fig 2: ROC curves from the pooled models
roc_df <- bind_rows(lapply(names(pooled$pred), function(m) bind_rows(lapply(c("Women", "Men"), function(s) {
  idx <- test$Sex == s
  r <- roc(test$diabetes[idx], pooled$pred[[m]][idx], quiet = TRUE, direction = "<")
  data.frame(model = m, sex = s, fpr = 1 - r$specificities, tpr = r$sensitivities)
}))))
p2 <- roc_df |>
  mutate(sex = factor(sex, levels = c("Women", "Men"))) |>
  arrange(fpr, tpr) |>
  ggplot(aes(fpr, tpr, colour = sex)) +
  geom_abline(linetype = 2, colour = "grey60") +
  geom_step(linewidth = 0.8) +
  facet_wrap(~model) +
  coord_equal() +
  scale_colour_manual(values = sex_cols, name = NULL) +
  labs(title = "ROC curves of the pooled models, split by sex",
       x = "False positive rate", y = "True positive rate")
save_fig(p2, "fig2_roc_by_sex", 8, 4.5)

# Fig 3: calibration by sex (quintiles of predicted risk)
calib <- bind_rows(lapply(names(pooled$pred), function(m)
  data.frame(model = m, sex = test$Sex, y = test$diabetes, p = pooled$pred[[m]]))) |>
  group_by(model, sex) |>
  mutate(bin = ntile(p, 5)) |>
  group_by(model, sex, bin) |>
  summarise(pred = mean(p), obs = mean(y), n = n(), .groups = "drop")
p3 <- ggplot(calib, aes(pred, obs, colour = sex)) +
  geom_abline(linetype = 2, colour = "grey60") +
  geom_line(linewidth = 0.8) + geom_point(size = 2.5) +
  facet_wrap(~model) +
  coord_equal(xlim = c(0, 0.45), ylim = c(0, 0.45)) +
  scale_colour_manual(values = sex_cols, name = NULL) +
  labs(title = "Calibration by sex (pooled models, quintiles of predicted risk)",
       x = "Mean predicted risk", y = "Observed diabetes rate")
save_fig(p3, "fig3_calibration_by_sex", 8, 4.5)

# ---- 3b. Fairness at fixed thresholds -----------------------------------------
# AUC says nothing about how a model behaves once a cut-off is applied. Apply
# the same risk threshold to women and men and compare error rates
# (equal opportunity = equal sensitivity; equalised odds = equal sens. and spec.).
thresholds <- c(0.05, 0.10, 0.20)
fair <- bind_rows(lapply(names(pooled$pred), function(m) bind_rows(lapply(c("Women", "Men"), function(s) {
  idx <- test$Sex == s
  bind_rows(lapply(thresholds, function(t)
    cbind(model = m, sex = s, threshold_metrics(test$diabetes[idx], pooled$pred[[m]][idx], t))))
}))))
write.csv(fair, "outputs/tables/threshold_metrics_by_sex.csv", row.names = FALSE)

# Women - men gap in sensitivity (TPR) and false positive rate, two-sample test
fair_gaps <- bind_rows(lapply(names(pooled$pred), function(m) bind_rows(lapply(thresholds, function(t) {
  pos <- pooled$pred[[m]] >= t
  gap <- function(cls) {
    k <- vapply(c("Women", "Men"), function(s) sum(pos & test$Sex == s & test$diabetes == cls), numeric(1))
    n <- vapply(c("Women", "Men"), function(s) sum(test$Sex == s & test$diabetes == cls), numeric(1))
    c(diff = unname(k[1] / n[1] - k[2] / n[2]), p = suppressWarnings(prop.test(k, n)$p.value))
  }
  tpr <- gap(1); fpr <- gap(0)
  data.frame(model = m, threshold = t,
             tpr_gap_women_minus_men = tpr[["diff"]], tpr_p = tpr[["p"]],
             fpr_gap_women_minus_men = fpr[["diff"]], fpr_p = fpr[["p"]])
}))))
write.csv(fair_gaps, "outputs/tables/equalized_odds_gaps.csv", row.names = FALSE)
cat("\nThreshold metrics by sex (pooled models):\n")
print(fair |> filter(metric %in% c("sensitivity", "specificity")) |>
        select(model, sex, threshold, metric, value) |>
        pivot_wider(names_from = c(metric, sex), values_from = value), digits = 3)
cat("\nEqualised-odds gaps (women - men):\n"); print(fair_gaps, digits = 3)

# Fig 8: sensitivity and specificity across thresholds, by sex
grid <- seq(0.02, 0.5, by = 0.01)
curves <- bind_rows(lapply(names(pooled$pred), function(m) bind_rows(lapply(c("Women", "Men"), function(s) {
  idx <- test$Sex == s
  y <- test$diabetes[idx]; p <- pooled$pred[[m]][idx]
  data.frame(model = m, sex = s, threshold = grid,
             Sensitivity = vapply(grid, function(t) mean(p[y == 1] >= t), numeric(1)),
             Specificity = vapply(grid, function(t) mean(p[y == 0] <  t), numeric(1)))
}))))
p8 <- curves |>
  pivot_longer(c(Sensitivity, Specificity), names_to = "metric") |>
  mutate(sex = factor(sex, levels = c("Women", "Men"))) |>
  ggplot(aes(threshold, value, colour = sex)) +
  geom_vline(xintercept = 0.10, linetype = 2, colour = "grey60") +
  geom_line(linewidth = 0.8) +
  facet_grid(metric ~ model) +
  scale_x_continuous(labels = scales::percent) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  scale_colour_manual(values = sex_cols, name = NULL) +
  labs(title = "Same threshold, different error rates? (pooled models, test set)",
       subtitle = "Dashed line: 10% predicted risk, the threshold reported in the README",
       x = "Risk threshold", y = NULL)
save_fig(p8, "fig8_threshold_metrics_by_sex", 9, 6)

# ---- 4. Elastic Net coefficients: women vs men --------------------------------
coefs <- bind_rows(lapply(names(by_sex), function(s)
  cbind(sex = s, enet_coefs(by_sex[[s]]$en))))
write.csv(coefs, "outputs/tables/enet_coefficients_by_sex.csv", row.names = FALSE)
coef_order <- coefs |> group_by(feature) |> summarise(m = max(abs(coef))) |>
  filter(m > 0) |> arrange(m) |> pull(feature)
p4 <- coefs |>
  filter(feature %in% coef_order) |>
  mutate(feature = factor(feature, levels = coef_order, labels = pretty_terms(coef_order)),
         sex = factor(sex, levels = c("Women", "Men"))) |>
  ggplot(aes(coef, feature, colour = sex)) +
  geom_vline(xintercept = 0, colour = "grey60") +
  geom_point(size = 2.8, position = position_dodge(width = 0.5)) +
  scale_colour_manual(values = sex_cols, name = NULL) +
  labs(title = "Sex-specific Elastic Net coefficients (log-odds)",
       subtitle = "Numeric features are standardised: effect per 1 SD. Zero = dropped by the penalty.",
       x = "Coefficient", y = NULL)
save_fig(p4, "fig4_enet_coefficients_by_sex", 8, 6)

# ---- 5. SHAP: feature importance, women vs men ------------------------------
imp <- bind_rows(lapply(names(by_sex), function(s) bind_rows(lapply(c("Elastic Net", "XGBoost"), function(m)
  cbind(sex = s, model = m, shap_importance(by_sex[[s]]$shap[[m]]$S))))))
write.csv(imp, "outputs/tables/shap_importance_by_sex.csv", row.names = FALSE)
imp_order <- imp |> group_by(feature) |> summarise(m = mean(mean_abs_shap)) |>
  arrange(m) |> pull(feature)
p5 <- imp |>
  mutate(feature = factor(feature, levels = imp_order, labels = feature_labels[imp_order]),
         sex = factor(sex, levels = c("Women", "Men"))) |>
  ggplot(aes(mean_abs_shap, feature, colour = sex)) +
  geom_pointrange(aes(xmin = lo, xmax = hi), position = position_dodge(width = 0.6),
                  size = 0.35, linewidth = 0.7) +
  facet_wrap(~model) +
  scale_colour_manual(values = sex_cols, name = NULL) +
  labs(title = "SHAP feature importance in sex-specific models",
       subtitle = "Mean |SHAP| on the log-odds scale, test set; bars = 95% bootstrap CI",
       x = "Mean |SHAP value|", y = NULL)
save_fig(p5, "fig5_shap_importance_by_sex", 10, 5.5)

# ---- 6. SHAP on the pooled XGBoost model -----------------------------------
sv_xg <- as_shapviz(pooled$shap$XGBoost, test)
sv_en <- as_shapviz(pooled$shap$`Elastic Net`, test)
beeswarm <- function(sv, title) sv_importance(sv, kind = "beeswarm", max_display = 12) +
  ggtitle(title) + theme(legend.position = "right")
p6 <- beeswarm(sv_en, "Elastic Net") + beeswarm(sv_xg, "XGBoost") +
  plot_annotation(title = "SHAP summary, pooled models (test set)",
                  theme = theme(plot.title = element_text(face = "bold")))
save_fig(p6, "fig6_shap_beeswarm_pooled", 13, 5.5)

# Dependence plots coloured by sex: does the same feature value push risk
# differently in women and men?
shap_long <- function(sh, model) {
  vars <- c("Age", "BMI", "BPSysAve", "DirectChol")
  bind_rows(lapply(vars, function(v) data.frame(
    model = model, feature = feature_labels[[v]], value = test[[v]],
    shap = sh$S[, v], sex = test$Sex)))
}
dep <- bind_rows(shap_long(pooled$shap$XGBoost, "XGBoost"),
                 shap_long(pooled$shap$`Elastic Net`, "Elastic Net"))
p7 <- dep |>
  filter(model == "XGBoost", !is.na(value)) |>
  ggplot(aes(value, shap, colour = sex)) +
  geom_hline(yintercept = 0, colour = "grey60") +
  geom_point(alpha = 0.35, size = 1.1) +
  geom_smooth(se = FALSE, method = "loess", formula = y ~ x, linewidth = 1) +
  facet_wrap(~feature, scales = "free_x") +
  scale_colour_manual(values = sex_cols, name = NULL) +
  labs(title = "SHAP dependence by sex (pooled XGBoost)",
       subtitle = "Each point is one test-set participant; lines are LOESS smooths",
       x = "Feature value", y = "SHAP value (log-odds)")
save_fig(p7, "fig7_shap_dependence_by_sex", 9, 6.5)

# How much does the Sex feature itself contribute in the pooled models?
sex_shap <- bind_rows(lapply(names(pooled$shap), function(m) data.frame(
  model = m, sex = test$Sex, shap = pooled$shap[[m]]$S[, "Sex"]))) |>
  group_by(model, sex) |>
  summarise(mean_shap_of_Sex = mean(shap), .groups = "drop")
write.csv(sex_shap, "outputs/tables/sex_feature_shap.csv", row.names = FALSE)
cat("\nContribution of the Sex feature itself (pooled models):\n"); print(sex_shap, digits = 3)

# ---- 7. Sensitivity analyses -------------------------------------------------
source("analyses/missing_data.R")
source("analyses/undiagnosed.R")

writeLines(capture.output(sessionInfo()), "outputs/sessionInfo.txt")
cat("\nDone. Figures in outputs/figures, tables in outputs/tables.\n")
