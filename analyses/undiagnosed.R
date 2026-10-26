# Undiagnosed diabetes: does training on *diagnosed* diabetes carry the
# diagnosis gap into the model?
# Sourced from run_all.R; uses `d` and the helpers in R/.
#
# The NHANES package has no lab glucose, but its ID is the original NHANES
# SEQN, so HbA1c and fasting plasma glucose from the CDC files can be merged in.
#   Lab-defined diabetes:  HbA1c >= 6.5% or fasting glucose >= 126 mg/dL (ADA)
#   Undiagnosed:           lab-defined diabetes but no reported diagnosis
#   Total diabetes:        diagnosed or undiagnosed

suppressPackageStartupMessages(library(haven))

cat("\n==== Undiagnosed diabetes ====\n")

# ---- Lab data from CDC (cached in data/cdc, not committed) -------------------
cdc_files <- c(GHB_F = "2009", GHB_G = "2011", GLU_F = "2009", GLU_G = "2011")
dir.create("data/cdc", recursive = TRUE, showWarnings = FALSE)
read_cdc <- function(name) {
  path <- file.path("data/cdc", paste0(name, ".xpt"))
  if (!file.exists(path)) {
    url <- sprintf("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/%s/DataFiles/%s.xpt",
                   cdc_files[[name]], name)
    download.file(url, path, mode = "wb", quiet = TRUE)
  }
  read_xpt(path)
}
hba1c <- bind_rows(read_cdc("GHB_F"), read_cdc("GHB_G")) |> select(ID = SEQN, hba1c = LBXGH)
fpg   <- bind_rows(read_cdc("GLU_F"), read_cdc("GLU_G")) |> select(ID = SEQN, fpg = LBXGLU)

lab <- d |>
  left_join(hba1c, by = "ID") |>
  left_join(fpg, by = "ID") |>
  mutate(lab_dm = (hba1c >= 6.5) %in% TRUE | (fpg >= 126) %in% TRUE,
         lab_known = !is.na(hba1c) | !is.na(fpg),
         undiagnosed = as.integer(diabetes == 0 & lab_dm),
         total_dm = as.integer(diabetes == 1 | lab_dm)) |>
  # Without a diagnosis and without any lab value, total status is unknown
  filter(diabetes == 1 | lab_known)
cat("Participants with lab data or a diagnosis:", nrow(lab), "of", nrow(d), "\n")

# ---- How much diabetes is undiagnosed, by sex? -------------------------------
ci_pct <- function(x, n) 100 * c(x / n, binom.test(x, n)$conf.int)
undx_tab <- bind_rows(lapply(c("Women", "Men"), function(s) {
  x <- filter(lab, Sex == s)
  share <- ci_pct(sum(x$undiagnosed), sum(x$total_dm))
  data.frame(sex = s, n = nrow(x),
             diagnosed_pct = 100 * mean(x$diabetes),
             undiagnosed_pct = 100 * mean(x$undiagnosed),
             total_pct = 100 * mean(x$total_dm),
             n_total_dm = sum(x$total_dm), n_undiagnosed = sum(x$undiagnosed),
             share_undiagnosed_pct = share[1], share_lo = share[2], share_hi = share[3])
}))
dm <- filter(lab, total_dm == 1)
undx_test <- chisq.test(table(dm$Sex, dm$undiagnosed))
undx_tab$share_p_value <- undx_test$p.value
write.csv(undx_tab, "outputs/tables/undiagnosed_by_sex.csv", row.names = FALSE)
cat("Undiagnosed diabetes by sex:\n"); print(undx_tab, digits = 3)

# ---- Out-of-fold predictions for two labels ----------------------------------
# 5-fold CV over the whole cohort, so every undiagnosed case gets a prediction
# from a model that never saw it.
oof_predict <- function(dat, label, k = 5, seed = 2026) {
  set.seed(seed)
  strata <- interaction(dat$Sex, dat[[label]])
  fold <- integer(nrow(dat))
  for (s in levels(strata)) {
    i <- which(strata == s)
    fold[i] <- sample(rep_len(seq_len(k), length(i)))
  }
  out <- list("Elastic Net" = numeric(nrow(dat)), "XGBoost" = numeric(nrow(dat)))
  for (f in seq_len(k)) {
    tr <- dat[fold != f, ]; te <- dat[fold == f, ]
    pp_f <- fit_preprocessor(tr)
    en <- fit_enet(make_matrix(tr, pp_f, impute_scale = TRUE), tr[[label]])
    xg <- fit_xgb(make_matrix(tr, pp_f, impute_scale = FALSE), tr[[label]])
    out[["Elastic Net"]][fold == f] <- predict_enet(en, make_matrix(te, pp_f, impute_scale = TRUE))
    out[["XGBoost"]][fold == f]     <- predict_xgb(xg, make_matrix(te, pp_f, impute_scale = FALSE))
  }
  out
}
cat("Cross-fitting models on the diagnosed label...\n")
oof_dx <- oof_predict(lab, "diabetes")
cat("Cross-fitting models on the total-diabetes label...\n")
oof_tot <- oof_predict(lab, "total_dm")

# ---- Evaluation --------------------------------------------------------------
auc_ci <- function(y, p) {
  ci <- as.numeric(ci.auc(roc(y, p, quiet = TRUE, direction = "<"), method = "delong"))
  c(auc = ci[2], lo = ci[1], hi = ci[3])
}
label_res <- bind_rows(lapply(names(oof_dx), function(m) bind_rows(lapply(c("Women", "Men"), function(s) {
  w <- lab$Sex == s
  nd <- w & lab$diabetes == 0
  rbind(
    data.frame(model = m, sex = s, task = "Diagnosed label: detect diagnosed diabetes",
               t(auc_ci(lab$diabetes[w], oof_dx[[m]][w]))),
    data.frame(model = m, sex = s, task = "Total label: detect any diabetes",
               t(auc_ci(lab$total_dm[w], oof_tot[[m]][w]))),
    data.frame(model = m, sex = s, task = "Diagnosed label: find undiagnosed among non-diagnosed",
               t(auc_ci(lab$undiagnosed[nd], oof_dx[[m]][nd]))),
    data.frame(model = m, sex = s, task = "Total label: find undiagnosed among non-diagnosed",
               t(auc_ci(lab$undiagnosed[nd], oof_tot[[m]][nd])))
  )
}))))
write.csv(label_res, "outputs/tables/undiagnosed_auc_by_label.csv", row.names = FALSE)
cat("\nAUC by label and sex (out-of-fold):\n"); print(label_res, digits = 3)

# Sensitivity at the 10% threshold: diagnosed vs undiagnosed cases
sens_res <- bind_rows(lapply(names(oof_dx), function(m) bind_rows(lapply(c("Women", "Men"), function(s) {
  w <- lab$Sex == s
  sens <- function(cases, p) {
    k <- sum(p[cases] >= 0.10); n <- sum(cases)
    c(k / n, binom.test(k, n)$conf.int, n)
  }
  r <- rbind(sens(w & lab$diabetes == 1,    oof_dx[[m]]),
             sens(w & lab$undiagnosed == 1, oof_dx[[m]]),
             sens(w & lab$undiagnosed == 1, oof_tot[[m]]))
  data.frame(model = m, sex = s,
             group = c("Diagnosed cases\n(diagnosed-label model)",
                       "Undiagnosed cases\n(diagnosed-label model)",
                       "Undiagnosed cases\n(total-label model)"),
             sensitivity = r[, 1], lo = r[, 2], hi = r[, 3], n_cases = r[, 4])
}))))
write.csv(sens_res, "outputs/tables/undiagnosed_sensitivity_by_sex.csv", row.names = FALSE)
cat("\nSensitivity at 10% risk:\n"); print(sens_res, digits = 3)

# ---- Fig 10 --------------------------------------------------------------------
p10a <- undx_tab |>
  select(sex, Diagnosed = diagnosed_pct, Undiagnosed = undiagnosed_pct) |>
  pivot_longer(-sex, names_to = "status", values_to = "pct") |>
  mutate(sex = factor(sex, levels = c("Women", "Men")),
         status = factor(status, levels = c("Undiagnosed", "Diagnosed"))) |>
  ggplot(aes(sex, pct, fill = status)) +
  geom_col(width = 0.6, colour = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.1f%%", pct)), position = position_stack(vjust = 0.5),
            colour = "white", size = 3.5) +
  scale_fill_manual(values = c(Diagnosed = "grey45", Undiagnosed = "#e34948"), name = NULL) +
  labs(title = "Prevalence", x = NULL, y = "% of adults")
p10b <- sens_res |>
  mutate(sex = factor(sex, levels = c("Women", "Men")),
         group = factor(group, levels = unique(group))) |>
  ggplot(aes(sensitivity, group, colour = sex)) +
  geom_pointrange(aes(xmin = lo, xmax = hi), position = position_dodge(width = 0.5),
                  size = 0.4, linewidth = 0.7) +
  facet_wrap(~model) +
  scale_x_continuous(labels = scales::percent, limits = c(0, 1)) +
  scale_colour_manual(values = sex_cols, name = NULL) +
  labs(title = "Share of cases flagged at 10% predicted risk (out-of-fold, 95% CI)",
       x = "Sensitivity", y = NULL)
p10 <- p10a + p10b + plot_layout(widths = c(1, 3.2)) +
  plot_annotation(title = "Undiagnosed diabetes (HbA1c >= 6.5% or fasting glucose >= 126 mg/dL, no diagnosis)",
                  theme = theme(plot.title = element_text(face = "bold")))
save_fig(p10, "fig10_undiagnosed_by_sex", 13, 5)
