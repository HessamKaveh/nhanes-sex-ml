# Data preparation: NHANES adults, diabetes outcome, train/test split.

suppressPackageStartupMessages({
  library(NHANES)
  library(dplyr)
  library(rsample)
})

numeric_vars <- c("Age", "BMI", "Pulse", "BPSysAve", "BPDiaAve", "TotChol",
                  "DirectChol", "Poverty", "SleepHrsNight", "AlcoholYear")
factor_vars  <- c("Race1", "Education", "PhysActive", "Smoke100")

load_nhanes <- function() {
  NHANES |>
    filter(Age >= 20, !is.na(Diabetes)) |>
    # The NHANES package resamples participants to mimic the survey design,
    # so the same person can appear several times. Keep one row per ID to
    # avoid the same individual landing in both train and test.
    distinct(ID, .keep_all = TRUE) |>
    transmute(
      ID,
      Sex      = factor(Gender, levels = c("female", "male"),
                        labels = c("Women", "Men")),
      diabetes = as.integer(Diabetes == "Yes"),
      across(all_of(numeric_vars), as.numeric),
      across(all_of(factor_vars), droplevels)
    ) |>
    filter(!is.na(Education))
}

split_data <- function(d, prop = 0.7, seed = 2026) {
  set.seed(seed)
  # Stratify on sex x outcome so both sexes keep the same prevalence in each part
  d$strata <- interaction(d$Sex, d$diabetes)
  s <- initial_split(d, prop = prop, strata = strata)
  list(train = select(training(s), -strata), test = select(testing(s), -strata))
}

# Imputation + scaling parameters are learned on the training data only.
fit_preprocessor <- function(train) {
  list(
    medians = vapply(train[numeric_vars], median, numeric(1), na.rm = TRUE),
    means   = vapply(train[numeric_vars], mean,   numeric(1), na.rm = TRUE),
    sds     = vapply(train[numeric_vars], sd,     numeric(1), na.rm = TRUE),
    modes   = lapply(train[factor_vars], function(x) names(which.max(table(x))))
  )
}

# Design matrix. `impute_scale = TRUE` for Elastic Net (needs complete,
# standardised inputs so coefficients are per-SD); XGBoost gets raw values
# with NA left in place, since it learns its own missing-value directions.
make_matrix <- function(d, pp, include_sex = TRUE, impute_scale = TRUE) {
  x <- d
  for (v in factor_vars) x[[v]][is.na(x[[v]])] <- pp$modes[[v]]
  if (impute_scale) {
    for (v in numeric_vars) {
      x[[v]][is.na(x[[v]])] <- pp$medians[[v]]
      x[[v]] <- (x[[v]] - pp$means[[v]]) / pp$sds[[v]]
    }
  }
  vars <- c(numeric_vars, factor_vars, if (include_sex) "Sex")
  f <- reformulate(vars)
  mf <- model.frame(f, x, na.action = na.pass)
  m <- model.matrix(f, mf)[, -1, drop = FALSE]
  colnames(m) <- sub("^Sex", "Sex_", colnames(m))
  m
}
