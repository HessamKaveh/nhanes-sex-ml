# Model fitting: Elastic Net (glmnet) and XGBoost, plus evaluation helpers.

suppressPackageStartupMessages({
  library(glmnet)
  library(xgboost)
  library(pROC)
})

# Elastic Net: tune alpha on a grid and lambda by CV, sharing the same folds.
fit_enet <- function(x, y, alphas = seq(0, 1, by = 0.25), nfolds = 10, seed = 2026) {
  set.seed(seed)
  foldid <- sample(rep(seq_len(nfolds), length.out = length(y)))
  fits <- lapply(alphas, function(a)
    cv.glmnet(x, y, family = "binomial", alpha = a, foldid = foldid,
              type.measure = "auc", standardize = FALSE))
  best <- which.max(vapply(fits, function(f) max(f$cvm), numeric(1)))
  fit <- fits[[best]]
  fit$alpha <- alphas[best]
  fit
}

predict_enet <- function(fit, x) {
  as.numeric(predict(fit, newx = x, s = "lambda.1se", type = "response"))
}

enet_coefs <- function(fit) {
  b <- as.matrix(coef(fit, s = "lambda.1se"))
  data.frame(feature = rownames(b), coef = b[, 1], row.names = NULL) |>
    subset(feature != "(Intercept)")
}

# XGBoost: fixed, fairly conservative hyper-parameters; number of rounds
# chosen by 5-fold CV with early stopping.
xgb_params <- list(
  objective = "binary:logistic", eval_metric = "auc",
  eta = 0.03, max_depth = 3, min_child_weight = 5,
  subsample = 0.8, colsample_bytree = 0.8, lambda = 1, nthread = 4
)

fit_xgb <- function(x, y, seed = 2026) {
  set.seed(seed)
  dtrain <- xgb.DMatrix(x, label = y)
  cv <- xgb.cv(params = xgb_params, data = dtrain, nrounds = 2000, nfold = 5,
               early_stopping_rounds = 50, verbose = 0, stratified = TRUE)
  best <- cv$early_stop$best_iteration
  fit <- xgb.train(params = xgb_params, data = dtrain, nrounds = best, verbose = 0)
  attr(fit, "best_nrounds") <- best
  fit
}

predict_xgb <- function(fit, x) predict(fit, xgb.DMatrix(x))

# AUC with DeLong 95% CI and Brier score
evaluate <- function(y, p) {
  r <- roc(y, p, quiet = TRUE, direction = "<")
  ci <- as.numeric(ci.auc(r, method = "delong"))
  data.frame(n = length(y), prevalence = mean(y),
             auc = ci[2], auc_lo = ci[1], auc_hi = ci[3],
             brier = mean((p - y)^2),
             mean_pred = mean(p))
}
