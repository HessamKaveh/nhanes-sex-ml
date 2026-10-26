# SHAP values for both models, collapsed back to the original variables.

suppressPackageStartupMessages(library(shapviz))

# Map each design-matrix column (e.g. "Race1Mexican", "Sex_Men") to the
# variable it came from, so one-hot dummies can be summed into one SHAP value.
column_groups <- function(cols) {
  vapply(cols, function(cn) {
    if (cn %in% numeric_vars) return(cn)
    if (startsWith(cn, "Sex_")) return("Sex")
    hit <- factor_vars[startsWith(cn, factor_vars)]
    if (length(hit) != 1) stop("Cannot map column: ", cn)
    hit
  }, character(1), USE.NAMES = FALSE)
}

collapse_shap <- function(S) {
  g <- column_groups(colnames(S))
  out <- sapply(unique(g), function(v) rowSums(S[, g == v, drop = FALSE]))
  if (is.null(dim(out))) out <- matrix(out, nrow = 1, dimnames = list(NULL, unique(g)))
  out
}

# XGBoost: exact TreeSHAP (log-odds scale) via predcontrib
shap_xgb <- function(fit, x) {
  S <- predict(fit, xgb.DMatrix(x), predcontrib = TRUE)
  list(S = collapse_shap(S[, colnames(x), drop = FALSE]),
       baseline = unname(S[1, ncol(S)]))  # last column = bias ("BIAS" / "(Intercept)")
}

# Elastic Net: for a linear model with independent features, the exact SHAP
# value of feature j is beta_j * (x_j - E[x_j]) on the log-odds scale, with the
# expectation taken over the training (background) data.
shap_enet <- function(fit, x, x_background) {
  b <- coef(fit, s = "lambda.1se")
  beta <- as.numeric(b[colnames(x), 1])
  mu <- colMeans(x_background)
  S <- sweep(sweep(x, 2, mu), 2, beta, `*`)
  list(S = collapse_shap(S),
       baseline = as.numeric(b[1, 1] + sum(beta * mu)))
}

# Display data for shapviz plots: original (unscaled) variables, same order as S
display_X <- function(d, S) as.data.frame(d[, colnames(S)])

as_shapviz <- function(sh, d) {
  S <- sh$S; X <- display_X(d, S)
  colnames(S) <- colnames(X) <- feature_labels[colnames(S)]
  shapviz(S, X = X, baseline = sh$baseline)
}

# Mean |SHAP| per feature with a percentile bootstrap CI over test rows
shap_importance <- function(S, B = 500, seed = 2026) {
  set.seed(seed)
  est <- colMeans(abs(S))
  boots <- replicate(B, colMeans(abs(S[sample(nrow(S), replace = TRUE), , drop = FALSE])))
  data.frame(feature = names(est), mean_abs_shap = est,
             lo = apply(boots, 1, quantile, 0.025),
             hi = apply(boots, 1, quantile, 0.975), row.names = NULL)
}
