test_that("threshold_metrics computes the confusion-matrix rates with exact CIs", {
  y <- c(1, 1, 1, 1, 0, 0, 0, 0, 0, 0)
  p <- c(.9, .8, .7, .1, .6, .2, .2, .1, .1, .1)  # TP 3, FN 1, FP 1, TN 5
  m <- defs$threshold_metrics(y, p, 0.5)
  v <- setNames(m$value, m$metric)
  expect_equal(v[["sensitivity"]], 3 / 4)
  expect_equal(v[["specificity"]], 5 / 6)
  expect_equal(v[["ppv"]], 3 / 4)
  expect_equal(v[["npv"]], 5 / 6)
  expect_equal(v[["flag_rate"]], 4 / 10)
  expect_true(all(m$lo <= m$value & m$value <= m$hi))
  expect_equal(unname(unlist(m[m$metric == "sensitivity", c("lo", "hi")])),
               binom.test(3, 4)$conf.int[1:2])
})

test_that("threshold_metrics returns NA when nobody is flagged", {
  m <- defs$threshold_metrics(c(0, 1, 0), c(.1, .2, .3), 0.9)
  expect_true(is.na(m$value[m$metric == "ppv"]))
  expect_equal(m$value[m$metric == "flag_rate"], 0)
})

test_that("evaluate reports AUC, DeLong CI and Brier score", {
  skip_if_not_installed("pROC")
  suppressPackageStartupMessages(library(pROC))
  set.seed(4)
  y <- rbinom(400, 1, 0.3)
  p <- plogis(2 * y - 1 + rnorm(400))
  e <- defs$evaluate(y, p)
  expect_equal(e$n, 400)
  expect_equal(e$brier, mean((p - y)^2))
  expect_true(e$auc_lo < e$auc && e$auc < e$auc_hi)
  expect_equal(e$auc, as.numeric(auc(roc(y, p, quiet = TRUE, direction = "<"))))
})
