test_that("column_groups maps design-matrix columns back to variables", {
  cols <- c("Age", "Race1Mexican", "Race1White", "Sex_Men", "Smoke100Yes", "EducationCollege Grad")
  expect_equal(defs$column_groups(cols), c("Age", "Race1", "Race1", "Sex", "Smoke100", "Education"))
  expect_error(defs$column_groups("Mystery"), "Cannot map column")
})

test_that("collapse_shap sums one-hot dummies, also for a single row", {
  S <- matrix(c(1, 2, 3, 4,
                5, 6, 7, 8), nrow = 2, byrow = TRUE,
              dimnames = list(NULL, c("Age", "Race1Mexican", "Race1White", "Sex_Men")))
  out <- defs$collapse_shap(S)
  expect_equal(colnames(out), c("Age", "Race1", "Sex"))
  expect_equal(unname(out[, "Race1"]), c(5, 13))
  one <- defs$collapse_shap(S[1, , drop = FALSE])
  expect_equal(dim(one), c(1L, 3L))
  expect_equal(unname(rowSums(one)), sum(S[1, ]))
})

test_that("linear SHAP is exact: baseline + sum(SHAP) equals the linear predictor", {
  d <- fake_cohort(50, seed = 5)
  pp <- defs$fit_preprocessor(d)
  x <- defs$make_matrix(d, pp)
  set.seed(6)
  beta <- setNames(rnorm(ncol(x)), colnames(x))
  fit <- structure(list(b = rbind("(Intercept)" = -1.5, cbind(beta))), class = "fake_glmnet")
  registerS3method("coef", "fake_glmnet", function(object, s, ...) object$b)
  sh <- defs$shap_enet(fit, x[1:10, ], x)
  eta <- -1.5 + as.numeric(x[1:10, ] %*% beta)
  expect_equal(sh$baseline + unname(rowSums(sh$S)), eta, tolerance = 1e-10)
  expect_equal(colnames(sh$S), unique(defs$column_groups(colnames(x))))
})

test_that("shap_importance gives mean |SHAP| with a bootstrap CI", {
  set.seed(7)
  S <- cbind(a = rnorm(200, sd = 2), b = rnorm(200, sd = 0.5))
  imp <- defs$shap_importance(S, B = 200)
  expect_equal(imp$mean_abs_shap, unname(colMeans(abs(S))))
  expect_true(all(imp$lo <= imp$mean_abs_shap & imp$mean_abs_shap <= imp$hi))
  expect_gt(imp$mean_abs_shap[1], imp$mean_abs_shap[2])
})

test_that("pretty_terms labels variables and factor levels", {
  expect_equal(defs$pretty_terms(c("BMI", "Race1White", "Smoke100Yes", "Unknown")),
               c("BMI", "Race/ethnicity: White", "Ever smoked (100+ cig.): Yes", "Unknown"))
})
