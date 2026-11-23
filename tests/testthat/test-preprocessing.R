test_that("fit_preprocessor learns medians, means, sds and modes ignoring NA", {
  d <- fake_cohort()
  d$BMI[1:5] <- NA
  d$Smoke100[] <- "Yes"; d$Smoke100[1] <- "No"
  pp <- defs$fit_preprocessor(d)
  expect_equal(pp$medians[["BMI"]], median(d$BMI, na.rm = TRUE))
  expect_equal(pp$means[["Age"]], mean(d$Age))
  expect_equal(pp$sds[["BMI"]], sd(d$BMI, na.rm = TRUE))
  expect_equal(pp$modes$Smoke100, "Yes")
})

test_that("make_matrix imputes with training statistics and standardises", {
  tr <- fake_cohort(80, seed = 2)
  te <- fake_cohort(30, seed = 3)
  te$BMI[1] <- NA; te$Race1[2] <- NA
  pp <- defs$fit_preprocessor(tr)

  xtr <- defs$make_matrix(tr, pp)
  expect_equal(unname(colMeans(xtr[, defs$numeric_vars])), rep(0, 10), tolerance = 1e-12)
  expect_equal(unname(apply(xtr[, defs$numeric_vars], 2, sd)), rep(1, 10), tolerance = 1e-12)

  xte <- defs$make_matrix(te, pp)
  expect_equal(nrow(xte), nrow(te))
  expect_false(anyNA(xte))
  expect_equal(xte[1, "BMI"], (pp$medians[["BMI"]] - pp$means[["BMI"]]) / pp$sds[["BMI"]])
  expect_true("Sex_Men" %in% colnames(xte))
  expect_identical(colnames(xte), colnames(xtr))
})

test_that("make_matrix options: no sex column, raw values with NA kept", {
  d <- fake_cohort()
  d$Pulse[3] <- NA
  pp <- defs$fit_preprocessor(d)
  x <- defs$make_matrix(d, pp, include_sex = FALSE, impute_scale = FALSE)
  expect_false(any(grepl("^Sex", colnames(x))))
  expect_true(is.na(x[3, "Pulse"]))
  expect_equal(unname(x[, "Age"]), d$Age)
})
