test_that("every R script in the project parses", {
  files <- c(list.files(file.path(project_root, c("R", "analyses")), pattern = "\\.R$", full.names = TRUE),
             file.path(project_root, "run_all.R"), file.path(project_root, "tests", "run_tests.R"))
  expect_gt(length(files), 5)
  for (f in files) expect_no_error(parse(f, keep.source = FALSE), message = basename(f))
})

test_that("helpers are defined", {
  for (fn in c("load_nhanes", "split_data", "fit_preprocessor", "make_matrix", "fit_enet", "fit_xgb",
               "evaluate", "threshold_metrics", "pretty_terms", "column_groups", "collapse_shap",
               "shap_enet", "shap_xgb", "shap_importance"))
    expect_true(is.function(defs[[fn]]), info = fn)
})
