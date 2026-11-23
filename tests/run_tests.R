#!/usr/bin/env Rscript
# Run the unit tests from the project root:  Rscript tests/run_tests.R
# The tests need only testthat (and pROC for one test); NHANES data are not used.
testthat::test_dir("tests/testthat", reporter = "summary", stop_on_failure = TRUE)
