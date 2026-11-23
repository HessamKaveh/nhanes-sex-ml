# Load the helper functions and constants from R/*.R *without* running the
# library() / theme_set() calls, so the pure logic can be tested without the
# NHANES data or the modelling packages.

find_root <- function(dir = getwd()) {
  while (!file.exists(file.path(dir, "run_all.R"))) {
    parent <- dirname(dir)
    if (identical(parent, dir)) stop("project root (run_all.R) not found")
    dir <- parent
  }
  dir
}
project_root <- find_root()

defs <- new.env()
for (f in sort(list.files(file.path(project_root, "R"), pattern = "\\.R$", full.names = TRUE))) {
  for (e in parse(f, keep.source = FALSE)) {
    if (is.call(e) && (identical(e[[1]], as.name("<-")) || identical(e[[1]], as.name("=")))) eval(e, defs)
  }
}

# Small synthetic cohort with the same columns as load_nhanes()
fake_cohort <- function(n = 60, seed = 1) {
  set.seed(seed)
  d <- data.frame(ID = seq_len(n),
                  Sex = factor(rep(c("Women", "Men"), length.out = n), levels = c("Women", "Men")),
                  diabetes = rbinom(n, 1, 0.2))
  for (v in defs$numeric_vars) d[[v]] <- rnorm(n, 50, 10)
  d$Race1 <- factor(sample(c("Black", "Hispanic", "Mexican", "White", "Other"), n, TRUE))
  d$Education <- factor(sample(c("8th Grade", "High School", "College Grad"), n, TRUE))
  d$PhysActive <- factor(sample(c("No", "Yes"), n, TRUE))
  d$Smoke100 <- factor(sample(c("No", "Yes"), n, TRUE))
  d
}
