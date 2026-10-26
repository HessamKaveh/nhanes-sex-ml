# Shared plotting theme and colours.

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

sex_cols   <- c(Women = "#2a78d6", Men = "#eb6834")
model_cols <- c("Elastic Net" = "#1baf7a", "XGBoost" = "#6250d6")

theme_set(
  theme_minimal(base_size = 11) +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major = element_line(colour = "grey90", linewidth = 0.3),
          plot.title = element_text(face = "bold"),
          plot.title.position = "plot",
          legend.position = "top",
          plot.background = element_rect(fill = "white", colour = NA))
)

save_fig <- function(p, name, w = 8, h = 5) {
  ggsave(file.path("outputs/figures", paste0(name, ".png")), p,
         width = w, height = h, dpi = 150, bg = "white")
}

# Human-readable feature labels
feature_labels <- c(
  Age = "Age", BMI = "BMI", Pulse = "Pulse", BPSysAve = "Systolic BP",
  BPDiaAve = "Diastolic BP", TotChol = "Total cholesterol",
  DirectChol = "HDL cholesterol", Poverty = "Income-to-poverty ratio",
  SleepHrsNight = "Sleep hours", AlcoholYear = "Alcohol days/year",
  Race1 = "Race/ethnicity", Education = "Education",
  PhysActive = "Physically active", Smoke100 = "Ever smoked (100+ cig.)",
  Sex = "Sex"
)

# Label design-matrix terms, e.g. "Race1White" -> "Race/ethnicity: White"
pretty_terms <- function(terms) {
  vapply(terms, function(t) {
    if (t %in% names(feature_labels)) return(feature_labels[[t]])
    for (v in c("Race1", "Education", "PhysActive", "Smoke100"))
      if (startsWith(t, v)) return(paste0(feature_labels[[v]], ": ", substring(t, nchar(v) + 1)))
    t
  }, character(1), USE.NAMES = FALSE)
}
