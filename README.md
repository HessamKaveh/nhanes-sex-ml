# Diabetes risk in NHANES: Elastic Net vs XGBoost, explained with SHAP, women vs men

A small, fully reproducible R project that predicts **diagnosed diabetes** in US adults
from the public [NHANES](https://cran.r-project.org/package=NHANES) data (2009–2012),
compares a penalised linear model (**Elastic Net**) with a gradient-boosted tree model
(**XGBoost**), explains both with **SHAP**, and asks whether the models behave
differently for **women and men**.

## Questions

1. Do the models discriminate equally well in women and in men?
2. Does training separate models for each sex help, compared with one pooled model?
3. Which risk factors drive predictions, and do they differ between the sexes?

## Data

- Source: the `NHANES` R package (10,000 resampled NHANES 2009–2012 participants).
- Adults aged 20+ with a known diabetes status. The package resamples participants,
  so the same person can appear more than once; **rows are de-duplicated by `ID`**
  to keep anyone from landing in both train and test.
- Final cohort: **4,644** adults (2,389 women, 2,255 men); diabetes prevalence
  11.1% in women and 12.4% in men.
- Features: age, BMI, pulse, systolic/diastolic BP, total and HDL cholesterol,
  income-to-poverty ratio, sleep hours, alcohol days/year, race/ethnicity,
  education, physical activity, smoking history. Lab glucose/HbA1c are not in the
  package and would leak the outcome anyway.

## Methods

| Step | Details |
|---|---|
| Split | 70/30 train/test, stratified by sex × outcome (seed 2026) |
| Pre-processing | Median/mode imputation and standardisation learned on the training set (Elastic Net). XGBoost gets raw values and handles missing values natively |
| Elastic Net | `glmnet`, α ∈ {0, .25, .5, .75, 1} and λ tuned with 10-fold CV (AUC), `lambda.1se` |
| XGBoost | depth 3, η = 0.03, subsample/colsample 0.8; number of rounds by 5-fold CV with early stopping |
| SHAP | Exact TreeSHAP for XGBoost (`predcontrib`); exact linear SHAP, β·(x − E[x]), for Elastic Net. Both on the log-odds scale; one-hot dummies summed back to their variable. Plots with `shapviz` |
| Sex comparison | (a) **Pooled** models with sex as a feature, evaluated separately in women and men; (b) **sex-specific** models trained on each sex alone |
| Missing data | Sensitivity analysis: median/mode, missingness indicators, complete cases, multiple imputation with `mice` (m = 5) |
| Evaluation | AUC with DeLong 95% CI, DeLong test women vs men, Brier score, calibration by quintile |
| Fairness | Sensitivity, specificity, PPV, NPV at fixed thresholds (5/10/20%) per sex; equal-opportunity and equalised-odds gaps |

## Results

### Discrimination

| Model | Training | Women AUC (95% CI) | Men AUC (95% CI) |
|---|---|---|---|
| Elastic Net | Pooled | 0.83 (0.78–0.87) | 0.76 (0.71–0.81) |
| Elastic Net | Sex-specific | 0.82 (0.78–0.86) | 0.77 (0.72–0.82) |
| XGBoost | Pooled | 0.84 (0.80–0.89) | 0.78 (0.73–0.83) |
| XGBoost | Sex-specific | 0.84 (0.80–0.88) | 0.78 (0.72–0.83) |

Overall test AUC: Elastic Net 0.79, XGBoost 0.81.

![AUC by sex](outputs/figures/fig1_auc_by_sex.png)

- Both models rank risk **better in women than in men** (≈0.06 AUC gap), but with
  ~80–100 cases per sex in the test set the difference is only borderline
  (DeLong p = 0.058 for Elastic Net, 0.069 for XGBoost).
- **Sex-specific models do not beat the pooled model.** The pooled model barely uses
  the sex variable itself (mean SHAP of `Sex` ≈ ±0.003 log-odds).
- XGBoost is slightly ahead of Elastic Net, with overlapping CIs.

![ROC by sex](outputs/figures/fig2_roc_by_sex.png)
![Calibration by sex](outputs/figures/fig3_calibration_by_sex.png)

### Fairness at a fixed threshold

AUC and calibration describe the whole risk ranking, but a screening tool applies a
cut-off. Using the **same 10% predicted-risk threshold** for everyone (pooled models,
test set, exact 95% CIs):

| Model | Sex | Sensitivity | Specificity |
|---|---|---|---|
| Elastic Net | Women | 0.96 (0.89–0.99) | 0.54 (0.50–0.57) |
| Elastic Net | Men | 0.89 (0.81–0.94) | 0.48 (0.43–0.52) |
| XGBoost | Women | 0.89 (0.80–0.95) | 0.71 (0.67–0.75) |
| XGBoost | Men | 0.74 (0.64–0.83) | 0.68 (0.64–0.71) |

- At the same threshold **the XGBoost model misses more diabetic men than women**
  (sensitivity gap 0.15, two-sample test p = 0.025), with similar false positive rates.
  That breaks *equal opportunity*.
- The Elastic Net gap goes the same way (0.08, p = 0.11) and its false positive rate is
  higher in men (p = 0.04).
- The pooled Elastic Net chose α = 0 (ridge) with strong shrinkage, so its predicted
  risks sit in a narrow band. Small threshold changes swing its error rates a lot (Fig. 8).
- Full tables for thresholds 5%, 10% and 20%, including PPV, NPV and flag rate:
  `outputs/tables/threshold_metrics_by_sex.csv` and `equalized_odds_gaps.csv`.

![Threshold metrics by sex](outputs/figures/fig8_threshold_metrics_by_sex.png)

### Missing data

31% of participants miss at least one predictor. Missingness is **not the same for
the two sexes**: alcohol use is missing for 28% of women vs 14% of men, and BP/pulse
for 5% vs 3%. Diabetes prevalence also differs between participants with and without
missing values (e.g. 16% vs 11% when cholesterol is missing), so values are probably
not missing completely at random.

The pooled models were refitted under five strategies (`analyses/missing_data.R`):

| Scenario | EN women | EN men | XGB women | XGB men |
|---|---|---|---|---|
| Baseline (EN: median/mode; XGB: native NA) | 0.83 | 0.76 | 0.84 | 0.78 |
| Median/mode for both | 0.83 | 0.76 | 0.85 | 0.78 |
| Median + missingness flags | 0.83 | 0.77 | 0.85 | 0.78 |
| **MICE**, m = 5, averaged predictions | 0.83 | 0.77 | 0.85 | 0.79 |
| Complete cases (n test = 966) | 0.78 | 0.77 | 0.79 | 0.76 |
| Baseline on the same complete-case test rows | 0.79 | 0.75 | 0.80 | 0.77 |

- Imputation choices barely move AUC (≤ 0.01), and the women-men gap stays at about 0.06.
- **Complete-case analysis is the outlier.** It discards 38% of the women in the test
  set but only 24% of the men. The women's AUC drops to about 0.79 and the sex gap
  almost disappears. The baseline model shows the same drop on those rows, so the
  drop comes from *who* is excluded, not from the modelling. Dropping incomplete rows
  would change the sex comparison.
- MICE: chained equations with predictive mean matching for numeric and logistic or
  polytomous regression for factor variables. The imputation model is estimated on
  training rows only (`ignore` = test), and the outcome is left out so test labels
  cannot leak.

![Missing-data sensitivity](outputs/figures/fig9_missing_data_sensitivity.png)

### What drives the predictions

![SHAP beeswarm](outputs/figures/fig6_shap_beeswarm_pooled.png)

Age and BMI dominate in both models, followed by systolic BP, cholesterol and income.
XGBoost picks up non-linear shapes the linear model cannot: the age effect rises
steeply between 35 and 60 and then flattens, and the BMI effect saturates above ~45.

![SHAP dependence by sex](outputs/figures/fig7_shap_dependence_by_sex.png)

### Women vs men

![SHAP importance by sex](outputs/figures/fig5_shap_importance_by_sex.png)
![Elastic Net coefficients by sex](outputs/figures/fig4_enet_coefficients_by_sex.png)

In the sex-specific models:

- **Age matters more in women** (mean |SHAP| for Elastic Net 0.70 vs 0.39; XGBoost 1.09 vs 0.96).
- **HDL cholesterol is protective mainly in women**: Elastic Net keeps it for women
  (−0.19 per SD) and drops it for men; XGBoost importance is 0.19 vs 0.06.
- Alcohol days/year has a negative association in women in the linear model; in men
  sleep hours and diastolic BP carry somewhat more weight in XGBoost.

## Caveats

- This is a teaching/demo analysis, not a clinical model. NHANES survey weights and
  design are not used; the `NHANES` package is already resampled to be roughly
  representative.
- "Diabetes" is self-reported diagnosis, so undiagnosed cases count as negatives.
- The bootstrap CIs on SHAP importance resample test rows only and ignore
  model-fitting uncertainty, so they are too narrow for formal inference.
- Elastic Net coefficients across the two sexes are shrunk by different λ values
  (women 0.020, men 0.066), so part of the size difference comes from regularisation.
- The selected α was 0 (ridge) for the pooled model and 0.5 / 0.25 for women / men.

## Reproduce

```bash
conda env create -f environment.yml
conda activate r-nhanes
Rscript -e 'install.packages("NHANES", repos = "https://cloud.r-project.org")'
Rscript run_all.R        # ~2 min; writes outputs/figures and outputs/tables
```

## Layout

```
run_all.R          main pipeline
R/01_data.R        loading, de-duplication, split, pre-processing
R/02_models.R      Elastic Net, XGBoost, evaluation
R/03_plots.R       theme, colours, labels
R/04_shap.R        SHAP for both models, importance with bootstrap CI
analyses/missing_data.R   missing-data sensitivity analysis (incl. MICE)
outputs/figures/   PNG figures
outputs/tables/    CSV results (performance, coefficients, SHAP importance, ...)
```

---

