# COSTAR-CVD: Colab reproduction and empirical simulation

## Files

- `COSTAR_CVD_Colab.ipynb`: self-contained Colab notebook (runs in normal CPU runtime; no GPU required).
- `COSTAR_CVD_colab.py`: the same pipeline as a standalone Python script.
- `requirements.txt`: packages needed in a fresh runtime.
- `test_methods.py`: focused mathematical/unit tests.

## Run on Google Colab

1. Open [Google Colab](https://colab.research.google.com), select **File > Upload notebook**, and upload `COSTAR_CVD_Colab.ipynb`.
2. Under **Runtime**, use **Python 3** (CPU is sufficient). Run **Runtime > Run all**.
3. The notebook installs dependencies, then prompts for **`Cardiovascular_Disease_Dataset.csv`**. Upload the original CSV unchanged (source [DOI:10.17632/dzz48mvjht.1](https://doi.org/10.17632/dzz48mvjht.1)).
4. Monitor the two live `tqdm` progress displays (overall stage progress with elapsed/ETA, and nested CV fold progress with elapsed/ETA). **ETAs are estimates and change as fitting speeds vary.**
5. On successful completion, a single `COSTAR_CVD_RESULTS.zip` is automatically downloaded. If the browser blocks the download, use Colab's **Files** pane and download `/content/COSTAR_CVD_RESULTS.zip`.
6. Return the output ZIP for the empirical Results, Discussion, and manuscript Tables/Figures sections.

No external service, API key, GPU, or patient-level web transmission is required by the script. Colab itself is a remote computing service.

## What the notebook does

- Fixed, non-stratified 600/200/200 partition: model development / threshold optimization / untouched evaluation.
- Nested 5-fold cross-fitting with inner 3-fold tuning on development only.
- Baseline elastic-net logistic regression, random forest, and histogram gradient boosting; 3-model convex Brier stack with regularization chosen using development OOF scores.
- Monotone sigmoid probability recalibration using OOF stacked scores.
- Initial-feature and full-information models with explicit information-acquisition sequence.
- Prespecified feature-access policy excluding the potentially downstream `noofmajorvessels` variable.
- A finite threshold grid, exact bounded-density-ratio robust support function, and constrained optimizer for testing, referral, and joint missed-case risk.
- Comparator policies: B-only tuned classifiers and stack, a basic logistic model with a fixed 0.5 threshold, nominal sequential policy, and random-panel allocation.
- Clopper–Pearson 95% familywise simultaneous one-sided certificates (three global risks plus two subgroup FNR rates).
- Paired percentile bootstrap (1,500 draws), exploratory AUROC differences and decision-loss differences.
- Gamma, two resource limits, joint missed-case tolerance, cost ratios, and descriptive subgroup penalty sensitivity.
- Separate retraining for `slope` removal, `slope`/`oldpeak` removal, cholesterol omission, missing-indicator omission; the downstream vessel variable is used for prediction-only retrospective stress testing, not for the primary policy.
- Saving all figures as both vector PDFs and high-resolution PNGs; tables as CSV and LaTeX.
- Exact sorting formula numerically checked against 28 independent linear-programming solutions.

## Default decision-analysis scenarios

The code declares the scenario before seeing evaluation outcomes:

```text
seed = 20261028
Gamma = 1.25
c_test = 0.055, c_referral = 0.18, c_missed = 1.0
capacity_test = 0.40, capacity_referral = 0.58
maximum_joint_miss = 0.42
lambda_gender_disparity = 0.0
alpha_familywise = 0.05
```

These are **hypothetical dimensionless loss weights and population-average resource budgets**, not measured hospital costs or clinical safety limits. Alter them only on substantive grounds as a **new prespecified experiment**; do not choose them after viewing evaluation results.

## Output ZIP

- `README_RESULTS.md`: results synopsis, caveats, achieved runtime.
- `configuration.json` / `package_versions.json` / `data_audit.json`: full provenance.
- `partition_*.csv`: exact split membership and observed class fractions.
- `selected_primary_policy.json` / `secondary_policy_parameters.json`: frozen thresholds.
- `tables/*.csv` and `tables/*.tex`: discriminative/calibration metrics, sequential decisions, certification, paired comparisons, subgroup analysis, ablations, cost- and robustness-sensitivity, and LP checks.
- `tables/complete_candidate_frontier.csv.gz`: all evaluated threshold/grid cases for all Gamma scenarios.
- `predictions/*.csv`: evaluation predictions and decisions, and optimization predictions.
- `figures/*.pdf` and `figures/*.png`: high-quality reusable figures.
- `models/*.joblib` (when serializing is available): sklearn models, which may require rerunning the source definitions and the same dependency versions to load.

## Important integrity caveats

1. The code does not force COSTAR to outperform its comparators. Its results may be neutral or unfavorable. These outcomes should be reported.
2. Evaluation-set outcome labels are not used for model fitting, hyperparameters, ensemble weights, calibration, optimizing the policy, or choosing the primary Gamma/cost/budget scenario. They **are** appropriately used to compute final evaluation statistics, confidence intervals, ablation comparisons, and descriptive scenario-reweighting summaries.
3. Because auxiliary analyses evaluate multiple candidate models on the same holdout, their CIs and differences are *exploratory*. The exact **single-policy** familywise theorem is separately reported for the frozen primary policy.
4. Robust reweighting of empirical cases is **not** a confidence region for the unknown population and is not a guarantee of external transportability.
5. This is an offline simulation of recorded status, **not** a prospective diagnostic study, patient referral recommendation, real-hospital scheduling experiment, or causal study of testing and treatment.
6. The high predictive separability of the data and the possible timing of stress-test variables need independent verification. The 53 zero serum-cholesterol entries are recoded as potential missing values, with the presence of a missingness indicator explicitly examined.
7. Colab execution time is hardware-dependent; the progress bar estimates time remaining using observed stage/fold rates.

## Local command-line mode

```bash
python -m pip install -r requirements.txt
python test_methods.py
python COSTAR_CVD_colab.py --file /path/to/Cardiovascular_Disease_Dataset.csv --output ./COSTAR_CVD_RESULTS --no-download
```

For rapid software smoke-testing **only**, use `--smoke-test`; it uses reduced folds/hyperparameters and only 90 bootstrap replications. **Do not use smoke-test results in a manuscript.** For debugging without sensitivity refits, `--skip-ablations` is available; default full runs perform all ablations.

## Error mode

If something fails, the code records a traceback in `ERROR_DIAGNOSTICS.json` and tries to create a partial diagnostic ZIP; do not interpret a partial run as a successful statistical analysis.
