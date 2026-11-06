# Late Deliveries

QuickBite can offer a $4 voucher when an order is placed, before a late delivery
disappoints the customer. This project predicts, at the moment an order is placed,
whether it will arrive more than ten minutes late, and flags it for a voucher when
that lowers the expected cost.

## Setup

```bash
conda env create -f environment.yml
conda activate lab05-late-deliveries
pip install -r requirements.txt
pip install -e .
```

## Run

```bash
python -m latedeliveries.train
```

This recreates `reports/cv_results.csv`, `reports/metrics.json`,
`models/late_orders.joblib` and `models/model_info.json` from nothing, in about ten
seconds. Every random step uses `SEED = 42`, so the numbers match the committed files.
`notebooks/analysis.ipynb` reads those files and shows the charts.

## Check it

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

## Decisions

### Columns

The model is used when the order is placed, so a column is a feature only if its value
is known at that moment.

| Used | Type |
|---|---|
| `distance_km`, `items`, `prep_minutes`, `courier_months` | numeric: median fill, then scaled |
| `hour`, `weather`, `day_type`, `vehicle` | category: one-hot, unseen values ignored |

| Left out | Reason |
|---|---|
| `customer_rating` | Given after delivery, and made from `late` itself (late orders score about 1.5 points lower). It does not exist when the order is placed, so using it is leakage. |
| `order_id` | A label with no meaning; a model could only learn noise from it. |
| `late` | The target. |

Assumptions behind the columns I kept:

- `prep_minutes` records how long the kitchen took, which is not known exactly when the
  order is placed. I keep it on the assumption that the restaurant gives an estimate at
  that moment. In real use the estimate would be noisier than this column.
- `vehicle` and `courier_months` assume the courier is assigned when the order is placed.
- `hour` is treated as a category, not a number: lateness rises at lunch and dinner
  rush and falls again, which a single linear slope on `hour` cannot represent.

**Leakage check.** With `customer_rating` added back, the chosen model's cross-validated
ROC-AUC rises from **0.811 to 0.915** (section 6 of the notebook). A jump that large from
one column is a warning, not good news, so the column stays out.

### Model

Five-fold stratified cross-validation on the training data only (2,250 orders), every
model inside the same pipeline. Mean ± standard deviation over the folds:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (`DummyClassifier`, most frequent) | 0.844 ± 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 |
| Logistic regression | 0.868 ± 0.011 | 0.688 ± 0.089 | 0.273 ± 0.048 | 0.391 ± 0.062 | 0.812 ± 0.034 |
| **Logistic regression, `class_weight="balanced"`** | 0.745 ± 0.013 | 0.345 ± 0.019 | 0.704 ± 0.038 | 0.463 ± 0.024 | 0.811 ± 0.034 |
| Random forest | 0.858 ± 0.007 | 0.731 ± 0.105 | 0.137 ± 0.036 | 0.229 ± 0.055 | 0.794 ± 0.031 |
| Gradient boosting | 0.862 ± 0.008 | 0.627 ± 0.063 | 0.279 ± 0.053 | 0.384 ± 0.059 | 0.789 ± 0.032 |

The baseline flags nobody, which is the company's current policy, and still reaches
84% accuracy because only 15.6% of orders are late. Accuracy is therefore the wrong
guide here.

I chose the **balanced logistic regression**. A missed late order costs $15 and a voucher
costs $4, so missing a late order is almost four times as expensive as a wasted voucher,
and recall matters more than precision. The balanced model has the highest recall and F1
with the smallest spread. Its ROC-AUC ties with the plain logistic regression (0.811 vs
0.812, far inside the 0.034 spread), so both rank orders equally well; the tree models
rank them slightly worse and are harder to explain.

### Tuning

`GridSearchCV` on the chosen model, 12 combinations:
`model__C` in {0.01, 0.1, 1, 10} × `model__l1_ratio` in {0, 0.5, 1}
(solver `saga`, five folds on the training data).

The scoring metric is **ROC-AUC**, because the threshold is chosen afterwards from costs.
Tuning should improve how well the model ranks risky orders, not its decisions at the
default 0.5 threshold.

Best settings: `C = 0.1`, `l1_ratio = 0` (an L2 penalty). Tuned ROC-AUC 0.812 ± 0.032
against 0.811 ± 0.034 untuned. **Tuning did not beat the untuned model by more than the
fold-to-fold spread**: the 0.001 gain is about thirty times smaller than the spread. I
keep the tuned settings because they are no worse, but the gain is not real evidence.

### Threshold

The cost rule needs its own threshold because the two mistakes cost different amounts.
The default 0.5 treats them as equal. Total cost = $4 × orders flagged + $15 × late
orders not flagged.

I took out-of-fold probabilities on the training data with `cross_val_predict`, so every
order is scored by a model that never saw it, and computed the total cost at every
threshold from 0.05 to 0.95 in steps of 0.05. The lowest was **0.70**, at $4,019,
against $5,265 for flagging nobody. The test set played no part in this choice.

0.70 looks high, given that a voucher pays off whenever the chance of lateness is above
4/15 ≈ 0.27. The reason is `class_weight="balanced"`, which pushes every predicted
probability up. A score of 0.70 from this model is not a 70% chance of being late.

## Result

The final pipeline was fitted on the whole training set and scored once on the 750 test
orders (117 of them late), at threshold 0.70.

|  | Predicted on time | Predicted late |
|---|---|---|
| **Actually on time** | 589 | 44 |
| **Actually late** | 64 | 53 |

| Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|
| 0.546 | 0.453 | 0.495 | 0.821 |

| Policy | Total cost on the test set |
|---|---|
| Flag nobody (current policy) | $1,755 |
| Flag everybody | $3,000 |
| **Model at 0.70** | **$1,348** |

The model saves **$407 (23%)** against flagging nobody.

I trust the direction of this saving more than its exact size: it matches the 24% saving
seen out-of-fold on the training data, but with only 117 late test orders a few
different outcomes would move the figure by tens to hundreds of dollars.

## What the tests cover

`python -m pytest` runs 14 tests in `tests/`, with shared fixtures in `conftest.py`.

- **Feature lists** (`test_pipeline.py`, parametrized): `order_id`, `customer_rating` and
  `late` are never features.
- **Split**: stratified 75/25 (2,250 and 750 rows, same late rate) and identical on
  every run.
- **Pipeline**: predicted probabilities lie between 0 and 1 and sum to 1, and an order
  with a missing `prep_minutes` and an unseen weather value (`"hail"`) still gets a
  prediction.
- **Cost and threshold** (`test_evaluation.py`): small examples worked out by hand,
  including both extremes (a case where flagging nobody is cheapest and one where
  flagging everybody is), and the metrics at a given threshold.
- **Saved model** (`test_saved_model.py`): a pipeline saved with joblib to `tmp_path`
  predicts exactly the same after reloading.

**Not tested:** the full run of `train.py`. It takes about ten seconds and fits many
models, which would make the test suite slow. Instead, its output is committed in
`models/` and `reports/`, and rerunning it reproduces those files exactly.

## Known limitations

- The data is invented by `data.py`, so the result says nothing about real deliveries.
- The test set is small (117 late orders), so the size of the saving is uncertain.
- `prep_minutes` is the recorded preparation time, used as a stand-in for the
  restaurant's estimate. With real estimates the model would likely do a little worse.
- The cost rule assumes a voucher always works and every missed late order costs exactly
  $15. If either changes, the threshold should be chosen again.
- `class_weight="balanced"` makes the probabilities unsuitable to read as real chances.
  With more time I would calibrate them (for example with `CalibratedClassifierCV`) so
  the threshold could be explained directly.
- The joblib file should be loaded only with the scikit-learn version in
  `models/model_info.json`, and only if it comes from a trusted source.
