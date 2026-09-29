# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A hands-on tutorial repo about credit-card approval risk modeling (Kaggle dataset `rikdifos/credit-card-approval-prediction`), built for students of FIAP's Tech Challenge Fase 2. It is a sequence of standalone, heavily-commented Jupyter notebooks — meant to be read like a guided essay, not just executed. Portuguese is the language throughout (markdown, comments, variable names). See `README.md` for the full narrative and headline findings; don't repeat that content here, just know it exists and is the source of truth for "why".

## Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Versions in `requirements.txt` are pinned deliberately to **current/latest** releases (e.g. `pandas==3.0.6`, `numpy==2.5.3`) — this is intentionally different from the older pins used elsewhere in the parent FIAP coursework repo (`pandas<2.0.0`), so don't "fix" it to match. Notebooks 2 and 3 need `imbalanced-learn`; notebook 3 also needs `xgboost`.

Raw CSVs (`application_record.csv`, `credit_record.csv`) are gitignored and must be placed in `data/raw/`, either via `notebooks/00_bootstrap_dados.ipynb` (downloads through `kagglehub`, requires Kaggle API credentials) or manual download from Kaggle.

## Notebook sequence (read/run in this order)

Each notebook assumes it runs from `notebooks/` with data in `../data/raw/`, and is designed to execute top-to-bottom with a restarted kernel. `RANDOM_STATE = 42` is fixed throughout for reproducibility (note in notebook 2: with `n_jobs=-1`, classification metrics can vary in the 2nd/3rd decimal across machines due to floating-point aggregation order across threads — ranking metrics and conclusions do not change).

0. **`00_bootstrap_dados.ipynb`** — one-time data download via `kagglehub` into `data/raw/`. Optional/idempotent.
1. **`demo_criacao_target.ipynb`** — builds TARGET from `credit_record.csv`'s `STATUS`/`MONTHS_BALANCE` history, comparing a permissive threshold (30+ days late) vs. a strict one (60+ days late). The core thesis: the TARGET definition is a business decision, and class balance is a *consequence* of it, never the criterion for choosing it. Surfaces the exposure bias of "ever late" aggregation, the inner-join selection bias against `application_record.csv`, and duplicate profiles in the data.
2. **`demo_modelagem_desbalanceada.ipynb`** — takes the strict TARGET as given and runs a full imbalanced-modeling pipeline: minimal preprocessing (see feature engineering below), **group-aware** train/test split and cross-validation (so identical-profile "twins" don't leak between train and test), baseline vs. `class_weight="balanced"` vs. SMOTENC, hyperparameter search across multiple scoring metrics (f1, recall, average precision), and decision-threshold calibration. Punchline: honest evaluation shows applicant profile data has almost no predictive power for 60+ day delinquency, and no balancing technique changes that.
3. **`demo_tecnicas_avancadas.ipynb`** — consolidates the problems found along the way (duplicate profiles/leakage, near-zero univariate AUC per attribute, exposure bias, CV optimism with few positives) and pivots to where signal actually lives: payment *behavior*. Reframes the problem as behavioral scoring with fixed observation/performance windows and rolling snapshots, evaluated the way a credit risk team would (capture rate among riskiest customers), not by F1 at a 0.5 threshold.

## Key architectural patterns (span multiple notebooks)

- **Group-aware splitting is load-bearing, not incidental.** ~91% of clients have an identical-profile "twin" under a different `ID` (same income, birth date in days, employment tenure in days, etc.) — a random split leaks twins across train/test and inflates ROC-AUC by ~0.10–0.18. Every split and every CV fold from notebook 2 onward groups by profile identity, not by `ID`.
- **Feature engineering for the applicant table** (repeated/refined across notebooks 2–3, notably in `codificar_cadastro()` in notebook 3): `DAYS_BIRTH` → `idade_anos`; `DAYS_EMPLOYED`'s sentinel value `365243` → `flag_sem_emprego_ativo` + `anos_empregado`; `FLAG_MOBIL` dropped (constant); binary categoricals mapped to 0/1; multi-class categoricals one-hot encoded via `pd.get_dummies` (encoded before the split — acceptable here since no column depends on TARGET, but noted in-notebook as a fragility since it couples train/test vocabulary).
- **Threshold selection is always out-of-fold, never on the test set** — see `limiar_f1_out_of_fold()` and `avaliar_honesto()` in notebook 3. This is the pattern to follow for any new evaluation code: pick thresholds from CV/OOF predictions, report test metrics only for the final honest read.
- **Behavioral scoring construction** (notebook 3, final section): 6-month observation window (features: fraction of months late, 30+ day lateness, worst status, on-time fraction, trend, last-month status) → 12-month performance window (target: reaches 60+ days late) → rolling snapshots every 3 months to multiply examples without leaking time → same group-based split discipline applies to snapshots too, since neighboring snapshots of one account overlap.

## Working conventions

- No test suite / linter — this is educational notebook content, correctness is judged by re-running end-to-end with a restarted kernel and checking the reported numbers match the narrative.
- When editing a notebook, preserve the "show uncomfortable results" tone: if a technique doesn't work or a metric is weak, that's discussed in-text, not hidden or smoothed over.
- Prefer extending the existing sequence (or adding a new notebook) over restructuring the existing four — the README's numbered list and cross-notebook references (`demo_X.ipynb` mentioned by name inside other notebooks) depend on filenames staying stable.
