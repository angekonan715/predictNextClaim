# Predicting Next-Year Car Insurance Claims

**Senior project · Data Science major, Statistics minor**

Which drivers are most likely to file a car insurance claim next year? This project estimates the probability that a policy has at least one claim, using three years of real, anonymised policy data from a Spanish motor insurer (2022–2024). It compares a formula (GLM), a logistic regression, a decision tree, a random forest, XGBoost, and LightGBM. LightGBM is the model in the web app.

> 🚧 **Status:** in progress. Results and app link will be added as the project develops.

---

## Live app

🔗 _Link coming soon (Streamlit Community Cloud)_

<!-- Add a screenshot once the app is running:
![App screenshot](reports/figures/app_screenshot.png)
-->

---

## Data

- **Source:** Espinosa, P., Lledó, J., & Atance del Olmo, D. (2026). *A detailed dataset of motor insurance policies with coverage-specific financial information.* Mendeley Data, V1. https://doi.org/10.17632/sw4jmdb2sm.1
- **Coverage:** anonymised policies from a Spanish insurer, January 2022 – December 2024
- **Size:** about 354,000 policy-year observations and 47 variables (policy, driver, vehicle, premiums, exposure, claims by coverage)

The raw data is **not** included in this repository. See [`data/README.md`](data/README.md) for download instructions.

---

## Approach

| Step | What | Where |
|---|---|---|
| 1. Data prep | Clean data, derive ages from dates, define the claim target | `src/data_prep.py`, `notebooks/01_eda.ipynb` |
| 2. Exploration | Exposure-weighted claim rates by risk factor | `notebooks/01_eda.ipynb` |
| 3. Modeling | GLM, logistic regression, decision tree, random forest, XGBoost, LightGBM | `src/train.py`, `notebooks/02_modeling.ipynb` |
| 4. Evaluation | AUC, Gini, calibration, and risk tiers on 2024 | `src/evaluate.py`, `notebooks/03_evaluation.ipynb` |
| 5. App | Streamlit form → predicted claim probability and risk tier | `app/app.py` |

**Validation design:** models are trained on 2022–2023 and tested on 2024 (out-of-time), which mirrors how an insurer would actually use the model and avoids leakage from the same policy appearing in both sets.

---

## Key results

Trained on 2022–2023, scored on 2024. Higher AUC and Gini mean better ranking. A smaller calibration gap means the predicted chance is closer to the claim rate that actually happened.

| Model | Test AUC | Gini | Calibration gap |
|---|---|---|---|
| GLM | 0.643 | 0.286 | 0.005 |
| Logistic regression | 0.642 | 0.283 | 0.010 |
| Decision tree | 0.638 | 0.275 | 0.006 |
| Random forest | 0.648 | 0.295 | 0.004 |
| XGBoost | 0.648 | 0.295 | 0.004 |
| **LightGBM** | **0.655** | **0.310** | **0.003** |

LightGBM is the saved model. It adds hundreds of small trees, and each new tree corrects the policies the earlier trees got wrong. The GLM is still the model to read when you want one percentage effect: a typical gap in vehicle value multiplies the claim rate by about 1.16, and a typical gap in vehicle age multiplies it by about 0.82.

---

## Repository structure

```
predictNextClaim/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/              # downloaded dataset (not committed)
│   ├── processed/        # cleaned data (not committed)
│   └── README.md         # how to get the data
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── config.py         # paths, column names, settings
│   ├── data_prep.py      # cleaning, features, target, year split
│   ├── train.py          # fits and saves models
│   ├── evaluate.py       # Gini, lift, calibration helpers
│   └── predict.py        # loads the model, returns a probability
├── models/               # saved model (claim_model.joblib)
├── app/
│   └── app.py            # Streamlit UI
└── reports/
    └── figures/
```

---

## How to run

```bash
# 1. Set up the environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Put the downloaded dataset in data/raw/ (see data/README.md)

# 3. Prepare data and train
python -m src.data_prep
python -m src.train

# 4. Launch the app
streamlit run app/app.py
```

---

## Author

**Ange Konan** · [GitHub](https://github.com/angekonan715)

_This is an educational project. Predictions are not insurance quotes._
