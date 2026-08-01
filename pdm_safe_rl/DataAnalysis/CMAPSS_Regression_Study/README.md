# C-MAPSS Regression Study

## Overview

This project compares multiple regression models for predicting the Remaining Useful Life (RUL) of turbofan aircraft engines using the NASA C-MAPSS dataset.

The study was developed as part of the course requirements for **IE 8623 – Advanced Data Analytics for Complex Systems** and also supports my ongoing PhD research on **uncertainty-aware reinforcement learning for predictive maintenance**.

The objective is to evaluate traditional regression methods and identify a suitable predictive model that can later be integrated into a reinforcement learning framework for maintenance decision-making.

---

## Dataset

- Dataset: NASA C-MAPSS FD001
- Response Variable: Remaining Useful Life (RUL)
- Predictors:
  - Operating cycle
  - Three operational settings
  - Twenty-one sensor measurements

---

## Regression Models

The following models are evaluated:

- Multiple Linear Regression
- Elastic Net Regression
- Principal Component Regression (PCR)
- Partial Least Squares (PLS)

---

## Performance Metrics

Models are compared using:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score
- Training Time
- Prediction Time

---

## Project Structure

```
CMAPSS_Regression_Study/
├── datasets/
├── notebooks/
├── scripts/
├── outputs/
│   ├── figures/
│   └── tables/
├── report/
├── README.md
└── requirements.txt
```

---

## Running the Project

1. Place the NASA C-MAPSS dataset in the appropriate data directory.
2. Open `notebooks/Regression_Comparison.ipynb`.
3. Run the notebook to:
   - Load and preprocess the dataset
   - Train all regression models
   - Compare model performance
   - Generate figures and tables

---

## Future Work

The regression models developed in this study will serve as the predictive component of a broader uncertainty-aware reinforcement learning framework for predictive maintenance.