# House Price Predictor

## Overview
This app predicts house prices based on various features such as overall quality, living area, basement area, garage capacity, and year built. It uses machine learning models to provide accurate price estimates for houses.

## Technical Highlights
- Performed feature engineering and categorical encoding to expand the dataset from 81 original columns (including `Id` and `SalePrice`) to 229 model-ready features.
- Analyzed feature importance to understand which variables most strongly influence house price prediction and which contribute little.
- Trained and compared multiple models: Linear Regression, LightGBM, and XGBoost.
- Used SHAP to interpret model behavior and identify the most impactful features for prediction.
- Applied GridSearchCV to tune XGBoost hyperparameters and improve performance.
- Observed that Linear Regression underperformed because many relationships in the dataset are non-linear.

## Model Comparison
| Model | RMSE |
|-------|------|
| Linear Regression | $52,720 |
| LightGBM | $29,952 |
| XGBoost | $25,278 |
| XGBoost (Tuned) | $24,027 |

## MLOps
The project ensures full ML lifecycle reliability by using MLflow to track models and their training data, GitHub Actions to automatically detect and prevent broken code, and drift detection to continuously monitor performance and prevent silent model degradation as data evolves.

- **Experiment Tracking:** All 4 models are logged with full hyperparameters and metrics using MLflow. Runs are comparable side by side in the MLflow UI.
- **Model Registry:** The best model (Tuned XGBoost) is registered in the MLflow Model Registry as `house-price-predictor v1`.
- **CI Pipeline:** GitHub Actions automatically runs the test suite on every push to `main`.
- **Drift Detection:** The KS test compares training data distributions against new incoming data and flags features that have drifted significantly.

## How to Run Locally

1. Clone the repository
```bash
   git clone https://github.com/Nikhil-Vaishnav-17/house-price-predictor.git
   cd house-price-ml
```

2. Create and activate virtual environment
```bash
   uv venv
   source .venv/bin/activate  # Mac/Linux
   .venv\Scripts\activate     # Windows
```

3. Install dependencies
```bash
   uv sync
```

4. Run the app
```bash
   python app.py
```

## Run Tests Locally
```bash
uv run pytest test_pipeline.py -v
```

## MLflow Experiment Tracking

1. Start the MLflow UI
```bash
uv run mlflow ui
```

2. Open `http://127.0.0.1:5000` in your browser

3. Run `mlflow_tracking.ipynb` to log all model runs

## Drift Detection

Run `drift_detection.ipynb` after placing your dataset in the `data/` directory. The notebook uses the Kolmogorov-Smirnov test to compare training vs. new data distributions and flags features that have drifted.

## Live Demo
You can access the live demo of the House Price Predictor app [here](https://huggingface.co/spaces/Nikhil-Vaishnav-17/house-price-predictor).

## Dataset
[Kaggle House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)