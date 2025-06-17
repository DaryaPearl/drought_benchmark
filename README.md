# 🌾 Drought Prediction Benchmark

This project benchmarks machine learning models for long-term agricultural drought prediction using the SPI-3 index derived from CHIRPS precipitation data.

## 📦 Dataset

- **Source:** CHIRPS v2.0 global precipitation  
- **Target:** SPI-3 index (Standardized Precipitation Index, 3-month scale)  
- **Temporal range:** 2003–2024  
- **Spatial extent:** East Africa, Central Asia, Central America (aggregated)

## 🧠 Models

| Model              | Description              |
|-------------------|--------------------------|
| Linear Regression | Baseline linear model    |
| Random Forest     | Nonlinear ensemble trees |
| XGBoost           | Gradient boosting trees  |

Each model predicts SPI-3 for:
- t + 1 month
- t + 2 months
- t + 3 months

## 🔁 Features

- Uses **12 lagged SPI3 values** as features  
- No external predictors (e.g., NDVI, SMAP) — SPI-only version  
- Outputs **MAE, RMSE, R²** + graphs

## 📊 Output

- Results saved to `results/`
  - `MAE_by_horizon.png`
  - `RMSE_by_horizon.png`
  - `R2_by_horizon.png`

## 🚀 Run

```bash
python src/train.py
