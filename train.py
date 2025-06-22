import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

# ────────────────────────
# Настройки
LAGS = 12
HORIZONS = [1, 2, 3]  # t+1, t+2, t+3 месяцев
TRAIN_YEARS = (2003, 2016)
TEST_YEARS  = (2020, 2024)
DATA_PATH = "data/processed/agro_cube.zarr"
SAVE_DIR = Path("results")
SAVE_DIR.mkdir(exist_ok=True)
# ────────────────────────

def load_spi_cube(path=DATA_PATH):
    print("📦 Загружаю SPI3 из Zarr...")
    ds = xr.open_zarr(path)
    return ds["spi3"]

def make_lagged_data(spi, start_year, end_year, horizon):
    df = spi.stack(grid=("latitude", "longitude")).transpose("time", "grid").to_pandas()
    df.index = spi.time.values
    df = df.loc[f"{start_year}-01-01":f"{end_year}-12-31"]

    X, y = [], []
    for t in range(LAGS, len(df) - horizon):
        X.append(df.iloc[t - LAGS:t].values.T)            # (pixels, lags)
        y.append(df.iloc[t + horizon].values)             # target через h месяцев
    X = np.concatenate(X)
    y = np.concatenate(y)

    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    return X[mask], y[mask]

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2   = r2_score(y_test, y_pred)

    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}

if __name__ == "__main__":
    spi = load_spi_cube()

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=20, n_jobs=-1, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05, n_jobs=-1, random_state=42)
    }

    results = []

    for horizon in HORIZONS:
        print(f"\n🕒 Прогноз на t+{horizon} месяцев:")

        X_train, y_train = make_lagged_data(spi, *TRAIN_YEARS, horizon)
        X_test,  y_test  = make_lagged_data(spi, *TEST_YEARS,  horizon)

        for name, model in models.items():
            scores = evaluate_model(name, model, X_train, y_train, X_test, y_test)
            scores["horizon"] = horizon
            results.append(scores)
            print(f"🔹 {name:<15}  MAE={scores['MAE']:.4f}  RMSE={scores['RMSE']:.4f}  R²={scores['R2']:.4f}")

    # 📊 Графики
    df = pd.DataFrame(results)

    for metric in ["MAE", "RMSE", "R2"]:
        plt.figure(figsize=(7, 4))
        for name in df["model"].unique():
            subset = df[df["model"] == name]
            plt.plot(subset["horizon"], subset[metric], label=name, marker="o")
        plt.title(f"{metric} by Forecast Horizon")
        plt.xlabel("Months Ahead (t+h)")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(SAVE_DIR / f"{metric}_by_horizon.png")
        plt.close()

    print(f"\n✅ Графики сохранены в: {SAVE_DIR.resolve()}")