"Random‑Forest wrapper for Lightning"
import pytorch_lightning as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch, numpy as np

class RandomForestModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = RandomForestRegressor(**cfg.params)

    def training_step(self, batch, batch_idx):
        X, y = batch
        X_np, y_np = X.numpy(), y.numpy()
        self.model.partial_fit(X_np, y_np)
        loss = torch.tensor(0.0)  # placeholder
        return loss

    def predict_step(self, batch, batch_idx):
        X, _ = batch
        return torch.tensor(self.model.predict(X.numpy()))

    def test_step(self, batch, batch_idx):
        X, y = batch
        preds = torch.tensor(self.model.predict(X.numpy()))
        mae = mean_absolute_error(y.numpy(), preds.numpy())
        rmse = mean_squared_error(y.numpy(), preds.numpy(), squared=False)
        self.log_dict({"test_mae": mae, "test_rmse": rmse})
