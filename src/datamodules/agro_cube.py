"AgroDataModule: wraps xarray Zarr cube for PyTorch Lightning"
from __future__ import annotations
import pytorch_lightning as pl
import xarray as xr
import torch
from torch.utils.data import DataLoader, TensorDataset

class AgroDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ds: xr.Dataset | None = None

    def setup(self, stage=None):
        self.ds = xr.open_zarr(self.cfg.path)  # path to agro_cube.zarr

    def _split(self, years):
        ds_years = self.ds.sel(time=self.ds.time.dt.year.isin(years))
        X = torch.tensor(ds_years[self.cfg.features].to_array().values)
        y = torch.tensor(ds_years[self.cfg.target].values)
        return TensorDataset(X, y)

    def train_dataloader(self):
        return DataLoader(self._split(self.cfg.train_years), batch_size=self.cfg.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self._split(self.cfg.val_years), batch_size=self.cfg.batch_size)

    def test_dataloader(self):
        return DataLoader(self._split(self.cfg.test_years), batch_size=self.cfg.batch_size)
