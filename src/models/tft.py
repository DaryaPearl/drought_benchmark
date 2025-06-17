"Temporal Fusion Transformer wrapper"
import pytorch_lightning as pl
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet

class TFTModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tft: TemporalFusionTransformer | None = None

    def setup(self, stage=None):
        # build TimeSeriesDataSet from DataModule's dataset
        dm = self.trainer.datamodule
        ds = dm.ds  # type: ignore
        ts = TimeSeriesDataSet.from_xarray(
            ds,
            time_idx="time",
            target=self.cfg.target,
            group_ids=["lat", "lon"],
            max_prediction_length=self.cfg.max_pred_len,
            max_encoder_length=self.cfg.max_enc_len,
        )
        self.tft = TemporalFusionTransformer.from_dataset(ts, **self.cfg.params)

    def forward(self, x):
        return self.tft(x)

    def configure_optimizers(self):
        return self.tft.configure_optimizers()
