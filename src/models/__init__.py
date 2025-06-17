from .rf import RandomForestModel
try:
    from .tft import TFTModel
except ImportError:
    TFTModel = None  # optional

def get_model(cfg):
    name = cfg.name.lower()
    if name == "rf":
        return RandomForestModel(cfg)
    if name == "tft":
        if TFTModel is None:
            raise ImportError("TFT dependencies missing")
        return TFTModel(cfg)
    raise ValueError(f"Unknown model: {name}")
