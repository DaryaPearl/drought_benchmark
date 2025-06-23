"""
SOTA модели для предсказания засухи - минимальная реализация
Сохраните как: src/models/sota_models.py
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any

class ConvLSTMModel(pl.LightningModule):
    """Простая ConvLSTM модель"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Простая архитектура
        self.conv1 = nn.Conv2d(config.get('input_channels', 5), 32, 3, padding=1)
        self.lstm = nn.LSTM(32, config.get('hidden_size', 64), batch_first=True)
        self.output_layer = nn.Linear(config.get('hidden_size', 64), 1)
        
    def forward(self, x):
        # x: (batch, time, channels, height, width)
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Процессируем каждый временной шаг
        conv_out = []
        for t in range(seq_len):
            conv_t = torch.relu(self.conv1(x[:, t]))  # (batch, 32, H, W)
            conv_t = conv_t.mean(dim=[2, 3])  # Global average pooling -> (batch, 32)
            conv_out.append(conv_t)
        
        # Стек для LSTM
        conv_seq = torch.stack(conv_out, dim=1)  # (batch, seq_len, 32)
        
        # LSTM
        lstm_out, _ = self.lstm(conv_seq)  # (batch, seq_len, hidden_size)
        
        # Выход
        output = self.output_layer(lstm_out[:, -1])  # Последний временной шаг
        
        return output
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # Приводим к одному размеру
        if y.dim() > y_hat.dim():
            y = y.view(y_hat.shape)
        
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        if y.dim() > y_hat.dim():
            y = y.view(y_hat.shape)
            
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.get('learning_rate', 0.001))

class EarthFormerModel(pl.LightningModule):
    """Заглушка для EarthFormer"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Простая замена
        self.backbone = nn.Sequential(
            nn.Conv3d(config.get('input_channels', 5), 64, (2, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(64, config.get('output_size', 1))
        )
        
    def forward(self, x):
        return self.backbone(x)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        # Добавляем канальное измерение если нужно
        if x.dim() == 4:  # (batch, time, H, W)
            x = x.unsqueeze(2)  # (batch, time, 1, H, W)
        
        y_hat = self(x)
        if y.dim() > y_hat.dim():
            y = y.view(y_hat.shape)
            
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        if x.dim() == 4:
            x = x.unsqueeze(2)
            
        y_hat = self(x)
        if y.dim() > y_hat.dim():
            y = y.view(y_hat.shape)
            
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.get('learning_rate', 0.001))

class TFTModel(pl.LightningModule):
    """Заглушка для Temporal Fusion Transformer"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        hidden_size = config.get('hidden_size', 128)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, nhead=4, batch_first=True),
            num_layers=2
        )
        self.input_projection = nn.Linear(config.get('input_size', 64), hidden_size)
        self.output_layer = nn.Linear(hidden_size, config.get('output_size', 1))
        
    def forward(self, x):
        # Flatten spatial dimensions
        batch_size, seq_len = x.size(0), x.size(1)
        x_flat = x.view(batch_size, seq_len, -1)
        
        # Project to hidden size
        x_proj = self.input_projection(x_flat)
        
        # Transformer
        encoded = self.encoder(x_proj)
        
        # Output
        output = self.output_layer(encoded[:, -1])  # Last timestep
        
        return output
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        if y.dim() > y_hat.dim():
            y = y.view(y_hat.shape)
            
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        if y.dim() > y_hat.dim():
            y = y.view(y_hat.shape)
            
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.get('learning_rate', 0.001))

class UNetModel(pl.LightningModule):
    """Простая U-Net для пространственно-временных данных"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Простая U-Net архитектура
        self.encoder = nn.Sequential(
            nn.Conv2d(config.get('input_channels', 5), 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, config.get('output_size', 1))
        )
        
    def forward(self, x):
        # Обрабатываем последний временной шаг
        if x.dim() == 5:  # (batch, time, channels, H, W)
            x = x[:, -1]  # Берем последний временной шаг
        
        encoded = self.encoder(x)
        output = self.decoder(encoded)
        
        return output
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        if y.dim() > y_hat.dim():
            y = y.view(y_hat.shape)
            
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        if y.dim() > y_hat.dim():
            y = y.view(y_hat.shape)
            
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.get('learning_rate', 0.001))

# Фабрика моделей
def get_model(model_name: str, config: Dict[str, Any]):
    """Создание модели по имени"""
    model_map = {
        'convlstm': ConvLSTMModel,
        'earthformer': EarthFormerModel,
        'tft': TFTModel,
        'unet': UNetModel
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_map.keys())}")
    
    return model_map[model_name](config)

# Конфигурации по умолчанию
CONFIGS = {
    'convlstm': {
        'input_channels': 5,
        'hidden_size': 64,
        'learning_rate': 0.001
    },
    'earthformer': {
        'input_channels': 5,
        'output_size': 1,
        'learning_rate': 0.0001
    },
    'tft': {
        'input_size': 64,
        'hidden_size': 128,
        'output_size': 1,
        'learning_rate': 0.001
    },
    'unet': {
        'input_channels': 5,
        'output_size': 1,
        'learning_rate': 0.001
    }
}