# src/__init__.py
"""
Система предсказания сельскохозяйственной засухи

Модули:
- data_pipeline: Загрузка и обработка данных
- models: ML/DL модели для предсказания
- utils: Утилиты и вспомогательные функции
"""

__version__ = "2.0.0"
__author__ = "Drought Prediction Team"
__description__ = "ML/AI система для предсказания сельскохозяйственной засухи"

# Основные импорты
from pathlib import Path

# Пути проекта
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Создание необходимых директорий
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / "raw").mkdir(exist_ok=True)
(DATA_DIR / "processed").mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# src/data_pipeline/__init__.py
"""
Пайплайн обработки данных для предсказания засухи

Модули:
- real_data_pipeline: Загрузка реальных данных (CHIRPS, ERA5, MODIS)
- drought_datamodule: PyTorch Lightning DataModule
"""

from .drought_datamodule import DroughtDataModule, DroughtDataset
from .drought_datamodule import create_drought_datamodule, get_datamodule_stats

__all__ = [
    'DroughtDataModule',
    'DroughtDataset', 
    'create_drought_datamodule',
    'get_datamodule_stats'
]



# src/models/__init__.py
"""
Модели машинного обучения для предсказания засухи

Модули:
- classical_models: Классические ML модели (RF, XGBoost, SVM и др.)
- sota_models: SOTA архитектуры (EarthFormer, ConvLSTM, TFT, UNet)
"""

# Классические модели
from .classical_models import (
    ClassicalModelFactory,
    LinearRegressionModel,
    RandomForestModel,
    XGBoostModel,
    EnsembleModel
)

# SOTA модели
from .sota_models import (
    get_model,
    EarthFormerModel,
    ConvLSTMModel,
    TFTModel,
    UNetModel,
    CONFIGS
)

__all__ = [
    # Классические модели
    'ClassicalModelFactory',
    'LinearRegressionModel', 
    'RandomForestModel',
    'XGBoostModel',
    'EnsembleModel',
    
    # SOTA модели
    'get_model',
    'EarthFormerModel',
    'ConvLSTMModel', 
    'TFTModel',
    'UNetModel',
    'CONFIGS'
]



# src/utils/__init__.py
"""
Утилиты для системы предсказания засухи

Модули:
- metrics: Метрики оценки качества моделей
- visualization: Визуализация результатов и данных
"""

from .metrics import DroughtMetrics, MetricsAggregator, RegionalMetrics
from .visualization import (
    DatasetVisualizer,
    TrainingVisualizer, 
    ResultsVisualizer,
    create_comprehensive_report
)

__all__ = [
    # Метрики
    'DroughtMetrics',
    'MetricsAggregator',
    'RegionalMetrics',
    
    # Визуализация
    'DatasetVisualizer',
    'TrainingVisualizer',
    'ResultsVisualizer', 
    'create_comprehensive_report'
]


# configs/__init__.py
"""
Конфигурации для системы предсказания засухи

Файлы:
- config.yaml: Базовая конфигурация
- quick_test.yaml: Быстрый тест
- earthformer.yaml: EarthFormer специализация
- research.yaml: Полное исследование
- production.yaml: Продакшн система
"""

import yaml
from pathlib import Path
from typing import Dict, Any

CONFIGS_DIR = Path(__file__).parent

def load_config(config_name: str) -> Dict[str, Any]:
    """Загрузка конфигурации по имени"""
    config_path = CONFIGS_DIR / f"{config_name}.yaml"
    if not config_path.exists():
        config_path = CONFIGS_DIR / config_name
        if not config_path.exists():
            raise FileNotFoundError(f"Конфигурация не найдена: {config_name}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def list_configs() -> list:
    """Список доступных конфигураций"""
    return [f.stem for f in CONFIGS_DIR.glob("*.yaml")]

__all__ = ['load_config', 'list_configs']