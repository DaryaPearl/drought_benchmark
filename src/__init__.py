# src/__init__.py
"""
Система предсказания сельскохозяйственной засухи
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