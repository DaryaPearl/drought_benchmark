"""
Классические модели машинного обучения для предсказания засухи

Включает:
- Linear Regression
- Ridge & Lasso Regression  
- Random Forest
- XGBoost & LightGBM
- Support Vector Regression
- Gaussian Process Regression
- Ensemble методы

Использование: from src.models.classical_models import ClassicalModelFactory
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
import joblib
from pathlib import Path
import time
import warnings

# Основные ML библиотеки
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA

# Продвинутые модели
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost не установлен")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM не установлен")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn("CatBoost не установлен")

warnings.filterwarnings("ignore")

class BaseClassicalModel(ABC):
    """Базовый класс для классических моделей"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.is_fitted = False
        self.training_time = 0.0
        self.feature_importance_ = None
        
    @abstractmethod
    def _create_model(self) -> Any:
        """Создание модели"""
        pass
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray = None, 
                       fit_transform: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Предобработка данных"""
        
        # Масштабирование признаков
        if self.config.get('scaling', 'standard') and fit_transform:
            scaler_type = self.config.get('scaling', 'standard')
            if scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif scaler_type == 'minmax':
                self.scaler = MinMaxScaler()
            elif scaler_type == 'robust':
                self.scaler = RobustScaler()
            
            X = self.scaler.fit_transform(X)
        elif self.scaler:
            X = self.scaler.transform(X)
        
        # Отбор признаков
        if self.config.get('feature_selection') and fit_transform and y is not None:
            n_features = self.config.get('n_features', min(50, X.shape[1]))
            method = self.config.get('feature_selection_method', 'f_regression')
            
            if method == 'f_regression':
                self.feature_selector = SelectKBest(f_regression, k=n_features)
            elif method == 'mutual_info':
                self.feature_selector = SelectKBest(mutual_info_regression, k=n_features)
            
            X = self.feature_selector.fit_transform(X, y)
        elif self.feature_selector:
            X = self.feature_selector.transform(X)
        
        return X, y
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseClassicalModel':
        """Обучение модели"""
        start_time = time.time()
        
        # Предобработка
        X_processed, y_processed = self.preprocess_data(X, y, fit_transform=True)
        
        # Создание и обучение модели
        if self.model is None:
            self.model = self._create_model()
        
        self.model.fit(X_processed, y_processed)
        
        # Сохранение важности признаков (если доступно)
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            self.feature_importance_ = np.abs(self.model.coef_)
        
        self.training_time = time.time() - start_time
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание"""
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")
        
        X_processed, _ = self.preprocess_data(X, fit_transform=False)
        return self.model.predict(X_processed)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Получение важности признаков"""
        return self.feature_importance_
    
    def save_model(self, path: str) -> None:
        """Сохранение модели"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'config': self.config,
            'training_time': self.training_time,
            'feature_importance_': self.feature_importance_
        }
        joblib.dump(model_data, path)
    
    @classmethod
    def load_model(cls, path: str) -> 'BaseClassicalModel':
        """Загрузка модели"""
        model_data = joblib.load(path)
        instance = cls(model_data['config'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_selector = model_data['feature_selector']
        instance.training_time = model_data['training_time']
        instance.feature_importance_ = model_data['feature_importance_']
        instance.is_fitted = True
        return instance

class LinearRegressionModel(BaseClassicalModel):
    """Линейная регрессия"""
    
    def _create_model(self):
        return LinearRegression(
            fit_intercept=self.config.get('fit_intercept', True),
            n_jobs=self.config.get('n_jobs', 1)
        )

class RidgeRegressionModel(BaseClassicalModel):
    """Ridge регрессия"""
    
    def _create_model(self):
        return Ridge(
            alpha=self.config.get('alpha', 1.0),
            fit_intercept=self.config.get('fit_intercept', True),
            random_state=self.config.get('random_state', 42)
        )

class LassoRegressionModel(BaseClassicalModel):
    """Lasso регрессия"""
    
    def _create_model(self):
        return Lasso(
            alpha=self.config.get('alpha', 1.0),
            fit_intercept=self.config.get('fit_intercept', True),
            random_state=self.config.get('random_state', 42),
            max_iter=self.config.get('max_iter', 1000)
        )

class ElasticNetModel(BaseClassicalModel):
    """ElasticNet регрессия"""
    
    def _create_model(self):
        return ElasticNet(
            alpha=self.config.get('alpha', 1.0),
            l1_ratio=self.config.get('l1_ratio', 0.5),
            fit_intercept=self.config.get('fit_intercept', True),
            random_state=self.config.get('random_state', 42),
            max_iter=self.config.get('max_iter', 1000)
        )

class RandomForestModel(BaseClassicalModel):
    """Random Forest регрессор"""
    
    def _create_model(self):
        return RandomForestRegressor(
            n_estimators=self.config.get('n_estimators', 100),
            max_depth=self.config.get('max_depth', None),
            min_samples_split=self.config.get('min_samples_split', 2),
            min_samples_leaf=self.config.get('min_samples_leaf', 1),
            max_features=self.config.get('max_features', 'sqrt'),
            bootstrap=self.config.get('bootstrap', True),
            oob_score=self.config.get('oob_score', False),
            random_state=self.config.get('random_state', 42),
            n_jobs=self.config.get('n_jobs', -1)
        )

class ExtraTreesModel(BaseClassicalModel):
    """Extra Trees регрессор"""
    
    def _create_model(self):
        return ExtraTreesRegressor(
            n_estimators=self.config.get('n_estimators', 100),
            max_depth=self.config.get('max_depth', None),
            min_samples_split=self.config.get('min_samples_split', 2),
            min_samples_leaf=self.config.get('min_samples_leaf', 1),
            max_features=self.config.get('max_features', 'sqrt'),
            bootstrap=self.config.get('bootstrap', False),
            random_state=self.config.get('random_state', 42),
            n_jobs=self.config.get('n_jobs', -1)
        )

class GradientBoostingModel(BaseClassicalModel):
    """Gradient Boosting регрессор"""
    
    def _create_model(self):
        return GradientBoostingRegressor(
            n_estimators=self.config.get('n_estimators', 100),
            learning_rate=self.config.get('learning_rate', 0.1),
            max_depth=self.config.get('max_depth', 3),
            subsample=self.config.get('subsample', 1.0),
            max_features=self.config.get('max_features', None),
            random_state=self.config.get('random_state', 42)
        )

class XGBoostModel(BaseClassicalModel):
    """XGBoost регрессор"""
    
    def _create_model(self):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost не установлен. Установите: pip install xgboost")
        
        return xgb.XGBRegressor(
            n_estimators=self.config.get('n_estimators', 100),
            learning_rate=self.config.get('learning_rate', 0.1),
            max_depth=self.config.get('max_depth', 6),
            subsample=self.config.get('subsample', 1.0),
            colsample_bytree=self.config.get('colsample_bytree', 1.0),
            reg_alpha=self.config.get('reg_alpha', 0),
            reg_lambda=self.config.get('reg_lambda', 1),
            random_state=self.config.get('random_state', 42),
            n_jobs=self.config.get('n_jobs', -1),
            tree_method=self.config.get('tree_method', 'auto')
        )

class LightGBMModel(BaseClassicalModel):
    """LightGBM регрессор"""
    
    def _create_model(self):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM не установлен. Установите: pip install lightgbm")
        
        return lgb.LGBMRegressor(
            n_estimators=self.config.get('n_estimators', 100),
            learning_rate=self.config.get('learning_rate', 0.1),
            max_depth=self.config.get('max_depth', -1),
            subsample=self.config.get('subsample', 1.0),
            colsample_bytree=self.config.get('colsample_bytree', 1.0),
            reg_alpha=self.config.get('reg_alpha', 0),
            reg_lambda=self.config.get('reg_lambda', 0),
            random_state=self.config.get('random_state', 42),
            n_jobs=self.config.get('n_jobs', -1),
            verbosity=self.config.get('verbosity', -1)
        )

class CatBoostModel(BaseClassicalModel):
    """CatBoost регрессор"""
    
    def _create_model(self):
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost не установлен. Установите: pip install catboost")
        
        return cb.CatBoostRegressor(
            iterations=self.config.get('iterations', 100),
            learning_rate=self.config.get('learning_rate', 0.1),
            depth=self.config.get('depth', 6),
            l2_leaf_reg=self.config.get('l2_leaf_reg', 3),
            random_state=self.config.get('random_state', 42),
            verbose=self.config.get('verbose', False)
        )

class SVRModel(BaseClassicalModel):
    """Support Vector Regression"""
    
    def _create_model(self):
        return SVR(
            kernel=self.config.get('kernel', 'rbf'),
            C=self.config.get('C', 1.0),
            epsilon=self.config.get('epsilon', 0.1),
            gamma=self.config.get('gamma', 'scale'),
            coef0=self.config.get('coef0', 0.0),
            degree=self.config.get('degree', 3)
        )

class GaussianProcessModel(BaseClassicalModel):
    """Gaussian Process регрессор"""
    
    def _create_model(self):
        # Создание ядра
        kernel_type = self.config.get('kernel', 'RBF')
        if kernel_type == 'RBF':
            kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-3)
        else:
            kernel = None
        
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.config.get('alpha', 1e-10),
            normalize_y=self.config.get('normalize_y', True),
            n_restarts_optimizer=self.config.get('n_restarts_optimizer', 0),
            random_state=self.config.get('random_state', 42)
        )

class EnsembleModel(BaseClassicalModel):
    """Ансамбль из нескольких моделей"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.models = {}
        self.weights = None
        
    def _create_model(self):
        # Создаем отдельные модели для ансамбля
        model_configs = self.config.get('models', {})
        
        for model_name, model_config in model_configs.items():
            if model_config.get('enabled', False):
                self.models[model_name] = self._create_single_model(model_name, model_config)
        
        return self.models
    
    def _create_single_model(self, model_name: str, model_config: Dict):
        """Создание отдельной модели для ансамбля"""
        model_map = {
            'linear_regression': LinearRegressionModel,
            'ridge': RidgeRegressionModel,
            'random_forest': RandomForestModel,
            'xgboost': XGBoostModel,
            'lightgbm': LightGBMModel
        }
        
        if model_name in model_map:
            return model_map[model_name](model_config)
        else:
            raise ValueError(f"Неизвестная модель: {model_name}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleModel':
        """Обучение ансамбля"""
        start_time = time.time()
        
        # Обучаем каждую модель
        for model_name, model in self.models.items():
            print(f"Обучение {model_name}...")
            model.fit(X, y)
        
        # Вычисляем веса на основе производительности валидации
        if self.config.get('weighted', True):
            self._compute_weights(X, y)
        
        self.training_time = time.time() - start_time
        self.is_fitted = True
        
        return self
    
    def _compute_weights(self, X: np.ndarray, y: np.ndarray):
        """Вычисление весов для ансамбля"""
        from sklearn.model_selection import cross_val_score
        
        scores = {}
        for model_name, model in self.models.items():
            # Кросс-валидация для получения оценки качества
            cv_scores = cross_val_score(model.model, X, y, cv=5, scoring='neg_mean_absolute_error')
            scores[model_name] = -cv_scores.mean()  # Конвертируем в положительные значения
        
        # Веса обратно пропорциональны ошибкам
        total_inverse_error = sum(1/score for score in scores.values())
        self.weights = {name: (1/score)/total_inverse_error for name, score in scores.items()}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание ансамбля"""
        if not self.is_fitted:
            raise ValueError("Ансамбль не обучен")
        
        predictions = {}
        for model_name, model in self.models.items():
            predictions[model_name] = model.predict(X)
        
        # Объединение предсказаний
        if self.weights:
            # Взвешенное усреднение
            final_pred = np.zeros_like(list(predictions.values())[0])
            for model_name, pred in predictions.items():
                final_pred += self.weights[model_name] * pred
        else:
            # Простое усреднение
            final_pred = np.mean(list(predictions.values()), axis=0)
        
        return final_pred

class HyperparameterOptimizer:
    """Оптимизатор гиперпараметров для классических моделей"""
    
    def __init__(self, model_class: type, param_grid: Dict, cv: int = 5):
        self.model_class = model_class
        self.param_grid = param_grid
        self.cv = cv
        self.best_model = None
        self.best_params = None
        self.best_score = None
    
    def optimize(self, X: np.ndarray, y: np.ndarray, search_type: str = 'grid') -> BaseClassicalModel:
        """Оптимизация гиперпараметров"""
        
        # Базовая конфигурация
        base_config = {'random_state': 42}
        base_model = self.model_class(base_config)._create_model()
        
        # Выбор метода поиска
        if search_type == 'grid':
            search = GridSearchCV(
                base_model, 
                self.param_grid, 
                cv=TimeSeriesSplit(n_splits=self.cv),
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
        elif search_type == 'random':
            search = RandomizedSearchCV(
                base_model,
                self.param_grid,
                n_iter=50,
                cv=TimeSeriesSplit(n_splits=self.cv),
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                random_state=42
            )
        else:
            raise ValueError("search_type должен быть 'grid' или 'random'")
        
        # Выполнение поиска
        search.fit(X, y)
        
        # Сохранение лучших результатов
        self.best_params = search.best_params_
        self.best_score = -search.best_score_  # Конвертируем обратно в положительное значение
        
        # Создание лучшей модели
        best_config = {**base_config, **self.best_params}
        self.best_model = self.model_class(best_config)
        self.best_model.fit(X, y)
        
        return self.best_model

class ClassicalModelFactory:
    """Фабрика для создания классических моделей"""
    
    MODEL_REGISTRY = {
        'linear_regression': LinearRegressionModel,
        'ridge': RidgeRegressionModel,
        'lasso': LassoRegressionModel,
        'elastic_net': ElasticNetModel,
        'random_forest': RandomForestModel,
        'extra_trees': ExtraTreesModel,
        'gradient_boosting': GradientBoostingModel,
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel,
        'catboost': CatBoostModel,
        'svr': SVRModel,
        'gaussian_process': GaussianProcessModel,
        'ensemble': EnsembleModel
    }
    
    @classmethod
    def create_model(cls, model_name: str, config: Dict[str, Any]) -> BaseClassicalModel:
        """Создание модели по имени"""
        if model_name not in cls.MODEL_REGISTRY:
            raise ValueError(f"Неизвестная модель: {model_name}. "
                           f"Доступные: {list(cls.MODEL_REGISTRY.keys())}")
        
        model_class = cls.MODEL_REGISTRY[model_name]
        return model_class(config)
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Получение списка доступных моделей"""
        return list(cls.MODEL_REGISTRY.keys())
    
    @classmethod
    def create_ensemble(cls, model_configs: Dict[str, Dict]) -> EnsembleModel:
        """Создание ансамбля моделей"""
        ensemble_config = {'models': model_configs}
        return EnsembleModel(ensemble_config)

# Предопределенные конфигурации моделей
DEFAULT_CONFIGS = {
    'linear_regression': {
        'fit_intercept': True,
        'scaling': 'standard'
    },
    'ridge': {
        'alpha': 1.0,
        'fit_intercept': True,
        'scaling': 'standard'
    },
    'random_forest': {
        'n_estimators': 300,
        'max_depth': 20,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42,
        'n_jobs': -1,
        'scaling': None  # RF не требует масштабирования
    },
    'xgboost': {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'scaling': None
    },
    'ensemble': {
        'models': {
            'random_forest': {
                'enabled': True,
                'n_estimators': 300,
                'max_depth': 20,
                'random_state': 42,
                'n_jobs': -1
            },
            'xgboost': {
                'enabled': True,
                'n_estimators': 500,
                'learning_rate': 0.05,
                'max_depth': 8,
                'random_state': 42,
                'n_jobs': -1
            }
        },
        'weighted': True
    }
}

def get_default_config(model_name: str) -> Dict[str, Any]:
    """Получение конфигурации по умолчанию для модели"""
    return DEFAULT_CONFIGS.get(model_name, {})

# Вспомогательные функции
def evaluate_model(model: BaseClassicalModel, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Оценка качества модели"""
    y_pred = model.predict(X_test)
    
    return {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'training_time': model.training_time
    }

def compare_models(models: Dict[str, BaseClassicalModel], 
                  X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """Сравнение нескольких моделей"""
    results = []
    
    for model_name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        metrics['model'] = model_name
        results.append(metrics)
    
    return pd.DataFrame(results).set_index('model')

if __name__ == "__main__":
    # Пример использования
    print("🤖 Классические модели машинного обучения для предсказания засухи")
    print(f"📊 Доступные модели: {ClassicalModelFactory.get_available_models()}")
    
    # Пример создания модели
    config = get_default_config('random_forest')
    model = ClassicalModelFactory.create_model('random_forest', config)
    print(f"✅ Создана модель Random Forest с конфигурацией: {config}")