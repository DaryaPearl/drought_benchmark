"""
Метрики и утилиты для оценки моделей предсказания засухи

Включает:
- Регрессионные метрики (MAE, RMSE, R², MAPE)
- Метрики классификации засухи
- Пространственно-временные метрики
- Климатологические метрики
- Экспертные метрики для засухи

Использование: from src.utils.metrics import DroughtMetrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report
)
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
import warnings

warnings.filterwarnings("ignore")

class DroughtMetrics:
    """Комплексный набор метрик для оценки предсказания засухи"""
    
    # Пороги засухи по SPI
    DROUGHT_THRESHOLDS = {
        'no_drought': (-0.5, 0.5),
        'mild_drought': (-1.0, -0.5),
        'moderate_drought': (-1.5, -1.0),
        'severe_drought': (-2.0, -1.5),
        'extreme_drought': (-np.inf, -2.0)
    }
    
    @staticmethod
    def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Основные регрессионные метрики"""
        
        # Удаляем NaN значения
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {name: np.nan for name in ['mae', 'rmse', 'r2', 'mape', 'smape', 'mase']}
        
        # Основные метрики
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        r2 = r2_score(y_true_clean, y_pred_clean)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / (y_true_clean + 1e-8))) * 100
        
        # SMAPE (Symmetric MAPE)
        smape = np.mean(2 * np.abs(y_pred_clean - y_true_clean) / 
                       (np.abs(y_true_clean) + np.abs(y_pred_clean) + 1e-8)) * 100
        
        # MASE (Mean Absolute Scaled Error)
        # Используем naive forecast (последнее значение) как базовую линию
        if len(y_true_clean) > 1:
            naive_errors = np.abs(y_true_clean[1:] - y_true_clean[:-1])
            mase = mae / (np.mean(naive_errors) + 1e-8)
        else:
            mase = np.nan
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape),
            'smape': float(smape),
            'mase': float(mase)
        }
    
    @staticmethod
    def drought_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Метрики классификации засухи"""
        
        # Преобразуем в категории засухи
        y_true_cat = DroughtMetrics._spi_to_drought_category(y_true)
        y_pred_cat = DroughtMetrics._spi_to_drought_category(y_pred)
        
        # Удаляем NaN
        mask = ~(pd.isna(y_true_cat) | pd.isna(y_pred_cat))
        y_true_clean = y_true_cat[mask]
        y_pred_clean = y_pred_cat[mask]
        
        if len(y_true_clean) == 0:
            return {name: np.nan for name in ['accuracy', 'precision', 'recall', 'f1']}
        
        # Метрики классификации
        accuracy = accuracy_score(y_true_clean, y_pred_clean)
        
        # Для мультиклассовой классификации используем macro average
        precision = precision_score(y_true_clean, y_pred_clean, average='macro', zero_division=0)
        recall = recall_score(y_true_clean, y_pred_clean, average='macro', zero_division=0)
        f1 = f1_score(y_true_clean, y_pred_clean, average='macro', zero_division=0)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    
    @staticmethod
    def drought_detection_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                threshold: float = -1.0) -> Dict[str, float]:
        """Метрики детекции засухи (бинарная классификация)"""
        
        # Бинарная классификация: засуха/не засуха
        drought_true = y_true < threshold
        drought_pred = y_pred < threshold
        
        # Удаляем NaN
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        drought_true_clean = drought_true[mask]
        drought_pred_clean = drought_pred[mask]
        
        if len(drought_true_clean) == 0:
            return {name: np.nan for name in ['pod', 'far', 'csi', 'bias']}
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(drought_true_clean, drought_pred_clean).ravel()
        
        # Метрики детекции
        pod = tp / (tp + fn) if (tp + fn) > 0 else 0  # Probability of Detection
        far = fp / (tp + fp) if (tp + fp) > 0 else 0  # False Alarm Ratio
        csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0  # Critical Success Index
        bias = (tp + fp) / (tp + fn) if (tp + fn) > 0 else 0  # Bias score
        
        return {
            'pod': float(pod),          # Probability of Detection
            'far': float(far),          # False Alarm Ratio
            'csi': float(csi),          # Critical Success Index
            'bias': float(bias)         # Bias score
        }
    
    @staticmethod
    def spatial_correlation(y_true: np.ndarray, y_pred: np.ndarray, 
                          coords: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Пространственные корреляции"""
        
        if y_true.ndim == 3:  # (time, lat, lon)
            # Средняя корреляция по времени
            correlations = []
            for t in range(y_true.shape[0]):
                true_flat = y_true[t].flatten()
                pred_flat = y_pred[t].flatten()
                
                mask = ~(np.isnan(true_flat) | np.isnan(pred_flat))
                if mask.sum() > 10:  # Минимум точек для корреляции
                    corr, _ = pearsonr(true_flat[mask], pred_flat[mask])
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            spatial_corr = np.mean(correlations) if correlations else np.nan
            
        else:  # Плоский массив
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            if mask.sum() > 10:
                spatial_corr, _ = pearsonr(y_true[mask], y_pred[mask])
            else:
                spatial_corr = np.nan
        
        return {'spatial_correlation': float(spatial_corr)}
    
    @staticmethod
    def temporal_consistency(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Временная согласованность предсказаний"""
        
        if y_true.ndim == 1:
            # Одномерный временной ряд
            true_series = y_true
            pred_series = y_pred
        else:
            # Многомерный: усредняем по пространству
            true_series = np.nanmean(y_true.reshape(y_true.shape[0], -1), axis=1)
            pred_series = np.nanmean(y_pred.reshape(y_pred.shape[0], -1), axis=1)
        
        # Удаляем NaN
        mask = ~(np.isnan(true_series) | np.isnan(pred_series))
        true_clean = true_series[mask]
        pred_clean = pred_series[mask]
        
        if len(true_clean) < 3:
            return {'temporal_correlation': np.nan, 'trend_agreement': np.nan}
        
        # Временная корреляция
        temp_corr, _ = pearsonr(true_clean, pred_clean)
        
        # Согласие трендов (корреляция первых разностей)
        true_diff = np.diff(true_clean)
        pred_diff = np.diff(pred_clean)
        
        if len(true_diff) > 0:
            trend_corr, _ = pearsonr(true_diff, pred_diff)
        else:
            trend_corr = np.nan
        
        return {
            'temporal_correlation': float(temp_corr),
            'trend_agreement': float(trend_corr)
        }
    
    @staticmethod
    def extreme_events_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                             extreme_threshold: float = -2.0) -> Dict[str, float]:
        """Метрики для экстремальных событий засухи"""
        
        # Выделяем экстремальные события
        extreme_true = y_true < extreme_threshold
        extreme_pred = y_pred < extreme_threshold
        
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        extreme_true_clean = extreme_true[mask]
        extreme_pred_clean = extreme_pred[mask]
        
        if len(extreme_true_clean) == 0 or extreme_true_clean.sum() == 0:
            return {name: np.nan for name in ['extreme_pod', 'extreme_far', 'extreme_intensity_error']}
        
        # Метрики детекции экстремальных событий
        tp = (extreme_true_clean & extreme_pred_clean).sum()
        fp = (~extreme_true_clean & extreme_pred_clean).sum()
        fn = (extreme_true_clean & ~extreme_pred_clean).sum()
        
        extreme_pod = tp / (tp + fn) if (tp + fn) > 0 else 0
        extreme_far = fp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Ошибка интенсивности экстремальных событий
        extreme_mask = extreme_true_clean
        if extreme_mask.sum() > 0:
            true_extreme_values = y_true[mask][extreme_mask]
            pred_extreme_values = y_pred[mask][extreme_mask]
            intensity_error = np.mean(np.abs(true_extreme_values - pred_extreme_values))
        else:
            intensity_error = np.nan
        
        return {
            'extreme_pod': float(extreme_pod),
            'extreme_far': float(extreme_far),
            'extreme_intensity_error': float(intensity_error)
        }
    
    @staticmethod
    def climatological_skill(y_true: np.ndarray, y_pred: np.ndarray,
                           climatology: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Климатологический скилл (по сравнению с климатической нормой)"""
        
        if climatology is None:
            # Используем среднее как климатологию
            climatology = np.full_like(y_true, np.nanmean(y_true))
        
        # Удаляем NaN
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(climatology))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        clim_clean = climatology[mask]
        
        if len(y_true_clean) == 0:
            return {'skill_score': np.nan, 'anomaly_correlation': np.nan}
        
        # Skill score (улучшение относительно климатологии)
        mse_pred = np.mean((y_true_clean - y_pred_clean) ** 2)
        mse_clim = np.mean((y_true_clean - clim_clean) ** 2)
        
        skill_score = 1 - (mse_pred / mse_clim) if mse_clim > 0 else np.nan
        
        # Корреляция аномалий
        true_anomaly = y_true_clean - clim_clean
        pred_anomaly = y_pred_clean - clim_clean
        
        if np.std(true_anomaly) > 0 and np.std(pred_anomaly) > 0:
            anomaly_corr, _ = pearsonr(true_anomaly, pred_anomaly)
        else:
            anomaly_corr = np.nan
        
        return {
            'skill_score': float(skill_score),
            'anomaly_correlation': float(anomaly_corr)
        }
    
    @staticmethod
    def drought_duration_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                threshold: float = -1.0) -> Dict[str, float]:
        """Метрики продолжительности засухи"""
        
        # Определяем периоды засухи
        drought_true = y_true < threshold
        drought_pred = y_pred < threshold
        
        # Находим длительности периодов засухи
        true_durations = DroughtMetrics._get_drought_durations(drought_true)
        pred_durations = DroughtMetrics._get_drought_durations(drought_pred)
        
        if len(true_durations) == 0 and len(pred_durations) == 0:
            return {name: np.nan for name in ['duration_bias', 'duration_correlation']}
        
        # Сравнение средних длительностей
        mean_true_duration = np.mean(true_durations) if true_durations else 0
        mean_pred_duration = np.mean(pred_durations) if pred_durations else 0
        
        duration_bias = (mean_pred_duration - mean_true_duration) / (mean_true_duration + 1e-8)
        
        # Корреляция длительностей (если достаточно событий)
        if len(true_durations) >= 3 and len(pred_durations) >= 3:
            # Используем гистограммы для сравнения распределений
            max_duration = max(max(true_durations, default=0), max(pred_durations, default=0))
            if max_duration > 0:
                bins = np.arange(1, max_duration + 2)
                hist_true, _ = np.histogram(true_durations, bins=bins, density=True)
                hist_pred, _ = np.histogram(pred_durations, bins=bins, density=True)
                
                duration_corr, _ = pearsonr(hist_true, hist_pred)
            else:
                duration_corr = np.nan
        else:
            duration_corr = np.nan
        
        return {
            'duration_bias': float(duration_bias),
            'duration_correlation': float(duration_corr)
        }
    
    @staticmethod
    def comprehensive_evaluation(y_true: np.ndarray, y_pred: np.ndarray,
                                coords: Optional[np.ndarray] = None,
                                climatology: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Комплексная оценка всех метрик"""
        
        metrics = {}
        
        # Регрессионные метрики
        metrics.update(DroughtMetrics.regression_metrics(y_true, y_pred))
        
        # Классификационные метрики
        metrics.update(DroughtMetrics.drought_classification_metrics(y_true, y_pred))
        
        # Детекция засухи
        metrics.update(DroughtMetrics.drought_detection_metrics(y_true, y_pred))
        
        # Пространственные метрики
        metrics.update(DroughtMetrics.spatial_correlation(y_true, y_pred, coords))
        
        # Временные метрики
        metrics.update(DroughtMetrics.temporal_consistency(y_true, y_pred))
        
        # Экстремальные события
        metrics.update(DroughtMetrics.extreme_events_metrics(y_true, y_pred))
        
        # Климатологический скилл
        metrics.update(DroughtMetrics.climatological_skill(y_true, y_pred, climatology))
        
        # Длительность засухи
        metrics.update(DroughtMetrics.drought_duration_metrics(y_true, y_pred))
        
        return metrics
    
    @staticmethod
    def _spi_to_drought_category(spi_values: np.ndarray) -> np.ndarray:
        """Преобразование SPI в категории засухи"""
        categories = np.full_like(spi_values, np.nan, dtype=object)
        
        for category, (min_val, max_val) in DroughtMetrics.DROUGHT_THRESHOLDS.items():
            mask = (spi_values >= min_val) & (spi_values < max_val)
            categories[mask] = category
        
        return categories
    
    @staticmethod
    def _get_drought_durations(drought_mask: np.ndarray) -> List[int]:
        """Получение длительностей периодов засухи"""
        durations = []
        current_duration = 0
        
        for is_drought in drought_mask:
            if is_drought:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
        
        # Добавляем последний период если он не закончился
        if current_duration > 0:
            durations.append(current_duration)
        
        return durations

class MetricsAggregator:
    """Агрегатор метрик для множественных экспериментов"""
    
    def __init__(self):
        self.experiments = {}
    
    def add_experiment(self, name: str, y_true: np.ndarray, y_pred: np.ndarray,
                      **kwargs) -> None:
        """Добавление эксперимента"""
        metrics = DroughtMetrics.comprehensive_evaluation(y_true, y_pred, **kwargs)
        self.experiments[name] = metrics
    
    def compare_experiments(self) -> pd.DataFrame:
        """Сравнение экспериментов"""
        return pd.DataFrame(self.experiments).T
    
    def rank_experiments(self, primary_metric: str = 'mae', 
                        ascending: bool = True) -> pd.DataFrame:
        """Ранжирование экспериментов по метрике"""
        df = self.compare_experiments()
        return df.sort_values(primary_metric, ascending=ascending)
    
    def get_best_experiment(self, metric: str = 'mae', 
                           ascending: bool = True) -> Tuple[str, Dict]:
        """Получение лучшего эксперимента"""
        ranking = self.rank_experiments(metric, ascending)
        best_name = ranking.index[0]
        return best_name, self.experiments[best_name]

class RegionalMetrics:
    """Метрики с разбивкой по регионам"""
    
    @staticmethod
    def evaluate_by_regions(y_true: np.ndarray, y_pred: np.ndarray,
                          region_mask: np.ndarray, 
                          region_names: List[str]) -> Dict[str, Dict]:
        """Оценка метрик по регионам"""
        
        regional_metrics = {}
        
        for i, region_name in enumerate(region_names):
            # Маска для текущего региона
            mask = region_mask == i
            
            if mask.sum() > 0:
                y_true_region = y_true[mask]
                y_pred_region = y_pred[mask]
                
                # Вычисляем метрики для региона
                metrics = DroughtMetrics.comprehensive_evaluation(y_true_region, y_pred_region)
                regional_metrics[region_name] = metrics
        
        return regional_metrics
    
    @staticmethod
    def compare_regional_performance(regional_metrics: Dict[str, Dict],
                                   metric: str = 'mae') -> pd.Series:
        """Сравнение производительности по регионам"""
        
        performance = {}
        for region, metrics in regional_metrics.items():
            performance[region] = metrics.get(metric, np.nan)
        
        return pd.Series(performance).sort_values()

def calculate_model_confidence(y_pred_ensemble: List[np.ndarray]) -> np.ndarray:
    """Вычисление уверенности модели на основе ансамбля предсказаний"""
    
    # Стандартное отклонение предсказаний как мера неопределенности
    pred_array = np.array(y_pred_ensemble)
    confidence = 1 / (1 + np.std(pred_array, axis=0))
    
    return confidence

def bootstrap_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     n_bootstrap: int = 1000) -> Dict[str, Dict[str, float]]:
    """Bootstrap оценка доверительных интервалов для метрик"""
    
    n_samples = len(y_true)
    bootstrap_metrics = {metric: [] for metric in ['mae', 'rmse', 'r2']}
    
    for _ in range(n_bootstrap):
        # Случайная выборка с возвращением
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Вычисляем метрики
        metrics = DroughtMetrics.regression_metrics(y_true_boot, y_pred_boot)
        
        for metric_name in bootstrap_metrics:
            bootstrap_metrics[metric_name].append(metrics[metric_name])
    
    # Вычисляем доверительные интервалы
    confidence_intervals = {}
    for metric_name, values in bootstrap_metrics.items():
        values = np.array(values)
        confidence_intervals[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'ci_2.5': np.percentile(values, 2.5),
            'ci_97.5': np.percentile(values, 97.5)
        }
    
    return confidence_intervals

if __name__ == "__main__":
    # Пример использования
    print("📊 Метрики для оценки предсказания засухи")
    
    # Генерация тестовых данных
    np.random.seed(42)
    n_samples = 1000
    
    # Симуляция SPI данных
    y_true = np.random.normal(0, 1, n_samples)  # SPI имеет нормальное распределение
    y_pred = y_true + np.random.normal(0, 0.3, n_samples)  # Добавляем шум
    
    # Комплексная оценка
    metrics = DroughtMetrics.comprehensive_evaluation(y_true, y_pred)
    
    print("🎯 Результаты оценки:")
    for metric_name, value in metrics.items():
        if not np.isnan(value):
            print(f"  {metric_name}: {value:.4f}")
    
    print("\n✅ Система метрик готова к использованию!")