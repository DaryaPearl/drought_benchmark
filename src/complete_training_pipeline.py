"""
Полный пайплайн обучения для предсказания засухи
Включает классические ML модели и SOTA архитектуры

Запуск:
    python src/complete_training_pipeline.py --model all --experiment drought_prediction_2024
    python src/complete_training_pipeline.py --model earthformer --config configs/earthformer.yaml
    python src/complete_training_pipeline.py --model classical --fast
"""

import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import yaml
import json
from datetime import datetime
import time

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xarray as xr
from torch.utils.data import DataLoader, Dataset

# Локальные импорты (предполагаем, что файлы созданы)
from src.models.sota_models import get_model, CONFIGS
from src.utils.metrics import DroughtMetrics
from src.utils.visualization import ResultsVisualizer, TrainingVisualizer

warnings.filterwarnings("ignore")

class DroughtDataset(Dataset):
    """PyTorch Dataset для данных засухи"""
    
    def __init__(self, data: xr.Dataset, 
                 input_vars: List[str],
                 target_var: str,
                 sequence_length: int = 12,
                 prediction_horizon: int = 3,
                 train_years: Tuple[int, int] = (2003, 2016),
                 spatial_subset: Optional[Dict] = None):
        
        self.data = data
        self.input_vars = input_vars
        self.target_var = target_var
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Фильтрация по годам
        start_year, end_year = train_years
        time_mask = (data.time.dt.year >= start_year) & (data.time.dt.year <= end_year)
        self.data = data.sel(time=time_mask)
        
        # Пространственная подвыборка (если нужно)
        if spatial_subset:
            self.data = self.data.sel(
                latitude=slice(spatial_subset['lat_min'], spatial_subset['lat_max']),
                longitude=slice(spatial_subset['lon_min'], spatial_subset['lon_max'])
            )
        
        # Подготовка данных
        self._prepare_sequences()
    
    def _prepare_sequences(self):
        """Подготовка последовательностей для обучения"""
        print(f"📊 Подготовка последовательностей (окно={self.sequence_length}, горизонт={self.prediction_horizon})")
        
        # Конвертация в numpy
        input_data = []
        for var in self.input_vars:
            if var in self.data.data_vars:
                input_data.append(self.data[var].values)
        
        input_array = np.stack(input_data, axis=1)  # (time, variables, lat, lon)
        target_array = self.data[self.target_var].values  # (time, lat, lon)
        
        # Создание последовательностей
        self.sequences = []
        self.targets = []
        
        T = input_array.shape[0]
        for t in range(self.sequence_length, T - self.prediction_horizon):
            # Входная последовательность
            seq = input_array[t - self.sequence_length:t]
            
            # Целевые значения (через prediction_horizon шагов)
            target_time_indices = range(t, t + self.prediction_horizon)
            target = target_array[target_time_indices]
            
            # Проверка на NaN
            if not (np.isnan(seq).any() or np.isnan(target).any()):
                self.sequences.append(seq)
                self.targets.append(target)
        
        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)
        
        print(f"✅ Создано {len(self.sequences)} последовательностей")
        print(f"📐 Форма входа: {self.sequences.shape}")
        print(f"📐 Форма цели: {self.targets.shape}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.targets[idx])
        )

class ClassicalModelsTrainer:
    """Тренер для классических моделей ML"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=config.get('ridge_alpha', 1.0)),
            'lasso': Lasso(alpha=config.get('lasso_alpha', 1.0)),
            'random_forest': RandomForestRegressor(
                n_estimators=config.get('rf_n_estimators', 300),
                max_depth=config.get('rf_max_depth', 20),
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=config.get('gb_n_estimators', 200),
                learning_rate=config.get('gb_learning_rate', 0.1),
                max_depth=config.get('gb_max_depth', 6),
                random_state=42
            ),
            'svr': SVR(
                kernel=config.get('svr_kernel', 'rbf'),
                C=config.get('svr_C', 1.0),
                epsilon=config.get('svr_epsilon', 0.1)
            )
        }
        
        self.scaler = StandardScaler()
        self.results = {}
    
    def prepare_classical_data(self, dataset: DroughtDataset) -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка данных для классических моделей"""
        print("🔄 Подготовка данных для классических моделей...")
        
        # Преобразование последовательностей в плоские признаки
        sequences = dataset.sequences  # (samples, time, vars, lat, lon)
        targets = dataset.targets     # (samples, horizon, lat, lon)
        
        # Flatten пространственных измерений и создание лагов
        n_samples, seq_len, n_vars, n_lat, n_lon = sequences.shape
        
        # Создание признаков: каждый пиксель и временной шаг становится признаком
        features = []
        labels = []
        
        for sample_idx in range(n_samples):
            seq = sequences[sample_idx]  # (time, vars, lat, lon)
            target = targets[sample_idx]  # (horizon, lat, lon)
            
            # Для каждого пикселя
            for i in range(n_lat):
                for j in range(n_lon):
                    # Признаки: временные ряды всех переменных для данного пикселя
                    pixel_features = seq[:, :, i, j].flatten()  # (time * vars,)
                    
                    # Целевые значения: средний SPI по горизонту для данного пикселя
                    pixel_target = target[:, i, j].mean()  # Среднее по горизонту
                    
                    if not (np.isnan(pixel_features).any() or np.isnan(pixel_target)):
                        features.append(pixel_features)
                        labels.append(pixel_target)
        
        X = np.array(features)
        y = np.array(labels)
        
        print(f"📊 Классические данные: X={X.shape}, y={y.shape}")
        
        # Нормализация признаков
        X = self.scaler.fit_transform(X)
        
        return X, y
    
    def train_classical_models(self, train_dataset: DroughtDataset, 
                             val_dataset: DroughtDataset) -> Dict[str, Dict]:
        """Обучение классических моделей"""
        print("\n🤖 Обучение классических моделей ML...")
        
        # Подготовка данных
        X_train, y_train = self.prepare_classical_data(train_dataset)
        X_val, y_val = self.prepare_classical_data(val_dataset)
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n🔧 Обучение {model_name}...")
            start_time = time.time()
            
            try:
                # Обучение
                model.fit(X_train, y_train)
                
                # Предсказания
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                
                # Метрики
                train_metrics = {
                    'mae': mean_absolute_error(y_train, y_train_pred),
                    'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                    'r2': r2_score(y_train, y_train_pred)
                }
                
                val_metrics = {
                    'mae': mean_absolute_error(y_val, y_val_pred),
                    'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
                    'r2': r2_score(y_val, y_val_pred)
                }
                
                training_time = time.time() - start_time
                
                results[model_name] = {
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'training_time': training_time,
                    'model': model
                }
                
                print(f"  ✅ {model_name}: Val MAE={val_metrics['mae']:.4f}, "
                      f"R²={val_metrics['r2']:.4f}, Time={training_time:.1f}s")
                
            except Exception as e:
                print(f"  ❌ Ошибка обучения {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results

class SOTAModelsTrainer:
    """Тренер для SOTA deep learning моделей"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.results = {}
    
    def create_data_loaders(self, train_dataset: DroughtDataset,
                           val_dataset: DroughtDataset,
                           test_dataset: DroughtDataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Создание DataLoader'ов"""
        batch_size = self.config.get('batch_size', 32)
        num_workers = self.config.get('num_workers', 4)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def train_sota_model(self, model_name: str, 
                        train_loader: DataLoader,
                        val_loader: DataLoader,
                        test_loader: DataLoader) -> Dict:
        """Обучение одной SOTA модели"""
        print(f"\n🚀 Обучение {model_name.upper()}...")
        
        # Конфигурация модели
        model_config = CONFIGS.get(model_name, {})
        model_config.update(self.config.get(f'{model_name}_config', {}))
        
        try:
            # Создание модели
            model = get_model(model_name, model_config)
            
            # Callbacks
            callbacks = [
                ModelCheckpoint(
                    monitor='val_loss',
                    dirpath=f'checkpoints/{model_name}',
                    filename='{epoch:02d}-{val_loss:.4f}',
                    save_top_k=3,
                    mode='min'
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.get('early_stopping_patience', 15),
                    mode='min'
                ),
                LearningRateMonitor(logging_interval='epoch')
            ]
            
            # Logger
            logger = TensorBoardLogger(
                save_dir='lightning_logs',
                name=model_name,
                version=datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            
            # Trainer
            trainer = pl.Trainer(
                max_epochs=self.config.get('max_epochs', 100),
                accelerator='auto',
                devices='auto',
                callbacks=callbacks,
                logger=logger,
                gradient_clip_val=self.config.get('gradient_clip_val', 1.0),
                accumulate_grad_batches=self.config.get('accumulate_grad_batches', 1),
                precision=self.config.get('precision', 32),
                log_every_n_steps=10,
            )
            
            # Обучение
            start_time = time.time()
            trainer.fit(model, train_loader, val_loader)
            training_time = time.time() - start_time
            
            # Тестирование
            test_results = trainer.test(model, test_loader)
            
            # Сбор результатов
            results = {
                'train_metrics': {
                    'final_train_loss': float(trainer.callback_metrics.get('train_loss', 0)),
                },
                'val_metrics': {
                    'final_val_loss': float(trainer.callback_metrics.get('val_loss', 0)),
                    'best_val_loss': float(callbacks[0].best_model_score),
                },
                'test_metrics': test_results[0] if test_results else {},
                'training_time': training_time,
                'best_model_path': callbacks[0].best_model_path,
                'model_config': model_config
            }
            
            print(f"  ✅ {model_name}: Val Loss={results['val_metrics']['final_val_loss']:.4f}, "
                  f"Time={training_time:.1f}s")
            
            return results
            
        except Exception as e:
            print(f"  ❌ Ошибка обучения {model_name}: {e}")
            return {'error': str(e)}

class ExperimentManager:
    """Менеджер экспериментов"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.experiment_name = config.get('experiment_name', f'drought_exp_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        self.results_dir = Path(config.get('results_dir', 'results')) / self.experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.all_results = {}
        
    def prepare_datasets(self) -> Tuple[DroughtDataset, DroughtDataset, DroughtDataset]:
        """Подготовка датасетов для обучения"""
        print("📦 Загрузка и подготовка данных...")
        
        # Загрузка данных
        data_path = self.config.get('data_path', 'data/processed/real_agro_cube.zarr')
        dataset = xr.open_zarr(data_path)
        
        # Конфигурация данных
        input_vars = self.config.get('input_variables', ['precipitation', 'temperature', 'ndvi', 'soil_moisture'])
        target_var = self.config.get('target_variable', 'spi3')
        sequence_length = self.config.get('sequence_length', 12)
        prediction_horizon = self.config.get('prediction_horizon', 3)
        
        # Разделение по годам
        train_years = self.config.get('train_years', (2003, 2016))
        val_years = self.config.get('val_years', (2017, 2019))
        test_years = self.config.get('test_years', (2020, 2024))
        
        # Пространственная подвыборка (для ускорения)
        spatial_subset = self.config.get('spatial_subset', None)
        if self.config.get('fast_mode', False):
            # Быстрый режим - меньше данных
            spatial_subset = {
                'lat_min': 40, 'lat_max': 50,
                'lon_min': -100, 'lon_max': -80
            }
        
        # Создание датасетов
        train_dataset = DroughtDataset(
            dataset, input_vars, target_var, sequence_length, prediction_horizon,
            train_years, spatial_subset
        )
        
        val_dataset = DroughtDataset(
            dataset, input_vars, target_var, sequence_length, prediction_horizon,
            val_years, spatial_subset
        )
        
        test_dataset = DroughtDataset(
            dataset, input_vars, target_var, sequence_length, prediction_horizon,
            test_years, spatial_subset
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def run_experiment(self, models_to_train: List[str]):
        """Запуск полного эксперимента"""
        print(f"🧪 Запуск эксперимента: {self.experiment_name}")
        print(f"📁 Результаты будут сохранены в: {self.results_dir}")
        
        # Подготовка данных
        train_dataset, val_dataset, test_dataset = self.prepare_datasets()
        
        # Классические модели
        if 'classical' in models_to_train or 'all' in models_to_train:
            print("\n" + "="*50)
            print("🤖 КЛАССИЧЕСКИЕ МОДЕЛИ МАШИННОГО ОБУЧЕНИЯ")
            print("="*50)
            
            classical_trainer = ClassicalModelsTrainer(self.config)
            classical_results = classical_trainer.train_classical_models(train_dataset, val_dataset)
            self.all_results['classical'] = classical_results
        
        # SOTA модели
        sota_models = ['earthformer', 'convlstm', 'tft', 'unet']
        sota_to_train = [m for m in models_to_train if m in sota_models] or (sota_models if 'all' in models_to_train else [])
        
        if sota_to_train:
            print("\n" + "="*50)
            print("🚀 SOTA DEEP LEARNING МОДЕЛИ")
            print("="*50)
            
            sota_trainer = SOTAModelsTrainer(self.config)
            train_loader, val_loader, test_loader = sota_trainer.create_data_loaders(
                train_dataset, val_dataset, test_dataset
            )
            
            for model_name in sota_to_train:
                if model_name in CONFIGS:
                    results = sota_trainer.train_sota_model(
                        model_name, train_loader, val_loader, test_loader
                    )
                    self.all_results[model_name] = results
                else:
                    print(f"⚠ Модель {model_name} не найдена в конфигурациях")
        
        # Сохранение результатов
        self.save_results()
        
        # Создание отчета
        self.create_report()
        
        # Визуализация результатов
        self.create_visualizations()
        
        print(f"\n🎉 Эксперимент завершен! Результаты в: {self.results_dir}")
    
    def save_results(self):
        """Сохранение результатов эксперимента"""
        print("💾 Сохранение результатов...")
        
        # Конвертация результатов в JSON-совместимый формат
        results_to_save = {}
        for model_name, results in self.all_results.items():
            if isinstance(results, dict):
                # Убираем объекты моделей из сохранения
                clean_results = {}
                for key, value in results.items():
                    if isinstance(value, dict) and 'model' not in key:
                        clean_results[key] = value
                    elif key != 'model':
                        clean_results[key] = str(value) if not isinstance(value, (int, float, str, bool, type(None))) else value
                results_to_save[model_name] = clean_results
        
        # Сохранение в JSON
        results_file = self.results_dir / 'experiment_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'experiment_name': self.experiment_name,
                'config': self.config,
                'results': results_to_save,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)
        
        print(f"✅ Результаты сохранены: {results_file}")
    
    def create_report(self):
        """Создание отчета эксперимента"""
        print("📄 Создание отчета...")
        
        report_file = self.results_dir / 'experiment_report.md'
        
        with open(report_file, 'w') as f:
            f.write(f"# Отчет по эксперименту: {self.experiment_name}\n\n")
            f.write(f"**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Конфигурация
            f.write("## Конфигурация эксперимента\n\n")
            f.write("```yaml\n")
            f.write(yaml.dump(self.config, default_flow_style=False))
            f.write("```\n\n")
            
            # Результаты по моделям
            f.write("## Результаты моделей\n\n")
            
            # Таблица сравнения
            f.write("### Сводная таблица\n\n")
            f.write("| Модель | Val MAE | Val RMSE | Val R² | Время обучения (с) |\n")
            f.write("|--------|---------|----------|--------|--------------------|\n")
            
            for model_name, results in self.all_results.items():
                if isinstance(results, dict) and 'error' not in results:
                    if model_name == 'classical':
                        # Для классических моделей берем лучшую по Val MAE
                        best_classical = min(results.items(), 
                                           key=lambda x: x[1].get('val_metrics', {}).get('mae', float('inf')))
                        best_name, best_results = best_classical
                        val_metrics = best_results.get('val_metrics', {})
                        f.write(f"| {best_name} (best classical) | "
                               f"{val_metrics.get('mae', 'N/A'):.4f} | "
                               f"{val_metrics.get('rmse', 'N/A'):.4f} | "
                               f"{val_metrics.get('r2', 'N/A'):.4f} | "
                               f"{best_results.get('training_time', 'N/A'):.1f} |\n")
                    else:
                        val_metrics = results.get('val_metrics', {})
                        val_loss = val_metrics.get('final_val_loss', 'N/A')
                        training_time = results.get('training_time', 'N/A')
                        f.write(f"| {model_name} | N/A | {val_loss:.4f} | N/A | {training_time:.1f} |\n")
            
            # Детальные результаты
            f.write("\n### Детальные результаты\n\n")
            
            for model_name, results in self.all_results.items():
                f.write(f"#### {model_name.upper()}\n\n")
                
                if isinstance(results, dict) and 'error' not in results:
                    if model_name == 'classical':
                        for sub_model, sub_results in results.items():
                            f.write(f"**{sub_model}**\n\n")
                            val_metrics = sub_results.get('val_metrics', {})
                            f.write(f"- Validation MAE: {val_metrics.get('mae', 'N/A'):.4f}\n")
                            f.write(f"- Validation RMSE: {val_metrics.get('rmse', 'N/A'):.4f}\n")
                            f.write(f"- Validation R²: {val_metrics.get('r2', 'N/A'):.4f}\n")
                            f.write(f"- Training time: {sub_results.get('training_time', 'N/A'):.1f}s\n\n")
                    else:
                        val_metrics = results.get('val_metrics', {})
                        test_metrics = results.get('test_metrics', {})
                        f.write(f"- Final validation loss: {val_metrics.get('final_val_loss', 'N/A'):.4f}\n")
                        f.write(f"- Best validation loss: {val_metrics.get('best_val_loss', 'N/A'):.4f}\n")
                        f.write(f"- Training time: {results.get('training_time', 'N/A'):.1f}s\n")
                        if test_metrics:
                            f.write(f"- Test metrics: {test_metrics}\n")
                        f.write("\n")
                else:
                    f.write(f"❌ Ошибка: {results.get('error', 'Unknown error')}\n\n")
        
        print(f"✅ Отчет создан: {report_file}")
    
    def create_visualizations(self):
        """Создание визуализаций результатов"""
        print("📊 Создание визуализаций...")
        
        try:
            visualizer = ResultsVisualizer(self.results_dir)
            
            # График сравнения моделей
            visualizer.plot_model_comparison(self.all_results)
            
            # График обучения (если есть логи)
            visualizer.plot_training_curves(self.results_dir)
            
            print("✅ Визуализации созданы")
            
        except Exception as e:
            print(f"⚠ Ошибка создания визуализаций: {e}")

def load_config(config_path: str) -> Dict:
    """Загрузка конфигурации из файла"""
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Конфигурация по умолчанию
        return {
            'experiment_name': f'drought_experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'data_path': 'data/processed/real_agro_cube.zarr',
            'input_variables': ['precipitation', 'temperature', 'ndvi', 'soil_moisture'],
            'target_variable': 'spi3',
            'sequence_length': 12,
            'prediction_horizon': 3,
            'train_years': [2003, 2016],
            'val_years': [2017, 2019],
            'test_years': [2020, 2024],
            'batch_size': 32,
            'max_epochs': 50,
            'early_stopping_patience': 10,
            'learning_rate': 0.001,
            'fast_mode': False,
            'results_dir': 'results'
        }

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Обучение моделей предсказания засухи')
    parser.add_argument('--model', type=str, choices=['all', 'classical', 'earthformer', 'convlstm', 'tft', 'unet'],
                       default='all', help='Модель для обучения')
    parser.add_argument('--config', type=str, help='Путь к файлу конфигурации')
    parser.add_argument('--experiment', type=str, help='Название эксперимента')
    parser.add_argument('--fast', action='store_true', help='Быстрый режим (меньше данных)')
    parser.add_argument('--data-path', type=str, help='Путь к данным')
    
    args = parser.parse_args()
    
    # Загрузка конфигурации
    config = load_config(args.config) if args.config else load_config('config.yaml')
    
    # Обновление конфигурации аргументами
    if args.experiment:
        config['experiment_name'] = args.experiment
    if args.fast:
        config['fast_mode'] = True
        config['max_epochs'] = 10  # Меньше эпох для быстрого тестирования
    if args.data_path:
        config['data_path'] = args.data_path
    
    # Определение моделей для обучения
    if args.model == 'all':
        models_to_train = ['classical', 'earthformer', 'convlstm', 'tft', 'unet']
    else:
        models_to_train = [args.model]
    
    print("🌾 Система предсказания засухи")
    print("=" * 50)
    print(f"🧪 Эксперимент: {config['experiment_name']}")
    print(f"🤖 Модели: {', '.join(models_to_train)}")
    print(f"⚡ Быстрый режим: {'Да' if config.get('fast_mode') else 'Нет'}")
    print("=" * 50)
    
    try:
        # Создание менеджера эксперимента
        experiment_manager = ExperimentManager(config)
        
        # Запуск эксперимента
        experiment_manager.run_experiment(models_to_train)
        
    except KeyboardInterrupt:
        print("\n⏹ Эксперимент прерван пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка выполнения эксперимента: {e}")
        raise

if __name__ == "__main__":
    main()