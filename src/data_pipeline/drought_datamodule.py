"""
PyTorch Lightning DataModule для данных засухи

Обеспечивает:
- Загрузку и предобработку данных из Zarr
- Создание временных последовательностей
- Разделение на train/val/test
- DataLoader'ы для PyTorch Lightning
- Трансформации и аугментации

Использование: from src.data_pipeline.drought_datamodule import DroughtDataModule
"""

import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

class DroughtDataset(Dataset):
    """Dataset для данных засухи"""
    
    def __init__(self, 
                 data: xr.Dataset,
                 input_vars: List[str],
                 target_var: str,
                 sequence_length: int = 12,
                 prediction_horizon: int = 3,
                 spatial_subset: Optional[Dict] = None,
                 transform: Optional[callable] = None,
                 augment: bool = False):
        
        self.data = data
        self.input_vars = input_vars
        self.target_var = target_var
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.transform = transform
        self.augment = augment
        
        # Пространственная подвыборка
        if spatial_subset:
            self.data = self.data.sel(
                latitude=slice(spatial_subset['lat_min'], spatial_subset['lat_max']),
                longitude=slice(spatial_subset['lon_min'], spatial_subset['lon_max'])
            )
        
        # Проверка наличия переменных
        available_vars = list(self.data.data_vars.keys())
        missing_input_vars = [var for var in self.input_vars if var not in available_vars]
        if missing_input_vars:
            raise ValueError(f"Отсутствуют входные переменные: {missing_input_vars}")
        
        if self.target_var not in available_vars:
            raise ValueError(f"Отсутствует целевая переменная: {self.target_var}")
        
        # Подготовка данных
        self._prepare_data()
    
    def _prepare_data(self):
        """Подготовка данных для обучения"""
        print(f"📊 Подготовка данных: {len(self.input_vars)} переменных, "
              f"окно={self.sequence_length}, горизонт={self.prediction_horizon}")
        
        # Извлечение массивов данных
        input_arrays = []
        for var in self.input_vars:
            input_arrays.append(self.data[var].values)
        
        self.input_data = np.stack(input_arrays, axis=1)  # (time, vars, lat, lon)
        self.target_data = self.data[self.target_var].values  # (time, lat, lon)
        
        # Размеры данных
        self.n_time, self.n_vars, self.n_lat, self.n_lon = self.input_data.shape
        
        # Создание последовательностей
        self.valid_indices = []
        
        for t in range(self.sequence_length, self.n_time - self.prediction_horizon + 1):
            # Проверяем наличие валидных данных в окне
            input_window = self.input_data[t - self.sequence_length:t]
            target_window = self.target_data[t:t + self.prediction_horizon]
            
            # Проверяем на NaN
            if not (np.isnan(input_window).any() or np.isnan(target_window).any()):
                self.valid_indices.append(t)
        
        print(f"✅ Создано {len(self.valid_indices)} валидных последовательностей")
        
        if len(self.valid_indices) == 0:
            raise ValueError("Нет валидных последовательностей в данных")
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Получение элемента датасета"""
        t = self.valid_indices[idx]
        
        # Входная последовательность
        input_sequence = self.input_data[t - self.sequence_length:t]  # (seq_len, vars, lat, lon)
        
        # Целевые значения
        target_sequence = self.target_data[t:t + self.prediction_horizon]  # (horizon, lat, lon)
        
        # Трансформации
        if self.transform:
            input_sequence = self.transform(input_sequence)
            target_sequence = self.transform(target_sequence)
        
        # Аугментация (только для обучения)
        if self.augment:
            input_sequence, target_sequence = self._apply_augmentation(
                input_sequence, target_sequence
            )
        
        # Преобразование в тензоры
        input_tensor = torch.FloatTensor(input_sequence)
        target_tensor = torch.FloatTensor(target_sequence)
        
        return input_tensor, target_tensor
    
    def _apply_augmentation(self, inputs: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Применение аугментации данных"""
        
        # Добавление шума
        if np.random.random() < 0.3:
            noise_std = 0.01
            inputs = inputs + np.random.normal(0, noise_std, inputs.shape)
        
        # Временной сдвиг (jitter)
        if np.random.random() < 0.2:
            shift = np.random.randint(-2, 3)
            if shift != 0:
                inputs = np.roll(inputs, shift, axis=0)
        
        # Пространственное отражение
        if np.random.random() < 0.3:
            if np.random.random() < 0.5:
                inputs = np.flip(inputs, axis=-2)  # Отражение по широте
                targets = np.flip(targets, axis=-2)
            if np.random.random() < 0.5:
                inputs = np.flip(inputs, axis=-1)  # Отражение по долготе
                targets = np.flip(targets, axis=-1)
        
        return inputs, targets
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Получение статистики данных"""
        stats = {}
        
        # Статистика по входным переменным
        for i, var in enumerate(self.input_vars):
            var_data = self.input_data[:, i, :, :]
            valid_data = var_data[~np.isnan(var_data)]
            
            if len(valid_data) > 0:
                stats[var] = {
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data)),
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data))
                }
        
        # Статистика по целевой переменной
        target_valid = self.target_data[~np.isnan(self.target_data)]
        if len(target_valid) > 0:
            stats[self.target_var] = {
                'mean': float(np.mean(target_valid)),
                'std': float(np.std(target_valid)),
                'min': float(np.min(target_valid)),
                'max': float(np.max(target_valid))
            }
        
        return stats

class DroughtDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule для данных засухи"""
    
    def __init__(self,
                 data_path: str,
                 input_vars: List[str],
                 target_var: str,
                 sequence_length: int = 12,
                 prediction_horizon: int = 3,
                 train_years: Tuple[int, int] = (2003, 2016),
                 val_years: Tuple[int, int] = (2017, 2019),
                 test_years: Tuple[int, int] = (2020, 2024),
                 batch_size: int = 32,
                 num_workers: int = 4,
                 spatial_subset: Optional[Dict] = None,
                 normalize: bool = True,
                 augment_train: bool = False):
        
        super().__init__()
        
        self.data_path = data_path
        self.input_vars = input_vars
        self.target_var = target_var
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.train_years = train_years
        self.val_years = val_years
        self.test_years = test_years
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.spatial_subset = spatial_subset
        self.normalize = normalize
        self.augment_train = augment_train
        
        # Будут инициализированы в setup()
        self.data = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.normalizer = None
        
    def prepare_data(self):
        """Подготовка данных (вызывается один раз)"""
        # Проверяем существование файла данных
        if not Path(self.data_path).exists():
            raise FileNotFoundError(f"Файл данных не найден: {self.data_path}")
        
        print(f"📁 Загрузка данных из: {self.data_path}")
    
    def setup(self, stage: Optional[str] = None):
        """Настройка датасетов"""
        
        # Загрузка данных
        print("📦 Загрузка данных засухи...")
        self.data = xr.open_zarr(self.data_path)
        
        print(f"📊 Размеры данных: {dict(self.data.dims)}")
        print(f"📋 Переменные: {list(self.data.data_vars.keys())}")
        
        # Нормализация (если включена)
        if self.normalize:
            self._setup_normalization()
        
        # Создание датасетов
        if stage == "fit" or stage is None:
            # Обучающий датасет
            train_data = self._filter_by_years(self.data, self.train_years)
            self.train_dataset = DroughtDataset(
                data=train_data,
                input_vars=self.input_vars,
                target_var=self.target_var,
                sequence_length=self.sequence_length,
                prediction_horizon=self.prediction_horizon,
                spatial_subset=self.spatial_subset,
                transform=self._get_transform(),
                augment=self.augment_train
            )
            
            # Валидационный датасет
            val_data = self._filter_by_years(self.data, self.val_years)
            self.val_dataset = DroughtDataset(
                data=val_data,
                input_vars=self.input_vars,
                target_var=self.target_var,
                sequence_length=self.sequence_length,
                prediction_horizon=self.prediction_horizon,
                spatial_subset=self.spatial_subset,
                transform=self._get_transform(),
                augment=False
            )
        
        if stage == "test" or stage is None:
            # Тестовый датасет
            test_data = self._filter_by_years(self.data, self.test_years)
            self.test_dataset = DroughtDataset(
                data=test_data,
                input_vars=self.input_vars,
                target_var=self.target_var,
                sequence_length=self.sequence_length,
                prediction_horizon=self.prediction_horizon,
                spatial_subset=self.spatial_subset,
                transform=self._get_transform(),
                augment=False
            )
        
        # Вывод статистики
        self._print_dataset_stats()
    
    def _filter_by_years(self, data: xr.Dataset, years: Tuple[int, int]) -> xr.Dataset:
        """Фильтрация данных по годам"""
        start_year, end_year = years
        time_mask = (data.time.dt.year >= start_year) & (data.time.dt.year <= end_year)
        return data.sel(time=time_mask)
    
    def _setup_normalization(self):
        """Настройка нормализации данных"""
        print("🔧 Настройка нормализации...")
        
        # Вычисляем статистики по обучающим данным
        train_data = self._filter_by_years(self.data, self.train_years)
        
        self.normalizer = {}
        
        # Для каждой входной переменной
        for var in self.input_vars:
            var_data = train_data[var].values
            valid_data = var_data[~np.isnan(var_data)]
            
            if len(valid_data) > 0:
                self.normalizer[var] = {
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data))
                }
            else:
                self.normalizer[var] = {'mean': 0.0, 'std': 1.0}
        
        # Для целевой переменной
        target_data = train_data[self.target_var].values
        valid_target = target_data[~np.isnan(target_data)]
        
        if len(valid_target) > 0:
            self.normalizer[self.target_var] = {
                'mean': float(np.mean(valid_target)),
                'std': float(np.std(valid_target))
            }
        else:
            self.normalizer[self.target_var] = {'mean': 0.0, 'std': 1.0}
        
        # Применяем нормализацию к данным
        self._apply_normalization()
    
    def _apply_normalization(self):
        """Применение нормализации к данным"""
        normalized_data = {}
        
        for var in self.data.data_vars:
            if var in self.normalizer:
                mean = self.normalizer[var]['mean']
                std = self.normalizer[var]['std']
                normalized_data[var] = (self.data[var] - mean) / (std + 1e-8)
            else:
                normalized_data[var] = self.data[var]
        
        # Создаем новый Dataset с нормализованными данными
        self.data = xr.Dataset(normalized_data, coords=self.data.coords, attrs=self.data.attrs)
    
    def _get_transform(self) -> Optional[callable]:
        """Получение функции трансформации"""
        # Здесь можно добавить дополнительные трансформации
        return None
    
    def _print_dataset_stats(self):
        """Вывод статистики датасетов"""
        print("\n📈 Статистика датасетов:")
        
        if self.train_dataset:
            print(f"  🏋️ Обучение: {len(self.train_dataset)} последовательностей")
        
        if self.val_dataset:
            print(f"  🔍 Валидация: {len(self.val_dataset)} последовательностей")
        
        if self.test_dataset:
            print(f"  🧪 Тест: {len(self.test_dataset)} последовательностей")
        
        # Статистика данных
        if self.train_dataset:
            stats = self.train_dataset.get_stats()
            print(f"\n📊 Статистика переменных (обучающий набор):")
            for var, var_stats in stats.items():
                print(f"  {var}: mean={var_stats['mean']:.3f}, std={var_stats['std']:.3f}")
    
    def train_dataloader(self) -> DataLoader:
        """DataLoader для обучения"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """DataLoader для валидации"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self) -> DataLoader:
        """DataLoader для тестирования"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def predict_dataloader(self) -> DataLoader:
        """DataLoader для предсказаний"""
        return self.test_dataloader()
    
    def get_sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Получение примера батча для отладки"""
        if self.train_dataset is None:
            raise ValueError("Датасеты не инициализированы. Вызовите setup() сначала.")
        
        sample_loader = DataLoader(self.train_dataset, batch_size=2, shuffle=False)
        return next(iter(sample_loader))
    
    def denormalize(self, data: torch.Tensor, var_name: str) -> torch.Tensor:
        """Денормализация данных"""
        if self.normalizer and var_name in self.normalizer:
            mean = self.normalizer[var_name]['mean']
            std = self.normalizer[var_name]['std']
            return data * std + mean
        return data
    
    def get_coords(self) -> Dict[str, np.ndarray]:
        """Получение координат"""
        if self.data is None:
            raise ValueError("Данные не загружены")
        
        coords = {}
        if self.spatial_subset:
            # Применяем пространственную подвыборку к координатам
            data_subset = self.data.sel(
                latitude=slice(self.spatial_subset['lat_min'], self.spatial_subset['lat_max']),
                longitude=slice(self.spatial_subset['lon_min'], self.spatial_subset['lon_max'])
            )
            coords['latitude'] = data_subset.latitude.values
            coords['longitude'] = data_subset.longitude.values
        else:
            coords['latitude'] = self.data.latitude.values
            coords['longitude'] = self.data.longitude.values
        
        coords['time'] = self.data.time.values
        return coords

# Функции-утилиты для работы с DataModule
def create_drought_datamodule(config: Dict) -> DroughtDataModule:
    """Фабрика для создания DroughtDataModule из конфигурации"""
    
    # Извлекаем параметры из конфигурации
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    
    # Создаем DataModule
    dm = DroughtDataModule(
        data_path=data_config.get('paths', {}).get('output_zarr', 'data/processed/real_agro_cube.zarr'),
        input_vars=data_config.get('input_variables', ['precipitation', 'temperature', 'ndvi']),
        target_var=data_config.get('target_variable', 'spi3'),
        sequence_length=training_config.get('sequence_length', 12),
        prediction_horizon=training_config.get('prediction_horizon', 3),
        train_years=tuple(training_config.get('train_years', [2003, 2016])),
        val_years=tuple(training_config.get('val_years', [2017, 2019])),
        test_years=tuple(training_config.get('test_years', [2020, 2024])),
        batch_size=training_config.get('batch_size', 32),
        num_workers=config.get('compute', {}).get('num_workers', 4),
        spatial_subset=config.get('spatial_subset'),
        normalize=data_config.get('normalize', True),
        augment_train=training_config.get('data_augmentation', {}).get('enabled', False)
    )
    
    return dm

def get_datamodule_stats(dm: DroughtDataModule) -> Dict:
    """Получение статистики DataModule"""
    
    if dm.train_dataset is None:
        raise ValueError("DataModule не настроен. Вызовите dm.setup() сначала.")
    
    stats = {
        'train_size': len(dm.train_dataset),
        'val_size': len(dm.val_dataset) if dm.val_dataset else 0,
        'test_size': len(dm.test_dataset) if dm.test_dataset else 0,
        'input_vars': dm.input_vars,
        'target_var': dm.target_var,
        'sequence_length': dm.sequence_length,
        'prediction_horizon': dm.prediction_horizon,
        'batch_size': dm.batch_size,
        'spatial_dims': None,
        'temporal_dims': None
    }
    
    # Получаем размеры данных из примера
    try:
        sample_input, sample_target = dm.get_sample_batch()
        stats['input_shape'] = list(sample_input.shape)
        stats['target_shape'] = list(sample_target.shape)
        stats['spatial_dims'] = sample_input.shape[-2:]  # (lat, lon)
        stats['temporal_dims'] = sample_input.shape[1]   # sequence_length
    except:
        pass
    
    # Статистика переменных
    if dm.train_dataset:
        stats['variable_stats'] = dm.train_dataset.get_stats()
    
    return stats

def validate_datamodule(dm: DroughtDataModule) -> Dict[str, bool]:
    """Валидация DataModule"""
    
    validation_results = {
        'data_loaded': False,
        'datasets_created': False,
        'dataloaders_work': False,
        'batch_shapes_correct': False,
        'no_nan_in_batches': False
    }
    
    try:
        # Проверяем загрузку данных
        if dm.data is not None:
            validation_results['data_loaded'] = True
        
        # Проверяем создание датасетов
        if dm.train_dataset and dm.val_dataset and dm.test_dataset:
            validation_results['datasets_created'] = True
        
        # Проверяем работу DataLoader'ов
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()
        
        # Пробуем получить батч
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        test_batch = next(iter(test_loader))
        
        validation_results['dataloaders_work'] = True
        
        # Проверяем форму батчей
        input_train, target_train = train_batch
        input_val, target_val = val_batch
        input_test, target_test = test_batch
        
        # Проверяем корректность размеров
        expected_input_dims = 5  # (batch, time, vars, lat, lon)
        expected_target_dims = 4  # (batch, horizon, lat, lon)
        
        if (len(input_train.shape) == expected_input_dims and 
            len(target_train.shape) == expected_target_dims):
            validation_results['batch_shapes_correct'] = True
        
        # Проверяем отсутствие NaN
        if (not torch.isnan(input_train).any() and 
            not torch.isnan(target_train).any() and
            not torch.isnan(input_val).any() and 
            not torch.isnan(target_val).any()):
            validation_results['no_nan_in_batches'] = True
        
    except Exception as e:
        print(f"Ошибка валидации: {e}")
    
    return validation_results

if __name__ == "__main__":
    # Пример использования
    print("📦 DroughtDataModule для PyTorch Lightning")
    
    # Пример конфигурации
    config = {
        'data': {
            'paths': {
                'output_zarr': 'data/processed/real_agro_cube.zarr'
            },
            'input_variables': ['precipitation', 'temperature', 'ndvi', 'soil_moisture'],
            'target_variable': 'spi3',
            'normalize': True
        },
        'training': {
            'sequence_length': 12,
            'prediction_horizon': 3,
            'train_years': [2003, 2016],
            'val_years': [2017, 2019],
            'test_years': [2020, 2024],
            'batch_size': 32,
            'data_augmentation': {
                'enabled': False
            }
        },
        'compute': {
            'num_workers': 4
        }
    }
    
    print("⚙️ Пример конфигурации:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n💡 Примеры использования:")
    print("# Создание DataModule из конфигурации")
    print("dm = create_drought_datamodule(config)")
    print("dm.setup()")
    print()
    print("# Получение статистики")
    print("stats = get_datamodule_stats(dm)")
    print()
    print("# Валидация DataModule")
    print("validation = validate_datamodule(dm)")
    print()
    print("# Использование в PyTorch Lightning")
    print("trainer = pl.Trainer()")
    print("trainer.fit(model, dm)")
    
    print("\n✅ DroughtDataModule готов к использованию!")
    
    # Доступные классы
    print("\n📋 Доступные классы:")
    print("  - DroughtDataset: Базовый датасет")
    print("  - DroughtDataModule: Основной DataModule")
    
    print("\n🔧 Основные функции:")
    print("  - create_drought_datamodule(): Создание из конфигурации")
    print("  - get_datamodule_stats(): Получение статистики")
    print("  - validate_datamodule(): Валидация корректности")