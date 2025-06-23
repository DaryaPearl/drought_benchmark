"""
Исправленный генератор синтетических данных без NaN
Замените содержимое файла src/data_pipeline/quick_data_pipeline.py
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def create_clean_synthetic_data():
    """Создание чистых синтетических данных без NaN"""
    print("🔄 Создание чистых синтетических данных...")
    
    # Временной диапазон
    time_range = pd.date_range('2020-01', '2022-12', freq='ME')
    
    # Пространственная сетка (небольшая для быстроты)
    lat_range = np.arange(40.0, 45.0, 1.0)  # 5 точек
    lon_range = np.arange(-100.0, -95.0, 1.0)  # 5 точек
    
    print(f"📐 Размеры: время={len(time_range)}, широта={len(lat_range)}, долгота={len(lon_range)}")
    
    # Инициализация данных
    n_time, n_lat, n_lon = len(time_range), len(lat_range), len(lon_range)
    
    # Случайные семена для воспроизводимости
    np.random.seed(42)
    
    # 1. Осадки (всегда положительные, без NaN)
    precipitation = np.zeros((n_time, n_lat, n_lon))
    for t, date in enumerate(time_range):
        # Сезонный компонент
        seasonal = 50 + 30 * np.sin(2 * np.pi * (date.month - 1) / 12)
        # Случайная компонента
        random_component = 20 * np.random.exponential(1, (n_lat, n_lon))
        precipitation[t] = seasonal + random_component
    
    # 2. Температура (реалистичная, без NaN)
    temperature = np.zeros((n_time, n_lat, n_lon))
    for t, date in enumerate(time_range):
        # Сезонность
        seasonal = 15 + 20 * np.sin(2 * np.pi * (date.month - 3) / 12)
        # Шум
        noise = 5 * np.random.randn(n_lat, n_lon)
        temperature[t] = seasonal + noise
    
    # 3. NDVI (зависит от осадков, без NaN)
    ndvi = np.zeros((n_time, n_lat, n_lon))
    for t in range(n_time):
        # Зависимость от осадков (нормализованная)
        precip_effect = np.tanh(precipitation[t] / 100) * 0.4
        # Базовый NDVI
        base_ndvi = 0.3
        # Сезонность
        seasonal_ndvi = 0.2 * np.sin(2 * np.pi * (time_range[t].month - 4) / 12)
        ndvi[t] = base_ndvi + precip_effect + seasonal_ndvi + 0.05 * np.random.randn(n_lat, n_lon)
        # Обрезаем в реалистичных пределах
        ndvi[t] = np.clip(ndvi[t], 0.1, 0.9)
    
    # 4. Влажность почвы (зависит от осадков, без NaN)
    soil_moisture = np.zeros((n_time, n_lat, n_lon))
    for t in range(n_time):
        # Базовая влажность + зависимость от осадков
        base_moisture = 0.3
        precip_effect = precipitation[t] / 500  # Влияние осадков
        soil_moisture[t] = base_moisture + precip_effect + 0.1 * np.random.randn(n_lat, n_lon)
        # Обрезаем в реалистичных пределах
        soil_moisture[t] = np.clip(soil_moisture[t], 0.1, 0.8)
    
    # 5. Расчет SPI-3 (простой, без NaN)
    spi3 = calculate_robust_spi(precipitation, window=3)
    
    # Проверяем на NaN
    for var_name, var_data in [
        ('precipitation', precipitation),
        ('temperature', temperature), 
        ('ndvi', ndvi),
        ('soil_moisture', soil_moisture),
        ('spi3', spi3)
    ]:
        nan_count = np.isnan(var_data).sum()
        if nan_count > 0:
            print(f"⚠ {var_name} содержит {nan_count} NaN значений - заменяем")
            # Заменяем NaN на средние значения
            var_data[np.isnan(var_data)] = np.nanmean(var_data)
    
    # Создание xarray Dataset
    dataset = xr.Dataset({
        'precipitation': (['time', 'latitude', 'longitude'], precipitation),
        'temperature': (['time', 'latitude', 'longitude'], temperature),
        'ndvi': (['time', 'latitude', 'longitude'], ndvi),
        'soil_moisture': (['time', 'latitude', 'longitude'], soil_moisture),
        'spi3': (['time', 'latitude', 'longitude'], spi3),
    }, coords={
        'time': time_range,
        'latitude': lat_range,
        'longitude': lon_range,
    })
    
    # Финальная проверка на NaN
    for var in dataset.data_vars:
        nan_count = np.isnan(dataset[var].values).sum()
        if nan_count > 0:
            print(f"🔧 Исправляем {nan_count} NaN в {var}")
            dataset[var] = dataset[var].fillna(dataset[var].mean())
    
    # Добавление метаданных
    dataset.attrs.update({
        'title': 'Clean Synthetic Drought Dataset',
        'description': 'Clean artificially generated data without NaN values',
        'created': pd.Timestamp.now().isoformat(),
        'spatial_resolution': '1.0 degrees',
        'temporal_resolution': 'monthly',
        'variables': list(dataset.data_vars.keys()),
        'quality': 'no_nan_values'
    })
    
    print("✅ Созданы чистые данные без NaN")
    return dataset

def calculate_robust_spi(precipitation, window=3):
    """Надежный расчет SPI без NaN"""
    print(f"🧮 Расчет надежного SPI-{window}...")
    
    T, H, W = precipitation.shape
    spi = np.zeros((T, H, W))  # Инициализируем нулями вместо NaN
    
    # Скользящая сумма
    for t in range(window, T):
        rolling_precip = np.sum(precipitation[t-window:t], axis=0)
        
        # Нормализация для каждого пикселя
        for i in range(H):
            for j in range(W):
                # Берем историю осадков для данного пикселя
                pixel_history = []
                for hist_t in range(window, t+1):
                    pixel_precip = np.sum(precipitation[hist_t-window:hist_t, i, j])
                    pixel_history.append(pixel_precip)
                
                if len(pixel_history) > 0:
                    mean_precip = np.mean(pixel_history)
                    std_precip = np.std(pixel_history)
                    
                    if std_precip > 0:
                        spi[t, i, j] = (rolling_precip[i, j] - mean_precip) / std_precip
                    else:
                        spi[t, i, j] = 0.0
                else:
                    spi[t, i, j] = 0.0
    
    # Обрезаем экстремальные значения
    spi = np.clip(spi, -3, 3)
    
    return spi

def main():
    """Главная функция для создания надежных тестовых данных"""
    print("🚀 Создание надежных тестовых данных без NaN")
    
    # Создание директорий
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Создание данных
    dataset = create_clean_synthetic_data()
    
    # Финальная проверка качества
    print("\n🔍 Финальная проверка качества данных:")
    total_nan = 0
    for var in dataset.data_vars:
        data = dataset[var].values
        nan_count = np.isnan(data).sum()
        total_nan += nan_count
        print(f"  {var}: {nan_count} NaN значений")
        print(f"    Диапазон: [{data.min():.3f}, {data.max():.3f}]")
        print(f"    Среднее: {data.mean():.3f}")
    
    if total_nan > 0:
        raise ValueError(f"Обнаружено {total_nan} NaN значений!")
    
    # Сохранение
    output_path = processed_dir / "real_agro_cube.zarr"
    print(f"\n💾 Сохранение данных в {output_path}...")
    
    dataset.to_zarr(output_path, mode='w')
    
    # Статистика
    print("\n📊 Статистика созданного датасета:")
    print(f"  📐 Размеры: {dict(dataset.dims)}")
    print(f"  📋 Переменные: {list(dataset.data_vars.keys())}")
    print(f"  📅 Временной диапазон: {dataset.time.min().values} - {dataset.time.max().values}")
    print(f"  🗺 Пространственный охват: {dataset.latitude.min().values:.1f}-{dataset.latitude.max().values:.1f}°N, {dataset.longitude.min().values:.1f}-{dataset.longitude.max().values:.1f}°E")
    print(f"  ✅ Качество: 0 NaN значений")
    
    # Проверка временных последовательностей
    print(f"\n🔄 Проверка последовательностей:")
    print(f"  Достаточно данных для sequence_length=12: {len(dataset.time) >= 12}")
    print(f"  Достаточно данных для обучения: {len(dataset.time) >= 20}")
    
    print(f"\n✅ Надежные тестовые данные успешно созданы!")
    print(f"📁 Местоположение: {output_path}")
    print(f"💡 Данные готовы для обучения моделей")
    
    return dataset

if __name__ == "__main__":
    main()