"""
Полный pipeline для реальных данных: CHIRPS + ERA5 + MODIS
Требует настройки:
1. CDS API для ERA5 (~/.cdsapirc)
2. Google Earth Engine для MODIS
3. NASA Earthdata для дополнительных источников

Запуск: python -m src.data_pipeline.real_data_pipeline
"""

import os
import sys
import time
from pathlib import Path
import warnings
import json
import datetime as dt
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import xarray as xr
import requests
from scipy.stats import gamma, norm

# Попытка импорта Google Earth Engine
try:
    import ee
    import geemap
    GEE_AVAILABLE = True
    print("✅ Google Earth Engine доступен")
except ImportError:
    GEE_AVAILABLE = False
    print("⚠ Google Earth Engine недоступен")

warnings.filterwarnings("ignore")

# Настройки
YEARS = range(2020, 2021)  # Полный период как в оригинале
OUT_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
ZARR_OUT = PROC_DIR / "real_agro_cube.zarr"

# Все регионы из оригинального проекта
REGIONS = {
    "us_plains": (35, 48, -104, -90),      # США - Великие равнины
    "br_cerrado": (-20, -6, -62, -46),     # Бразилия - Серрадо  
    "in_ganga": (21, 31, 73, 90),          # Индия - Ганг
    "ru_steppe": (50, 55, 37, 47),         # Россия - Центрально-Черноземный
}

# Глобальный bbox охватывающий все регионы
LAT_MIN = min(r[0] for r in REGIONS.values()) - 1
LAT_MAX = max(r[1] for r in REGIONS.values()) + 1  
LON_MIN = min(r[2] for r in REGIONS.values()) - 1
LON_MAX = max(r[3] for r in REGIONS.values()) + 1

GLOBAL_BOUNDS = {
    'lat_min': LAT_MIN,
    'lat_max': LAT_MAX,
    'lon_min': LON_MIN,
    'lon_max': LON_MAX
}

class GoogleEarthEngineSetup:
    """Настройка и инициализация Google Earth Engine"""
    
    @staticmethod
    def initialize_gee(project_id: Optional[str] = None) -> bool:
        """Инициализация GEE"""
        if not GEE_AVAILABLE:
            return False
            
        try:
            # Попытка инициализации с проектом
            if project_id:
                ee.Initialize(project=project_id)
                print(f"✅ GEE инициализирован с проектом: {project_id}")
            else:
                ee.Initialize()
                print("✅ GEE инициализирован")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка инициализации GEE: {e}")
            print("\n📋 Для настройки Google Earth Engine:")
            print("1. Зарегистрируйтесь: https://earthengine.google.com/")
            print("2. Создайте Google Cloud Project: https://console.cloud.google.com/")
            print("3. Включите Earth Engine API для проекта")
            print("4. Установите: pip install earthengine-api geemap")
            print("5. Выполните: earthengine authenticate")
            print("6. Установите переменную: export GEE_PROJECT_ID='your-project-id'")
            return False

class RealMODISDownloader:
    """Загрузчик реальных MODIS данных через Google Earth Engine"""
    
    def __init__(self, global_bounds: Dict):
        self.global_bounds = global_bounds
        self.roi = ee.Geometry.Rectangle([
            global_bounds['lon_min'], global_bounds['lat_min'],
            global_bounds['lon_max'], global_bounds['lat_max']
        ])
        
    def download_modis_ndvi(self, years: range) -> Optional[xr.Dataset]:
        """Загрузка реальных MODIS NDVI данных для всех регионов"""
        print("🛰 Загрузка реальных MODIS NDVI через Google Earth Engine...")
        print(f"🗺 Глобальная область: {self.global_bounds}")
        print(f"📍 Регионы: {list(REGIONS.keys())}")
        
        # Создаем список дат
        start_date = f"{years[0]}-01-01"
        end_date = f"{years[-1]}-12-31"
        
        print(f"📅 Период: {start_date} - {end_date}")
        
        try:
            # MODIS Terra Vegetation Indices (MOD13A2) - 16-дневные композиты 1км
            modis = ee.ImageCollection("MODIS/061/MOD13A2") \
                .filterDate(start_date, end_date) \
                .filterBounds(self.roi) \
                .select(['NDVI', 'EVI', ])
            
            print(f"📊 Найдено {modis.size().getInfo()} MODIS изображений")
            
            if modis.size().getInfo() == 0:
                print("❌ Нет доступных MODIS данных для указанного периода")
                return None
            
            # Функция обработки изображений
            def process_modis(image):
                # Масштабирование NDVI и EVI (scale factor = 0.0001)
                ndvi = image.select('NDVI').multiply(0.0001)
                evi = image.select('EVI').multiply(0.0001)
                
                # Маска качества (используем только надежные пиксели)
                quality = image.select('DetailedQA')
                good_pixels = quality.eq(0)  # 0 = хорошее качество
                
                # Применяем маску
                ndvi_masked = ndvi.updateMask(good_pixels)
                evi_masked = evi.updateMask(good_pixels)
                
                return ee.Image.cat([
                    ndvi_masked.rename('ndvi'),
                    evi_masked.rename('evi')
                ]).copyProperties(image, ['system:time_start'])
            
            # Обрабатываем коллекцию
            modis_processed = modis.map(process_modis)
            
            # Создаем месячные композиты
            def create_monthly_composite(year_month):
                year = ee.Number(year_month).divide(100).floor()
                month = ee.Number(year_month).mod(100)
                
                start = ee.Date.fromYMD(year, month, 1)
                end = start.advance(1, 'month')
                
                monthly = modis_processed.filterDate(start, end)
                
                return ee.Algorithms.If(
                    monthly.size().gt(0),
                    monthly.median().set({
                        'year': year,
                        'month': month,
                        'system:time_start': start.millis()
                    }),
                    None
                )
            
            # Список всех месяцев
            months_list = []
            for year in years:
                for month in range(1, 13):
                    months_list.append(year * 100 + month)
            
            monthly_images = [create_monthly_composite(ym) for ym in months_list]
            
            # Фильтруем None значения
            monthly_collection = ee.ImageCollection(
                ee.List(monthly_images).filter(ee.Filter.neq('item', None))
            )
            
            actual_count = monthly_collection.size().getInfo()
            print(f"📊 Создано {actual_count} месячных композитов")
            
            if actual_count == 0:
                print("❌ Не удалось создать месячные композиты")
                return None
            
            # Экспорт данных
            return self._export_to_xarray(monthly_collection, years)
            
        except Exception as e:
            print(f"❌ Ошибка загрузки MODIS: {e}")
            return None
    
    def _export_to_xarray(self, collection: ee.ImageCollection, years: range) -> xr.Dataset:
        """Экспорт GEE коллекции в xarray"""
        print("📦 Экспорт MODIS данных в xarray...")
        
        # Определяем пространственную сетку для всех регионов
        scale = 1000  # 1км в метрах
        
        # Создаем координаты для глобального охвата
        lon_coords = np.arange(
            self.global_bounds['lon_min'], 
            self.global_bounds['lon_max'], 
            0.01  # ~1км
        )
        lat_coords = np.arange(
            self.global_bounds['lat_max'], 
            self.global_bounds['lat_min'], 
            -0.01  # Убывающая последовательность
        )
        
        # Временные координаты
        time_coords = pd.date_range(
            f"{years[0]}-01-01", 
            f"{years[-1]}-12-31", 
            freq='MS'
        )
        
        # Инициализируем массивы данных
        ndvi_data = np.full((len(time_coords), len(lat_coords), len(lon_coords)), np.nan)
        evi_data = np.full((len(time_coords), len(lat_coords), len(lon_coords)), np.nan)
        
        # Получаем список изображений
        img_list = collection.toList(collection.size())
        n_images = collection.size().getInfo()
        
        print(f"🔄 Обработка {n_images} изображений...")
        
        for i in range(min(n_images, len(time_coords))):
            try:
                print(f"  📅 Обработка {time_coords[i].strftime('%Y-%m')} ({i+1}/{n_images})")
                
                # Получаем изображение
                img = ee.Image(img_list.get(i))
                
                # Экспорт через geemap
                try:
                    # NDVI
                    ndvi_array = geemap.ee_to_numpy(
                        img.select('ndvi'),
                        region=self.roi,
                        scale=scale,
                        crs='EPSG:4326'
                    )
                    
                    # EVI  
                    evi_array = geemap.ee_to_numpy(
                        img.select('evi'),
                        region=self.roi,
                        scale=scale,
                        crs='EPSG:4326'
                    )
                    
                    if ndvi_array is not None and evi_array is not None:
                        # Изменяем размер если нужно
                        if ndvi_array.shape != (len(lat_coords), len(lon_coords)):
                            from scipy.ndimage import zoom
                            zoom_factors = (
                                len(lat_coords) / ndvi_array.shape[0],
                                len(lon_coords) / ndvi_array.shape[1]
                            )
                            ndvi_array = zoom(ndvi_array, zoom_factors, order=1)
                            evi_array = zoom(evi_array, zoom_factors, order=1)
                        
                        ndvi_data[i] = ndvi_array
                        evi_data[i] = evi_array
                        
                        print(f"    ✅ Успешно обработано")
                    else:
                        print(f"    ⚠ Пустые данные")
                        
                except Exception as export_error:
                    print(f"    ❌ Ошибка экспорта: {export_error}")
                    continue
                    
            except Exception as img_error:
                print(f"    ❌ Ошибка изображения: {img_error}")
                continue
        
        # Подсчет успешно загруженных месяцев
        valid_ndvi = ~np.isnan(ndvi_data).all(axis=(1, 2))
        success_rate = valid_ndvi.sum() / len(time_coords) * 100
        
        print(f"📊 Успешно загружено: {valid_ndvi.sum()}/{len(time_coords)} месяцев ({success_rate:.1f}%)")
        
        # Создаем xarray Dataset
        modis_ds = xr.Dataset({
            'ndvi': (['time', 'latitude', 'longitude'], ndvi_data),
            'evi': (['time', 'latitude', 'longitude'], evi_data),
        }, coords={
            'time': time_coords,
            'latitude': lat_coords,
            'longitude': lon_coords,
        })
        
        # Метаданные
        modis_ds.attrs.update({
            'title': 'Real MODIS NDVI/EVI from Google Earth Engine',
            'source': 'MODIS/061/MOD13A2',
            'description': 'Monthly composites of MODIS NDVI and EVI',
            'spatial_resolution': '1km',
            'temporal_resolution': 'monthly',
            'download_method': 'Google Earth Engine',
            'success_rate': f'{success_rate:.1f}%',
            'global_bounds': self.global_bounds,
            'regions_covered': list(REGIONS.keys())
        })
        
        print(f"✅ MODIS данные готовы: {dict(modis_ds.dims)}")
        return modis_ds

class CHIRPSDownloader:
    """Загрузчик реальных CHIRPS данных для всех регионов"""
    
    BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p25"
    
    @staticmethod
    def download_and_process(years: range, global_bounds: Dict, dest_dir: Path) -> xr.Dataset:
        """Загрузка и обработка CHIRPS для всех регионов"""
        print("📦 Загрузка реальных данных CHIRPS для всех регионов...")
        print(f"🌍 Глобальная область: {global_bounds}")
        print(f"📍 Регионы: {list(REGIONS.keys())}")
        
        datasets = []
        
        for year in years:
            dest_file = dest_dir / f"chirps_{year}.nc"
            
            if not dest_file.exists():
                url = f"{CHIRPSDownloader.BASE_URL}/chirps-v2.0.{year}.days_p25.nc"
                print(f"⬇ Загрузка CHIRPS {year}...")
                
                try:
                    response = requests.get(url, stream=True, timeout=1800)  # 30 минут
                    response.raise_for_status()
                    
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    
                    with open(dest_file, "wb") as f:
                        for chunk in response.iter_content(chunk_size=1024*1024):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                
                                # Показываем прогресс каждые 100MB
                                if downloaded % (100 * 1024 * 1024) == 0:
                                    if total_size > 0:
                                        progress = (downloaded / total_size) * 100
                                        print(f"    📥 {progress:.0f}%", end="", flush=True)
                    
                    file_size = dest_file.stat().st_size / (1024**2)
                    print(f" ✅ ({file_size:.1f} MB)")
                    
                except Exception as e:
                    print(f"❌ Ошибка загрузки CHIRPS {year}: {e}")
                    continue
            else:
                file_size = dest_file.stat().st_size / (1024**2)
                print(f"📁 CHIRPS {year} уже загружен ({file_size:.1f} MB)")
            
            # Загружаем файл
            try:
                ds = xr.open_dataset(dest_file)
                datasets.append(ds)
                print(f"✅ CHIRPS {year} прочитан")
            except Exception as e:
                print(f"⚠ Ошибка чтения CHIRPS {year}: {e}")
        
        if not datasets:
            raise RuntimeError("❌ Не удалось загрузить CHIRPS данные")
        
        print(f"🔗 Объединение {len(datasets)} файлов CHIRPS...")
        combined = xr.concat(datasets, dim="time")
        
        # Обрезка по глобальной области всех регионов
        lat_slice = slice(global_bounds['lat_min'], global_bounds['lat_max'])
        if combined.latitude[0] > combined.latitude[-1]:
            lat_slice = slice(global_bounds['lat_max'], global_bounds['lat_min'])
            
        combined = combined.sel(
            latitude=lat_slice,
            longitude=slice(global_bounds['lon_min'], global_bounds['lon_max'])
        )
        
        # Месячные суммы
        print("📊 Вычисление месячных сумм осадков...")
        monthly = combined.resample(time="1M").sum()
        
        print(f"✅ CHIRPS готов: {dict(monthly.dims)}")
        print(f"📊 Временной диапазон: {monthly.time.min().values} - {monthly.time.max().values}")
        
        # Проверяем покрытие регионов
        print("🗺 Проверка покрытия регионов:")
        for region_name, (lat_min, lat_max, lon_min, lon_max) in REGIONS.items():
            try:
                region_data = monthly.sel(
                    latitude=slice(lat_min, lat_max),
                    longitude=slice(lon_min, lon_max)
                )
                if len(region_data.latitude) > 0 and len(region_data.longitude) > 0:
                    coverage = 1 - np.isnan(region_data['precip'].values).mean()
                    print(f"  {region_name}: {coverage*100:.1f}% покрытие, "
                          f"{len(region_data.latitude)}x{len(region_data.longitude)} пикселей")
                else:
                    print(f"  {region_name}: ❌ нет данных")
            except Exception as e:
                print(f"  {region_name}: ⚠ ошибка проверки: {e}")
        
        return monthly

class ERA5Downloader:
    """Загрузчик ERA5 для всех регионов"""
    
    @staticmethod
    def check_and_download(years: range, global_bounds: Dict, dest_dir: Path) -> Optional[xr.Dataset]:
        """Проверка и загрузка ERA5 для всех регионов"""
        
        # Проверяем CDS API
        cds_rc = Path.home() / ".cdsapirc"
        if not cds_rc.exists():
            print("⚠ ERA5 пропущен: нет ~/.cdsapirc")
            print("💡 Создайте файл ~/.cdsapirc с вашим CDS API ключом")
            return None
            
        try:
            import cdsapi
        except ImportError:
            print("⚠ ERA5 пропущен: нет cdsapi")
            print("💡 Установите: pip install cdsapi")
            return None
        
        era5_file = dest_dir / "era5_global_all_regions.nc"
        
        if era5_file.exists():
            print("📁 ERA5 уже загружен")
            return xr.open_dataset(era5_file)
        
        print("⬇ Загрузка ERA5-Land для всех регионов...")
        print(f"🌍 Глобальная область: {global_bounds}")
        
        client = cdsapi.Client()
        
        try:
            # Запрашиваем больше переменных для всех регионов
            client.retrieve(
                'reanalysis-era5-land',
                {
                    'variable': [
                        '2m_temperature',
                        'volumetric_soil_water_layer_1',
                        'potential_evaporation',
                        'total_precipitation'
                    ],
                    'year': [str(y) for y in years],
                    'month': [f'{m:02d}' for m in range(1, 13)],
                    'day': '15',  # Середина месяца
                    'time': '12:00',
                    'area': [
                        global_bounds['lat_max'], global_bounds['lon_min'],
                        global_bounds['lat_min'], global_bounds['lon_max']
                    ],
                    'format': 'netcdf',
                },
                str(era5_file)
            )
            
            ds = xr.open_dataset(era5_file)
            print(f"✅ ERA5 готов: {dict(ds.dims)}")
            
            # Проверяем покрытие регионов
            print("🗺 Проверка покрытия ERA5 по регионам:")
            for region_name, (lat_min, lat_max, lon_min, lon_max) in REGIONS.items():
                try:
                    region_data = ds.sel(
                        latitude=slice(lat_min, lat_max),
                        longitude=slice(lon_min, lon_max)
                    )
                    if len(region_data.latitude) > 0 and len(region_data.longitude) > 0:
                        print(f"  {region_name}: ✅ {len(region_data.latitude)}x{len(region_data.longitude)} пикселей")
                    else:
                        print(f"  {region_name}: ❌ нет данных")
                except Exception as e:
                    print(f"  {region_name}: ⚠ ошибка: {e}")
            
            return ds
            
        except Exception as e:
            print(f"❌ Ошибка загрузки ERA5: {e}")
            print("💡 Проверьте настройки CDS API и квоты")
            return None

def calculate_drought_indices(precip_data: np.ndarray) -> Dict[str, np.ndarray]:
    """Расчет индексов засухи"""
    print("🧮 Расчет индексов засухи...")
    
    T, H, W = precip_data.shape
    
    # SPI-3
    spi3 = np.zeros((T, H, W))
    
    for t in range(3, T):
        rolling_sum = np.sum(precip_data[t-3:t], axis=0)
        
        for i in range(H):
            for j in range(W):
                # История для пикселя
                history = []
                for ht in range(3, t+1):
                    pixel_sum = np.sum(precip_data[ht-3:ht, i, j])
                    history.append(pixel_sum)
                
                if len(history) > 10:
                    mean_val = np.mean(history)
                    std_val = np.std(history)
                    if std_val > 0:
                        spi3[t, i, j] = (rolling_sum[i, j] - mean_val) / std_val
    
    spi3 = np.clip(spi3, -3, 3)
    
    return {'spi3': spi3}

def main():
    """Главная функция с реальными данными для всех регионов"""
    print("🌍 Загрузка РЕАЛЬНЫХ данных для всех регионов: CHIRPS + ERA5 + MODIS")
    print("=" * 70)
    print(f"📍 Регионы: {list(REGIONS.keys())}")
    print(f"📅 Период: {YEARS[0]}-{YEARS[-1]}")
    print(f"🌍 Глобальная область: {GLOBAL_BOUNDS}")
    
    # Проверяем настройки
    gee_project = os.getenv('GEE_PROJECT_ID')
    if not gee_project:
        print("⚠ Переменная GEE_PROJECT_ID не установлена")
        print("💡 Установите: export GEE_PROJECT_ID='your-google-cloud-project-id'")
    
    # Создаем директории
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Инициализация Google Earth Engine
        if GEE_AVAILABLE:
            gee_ready = GoogleEarthEngineSetup.initialize_gee(gee_project)
        else:
            gee_ready = False
        
        # 2. Загрузка CHIRPS (реальные данные для всех регионов)
        print("\n1️⃣ Загрузка реальных CHIRPS данных для всех регионов...")
        chirps_ds = CHIRPSDownloader.download_and_process(YEARS, GLOBAL_BOUNDS, OUT_DIR)
        
        # 3. Загрузка MODIS (реальные данные через GEE для всех регионов)
        print("\n2️⃣ Загрузка реальных MODIS данных для всех регионов...")
        if gee_ready:
            modis_downloader = RealMODISDownloader(GLOBAL_BOUNDS)
            modis_ds = modis_downloader.download_modis_ndvi(YEARS)
        else:
            print("❌ MODIS пропущен: GEE недоступен")
            modis_ds = None
        
        # 4. Загрузка ERA5 (для всех регионов)
        print("\n3️⃣ Попытка загрузки ERA5 для всех регионов...")
        era5_ds = ERA5Downloader.check_and_download(YEARS, GLOBAL_BOUNDS, OUT_DIR)
        
        # 5. Объединение данных
        print("\n4️⃣ Объединение всех данных...")
        
        # Базовые координаты от CHIRPS
        target_coords = {
            'time': chirps_ds.time,
            'latitude': chirps_ds.latitude,
            'longitude': chirps_ds.longitude
        }
        
        # Начинаем с CHIRPS
        final_vars = {
            'precipitation': (['time', 'latitude', 'longitude'], chirps_ds['precip'].values)
        }
        
        # Добавляем MODIS если есть
        if modis_ds is not None:
            modis_interp = modis_ds.interp(
                latitude=target_coords['latitude'],
                longitude=target_coords['longitude'],
                time=target_coords['time'],
                method='linear'
            )
            final_vars['ndvi'] = (['time', 'latitude', 'longitude'], modis_interp['ndvi'].values)
            final_vars['evi'] = (['time', 'latitude', 'longitude'], modis_interp['evi'].values)
        
        # Добавляем ERA5 если есть
        if era5_ds is not None:
            era5_interp = era5_ds.interp(
                latitude=target_coords['latitude'],
                longitude=target_coords['longitude'],
                time=target_coords['time'],
                method='linear'
            )
            final_vars['temperature'] = (['time', 'latitude', 'longitude'], era5_interp['t2m'].values - 273.15)
            final_vars['soil_moisture'] = (['time', 'latitude', 'longitude'], era5_interp['swvl1'].values)
        
        # 6. Расчет индексов засухи
        print("\n5️⃣ Расчет индексов засухи...")
        drought_indices = calculate_drought_indices(chirps_ds['precip'].values)
        
        for idx_name, idx_data in drought_indices.items():
            final_vars[idx_name] = (['time', 'latitude', 'longitude'], idx_data)
        
        # 7. Создание финального датасета
        final_ds = xr.Dataset(final_vars, coords=target_coords)
        
        # Метаданные
        data_sources = {
            'precipitation': 'CHIRPS v2.0 (real)',
            'drought_indices': 'Calculated from CHIRPS'
        }
        
        if modis_ds is not None:
            data_sources['ndvi'] = 'MODIS MOD13A2 via Google Earth Engine (real)'
            data_sources['evi'] = 'MODIS MOD13A2 via Google Earth Engine (real)'
            
        if era5_ds is not None:
            data_sources['temperature'] = 'ERA5-Land reanalysis (real)'
            data_sources['soil_moisture'] = 'ERA5-Land reanalysis (real)'
            data_sources['potential_evaporation'] = 'ERA5-Land reanalysis (real)'
        
        final_ds.attrs.update({
            'title': 'Real Multi-Source Multi-Region Drought Dataset',
            'description': 'Combined real satellite and reanalysis data for multiple agricultural regions',
            'data_sources': data_sources,
            'regions': REGIONS,
            'region_bounds': REGION_BOUNDS,
            'global_bounds': GLOBAL_BOUNDS,
            'spatial_resolution': '0.25 degrees',
            'temporal_resolution': 'monthly',
            'time_range': f'{YEARS[0]}-{YEARS[-1]}',
            'creation_date': dt.datetime.now().isoformat(),
            })

        
        # 8. Проверка качества по регионам
        print("\n6️⃣ Проверка качества данных по регионам...")
        
        print("\n📊 Общая статистика:")
        for var in final_ds.data_vars:
            data = final_ds[var].values
            nan_pct = np.isnan(data).mean() * 100
            print(f"  {var}: {nan_pct:.1f}% NaN, диапазон [{np.nanmin(data):.2f}, {np.nanmax(data):.2f}]")
        
        print("\n🗺 Покрытие данными по регионам:")
        for region_name, (lat_min, lat_max, lon_min, lon_max) in REGIONS.items():
            try:
                region_data = final_ds.sel(
                    latitude=slice(lat_min, lat_max),
                    longitude=slice(lon_min, lon_max)
                )
                
                if len(region_data.latitude) > 0 and len(region_data.longitude) > 0:
                    # Проверяем покрытие для каждой переменной
                    print(f"  📍 {region_name}:")
                    print(f"    Размер: {len(region_data.latitude)}x{len(region_data.longitude)} пикселей")
                    print(f"    Временной диапазон: {region_data.time.min().values} - {region_data.time.max().values}")
                    
                    for var in final_ds.data_vars:
                        if var in region_data.data_vars:
                            var_data = region_data[var].values
                            coverage = (1 - np.isnan(var_data).mean()) * 100
                            print(f"    {var}: {coverage:.1f}% покрытие")
                        
                else:
                    print(f"  📍 {region_name}: ❌ нет данных в этом регионе")
                    
            except Exception as e:
                print(f"  📍 {region_name}: ⚠ ошибка проверки: {e}")
        
        # 9. Сохранение
        print(f"\n💾 Сохранение в {ZARR_OUT}...")
        final_ds.to_zarr(ZARR_OUT, mode='w')
        
        # Финальный отчет
        file_size = ZARR_OUT.stat().st_size / (1024**2)  # MB
        print(f"\n🎉 Реальный мультирегиональный датасет успешно создан!")
        print(f"📁 Местоположение: {ZARR_OUT}")
        print(f"💽 Размер: {file_size:.1f} MB")
        print(f"📊 Переменные: {list(final_ds.data_vars.keys())}")
        print(f"📐 Размерность: {dict(final_ds.dims)}")
        print(f"📍 Регионы: {list(REGIONS.keys())}")
        print(f"🌍 Источники данных:")
        for var, source in data_sources.items():
            print(f"  • {var}: {source}")
        
        # Сохраняем метаданные отдельно
        metadata_file = PROC_DIR / "dataset_metadata.json"
        metadata = {
            'regions': REGIONS,
            'global_bounds': GLOBAL_BOUNDS,
            'years': list(YEARS),
            'data_sources': data_sources,
            'variables': list(final_ds.data_vars.keys()),
            'dimensions': dict(final_ds.dims),
            'creation_date': dt.datetime.now().isoformat(),
            'file_size_mb': file_size
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"📋 Метаданные сохранены: {metadata_file}")
        print("\n✅ Готов к использованию для обучения моделей на всех регионах!")
        
        return final_ds
        
    except KeyboardInterrupt:
        print("\n⏹ Процесс прерван пользователем")
        return None
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()