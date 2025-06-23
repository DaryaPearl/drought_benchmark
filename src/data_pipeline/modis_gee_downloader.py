"""
Дополнение к src/data_pipeline/real_data_pipeline.py для реальных MODIS данных
Сохраните как: src/data_pipeline/modis_gee_downloader.py
"""

import ee
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import geemap
import warnings
warnings.filterwarnings("ignore")

class MODISGEEDownloader:
    """Загрузчик реальных MODIS данных через Google Earth Engine"""
    
    def __init__(self, project_id=None):
        """
        project_id: Google Cloud Project ID
        Создайте проект на https://console.cloud.google.com/
        """
        self.project_id = project_id
        self._initialize_ee()
    
    def _initialize_ee(self):
        """Инициализация Earth Engine"""
        try:
            if self.project_id:
                ee.Initialize(project=self.project_id)
            else:
                ee.Initialize()
            print("✅ Google Earth Engine инициализирован")
        except Exception as e:
            print(f"❌ Ошибка инициализации Earth Engine: {e}")
            print("💡 Запустите: earthengine authenticate")
            print("💡 Создайте проект: https://console.cloud.google.com/")
            raise
    
    def download_modis_ndvi_real(self, chirps_file='data/raw/chirps_spi.nc', 
                                output_file='data/raw/modis_ndvi_real.nc'):
        """Загрузка реальных MODIS NDVI данных"""
        
        print("🛰️ Загрузка реальных MODIS NDVI данных через Google Earth Engine...")
        
        # Загружаем CHIRPS для координат и временного диапазона
        chirps = xr.open_dataset(chirps_file)
        
        # Определяем область интереса
        lon_min, lon_max = float(chirps.longitude.min()), float(chirps.longitude.max())
        lat_min, lat_max = float(chirps.latitude.min()), float(chirps.latitude.max())
        
        print(f"📍 Область: {lat_min:.1f}-{lat_max:.1f}°N, {lon_min:.1f}-{lon_max:.1f}°E")
        
        # Создаем геометрию области
        roi = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
        
        # Временной диапазон
        start_date = chirps.time.min().values
        end_date = chirps.time.max().values
        
        print(f"📅 Период: {pd.Timestamp(start_date).strftime('%Y-%m')} - {pd.Timestamp(end_date).strftime('%Y-%m')}")
        
        # Загружаем MODIS NDVI (MOD13A2 - 16-дневные композиты 1км)
        modis = ee.ImageCollection('MODIS/061/MOD13A2') \
            .filterDate(pd.Timestamp(start_date).strftime('%Y-%m-%d'), 
                       pd.Timestamp(end_date).strftime('%Y-%m-%d')) \
            .filterBounds(roi) \
            .select(['NDVI', 'EVI'])
        
        print(f"📊 Найдено {modis.size().getInfo()} MODIS изображений")
        
        # Функция для обработки MODIS данных
        def process_modis(image):
            # Масштабирование (MODIS NDVI имеет scale factor 0.0001)
            ndvi = image.select('NDVI').multiply(0.0001)
            evi = image.select('EVI').multiply(0.0001)
            
            # Обрезка по области
            ndvi = ndvi.clip(roi)
            evi = evi.clip(roi)
            
            return ee.Image.cat([ndvi.rename('ndvi'), evi.rename('evi')]) \
                .copyProperties(image, ['system:time_start'])
        
        # Обрабатываем коллекцию
        modis_processed = modis.map(process_modis)
        
        # Создаем месячные композиты
        def create_monthly_composite(year, month):
            start = ee.Date.fromYMD(year, month, 1)
            end = start.advance(1, 'month')
            
            monthly = modis_processed.filterDate(start, end)
            
            # Если есть данные за месяц, создаем композит
            return ee.Algorithms.If(
                monthly.size().gt(0),
                monthly.median().set({
                    'year': year,
                    'month': month,
                    'system:time_start': start.millis()
                }),
                None
            )
        
        # Создаем список всех месяцев
        time_range = pd.date_range(
            pd.Timestamp(start_date).strftime('%Y-%m'),
            pd.Timestamp(end_date).strftime('%Y-%m'),
            freq='MS'
        )
        
        monthly_images = []
        for date in time_range:
            composite = create_monthly_composite(date.year, date.month)
            monthly_images.append(composite)
        
        # Фильтруем None значения
        monthly_collection = ee.ImageCollection(
            ee.List(monthly_images).filter(ee.Filter.neq('item', None))
        )
        
        print(f"📊 Создано {monthly_collection.size().getInfo()} месячных композитов")
        
        # Экспорт данных
        print("⬇️ Экспорт данных из Google Earth Engine...")
        
        # Определяем разрешение на основе CHIRPS (0.25 градуса = ~27.5 км)
        scale = 0.25 * 111320  # метры
        
        # Создаем координатную сетку как в CHIRPS
        lat_coords = chirps.latitude.values
        lon_coords = chirps.longitude.values
        time_coords = time_range
        
        # Инициализируем массивы данных
        ndvi_data = np.full((len(time_coords), len(lat_coords), len(lon_coords)), np.nan)
        evi_data = np.full((len(time_coords), len(lat_coords), len(lon_coords)), np.nan)
        
        # Экспортируем данные по месяцам (для избежания лимитов GEE)
        monthly_list = monthly_collection.toList(monthly_collection.size())
        
        for i in range(len(time_coords)):
            try:
                print(f"  📅 Обработка {time_coords[i].strftime('%Y-%m')} ({i+1}/{len(time_coords)})")
                
                # Получаем изображение для месяца
                image = ee.Image(monthly_list.get(i))
                
                # Проверяем что изображение валидно
                try:
                    image_info = image.getInfo()
                    if not image_info:
                        print(f"    ⚠️ Пустое изображение для {time_coords[i].strftime('%Y-%m')}")
                        continue
                except:
                    print(f"    ⚠️ Ошибка доступа к изображению для {time_coords[i].strftime('%Y-%m')}")
                    continue
                
                # Экспортируем как numpy array
                try:
                    # Используем geemap для экспорта
                    ndvi_array = geemap.ee_to_numpy(
                        image.select('ndvi'),
                        region=roi,
                        scale=scale,
                        bands=['ndvi']
                    )
                    
                    evi_array = geemap.ee_to_numpy(
                        image.select('evi'),
                        region=roi,
                        scale=scale,
                        bands=['evi']
                    )
                    
                    if ndvi_array is not None and evi_array is not None:
                        # Изменяем размер если нужно
                        if ndvi_array.shape != (len(lat_coords), len(lon_coords)):
                            from scipy.ndimage import zoom
                            zoom_factors = (len(lat_coords) / ndvi_array.shape[0], 
                                          len(lon_coords) / ndvi_array.shape[1])
                            ndvi_array = zoom(ndvi_array, zoom_factors, order=1)
                            evi_array = zoom(evi_array, zoom_factors, order=1)
                        
                        ndvi_data[i] = ndvi_array
                        evi_data[i] = evi_array
                        print(f"    ✅ {time_coords[i].strftime('%Y-%m')} обработан")
                    
                except Exception as e:
                    print(f"    ❌ Ошибка экспорта для {time_coords[i].strftime('%Y-%m')}: {e}")
                    continue
                    
            except Exception as e:
                print(f"    ❌ Общая ошибка для {time_coords[i].strftime('%Y-%m')}: {e}")
                continue
        
        # Подсчет успешно загруженных месяцев
        valid_ndvi = ~np.isnan(ndvi_data).all(axis=(1,2))
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
        
        # Добавляем метаданные
        modis_ds.attrs.update({
            'title': 'Real MODIS NDVI/EVI Data from Google Earth Engine',
            'source': 'MODIS/061/MOD13A2',
            'description': 'Monthly composites of MODIS NDVI and EVI',
            'spatial_resolution': '1km (resampled to CHIRPS grid)',
            'temporal_resolution': 'monthly',
            'download_method': 'Google Earth Engine',
            'success_rate': f'{success_rate:.1f}%'
        })
        
        # Сохраняем
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        modis_ds.to_netcdf(output_file)
        
        file_size = Path(output_file).stat().st_size / 1024 / 1024
        print(f"\n✅ Реальные MODIS данные сохранены:")
        print(f"  📁 Файл: {output_file}")
        print(f"  💽 Размер: {file_size:.1f} MB")
        print(f"  📊 Переменные: NDVI, EVI")
        print(f"  🎯 Успешность: {success_rate:.1f}%")
        
        return modis_ds

def main():
    """Основная функция для загрузки MODIS данных"""
    
    # Замените на ваш Google Cloud Project ID
    PROJECT_ID = "abstract-maker-450111-n5"  # Например: "drought-prediction-12345"
    
    try:
        downloader = MODISGEEDownloader(project_id=PROJECT_ID)
        modis_data = downloader.download_modis_ndvi_real()
        
        print("\n🎉 Реальные MODIS данные успешно загружены!")
        print("💡 Теперь можете запустить полный pipeline с реальными данными")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("\n💡 Шаги для настройки Google Earth Engine:")
        print("1. Зарегистрируйтесь: https://earthengine.google.com/")
        print("2. Создайте проект: https://console.cloud.google.com/")
        print("3. Запустите: earthengine authenticate")
        print("4. Установите: pip install earthengine-api geemap")

if __name__ == "__main__":
    main()