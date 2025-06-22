"""
Google Earth Engine pipeline for drought dataset with CHIRPS, MODIS, ERA5, and soil moisture
Calculates SPI, SPEI, and PDSI for multiple regions including Russia

Setup:
1. pip install earthengine-api geemap xarray rioxarray
2. earthengine authenticate
3. python -m src.data_pipeline.gee_drought_pipeline
"""

import ee
import os
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
import geemap
from typing import Dict, Tuple, List
import warnings

warnings.filterwarnings("ignore")

# # Initialize Earth Engine
# try:
#     ee.Initialize()
# except:
#     ee.Authenticate()
#     ee.Initialize()

# Initialize Earth Engine
PROJECT_ID = 'abstract-maker-450111-n5'  # замени на свой реальный GCP project ID

try:
    ee.Initialize(project=PROJECT_ID)
except Exception as e:
    print(f'Initial EE initialization failed: {e}')
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)


# Configuration
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Regions including Russia
REGIONS = {
    "us_plains": (35, 48, -104, -90),
    # "br_cerrado": (-20, -6, -62, -46),
    # "in_ganga": (21, 31, 73, 90),
    # "ru_volga": (48, 58, 40, 55),
    # "ru_south": (43, 48, 36, 48),
    # "ru_siberia": (52, 56, 80, 95),
    # "ru_altai": (50, 54, 82, 88),
}

# Time range
START_DATE = "2022-01-01"
END_DATE = "2023-12-31"

# Output resolution in degrees
SCALE = 0.25  # ~27.5 km at equator


class GEEDroughtData:
    """Download and process drought-related data using Google Earth Engine"""
    
    def __init__(self, region_name: str, bbox: Tuple[float, float, float, float]):
        self.region_name = region_name
        self.bbox = bbox
        self.roi = ee.Geometry.Rectangle([bbox[2], bbox[0], bbox[3], bbox[1]])
        self.dates = pd.date_range(START_DATE, END_DATE, freq='MS')
        self.cache_dir = OUT_DIR / f"{self.region_name}_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True) # Ensure cache directory exists
        
    def get_chirps_precipitation(self) -> xr.Dataset:
        """Get CHIRPS precipitation data"""
        print(f"  📥 Getting CHIRPS precipitation from GEE...")
        
        # CHIRPS daily collection
        chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
            .filterBounds(self.roi) \
            .select(['precipitation'])
        
        # Convert to monthly sums
        def monthly_sum(start_date):
            start = ee.Date(start_date)
            end = start.advance(1, 'month')
            
            monthly = chirps.filterDate(start, end).sum()
            
            # Set time property
            return monthly.set({
                'system:time_start': start.millis(),
                'date': start.format('YYYY-MM-dd')
            })
        
        # Map over all months
        monthly_list = []
        for date in self.dates:
            img = monthly_sum(date.strftime('%Y-%m-%d'))
            monthly_list.append(img)
        
        # Create collection
        monthly_col = ee.ImageCollection(monthly_list)
        
        # Convert to xarray
        ds = self._collection_to_xarray(monthly_col, 'precipitation', self.dates)
        
        return ds
    
    def get_modis_ndvi(self) -> xr.Dataset:
        """Get MODIS NDVI data"""
        print(f"  📥 Getting MODIS NDVI from GEE...")

        all_years_data = []

        # Process year by year to enable saving intermediate results
        unique_years = sorted(self.dates.to_series().dt.year.unique())

        for year in unique_years:
            year_start = datetime(year, 1, 1)
            year_end = datetime(year + 1, 1, 1) if year < unique_years[-1] else (self.dates[-1] + pd.DateOffset(months=1))

            # Define cache file path for the current year
            year_cache_file = self.cache_dir / f"modis_ndvi_{year}.nc"

            if year_cache_file.exists():
                print(f"  ✅ Loading MODIS NDVI for {year} from cache: {year_cache_file}")
                try:
                    year_ds = xr.open_dataset(year_cache_file)
                    all_years_data.append(year_ds)
                    continue # Skip download and go to next year
                except Exception as e:
                    print(f"    ❌ Error loading cache file for {year}: {e}. Re-downloading.")
                    # If corrupted, re-download this year's data

            print(f"  ⬇️ Downloading MODIS NDVI for year {year} from GEE...")

            # MODIS 16-day NDVI for the current year, filtered by date
            modis_yearly = ee.ImageCollection("MODIS/061/MOD13A2") \
                .filterBounds(self.roi) \
                .filterDate(ee.Date(year_start.strftime('%Y-%m-%d')), ee.Date(year_end.strftime('%Y-%m-%d'))) \
                .select(['NDVI'])

            # Scale factor
            def scale_ndvi(img):
                return img.multiply(0.0001) \
                    .copyProperties(img, ['system:time_start'])

            modis_yearly = modis_yearly.map(scale_ndvi)

            # Monthly composites
            def monthly_composite(start_date):
                start = ee.Date(start_date)
                end = start.advance(1, 'month')
                
                monthly = modis_yearly.filterDate(start, end).mean()
                
                return monthly.set({
                    'system:time_start': start.millis(),
                    'date': start.format('YYYY-MM-dd')
                })

            dates_for_year = [d for d in self.dates if d.year == year]

            # ИСПРАВЛЕНИЕ 1: Проверяем, есть ли даты для обработки
            if not dates_for_year:
                print(f"    ⚠️ No dates to process for year {year}.")
                continue

            monthly_list = []
            for date in dates_for_year:
                img = monthly_composite(date.strftime('%Y-%m-%d'))
                monthly_list.append(img)

            # ИСПРАВЛЕНИЕ 2: Проверяем, что список не пустой
            if not monthly_list:
                print(f"    ⚠️ No monthly images to process for year {year}.")
                continue

            # ИСПРАВЛЕНИЕ 3: Создаем коллекцию с правильным именем переменной
            monthly_col_yearly = ee.ImageCollection(monthly_list)

            try:
                # Convert to xarray for the current year's collection
                year_ds = self._collection_to_xarray(monthly_col_yearly, 'NDVI', pd.DatetimeIndex(dates_for_year))
                
                # ИСПРАВЛЕНИЕ 4: Проверяем, что данные не пустые
                if year_ds is None or len(year_ds.data_vars) == 0:
                    print(f"    ⚠️ No valid data returned for year {year}")
                    continue
                    
                all_years_data.append(year_ds)
                
                # Save the yearly data to NetCDF
                print(f"  💾 Saving MODIS NDVI for {year} to {year_cache_file}")
                year_ds.to_netcdf(year_cache_file)
                print(f"  🎉 Successfully saved {year_cache_file}")

            except ee.EEException as e:
                print(f"    ❌ Earth Engine Error for year {year}: {e}")
                print(f"    Continuing to next year...")
                continue
            except Exception as e:
                print(f"    ❌ General Error processing year {year}: {e}")
                print(f"    Continuing to next year...")
                continue

        if not all_years_data:
            print("  ❌ No MODIS NDVI data could be retrieved or processed.")
            return xr.Dataset()

        # Concatenate all yearly datasets into a single xarray Dataset
        try:
            ds = xr.concat(all_years_data, dim="time").sortby("time")
            print("  ✅ MODIS NDVI data retrieval complete.")
            return ds
        except Exception as e:
            print(f"  ❌ Error concatenating yearly datasets: {e}")
            return xr.Dataset()
    
    
    
    def get_soil_moisture(self) -> xr.Dataset:
        """Get soil moisture data from NASA SMAP or ERA5"""
        print(f"  📥 Getting soil moisture data from GEE...")
        
        # Try SMAP first (available from 2015)
        smap_start = "2015-04-01"
        
        if pd.Timestamp(START_DATE) >= pd.Timestamp(smap_start):
            # Use SMAP
            smap = ee.ImageCollection("NASA_USDA/HSL/SMAP10KM_soil_moisture") \
                .filterBounds(self.roi) \
                .select(['ssm', 'susm'])  # surface and subsurface
            
            # Monthly means
            def monthly_mean(start_date):
                start = ee.Date(start_date)
                end = start.advance(1, 'month')
                
                monthly = smap.filterDate(start, end).mean()
                
                return monthly.set({
                    'system:time_start': start.millis(),
                    'date': start.format('YYYY-MM-dd')
                })
            
            # Only for dates after SMAP start
            smap_dates = [d for d in self.dates if d >= pd.Timestamp(smap_start)]
            
            monthly_list = []
            for date in smap_dates:
                img = monthly_mean(date.strftime('%Y-%m-%d'))
                monthly_list.append(img)
            
            if monthly_list:
                monthly_col = ee.ImageCollection(monthly_list)
                ds_smap = self._collection_to_xarray(monthly_col, 'sm_surface', smap_dates)
                ds_smap = ds_smap.rename({'sm_surface': 'sm_surface'})
        
        # Use ERA5 soil moisture for full period
        era5_soil = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY") \
            .filterBounds(self.roi) \
            .select([
                'volumetric_soil_water_layer_1',  # 0-7cm
                'volumetric_soil_water_layer_2'   # 7-28cm
            ])
        
        def process_soil(img):
            # Average top two layers for surface
            surface = img.select('volumetric_soil_water_layer_1')
            rootzone = img.select(['volumetric_soil_water_layer_1', 
                                  'volumetric_soil_water_layer_2']).mean()
            
            return ee.Image.cat([
                surface.rename('sm_surface'),
                rootzone.rename('sm_rootzone')
            ]).copyProperties(img, ['system:time_start'])
        
        era5_soil = era5_soil.map(process_soil).filterDate(START_DATE, END_DATE)
        
        # Convert to xarray
        ds_surface = self._collection_to_xarray(era5_soil.select('sm_surface'), 
                                               'sm_surface', self.dates)
        ds_rootzone = self._collection_to_xarray(era5_soil.select('sm_rootzone'), 
                                                'sm_rootzone', self.dates)
        
        ds = xr.merge([ds_surface, ds_rootzone])
        
        return ds
    

    def get_era5_climate(self) -> xr.Dataset:
        """Get ERA5 climate data from GEE"""
        print(f"  📥 Getting ERA5 climate data from GEE...")
        
        # ERA5 monthly data
        era5 = ee.ImageCollection("ECMWF/ERA5/MONTHLY") \
            .filterBounds(self.roi) \
            .filterDate(START_DATE, END_DATE) \
            .select([
                'mean_2m_air_temperature',
                'minimum_2m_air_temperature', 
                'maximum_2m_air_temperature',
                'total_precipitation',
                'mean_2m_dewpoint_temperature',
                'surface_solar_radiation_downwards'
            ])
        
        # Проверим доступность данных
        collection_size = era5.size().getInfo()
        print(f"    Found {collection_size} ERA5 images")
        
        if collection_size == 0:
            print("    ⚠️ No ERA5 data found for the specified period")
            return xr.Dataset()
        
        # Convert temperature from K to C and calculate additional variables
        def process_era5(img):
            # Temperature conversion
            t2m = img.select('mean_2m_air_temperature').subtract(273.15)
            tmin = img.select('minimum_2m_air_temperature').subtract(273.15)
            tmax = img.select('maximum_2m_air_temperature').subtract(273.15)
            dewpoint = img.select('mean_2m_dewpoint_temperature').subtract(273.15)
            
            # Calculate VPD
            # Saturation vapor pressure
            es = ee.Image(6.112).multiply(
                ee.Image(17.67).multiply(t2m).divide(t2m.add(243.5)).exp()
            )
            # Actual vapor pressure
            ea = ee.Image(6.112).multiply(
                ee.Image(17.67).multiply(dewpoint).divide(dewpoint.add(243.5)).exp()
            )
            vpd = es.subtract(ea)
            
            # Solar radiation: J/m2 to MJ/m2/day
            srad = img.select('surface_solar_radiation_downwards').divide(1e6)
            
            # Combine bands
            return ee.Image.cat([
                t2m.rename('t2m'),
                tmin.rename('tmin'),
                tmax.rename('tmax'),
                vpd.rename('vpd'),
                srad.rename('srad'),
                img.select('total_precipitation').rename('tp')
            ]).copyProperties(img, ['system:time_start'])
        
        era5_processed = era5.map(process_era5)
        
        # Convert to xarray for each variable separately
        variables = ['t2m', 'tmin', 'tmax', 'vpd', 'srad', 'tp']
        datasets = []
        
        for var in variables:
            print(f"    Processing variable: {var}")
            try:
                ds = self._collection_to_xarray_era5(
                    era5_processed.select(var), 
                    var, 
                    self.dates
                )
                if ds is not None and var in ds.data_vars:
                    datasets.append(ds)
                else:
                    print(f"      ⚠️ No data for variable {var}")
            except Exception as e:
                print(f"      ❌ Error processing {var}: {e}")
                continue
        
        # Merge all variables
        if datasets:
            ds_combined = xr.merge(datasets)
            return ds_combined
        else:
            print("    ❌ No ERA5 variables could be processed")
            return xr.Dataset()

    def _collection_to_xarray_era5(self, collection: ee.ImageCollection, 
                                var_name: str, dates: pd.DatetimeIndex) -> xr.Dataset:
        """Convert ERA5 ImageCollection to xarray Dataset with better error handling"""
        
        # Get the region bounds
        lon_min, lat_min, lon_max, lat_max = self.bbox[2], self.bbox[0], self.bbox[3], self.bbox[1]
        
        # Create coordinate arrays
        scale_deg = SCALE
        lons = np.arange(lon_min, lon_max, scale_deg)
        lats = np.arange(lat_max, lat_min, -scale_deg)  # Descending
        
        # Initialize empty data array
        data = np.full((len(dates), len(lats), len(lons)), np.nan)
        successful_downloads = 0
        
        print(f"      Processing {len(dates)} time steps for {var_name}...")
        
        for i, date in enumerate(dates):
            if i % 6 == 0:  # Print every 6 months
                print(f"        {date.strftime('%Y-%m')}...")
            
            # Get image for the date - use exact date matching for ERA5
            start_date = date.strftime('%Y-%m-01')
            end_date = (date + pd.DateOffset(months=1)).strftime('%Y-%m-01')
            
            # Filter for exact month
            monthly_imgs = collection.filterDate(start_date, end_date)
            img_count = monthly_imgs.size().getInfo()
            
            if img_count == 0:
                print(f"        ⚠️ No {var_name} image for {date.strftime('%Y-%m')}")
                continue
            
            # Get the first (and usually only) image for the month
            img = monthly_imgs.first()
            
            # Check if image exists
            try:
                # Simple check - try to get image info
                img_info = img.getInfo()
                if img_info is None:
                    print(f"        ⚠️ Null image for {date.strftime('%Y-%m')}")
                    continue
            except Exception as e:
                print(f"        ⚠️ Cannot access image for {date.strftime('%Y-%m')}: {e}")
                continue
            
            # Check bands
            try:
                bands = img.bandNames().getInfo()
                if not bands or var_name not in bands:
                    print(f"        ⚠️ No '{var_name}' band in {date.strftime('%Y-%m')}. Available: {bands}")
                    continue
            except Exception as e:
                print(f"        ⚠️ Cannot get bands for {date.strftime('%Y-%m')}: {e}")
                continue
            
            # Select the variable
            img = img.select(var_name)
            
            # Download data
            try:
                arr = geemap.ee_to_numpy(
                    img,
                    region=self.roi,
                    scale=SCALE * 111320  # Convert degrees to meters
                )
                
                if arr is None:
                    print(f"        ⚠️ No array returned for {date.strftime('%Y-%m')}")
                    continue
                
                # Handle 3D arrays
                if arr.ndim == 3 and arr.shape[2] == 1:
                    arr = arr[:, :, 0]
                
                if arr.ndim != 2:
                    print(f"        ⚠️ Unexpected array shape for {date.strftime('%Y-%m')}: {arr.shape}")
                    continue
                
                # Resize if needed
                if arr.shape != (len(lats), len(lons)):
                    from scipy.ndimage import zoom
                    zoom_factors = (len(lats) / arr.shape[0], len(lons) / arr.shape[1])
                    arr = zoom(arr, zoom_factors, order=1)
                
                data[i] = arr
                successful_downloads += 1
                
            except Exception as e:
                print(f"        ⚠️ Download failed for {date.strftime('%Y-%m')}: {e}")
                continue
        
        print(f"      Successfully downloaded {successful_downloads}/{len(dates)} time steps for {var_name}")
        
        if successful_downloads == 0:
            print(f"      ❌ No successful downloads for {var_name}")
            return None
        
        # Create xarray dataset
        ds = xr.Dataset(
            {var_name: (['time', 'lat', 'lon'], data)},
            coords={
                'time': dates,
                'lat': lats,
                'lon': lons
            }
        )
        
        return ds

    # Также улучшим основную функцию _collection_to_xarray для лучшей обработки ошибок
    def _collection_to_xarray(self, collection: ee.ImageCollection, 
                            var_name: str, dates: pd.DatetimeIndex) -> xr.Dataset:
        """Convert Earth Engine ImageCollection to xarray Dataset"""

        # Get the region bounds
        lon_min, lat_min, lon_max, lat_max = self.bbox[2], self.bbox[0], self.bbox[3], self.bbox[1]

        # Create coordinate arrays
        scale_deg = SCALE
        lons = np.arange(lon_min, lon_max, scale_deg)
        lats = np.arange(lat_max, lat_min, -scale_deg)  # Descending

        # Initialize empty data array
        data = np.full((len(dates), len(lats), len(lons)), np.nan)
        successful_downloads = 0

        print(f"    Processing {len(dates)} images...")

        for i, date in enumerate(dates):
            if i % 12 == 0:
                print(f"      Year {date.year}...")

            # Get image for the date
            start_date = date.strftime('%Y-%m-%d')
            end_date = (date + pd.DateOffset(months=1)).strftime('%Y-%m-%d')
            
            monthly_imgs = collection.filterDate(start_date, end_date)
            
            # Check if any images exist for this period
            try:
                img_count = monthly_imgs.size().getInfo()
                if img_count == 0:
                    print(f"        ⚠️ No images for {date.strftime('%Y-%m')}")
                    continue
            except Exception as e:
                print(f"        ⚠️ Cannot check image count for {date.strftime('%Y-%m')}: {e}")
                continue
            
            img = monthly_imgs.first()

            # Check if image is valid
            try:
                img_info = img.getInfo()
                if img_info is None:
                    print(f"        ⚠️ Null image for {date.strftime('%Y-%m')}")
                    continue
            except Exception as e:
                print(f"        ⚠️ Cannot access image for {date.strftime('%Y-%m')}: {e}")
                continue

            # Check bands
            try:
                bands = img.bandNames().getInfo()
                if not bands or var_name not in bands:
                    print(f"        ⚠️ No '{var_name}' band for {date.strftime('%Y-%m')}. Available: {bands}")
                    continue
            except Exception as e:
                print(f"        ⚠️ Cannot get bands for {date.strftime('%Y-%m')}: {e}")
                continue

            img = img.select(var_name)

            try:
                arr = geemap.ee_to_numpy(
                    img,
                    region=self.roi,
                    scale=SCALE * 111320
                )

                if arr is None:
                    print(f"        ⚠️ No array returned for {date.strftime('%Y-%m')}")
                    continue

                # If array is 3D with shape (H, W, 1) → squeeze to (H, W)
                if arr.ndim == 3 and arr.shape[2] == 1:
                    arr = arr[:, :, 0]

                if arr.ndim != 2:
                    print(f"        ⚠️ Unexpected array shape for {date.strftime('%Y-%m')}: {arr.shape}")
                    continue

                # Resize if needed
                if arr.shape != (len(lats), len(lons)):
                    from scipy.ndimage import zoom
                    zoom_factors = (len(lats) / arr.shape[0], len(lons) / arr.shape[1])
                    arr = zoom(arr, zoom_factors, order=1)

                data[i] = arr
                successful_downloads += 1

            except Exception as e:
                print(f"        ⚠️ Failed to process {date.strftime('%Y-%m')}: {e}")
                continue

        print(f"    Successfully processed {successful_downloads}/{len(dates)} images for {var_name}")
        
        if successful_downloads == 0:
            print(f"    ❌ No successful downloads for {var_name}")
            return xr.Dataset()

        # Create xarray dataset
        ds = xr.Dataset(
            {var_name: (['time', 'lat', 'lon'], data)},
            coords={
                'time': dates,
                'lat': lats,
                'lon': lons
            }
        )

        return ds


    def calculate_pet_hargreaves(self, tmin: xr.DataArray, tmax: xr.DataArray, 
                                tmean: xr.DataArray) -> xr.DataArray:
        """Calculate PET using Hargreaves method"""
        print("  📊 Calculating PET (Hargreaves)...")
        
        # Solar radiation calculation
        lat_rad = np.deg2rad(self.ds.lat)
        
        # Initialize PET array
        pet = xr.zeros_like(tmean)
        
        for i, time in enumerate(tmean.time):
            # Day of year
            doy = pd.Timestamp(time.values).dayofyear
            
            # Solar parameters
            P = 2 * np.pi * doy / 365
            solar_decl = 0.006918 - 0.399912 * np.cos(P) + 0.070257 * np.sin(P)
            
            # Calculate for each latitude
            for j, lat in enumerate(lat_rad):
                # Sunset hour angle
                ws = np.arccos(np.clip(-np.tan(lat) * np.tan(solar_decl), -1, 1))
                
                # Extraterrestrial radiation
                dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
                Ra = 24 * 60 * 0.0820 / np.pi * dr * (
                    ws * np.sin(lat) * np.sin(solar_decl) +
                    np.cos(lat) * np.cos(solar_decl) * np.sin(ws)
                )
                
                # Hargreaves equation
                td = tmax.isel(time=i, lat=j) - tmin.isel(time=i, lat=j)
                pet_val = 0.0023 * Ra * np.sqrt(np.abs(td)) * (tmean.isel(time=i, lat=j) + 17.8)
                
                pet[i, j, :] = np.maximum(pet_val, 0)
        
        return pet
    
    def calculate_indices(self, ds: xr.Dataset) -> xr.Dataset:
        """Calculate SPI, SPEI, and PDSI"""
        
        # Calculate PET first
        pet = self.calculate_pet_hargreaves(ds['tmin'], ds['tmax'], ds['t2m'])
        ds['pet'] = pet
        
        # SPI-3
        print("  📊 Calculating SPI-3...")
        spi3 = self._calculate_spi(ds['precipitation'], scale=3)
        ds['spi3'] = spi3
        
        # SPEI-3
        print("  📊 Calculating SPEI-3...")
        water_balance = ds['precipitation'] - ds['pet']
        spei3 = self._calculate_spei(water_balance, scale=3)
        ds['spei3'] = spei3
        
        # PDSI (simplified)
        print("  📊 Calculating PDSI...")
        pdsi = self._calculate_pdsi_simple(ds['precipitation'], ds['pet'], 
                                          ds.get('sm_surface', None))
        ds['pdsi'] = pdsi
        
        return ds
    
    def _calculate_spi(self, precip: xr.DataArray, scale: int = 3) -> xr.DataArray:
        """Calculate SPI"""
        # Rolling sum
        precip_acc = precip.rolling(time=scale, min_periods=scale).sum()
        
        # Initialize output
        spi = xr.full_like(precip_acc, np.nan)
        
        # Fit gamma distribution for each pixel
        for i in range(len(precip.lat)):
            for j in range(len(precip.lon)):
                ts = precip_acc.isel(lat=i, lon=j).values
                valid = ~np.isnan(ts) & (ts > 0)
                
                if np.sum(valid) > 30:
                    # Fit gamma
                    params = stats.gamma.fit(ts[valid], floc=0)
                    
                    # Transform to SPI
                    cdf = stats.gamma.cdf(ts[valid], *params)
                    spi_values = stats.norm.ppf(cdf)
                    
                    # Assign back
                    spi[valid, i, j] = spi_values
        
        return spi
    
    def _calculate_spei(self, water_balance: xr.DataArray, scale: int = 3) -> xr.DataArray:
        """Calculate SPEI"""
        # Rolling sum
        wb_acc = water_balance.rolling(time=scale, min_periods=scale).sum()
        
        # Initialize output  
        spei = xr.full_like(wb_acc, np.nan)
        
        # Process each pixel
        for i in range(len(water_balance.lat)):
            for j in range(len(water_balance.lon)):
                ts = wb_acc.isel(lat=i, lon=j).values
                valid = ~np.isnan(ts)
                
                if np.sum(valid) > 30:
                    # Shift to positive
                    ts_shifted = ts[valid] - np.min(ts[valid]) + 0.01
                    
                    # Fit Pearson III
                    params = stats.pearson3.fit(ts_shifted)
                    
                    # Transform
                    cdf = stats.pearson3.cdf(ts_shifted, *params)
                    spei_values = stats.norm.ppf(cdf)
                    
                    spei[valid, i, j] = spei_values
        
        return spei
    
    def _calculate_pdsi_simple(self, precip: xr.DataArray, pet: xr.DataArray,
                              soil_moisture: xr.DataArray = None) -> xr.DataArray:
        """Simplified PDSI calculation"""
        # Initialize
        pdsi = xr.zeros_like(precip)
        awc = 100.0  # Available water capacity (mm)
        
        # Initial soil moisture
        if soil_moisture is not None:
            sm = soil_moisture.isel(time=0) * awc
        else:
            sm = xr.full_like(precip.isel(time=0), awc/2)
        
        # Calculate monthly
        for t in range(len(precip.time)):
            # Water balance
            p = precip.isel(time=t)
            pe = pet.isel(time=t)
            
            # Update soil moisture
            recharge = p - pe
            sm_new = np.clip(sm + recharge, 0, awc)
            
            # Moisture anomaly
            z_index = (sm_new - awc/2) / (awc/2)
            
            # PDSI (auto-regressive)
            if t == 0:
                pdsi[t] = z_index
            else:
                pdsi[t] = 0.897 * pdsi[t-1] + z_index / 3
            
            sm = sm_new
        
        return pdsi
    
    def process_region(self) -> xr.Dataset:
        """Process all data for the region"""
        print(f"\n🌍 Processing region: {self.region_name}")
        print(f"   Bbox: {self.bbox}")
        
        # Get all data
        ds_list = []
        
        # Precipitation
        try:
            precip = self.get_chirps_precipitation()
            ds_list.append(precip)
        except Exception as e:
            print(f"   ⚠️  Failed to get precipitation: {e}")
            return None
        
        # NDVI
        try:
            ndvi = self.get_modis_ndvi()
            ds_list.append(ndvi)
        except Exception as e:
            print(f"   ⚠️  Failed to get NDVI: {e}")
        
        # Climate
        try:
            climate = self.get_era5_climate()
            ds_list.append(climate)
        except Exception as e:
            print(f"   ⚠️  Failed to get climate: {e}")
            return None
        
        # Soil moisture
        try:
            soil = self.get_soil_moisture()
            ds_list.append(soil)
        except Exception as e:
            print(f"   ⚠️  Failed to get soil moisture: {e}")
        
        # Merge all datasets
        self.ds = xr.merge(ds_list)
        
        # Calculate indices
        self.ds = self.calculate_indices(self.ds)
        
        # Add metadata
        self.ds.attrs.update({
            'region': self.region_name,
            'bbox': self.bbox,
            'created': datetime.now().isoformat(),
            'source': 'Google Earth Engine'
        })
        
        return self.ds


def main():
    """Main processing pipeline"""
    print("🚀 Starting Google Earth Engine drought dataset pipeline")
    print(f"   Regions: {len(REGIONS)}")
    print(f"   Time range: {START_DATE} to {END_DATE}")
    print(f"   Resolution: {SCALE}° (~{SCALE*111:.1f} km)")
    
    # Process each region
    all_regions = []
    
    for region_name, bbox in REGIONS.items():
        try:
            # Check if already processed
            region_file = OUT_DIR / f"gee_{region_name}.nc"
            
            if region_file.exists():
                print(f"\n✅ Loading existing: {region_name}")
                ds = xr.open_dataset(region_file)
            else:
                # Process region
                processor = GEEDroughtData(region_name, bbox)
                ds = processor.process_region()
                
                if ds is not None:
                    # Save
                    ds.to_netcdf(region_file)
                    print(f"   💾 Saved to {region_file}")
                else:
                    print(f"   ❌ Failed to process {region_name}")
                    continue
            
            all_regions.append(ds)
            
        except Exception as e:
            print(f"\n❌ Error with {region_name}: {e}")
            continue
    
    # Combine all regions
    if all_regions:
        print("\n📦 Combining all regions...")
        
        # Add region dimension
        for i, (name, ds) in enumerate(zip(REGIONS.keys(), all_regions)):
            ds = ds.expand_dims({'region': [name]})
            all_regions[i] = ds
        
        # Concatenate
        ds_combined = xr.concat(all_regions, dim='region')
        
        # Save combined dataset
        out_file = OUT_DIR / "drought_dataset_gee.zarr"
        print(f"\n💾 Saving combined dataset to {out_file}")
        ds_combined.to_zarr(out_file, mode='w', consolidated=True)
        
        # Summary
        print("\n📊 Dataset summary:")
        print(f"   Shape: {dict(ds_combined.dims)}")
        print(f"   Variables: {list(ds_combined.data_vars)}")
        print(f"   Regions: {list(ds_combined.region.values)}")
        print(f"   Time range: {ds_combined.time.min().values} to {ds_combined.time.max().values}")
        
        # Create visualization
        create_summary_plots(ds_combined)
        
        print("\n✅ Google Earth Engine pipeline complete!")
        
        return ds_combined
    else:
        print("\n❌ No regions successfully processed")
        return None


def create_summary_plots(ds: xr.Dataset):
    """Create summary visualizations"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Mean SPI-3 time series by region
    ax = axes[0, 0]
    for region in ds.region.values:
        spi_mean = ds.spi3.sel(region=region).mean(['lat', 'lon'])
        ax.plot(spi_mean.time, spi_mean, label=region, alpha=0.8)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_title('Mean SPI-3 by Region', fontsize=14)
    ax.set_ylabel('SPI-3')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Precipitation vs NDVI correlation
    ax = axes[0, 1]
    region_sample = ds.sel(region='us_plains')
    precip_mean = region_sample.precipitation.mean(['lat', 'lon'])
    ndvi_mean = region_sample.ndvi.mean(['lat', 'lon'])
    
    # Normalize for comparison
    precip_norm = (precip_mean - precip_mean.mean()) / precip_mean.std()
    ndvi_norm = (ndvi_mean - ndvi_mean.mean()) / ndvi_mean.std()
    
    ax.plot(precip_norm.time, precip_norm, label='Precipitation', alpha=0.8)
    ax.plot(ndvi_mean.time, ndvi_norm, label='NDVI', alpha=0.8)
    ax.set_title('Precipitation vs NDVI (US Plains)', fontsize=14)
    ax.set_ylabel('Normalized values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: SPEI-3 spatial pattern
    ax = axes[1, 0]
    spei_latest = ds.spei3.sel(region='ru_volga').isel(time=-1)
    im = ax.imshow(spei_latest, cmap='RdBu', vmin=-3, vmax=3, aspect='auto')
    ax.set_title(f'SPEI-3 Spatial Pattern - Volga Region\n{ds.time.isel(time=-1).values}', 
                 fontsize=14)
    ax.set_xlabel('Longitude index')
    ax.set_ylabel('Latitude index')
    plt.colorbar(im, ax=ax, label='SPEI-3')
    
    # Plot 4: Drought indices comparison
    ax = axes[1, 1]
    region_sample = ds.sel(region='ru_south').isel(lat=len(ds.lat)//2, lon=len(ds.lon)//2)
    
    # Plot last 5 years
    time_slice = slice('2020-01-01', '2024-12-31')
    ax.plot(region_sample.time.sel(time=time_slice), 
            region_sample.spi3.sel(time=time_slice), label='SPI-3', alpha=0.8)
    ax.plot(region_sample.time.sel(time=time_slice), 
            region_sample.spei3.sel(time=time_slice), label='SPEI-3', alpha=0.8)
    ax.plot(region_sample.time.sel(time=time_slice), 
            region_sample.pdsi.sel(time=time_slice), label='PDSI', alpha=0.8)
    
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_title('Drought Indices Comparison - Southern Russia\n(Single pixel)', fontsize=14)
    ax.set_ylabel('Index value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'gee_dataset_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   📈 Saved summary plots to {OUT_DIR / 'gee_dataset_summary.png'}")


if __name__ == "__main__":
    dataset = main()