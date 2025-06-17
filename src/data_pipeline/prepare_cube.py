# """
# Собирает финальный датакуб:
#   • MODIS MOD13Q1  NDVI, EVI
#   • SMAP L4        soil moisture (surface + root)
#   • ERA5-Land      t2m, vpd, swrad
#   + spi3 (из chirps_spi.nc)

# Вывод: data/processed/agro_cube.zarr
# Запускать из корня проекта:
#     python -m src.data_pipeline.prepare_cube
# """

# import xarray as xr, rioxarray as rio, numpy as np, os, tempfile, shutil
# from pathlib import Path
# import warnings, datetime as dt

# RAW   = Path("data/raw")
# PROC  = Path("data/processed")
# SPI_NC = RAW / "chirps_spi.nc"
# ZARR  = PROC / "agro_cube.zarr"

# # ---- ROI границы (тот же словарь) ----
# REGIONS = {
#     "us_plains":  (35, 48, -104, -90),
#     "br_cerrado": (-20, -6, -62,  -46),
#     "in_ganga":   (21, 31,  73,   90),
# }
# LAT_MIN = min(r[0] for r in REGIONS.values())
# LAT_MAX = max(r[1] for r in REGIONS.values())
# LON_MIN = min(r[2] for r in REGIONS.values())
# LON_MAX = max(r[3] for r in REGIONS.values())

# def clip_roi(ds: xr.Dataset) -> xr.Dataset:
#     lat_asc = float(ds.latitude[1]) > float(ds.latitude[0])
#     lat_slice = slice(LAT_MIN, LAT_MAX) if lat_asc else slice(LAT_MAX, LAT_MIN)
#     return ds.sel(latitude=lat_slice, longitude=slice(LON_MIN, LON_MAX))

# def load_spi() -> xr.Dataset:
#     ds = xr.open_dataset(SPI_NC)
#     return ds.rename(time="time")   # гарантируем единое имя координаты

# def load_modis() -> xr.Dataset:
#     """
#     Скачивает только нужные тайлы MOD13Q1 (NDVI) через HTTPS LP-DAAC.
#     Без earthaccess, без S3.  Работает ~15-20 мин на 300 HDF.
#     """
#     import datetime as dt, itertools, tempfile, rioxarray as rio, xarray as xr
#     from pathlib import Path, PurePosixPath
#     import requests, tqdm, warnings, os

#     TILES = ["h09v04","h10v04","h11v04", "h13v10","h14v10",
#              "h25v07","h26v07"]
#     start = dt.date(2003,1,1)
#     end   = dt.date(2024,12,31)

#     # каждые 16 дней (MOD13Q1)
#     dates = [start + dt.timedelta(days=i)
#              for i in range(0, (end-start).days+1, 16)]

#     tmp_dir = Path(tempfile.mkdtemp())
#     sess = requests.Session()
#     sess.auth = (os.getenv("ED_USERNAME"), os.getenv("ED_PASSWORD"))  # или ~/.netrc

#     hdf_paths = []

#     for d in tqdm.tqdm(dates, desc="MODIS days"):
#         folder = f"{d:%Y.%m.%d}"
#         doy    = d.timetuple().tm_yday
#         for tile in TILES:
#             # имя файла без хвоста после .061 — найдём запросом на HTML-листинг
#             prefix = f"MOD13Q1.A{d.year}{doy:03d}.{tile}.061"
#             url_list = (f"https://e4ftl01.cr.usgs.gov/MOLT/MOD13Q1.061/"
#                         f"{folder}/")
#             try:
#                 r = sess.get(url_list, timeout=30)
#                 r.raise_for_status()
#                 fname = next(l.split('">')[0]
#                              for l in r.text.split()
#                              if l.startswith(prefix) and l.endswith(".hdf"))
#                 url = PurePosixPath(url_list) / fname
#                 out = tmp_dir / fname
#                 if not out.exists():
#                     with sess.get(str(url), stream=True, timeout=120) as resp:
#                         resp.raise_for_status()
#                         with open(out, "wb") as f:
#                             for chunk in resp.iter_content(1024*1024):
#                                 f.write(chunk)
#                 hdf_paths.append(out)
#             except Exception:
#                 # пропускаем отсутствующие или рваные файлы
#                 continue

#     # читаем NDVI слой
#     stacks = []
#     for f in hdf_paths:
#         sds = (f"HDF4_EOS:EOS_GRID:{f}:"
#                "MODIS_Grid_16DAY_250m_500m_VI:1 km VI NDVI")
#         da  = rio.open_rasterio(sds, masked=True).squeeze()
#         date_code = f.name.split('.')[1]   # A2003353
#         dtc = dt.datetime.strptime(date_code[1:], "%Y%j")
#         da = da.assign_coords({"time": dtc}).expand_dims("time")
#         stacks.append(da)

#     if not stacks:
#         raise RuntimeError("Не удалось скачать ни одного NDVI-файла")

#     ndvi = xr.concat(stacks, dim="time").rename({"y":"latitude","x":"longitude"})
#     ndvi = clip_roi(ndvi).resample(time="1M").mean(keep_attrs=True)
#     ndvi.name = "ndvi"
#     warnings.filterwarnings("ignore")
#     return ndvi.to_dataset()

# def load_smap() -> xr.Dataset:
#     url = "https://smappub.jpl.nasa.gov/data/smap_level4/spl4smgp/..."  # урезанный пример
#     sm = xr.open_dataset(url, chunks={"time":31})
#     sm = clip_roi(sm)
#     sm = sm.resample(time="1M").mean()
#     sm = sm[["sm_surface","sm_rootzone"]]
#     return sm

# def load_era5() -> xr.Dataset:
#     import cdsapi, pandas as pd
#     c = cdsapi.Client()
#     out = RAW / "era5_land.nc"
#     if not out.exists():
#         c.retrieve(
#             "reanalysis-era5-land",
#             {"variable":["2m_temperature","surface_net_solar_radiation","vpd"],
#              "year":[str(y) for y in range(2003,2025)],
#              "month":[f"{m:02d}" for m in range(1,13)],
#              "day":"01","time":"00:00",
#              "area":[LAT_MAX, LON_MIN, LAT_MIN, LON_MAX],
#              "format":"netcdf"}, str(out))
#     era = xr.open_dataset(out)
#     era = era.resample(time="1M").mean()
#     era = era.rename({"msdwlwrf":"srad"})
#     return era

# def main():
#     warnings.filterwarnings("ignore")
#     spi  = load_spi()
#     ndvi = load_modis()
#     sm   = load_smap()
#     era  = load_era5()

#     cube = xr.merge([spi, ndvi, sm, era], compat="override")
#     PROC.mkdir(parents=True, exist_ok=True)
#     cube.to_zarr(ZARR, mode="w")
#     print(f"✅  Saved → {ZARR}")

# if __name__ == "__main__":
#     main()

from pathlib import Path
import xarray as xr

RAW = Path("data/raw/chirps_spi.nc")
PROC = Path("data/processed")
ZARR = PROC / "agro_cube.zarr"

def main():
    print("📦 Загружаю SPI файл…")
    ds = xr.open_dataset(RAW)

    print("📦 Проверяю структуру…")
    assert "spi3" in ds.data_vars, "Нет spi3 в датасете!"
    assert "time" in ds.dims, "Нет оси времени!"

    print("📦 Сохраняю в Zarr…")
    PROC.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(ZARR, mode="w")

    print(f"✅ Saved → {ZARR}")

if __name__ == "__main__":
    main()