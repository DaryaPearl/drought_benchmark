"""
END-TO-END: качаем CHIRPS 2003-2024, режем под 3 ROI, считаем SPI-3
(через чистый SciPy) и сохраняем NetCDF → data/raw/chirps_spi.nc

Запускать из корня репо:
    python -m src.data_pipeline.chirps_to_spi
"""

import os, shutil, tempfile, requests, warnings
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.stats import gamma, norm

# ─────── настройки ───────
YEARS = range(2003, 2025)
BASE  = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p25"
OUT   = Path("data/raw/chirps_spi.nc")
REGIONS = {                         # lat_min, lat_max, lon_min, lon_max
    "us_plains":  (35, 48, -104, -90),
    "br_cerrado": (-20, -6, -62,  -46),
    "in_ganga":   (21, 31,  73,   90),
}

warnings.filterwarnings("ignore", category=RuntimeWarning)  # NaNs по краям океана

def download_one(year: int, dest: Path) -> None:
    url = f"{BASE}/chirps-v2.0.{year}.days_p25.nc"
    print(f"⬇  {year} …", end="", flush=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    print(" ok")

def calc_spi3(monthly: np.ndarray) -> np.ndarray:
    """
    monthly  –  numpy 3-D  (time, lat, lon), mm / мес
    Возвращает SPI-3 с той же формой.
    """
    T, H, W  = monthly.shape
    # скользящая сумма по 3 месяцам
    three = np.apply_along_axis(lambda m: np.convolve(m, np.ones(3), "same"), 0, monthly)
    three = three[2:]             # первые 2 месяца ∅
    T2     = three.shape[0]

    flat   = three.reshape(T2, -1)        # (T2, N)
    spi    = np.empty_like(flat, dtype=np.float32)

    for px in range(flat.shape[1]):
        s = flat[:, px]
        if np.all(np.isnan(s)):           # весь океан → NaN
            spi[:, px] = np.nan
            continue
        valid = ~np.isnan(s)
        data  = s[valid]
        if data.min() <= 0:               # гамма только >0
            data = data + 0.1
        α, loc, β = gamma.fit(data, floc=0)
        cdf = gamma.cdf(data, α, loc=0, scale=β)
        z   = norm.ppf(cdf)               # перевод в стандартные отклонения
        spi[:, px][valid] = z
        spi[:, px][~valid] = np.nan

    return spi.reshape(T2, H, W)

def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        ds_list = []
        for y in YEARS:
            f = tmp / f"{y}.nc"
            download_one(y, f)
            ds_list.append(xr.open_dataset(f))
        ds = xr.concat(ds_list, dim="time")

        # общее окно
        lat_min = min(r[0] for r in REGIONS.values())
        lat_max = max(r[1] for r in REGIONS.values())
        lon_min = min(r[2] for r in REGIONS.values())
        lon_max = max(r[3] for r in REGIONS.values())

        lat_asc = float(ds.latitude[1]) > float(ds.latitude[0])
        lat_slice = slice(lat_min, lat_max) if lat_asc else slice(lat_max, lat_min)
        ds = ds.sel(latitude=lat_slice, longitude=slice(lon_min, lon_max))

        # дневные → месячные суммы осадков
        mon = ds.resample(time="1M").sum()
        spi = calc_spi3(mon["precip"].values)      # (T-2, H, W)

        mon = mon.isel(time=slice(2, None))        # подрезаем те же 2 месяца
        mon["spi3"] = (("time", "latitude", "longitude"), spi)

        OUT.parent.mkdir(parents=True, exist_ok=True)
        mon.to_netcdf(OUT)
        sz = OUT.stat().st_size / 1e6
        print(f"\n✅  Saved → {OUT}  ({sz:.1f} MB)")
    finally:
        shutil.rmtree(tmp)

if __name__ == "__main__":
    main()