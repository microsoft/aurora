"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import os
import time
from pathlib import Path

import numpy as np
import requests
import scipy.interpolate.RegularGridInterpolator as RGI

__all__ = [
    "try_download",
    "download_hres_rda_surf",
    "download_hres_rda_atmos",
]


def try_download(url: str, outpath: Path):
    """
    Download a file from a URL to a specified path. If the file already exists and is not empty,
    skip the download.

    Args:
        url (Path): URL to download the file from.
        outpath (Path): Path to save the downloaded file to.
    """
    # Check if the file exists and is not empty
    if os.path.isfile(outpath) and os.path.getsize(outpath) > 0:
        print(f"File {outpath} exists and is not empty. Skipping download.")

    directory = os.path.dirname(outpath)

    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors (status codes != 2xx)

        with open(outpath, "wb") as f:
            f.write(response.content)
            os.chmod(outpath, 0o755)  # Change file permissions to make it executable

    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}. Error: {e}")


def download_hres_rda_surf(
    save_dir: Path, year: str, month: str, day: str, variable: str, var_dict: dict[str, str]
):
    """
    Download IFS HRES 0.1 deg surface data from the RDA website: https://rda.ucar.edu/datasets/d113001/#

    Args:
        save_dir (Path): Path to save the downloaded files to.
        year (str): Year to download data for.
        month (str): Month to download data for.
        day (str): Day to download data for.
        variable (str): Variable to download data for
        var_dict (dict[str, str]): Dictionary mapping variables to their corresponding RDA numbers.
        See `docs/example_0.1deg.ipynb` for an example.
    """
    v_num = var_dict[variable]
    print(f"Downloading {variable} for {year}-{month}-{day}...")

    filename = (
        f"ec.oper.an.sfc/{year}{month}/ec.oper.an.sfc.128_{v_num}_{variable}."
        f"regn1280sc.{year}{month}{day}.grb"
    )

    outpath = save_dir / f"{variable}_{year}_{month}_{day}.grb"
    url = "https://data.rda.ucar.edu/ds113.1/" + filename

    start_time = time.time()
    try_download(url, outpath)
    print(f"Downloaded {filename} to {outpath} in {time.time() - start_time:.2f} seconds.")


def download_hres_rda_atmos(
    save_dir: Path,
    year: str,
    month: str,
    day: str,
    variable: str,
    var_dict: dict[str, str],
    timeofday: str,
):
    """
    Download IFS HRES 0.1 deg atmospheric data from the RDA website: https://rda.ucar.edu/datasets/d113001/#

    Args:
        save_dir (Path): Path to save the downloaded files to.
        year (str): Year to download data for.
        month (str): Month to download data for.
        day (str): Day to download data for.
        variable (str): Variable to download data for.
        var_dict (dict[str, str]): Dictionary mapping variables to their RDA numbers.
                                   See `docs/example_0.1deg.ipynb` for an example.
        timeofday (str): Time of day to download data for. Format: "00", "06", "12", or "18".
    """
    v_num = var_dict[variable]
    print(f"Downloading {variable} for {year}-{month}-{day}-{timeofday}...")

    if variable in ["z", "t", "q", "w"]:
        filename = (
            f"ec.oper.an.pl/{year}{month}/ec.oper.an.pl.128_{v_num}_{variable}."
            f"regn1280sc.{year}{month}{day}{timeofday}.grb"
        )
    else:  # u and v have different filenames than z, t, and q.
        filename = (
            f"ec.oper.an.pl/{year}{month}/ec.oper.an.pl.128_{v_num}_{variable}."
            f"regn1280uv.{year}{month}{day}{timeofday}.grb"
        )

    url = "https://data.rda.ucar.edu/ds113.1/" + filename
    outpath = save_dir / f"{variable}_{year}_{month}_{day}_{timeofday}.grb"

    start_time = time.time()
    try_download(url, outpath)
    print(f"Downloaded {filename} to {outpath} in {time.time() - start_time:.2f} seconds.")


def download_hres_rda_static(
    save_dir: Path, year: str, month: str, day: str, variable: str, var_dict: dict[str, str]
):
    """
    Download IFS HRES 0.1 deg surface data from the RDA website: https://rda.ucar.edu/datasets/d113001/#

    Args:
        save_dir (Path): Path to save the downloaded files to.
        year (str): Year to download data for.
        month (str): Month to download data for.
        day (str): Day to download data for.
        variable (str): Variable to download data for
        var_dict (dict[str, str]): Dictionary mapping variables to their corresponding RDA numbers.
        See `docs/example_0.1deg.ipynb` for an example.
    """

    v_num = var_dict[variable]
    print(f"Downloading {variable} for {year}-{month}-{day}...")

    filename = (
        f"ec.oper.an.sfc/{year}{month}/ec.oper.an.sfc.128_{v_num}_{variable}."
        f"regn1280sc.{year}{month}{day}.grb"
    )

    outpath = save_dir / f"static_{variable}.grb"
    url = "https://data.rda.ucar.edu/d113001/" + filename

    start_time = time.time()
    try_download(url, outpath)
    print(f"Downloaded {filename} to {outpath} in {time.time() - start_time:.2f} seconds.")


def interpolate(
    v: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    res: float,
) -> np.ndarray:
    """Interpolate a variable `v` with latitudes `lat` and longitudes `lon` to resolution `res`."""
    shape = (round(180 / res) + 1, round(360 / res))

    # Implement periodic longitudes.
    assert (np.diff(lon) > 0).all()
    lon = np.concatenate((lon[-1:] - 360, lon, lon[:1] + 360))
    v = np.concatenate((v[:, -1:], v, v[:, :1]), axis=1)

    rgi = RGI(
        (lat, lon),
        v,
        method="linear",
        bounds_error=False,  # Allow out of bounds, for the latitudes.
        fill_value=None,  # Extrapolate latitudes if they are out of bounds.
    )
    lat_new, lon_new = np.meshgrid(
        np.linspace(90, -90, shape[0]),
        np.linspace(0, 360, shape[1], endpoint=False),
        indexing="ij",
        sparse=True,
    )
    return rgi((lat_new, lon_new))
