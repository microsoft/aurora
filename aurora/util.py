from pathlib import Path
import os
import requests
import time

def try_download(url: Path, outpath: Path):
    """
    Download a file from a URL to a specified path. If the file already exists and is not empty, skip the download.

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
        response.raise_for_status()  # Raise an exception for HTTP errors (status codes other than 2xx)

        with open(outpath, "wb") as f:
            f.write(response.content)
            os.chmod(outpath, 0o755)  # Change file permissions to make it executable

    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}. Error: {e}")


def download_hres_rda_surf(save_dir: Path, year: str, month: str, day: str, variable: str, var_dict: dict[str, str]):
    """
    Download IFS HRES 0.1 deg surface data from the RDA website: https://rda.ucar.edu/datasets/d113001/# 
    
    Args:
        save_dir (Path): Path to save the downloaded files to.
        year (str): Year to download data for.
        month (str): Month to download data for.
        day (str): Day to download data for.
        variable (str): Variable to download data for.
        var_dict (dict[str, str]): Dictionary mapping variable names to their corresponding RDA variable numbers. See `docs/example_0.1deg.ipynb` for an example. 
    """
    v_num = var_dict[variable]
    print(f"Downloading {variable} for {year}-{month}-{day}...")


    filename = f"ec.oper.an.sfc/{year}{month}/ec.oper.an.sfc.128_{v_num}_{variable}.regn1280sc.{year}{month}{day}.grb"

    outpath = os.path.join(save_dir, f"{variable}_{year}_{month}_{day}.grb")
    print(outpath)
    url = "https://data.rda.ucar.edu/ds113.1/" + filename

    start_time = time.time()
    try_download(url, outpath)
    print(f"Downloaded {filename} to {outpath} in {time.time() - start_time:.2f} seconds.")


def download_hres_rda_atmos(save_dir: Path, year: str, month: str, day: str, variable: str, var_dict: dict[str, str], timeofday: str):
    """
    Download IFS HRES 0.1 deg atmospheric data from the RDA website: https://rda.ucar.edu/datasets/d113001/#

    Args:
        save_dir (Path): Path to save the downloaded files to.
        year (str): Year to download data for.
        month (str): Month to download data for.
        day (str): Day to download data for.
        variable (str): Variable to download data for.
        var_dict (dict[str, str]): Dictionary mapping variable names to their corresponding RDA variable numbers. See `docs/example_0.1deg.ipynb` for an example. 
        timeofday (str): Time of day to download data for. Format: "00", "06", "12", or "18".
    """
    v_num = var_dict[variable]
    print(f"Downloading {variable} for {year}-{month}-{day}-{timeofday}...")

    if variable in ["z", "t", "q", "w"]:
        filename = f"ec.oper.an.pl/{year}{month}/ec.oper.an.pl.128_{v_num}_{variable}.regn1280sc.{year}{month}{day}{timeofday}.grb"
    else:  # u and v have different filenames than z, t, and q.
        filename = f"ec.oper.an.pl/{year}{month}/ec.oper.an.pl.128_{v_num}_{variable}.regn1280uv.{year}{month}{day}{timeofday}.grb"

    url = "https://data.rda.ucar.edu/ds113.1/" + filename
    outpath = os.path.join(save_dir, f"{variable}_{year}_{month}_{day}_{timeofday}.grb")

    start_time = time.time()
    try_download(url, outpath)
    print(f"Downloaded {filename} to {outpath} in {time.time() - start_time:.2f} seconds.")