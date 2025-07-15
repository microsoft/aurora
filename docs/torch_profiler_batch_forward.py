import xarray as xr
import datetime
from aurora import Batch, Metadata, AuroraWave
import torch
import pickle
from typing import Dict, Tuple, Optional
from torch.profiler import profile, record_function, ProfilerActivity
import os
import psutil

AURORA_TO_ERA5_ATM = {
    't': 'temperature',
}
AURORA_TO_ERA5_SURF = {
    '10u': '10m_u_component_of_wind',
    '10v': '10m_v_component_of_wind',
    'swh': 'significant_height_of_combined_wind_waves_and_swell',
    'mwd': 'mean_wave_direction',
    'mwp': 'mean_wave_period',
}
AURORA_TO_ERA5_FULL = {
    '2t': '2m_temperature',
    '10u': '10m_u_component_of_wind',
    '10v': '10m_v_component_of_wind',
    'swh': 'significant_height_of_combined_wind_waves_and_swell',
    'mwd': 'mean_wave_direction',
    'mwp': 'mean_wave_period',

    't': 'temperature',
    'u': 'u_component_of_wind',
    'v': 'v_component_of_wind',
    'q': 'specific_humidity',
    'z': 'geopotential_at_surface',
    'slt': 'soil_type',
    'lsm': 'land_sea_mask'
}
PRESSURE_LEVELS = [50]
ATMOSPHERIC_VARS = ['t', 'u', 'v', 'q']

def print_mem(prefix):
    process = psutil.Process(os.getpid())
    cpu = process.memory_info().rss / 1024**2
    gpu = (torch.cuda.memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0
    print(f"[{prefix}] CPU Memory: {cpu:.2f} MB, GPU Memory: {gpu:.2f} MB")

def _open_era5_dataset(path: str, chunks: Optional[Dict] = None) -> xr.Dataset:
    return xr.open_zarr(
        path,
        chunks=chunks,
        storage_options=dict(token='anon'),
        decode_times=True
    )

def _get_time_range(year: int, month: int, day: int, hour: int) -> Tuple[datetime.datetime, datetime.datetime]:
    start_time = datetime.datetime(year, month, day, hour)
    end_time = start_time + datetime.timedelta(hours=6)
    return start_time, end_time

def _select_time_slice(ds: xr.Dataset, start_time: datetime.datetime, end_time: datetime.datetime) -> xr.Dataset:
    analysis_ready = ds.sel(time=slice(ds.attrs['valid_time_start'], ds.attrs['valid_time_stop']))
    return analysis_ready.sel(time=[start_time, end_time], method='nearest')

def _process_atmospheric_variable(data: xr.DataArray, var_name: str) -> xr.DataArray:
    if var_name in ATMOSPHERIC_VARS:
        return data[:, :len(PRESSURE_LEVELS), :, :]
    elif var_name == 'z':
        data = data.expand_dims({'level': len(PRESSURE_LEVELS)}, axis=2)
        dims_order = [data.dims[0], data.dims[2], data.dims[1], data.dims[3]]
        return data.transpose(*dims_order)
    return data

def get_two_timesteps_era5(path: str, var_name: str, year: int, month: int, day: int, hour: int, device=None) -> torch.Tensor:
    ds = _open_era5_dataset(path)
    start_time, end_time = _get_time_range(year, month, day, hour)
    analysis_ready = _select_time_slice(ds, start_time, end_time)
    era5_var_name = AURORA_TO_ERA5_FULL[var_name]
    data = analysis_ready[era5_var_name]
    arr = _process_atmospheric_variable(data, var_name)
    t = torch.from_numpy(arr.values[:2][None])
    if device:
        t = t.to(device)
    return t

def get_metadata_era5(path: str, year: int, month: int, day: int, hour_start: int, device=None) -> Metadata:
    ds = _open_era5_dataset(path, chunks={'time': 48})
    start_time, end_time = _get_time_range(year, month, day, hour_start)
    analysis_ready = _select_time_slice(ds, start_time, end_time)
    latitudes = torch.from_numpy(analysis_ready['latitude'].values) if 'latitude' in analysis_ready else None
    longitudes = torch.from_numpy(analysis_ready['longitude'].values) if 'longitude' in analysis_ready else None
    if latitudes is not None and device: latitudes = latitudes.to(device)
    if longitudes is not None and device: longitudes = longitudes.to(device)
    return Metadata(
        time=tuple([end_time]),
        atmos_levels=tuple(PRESSURE_LEVELS),
        lat=latitudes,
        lon=longitudes,
        rollout_step=0
    )

def get_static_vars_era5(path: str, device=None) -> torch.Tensor:
    ds = _open_era5_dataset(path)
    reference_time = datetime.datetime(2025, 1, 1, 0)
    ds_time_selected = ds.sel(time=[reference_time], method='nearest')
    lsm = torch.from_numpy(ds_time_selected[AURORA_TO_ERA5_FULL['lsm']].values)[0]
    if device:
        lsm = lsm.to(device)
    return lsm

def _load_variables(path: str, var_dict: Dict[str, str], year: int, month: int, day: int, hour_start: int, device=None) -> Dict[str, torch.Tensor]:
    result = {}
    for var_name in var_dict.keys():
        t = get_two_timesteps_era5(path, var_name, year, month, day, hour_start, device)
        result[var_name] = t
    return result

def get_batch_era5(path: str, year: int, month: int, day: int, hour_start: int, device=None) -> Batch:
    surface_vars = _load_variables(path, AURORA_TO_ERA5_SURF, year, month, day, hour_start, device)
    atm_vars = _load_variables(path, AURORA_TO_ERA5_ATM, year, month, day, hour_start, device)
    lsm = get_static_vars_era5(path, device)
    metadata = get_metadata_era5(path, year, month, day, hour_start, device)
    batch = Batch(
        surf_vars=surface_vars,
        atmos_vars=atm_vars,
        metadata=metadata,
        static_vars={'lsm': lsm}
    )
    # .to(device) if your Batch class provides such method, otherwise data is already on correct device.
    return batch

def create_aurora_model(device=None) -> AuroraWave:
    model = AuroraWave(
        surf_vars=tuple(AURORA_TO_ERA5_SURF.keys()),
        angle_surf_vars=['mwd']
    )
    if device:
        model.to(device)
    print("Model created successfully.")
    model.load_checkpoint(strict=False)
    print("Checkpoint loaded successfully.")
    model.eval()
    return model

def main():
    path = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
    year, month, day, hour = 2020, 1, 1, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = [("batch_creation", None), ("batch_saving", None), ("model_creation", None), ("model_inference", None)]

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_output')
    ) as prof:
        # --- Batch creation ---
        with record_function("batch_creation"):
            print("\nCreating batch from ERA5 data...")
            batch = get_batch_era5(path, year, month, day, hour, device)
            print_mem("Batch Creation")

        # --- Batch saving ---
        with record_function("batch_saving"):
            print("\nSaving batch to file...")
            with open("saved_batch.pkl", "wb") as f:
                pickle.dump(batch, f)
            print_mem("Batch Saving")

        # --- Model creation ---
        with record_function("model_creation"):
            print("\nCreating model...")
            model = create_aurora_model(device)
            print_mem("Model Creation")

        # --- Model inference ---
        with record_function("model_inference"):
            print("\nPerforming model inference...")
            pred = model.forward(batch)
            print("Prediction completed.")
            print_mem("Model Inference")
        prof.step()

    print("\nProfiling complete. Results in ./profiler_output\nSee per-step memory stats in TensorBoard (run: tensorboard --logdir=./profiler_output )")

if __name__ == "__main__":
    main()
