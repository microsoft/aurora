import xarray as xr
import datetime
from aurora import Batch, Metadata, AuroraWave
import torch
import pickle
from typing import Dict, Tuple, Optional

# Variable mapping dictionaries
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
    'pp1d': 'peak_wave_period',
    'shww': 'significant_height_of_wind_waves',
    'mdww': 'mean_direction_of_wind_waves',
    'mpww': 'mean_period_of_wind_waves',
    'shts': 'significant_height_of_total_swell',
    'mdts': 'mean_direction_of_total_swell',
    'mpts': 'mean_period_of_total_swell',
    'swh1': 'significant_wave_height_of_first_swell_partition',
    'mwd1': 'mean_wave_direction_of_first_swell_partition',
    'mwp1': 'mean_wave_period_of_first_swell_partition',
    'swh2': 'significant_wave_height_of_second_swell_partition',
    'mwd2': 'mean_wave_direction_of_second_swell_partition',
    'mwp2': 'mean_wave_period_of_second_swell_partition',
    '10u_wave': 'u_component_stokes_drift',
    '10v_wave': 'v_component_stokes_drift',
    'wind': 'ocean_surface_stress_equivalent_10m_neutral_wind_speed',
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


def _open_era5_dataset(path: str, chunks: Optional[Dict] = None) -> xr.Dataset:
    """Open ERA5 dataset with consistent parameters."""
    return xr.open_zarr(
        path,
        chunks=chunks,
        storage_options=dict(token='anon'),
        decode_times=True
    )


def _get_time_range(year: int, month: int, day: int, hour: int) -> Tuple[datetime.datetime, datetime.datetime]:
    """Get start and end times for query."""
    start_time = datetime.datetime(year, month, day, hour)
    end_time = start_time + datetime.timedelta(hours=6)
    return start_time, end_time


def _select_time_slice(ds: xr.Dataset, start_time: datetime.datetime, end_time: datetime.datetime) -> xr.Dataset:
    """Select time slice from dataset."""
    analysis_ready = ds.sel(time=slice(ds.attrs['valid_time_start'], ds.attrs['valid_time_stop']))
    return analysis_ready.sel(time=[start_time, end_time], method='nearest')


def _process_atmospheric_variable(data: xr.DataArray, var_name: str) -> xr.DataArray:
    """Process atmospheric variables with level selection."""
    if var_name in ATMOSPHERIC_VARS:
        return data[:, :len(PRESSURE_LEVELS), :, :]
    elif var_name == 'z':
        data = data.expand_dims({'level': len(PRESSURE_LEVELS)}, axis=2)
        dims_order = [data.dims[0], data.dims[2], data.dims[1], data.dims[3]]
        return data.transpose(*dims_order)
    return data


def get_two_timesteps_era5(path: str, var_name: str, year: int, month: int, day: int, hour: int) -> xr.DataArray:
    """Get two consecutive timesteps for a specific variable."""
    ds = _open_era5_dataset(path)
    start_time, end_time = _get_time_range(year, month, day, hour)
    analysis_ready = _select_time_slice(ds, start_time, end_time)
    
    era5_var_name = AURORA_TO_ERA5_FULL[var_name]
    data = analysis_ready[era5_var_name]
    
    return _process_atmospheric_variable(data, var_name)


def get_metadata_era5(path: str, year: int, month: int, day: int, hour_start: int) -> Metadata:
    """Extract metadata from ERA5 dataset."""
    ds = _open_era5_dataset(path, chunks={'time': 48})
    start_time, end_time = _get_time_range(year, month, day, hour_start)
    analysis_ready = _select_time_slice(ds, start_time, end_time)
    
    # Extract coordinate information
    latitudes = torch.from_numpy(analysis_ready['latitude'].values) if 'latitude' in analysis_ready else None
    longitudes = torch.from_numpy(analysis_ready['longitude'].values) if 'longitude' in analysis_ready else None
    
    return Metadata(
        time=tuple([end_time]),
        atmos_levels=tuple(PRESSURE_LEVELS),
        lat=latitudes,
        lon=longitudes,
        rollout_step=0
    )


def get_static_vars_era5(path: str) -> torch.Tensor:
    """Get static variables (land-sea mask) from ERA5 dataset."""
    ds = _open_era5_dataset(path)
    reference_time = datetime.datetime(2025, 1, 1, 0)
    ds_time_selected = ds.sel(time=[reference_time], method='nearest')
    
    return torch.from_numpy(ds_time_selected[AURORA_TO_ERA5_FULL['lsm']].values)[0]


def _load_variables(path: str, var_dict: Dict[str, str], year: int, month: int, day: int, hour_start: int) -> Dict[str, torch.Tensor]:
    """Load and convert variables to torch tensors."""
    result = {}
    for var_name in var_dict.keys():
        data = get_two_timesteps_era5(path, var_name, year, month, day, hour_start)
        result[var_name] = torch.from_numpy(data.values[:2][None])
    return result


def get_batch_era5(path: str, year: int, month: int, day: int, hour_start: int) -> Batch:
    """Create a complete batch from ERA5 data."""
    # Load surface and atmospheric variables
    surface_vars = _load_variables(path, AURORA_TO_ERA5_SURF, year, month, day, hour_start)
    atm_vars = _load_variables(path, AURORA_TO_ERA5_ATM, year, month, day, hour_start)
    
    # Get static variables and metadata
    lsm = get_static_vars_era5(path)
    metadata = get_metadata_era5(path, year, month, day, hour_start)
    
    return Batch(
        surf_vars=surface_vars,
        atmos_vars=atm_vars,
        metadata=metadata,
        static_vars={'lsm': lsm}
    )


def create_aurora_model() -> AuroraWave:
    """Create and initialize Aurora model."""
    model = AuroraWave(
        surf_vars=tuple(AURORA_TO_ERA5_SURF.keys()),
        angle_surf_vars=['mwd']
    )
    print("Model created successfully.")
    
    model.load_checkpoint(strict=False)
    print("Checkpoint loaded successfully.")
    
    model.eval()
    return model


def main():
    """Main execution function."""
    # Configuration
    path = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
    year, month, day, hour = 2020, 1, 1, 0
    
    # Create batch and save it
    print("Creating batch from ERA5 data...")
    batch = get_batch_era5(path, year, month, day, hour)
    
    with open("saved_batch.pkl", "wb") as f:
        pickle.dump(batch, f)
    print("Batch saved successfully.")
    
    # Create and run model
    model = create_aurora_model()
    
    print("Feeding the batch to the model...")
    pred = model.forward(batch)
    print("Prediction completed.")


if __name__ == "__main__":
    main()