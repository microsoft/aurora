"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, minimum_filter

from aurora.batch import Batch

__all__ = ["Tracker"]

logger = logging.getLogger(__file__)


class NoEyeException(Exception):
    """Raised when no eye can be found."""


def get_box(
    variable: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
):
    """Get a square box for a variable."""
    # Make latitude selection.
    lat_mask = (lat_min <= lats) & (lats <= lat_max)
    box = variable[..., lat_mask, :]
    lats = lats[lat_mask]

    # Make longitude selection. Be careful when wrapping around.
    lon_min = lon_min % 360
    lon_max = lon_max % 360
    if lon_min <= lon_max:
        lon_mask = (lon_min <= lons) & (lons <= lon_max)
        box = box[..., lon_mask]
        lons = lons[lon_mask]
    else:
        lon_mask1 = lon_min <= lons
        lon_mask2 = lons <= lon_max
        box = np.concatenate((box[..., lon_mask1], box[..., lon_mask2]), axis=-1)
        lons = np.concatenate((lons[lon_mask1], lons[lon_mask2]))

    return lats, lons, box


def havdist(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance between two latitude-longitude coordinates."""
    lat1, lat2 = np.deg2rad(lat1), np.deg2rad(lat2)
    lon1, lon2 = np.deg2rad(lon1), np.deg2rad(lon2)
    rad_earth_km = 6371
    inner = 1 - np.cos(lat2 - lat1) + np.cos(lat1) * np.cos(lat2) * (1 - np.cos(lon2 - lon1))
    return 2 * rad_earth_km * np.arcsin(np.sqrt(0.5 * inner))


def get_closest_min(
    variable: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    lat: float,
    lon: float,
    delta_lat: float = 5,
    delta_lon: float = 5,
    minimum_cap_size: int = 8,
) -> tuple[float, float]:
    """Get the minimum in `variable` that is closest to `lat` and `lon`."""
    # Create a box centred around the current latitude and longitude.
    lats, lons, box = get_box(
        variable,
        lats,
        lons,
        lat - delta_lat,
        lat + delta_lat,
        lon - delta_lon,
        lon + delta_lon,
    )

    # Smooth to avoid local minima due to noise.
    box = gaussian_filter(box, sigma=1)

    # Find local minima.
    local_minima = minimum_filter(box, size=(minimum_cap_size, minimum_cap_size)) == box

    # Remove minima at the edges: these occur when the tracker fails.
    local_minima[0, :] = 0
    local_minima[-1, :] = 0
    local_minima[:, 0] = 0
    local_minima[:, -1] = 0

    # If no local minima are left, no eye can be found. Try the next one.
    if local_minima.sum() == 0:
        raise NoEyeException()

    # Return the latitude and longitude of the closest local minimum.
    lat_inds, lon_inds = zip(*np.argwhere(local_minima))
    dists = havdist(lats[list(lat_inds)], lons[list(lon_inds)], lat, lon)
    i = np.argmin(dists)

    return lats[lat_inds[i]], lons[lon_inds[i]]


def extrapolate(lats: list[float], lons: list[float]) -> tuple[float, float]:
    """Guess an initial latitude and longitude by extrapolating `lats` and `lons`."""
    assert len(lats) == len(lons)
    if len(lats) == 0:
        raise ValueError("Cannot extrapolate from empty lists.")
    elif len(lats) == 1:
        return lats[0], lons[0]
    else:
        # Linearly extrapolate using the last eight points.
        lats = lats[-8:]
        lons = lons[-8:]
        n = len(lats)
        fit = np.polyfit(np.arange(n), np.stack((lats, lons), axis=-1), 1)
        return np.polyval(fit, n)


class Tracker:
    """Simple tropical cyclone tracker.

    This algorithm was originally designed and implemented by Anna Allen. This particular
    implementation is by Wessel Bruinsma and features various improvements over the original design.
    """

    def __init__(
        self,
        init_lat: float,
        init_lon: float,
        init_time: datetime,
    ) -> None:
        self.tracked_times: list[datetime] = [init_time]
        self.tracked_lats: list[float] = [init_lat]
        self.tracked_lons: list[float] = [init_lon]
        self.tracked_msls: list[float] = [np.nan]
        self.tracked_winds: list[float] = [np.nan]
        self.fails: int = 0

    def results(self) -> pd.DataFrame:
        """Assemble the track into a convenient DataFrame."""
        return pd.DataFrame(
            {
                "time": self.tracked_times,
                "lat": self.tracked_lats,
                "lon": self.tracked_lons,
                "msl": self.tracked_msls,
                "wind": self.tracked_winds,
            }
        )

    def step(self, batch: Batch) -> None:
        """Track the next step.

        Args:
            batch (:class:`aurora.batch.Batch`): Prediction.
        """
        # Check that there is only one prediction. We don't support batched tracking.
        if len(batch.metadata.time) != 1:
            raise RuntimeError("Predictions don't have batch size one.")

        # No need to do tracking on the GPU. It's cheap.
        batch = batch.to("cpu")

        # Extract the relevant variables from the prediction.
        z700_index = list(batch.metadata.atmos_levels).index(700)
        z700 = batch.atmos_vars["z"][0, 0, z700_index].numpy()
        msl = batch.surf_vars["msl"][0, 0].numpy()
        u10 = batch.surf_vars["10u"][0, 0].numpy()
        v10 = batch.surf_vars["10v"][0, 0].numpy()
        wind = np.sqrt(u10 * u10 + v10 * v10)
        lsm = batch.static_vars["lsm"].numpy()
        lats = batch.metadata.lat.numpy()
        lons = batch.metadata.lon.numpy()
        time = batch.metadata.time[0]

        # Provide an initial guess by extrapolating.
        lat, lon = extrapolate(self.tracked_lats, self.tracked_lons)
        lat = max(min(lat, 90), -90)
        lon = lon % 360

        def is_clear(lat: float, lon: float, delta: float) -> bool:
            """Is a box centred at `lat` and `lon` with "radius" `delta` clear of land?"""
            _, _, lsm_box = get_box(
                lsm,
                lats,
                lons,
                lat - delta,
                lat + delta,
                lon - delta,
                lon + delta,
            )
            return lsm_box.max() < 0.5

        # Did we "snap" from the guess to a real nearby minimum?
        snap = False

        # Try MSL with increasingly small boxes.
        for delta in [5, 4, 3, 2, 1.5]:
            try:
                if is_clear(lat, lon, delta):
                    lat, lon = get_closest_min(
                        msl,
                        lats,
                        lons,
                        lat,
                        lon,
                        delta_lat=delta,
                        delta_lon=delta,
                    )
                    snap = True
                    break
            except NoEyeException:
                pass

        if not snap:
            # MSL didn't work. Try Z700. If it works, try to refine with MSL.
            try:
                lat, lon = get_closest_min(
                    z700,
                    lats,
                    lons,
                    lat,
                    lon,
                    delta_lat=5,
                    delta_lon=5,
                )
                snap = True

                for delta in [5, 4, 3, 2, 1.5]:
                    try:
                        if is_clear(lat, lon, delta):
                            lat, lon = get_closest_min(
                                msl,
                                lats,
                                lons,
                                lat,
                                lon,
                                delta_lat=delta,
                                delta_lon=delta,
                            )
                            break
                    except NoEyeException:
                        pass
            except NoEyeException:
                pass

        if not snap:
            self.fails += 1
            if len(self.tracked_lats) > 1:
                logger.info(f"Failed at time {time}. Extrapolating in a silly way.")
            else:
                raise NoEyeException("Completely failed at the first step.")

        self.tracked_times.append(time)
        self.tracked_lats.append(lat)
        self.tracked_lons.append(lon)

        # Extract minimum MSL and maximum wind speed from a crop around the TC.
        _, _, msl_crop = get_box(
            msl,
            lats,
            lons,
            lat - 1.5,
            lat + 1.5,
            lon - 1.5,
            lon + 1.5,
        )
        _, _, wind_crop = get_box(
            wind,
            lats,
            lons,
            lat - 1.5,
            lat + 1.5,
            lon - 1.5,
            lon + 1.5,
        )
        self.tracked_msls.append(msl_crop.min())
        self.tracked_winds.append(wind_crop.max())
