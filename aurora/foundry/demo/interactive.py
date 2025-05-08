"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import base64
import tempfile

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from ipyleaflet import (
    ImageOverlay,
    LayersControl,
    Map,
    projections,
)
from ipyleaflet.velocity import Velocity
from ipywidgets import Layout

from aurora import Batch

__all__ = ["interactive_plot"]


def variable_to_urldata(variable: xr.DataArray, cmap: str, vmin: float, vmax: float) -> str:
    """Encode a variable in an URL to show it on a plot."""
    # Center the variable correctly.
    field = variable.values
    first_half = variable.longitude.values <= 180
    field = np.concatenate((field[:, ~first_half], field[:, first_half]), axis=1)

    with tempfile.NamedTemporaryFile(suffix=".png") as tf:
        tf.close()  # We will only use the name.

        # Make the image that will be overlayed.
        fig = plt.figure()
        ax = fig.add_subplot(projection=ccrs.PlateCarree(), frameon=False)
        ax.set_global()
        ax.coastlines(lw=0.1)
        plt.imshow(
            field,
            extent=(-180, 180, -90, 90),
            transform=ccrs.PlateCarree(),
            aspect=1,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        plt.savefig(tf.name, bbox_inches="tight", pad_inches=0, dpi=1200)
        plt.close("all")

        # Produce the URL data.
        with open(tf.name, "rb") as f:
            base64_utf8_str = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/png;base64,{base64_utf8_str}"


def interactive_plot(prediction: Batch, width: str = "1000px", height: str = "500px") -> Map:
    m = Map(
        layers=[],
        center=(20, 10),
        zoom=2,
        interpolation="nearest",
        crs=projections.EPSG4326,
        layout=Layout(width=width, height=height),
    )

    with tempfile.NamedTemporaryFile(suffix=".nc") as tf:
        tf.close()  # We will only use the name.

        prediction.to_netcdf(tf.name)
        ds = xr.load_dataset(tf.name).isel(batch=0, history=0, time=0)
        dt = ds.time.values.astype("datetime64[s]").tolist()
        print("Prediction for " + dt.strftime("%Y-%m-%d %H:%M"))

        wind = Velocity(
            data=ds,
            name="Wind",
            zonal_speed="surf_10u",
            meridional_speed="surf_10v",
            latitude_dimension="latitude",
            longitude_dimension="longitude",
            velocity_scale=0.01,
            max_velocity=20,
            display_options={
                "velocityType": "Global wind",
                "displayPosition": "bottomleft",
                "displayEmptyString": "No wind data",
            },
        )
        m.add(wind)

        layer = ImageOverlay(
            name="Mean pressure at sea level",
            url=variable_to_urldata(
                ds["surf_msl"], "viridis", 100 * (1000 - 20), 100 * (1000 + 20)
            ),
            bounds=((-90, -180), (90, 180)),
        )
        m.add_layer(layer)

        layer = ImageOverlay(
            name="Temperature",
            url=variable_to_urldata(ds["surf_2t"], "RdBu_r", 273.15 - 50, 273.15 + 50),
            bounds=((-90, -180), (90, 180)),
        )
        m.add_layer(layer)

        m.add_control(LayersControl())

    return m
