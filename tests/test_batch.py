"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import numpy as np

from tests.conftest import SavedBatch

from aurora import Batch


def test_interpolation(test_input_output: tuple[Batch, SavedBatch]) -> None:
    batch, _ = test_input_output

    # Regridding to the same resolution shouldn't change the data.
    batch_regridded = batch.regrid(0.45)
    batch_regridded = batch_regridded.crop(4)  # Regridding added the south pole. Remove it again.

    for k in batch.surf_vars:
        np.testing.assert_allclose(
            batch.surf_vars[k],
            batch_regridded.surf_vars[k],
            rtol=5e-6,
        )
    for k in batch.static_vars:
        np.testing.assert_allclose(
            batch.static_vars[k],
            batch_regridded.static_vars[k],
            atol=1e-7,
        )
    for k in batch.atmos_vars:
        np.testing.assert_allclose(
            batch.atmos_vars[k],
            batch_regridded.atmos_vars[k],
            rtol=5e-6,
        )

    np.testing.assert_allclose(batch.metadata.lat, batch_regridded.metadata.lat, atol=1e-10)
    np.testing.assert_allclose(batch.metadata.lon, batch_regridded.metadata.lon, atol=1e-10)
