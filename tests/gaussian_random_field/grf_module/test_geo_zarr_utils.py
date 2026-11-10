"""Tests for legacy-compatible GeoZarr serialization."""

import numpy as np
import pytest
import xarray as xr
import zarr

from gaussian_random_field.grf_module.geo_zarr_utils import (
    prepare_cf_data_variable_arrays,
    prepare_cf_time_data_array,
    save_geo_zarr,
)


@pytest.fixture
def runoff_dataset():
    values = np.array([[[np.nan, 2.0], [3.0, 4.0]]])
    ds = xr.Dataset(
        {
            "land_ice_runoff_flux": (("time", "x", "y"), values),
            "land_ice_runoff_flux_uncertainty": (("time", "x", "y"), values / 10),
            "land_ice_runoff_flux_mean": ("time", [3.0]),
            "land_ice_runoff_flux_uncertainty_mean": ("time", [0.3]),
        },
        coords={
            "time": [np.datetime64("2019-08-15T00:00:00.123456789")],
            "x": [100.0, 200.0],
            "y": [300.0, 400.0],
        },
        attrs={"source": "not copied by the legacy writer"},
    ).rio.write_crs("EPSG:3013")
    for name in ds.data_vars:
        ds[name].attrs = {"standard_name": "land_ice_runoff_flux", "units": "kg m-2 s-1"}
        # A decoded input may retain conflicting serialization instructions.
        ds[name].attrs["grid_mapping"] = "spatial_ref"
        ds[name].encoding.update(_FillValue=np.nan, scale_factor=0.1, add_offset=10.0, dtype="int16", chunks=(1,))
    ds.time.encoding.update(units="days since 2000-01-01", calendar="proleptic_gregorian")
    return ds


def test_save_geo_zarr_does_not_mutate_input(runoff_dataset, tmp_zarr_output_path):
    before = runoff_dataset.copy(deep=True)
    save_geo_zarr(runoff_dataset, tmp_zarr_output_path)
    xr.testing.assert_identical(runoff_dataset, before)
    for name in runoff_dataset.variables:
        assert runoff_dataset[name].encoding == before[name].encoding


def test_save_geo_zarr_respects_target_chunks(runoff_dataset, tmp_zarr_output_path):
    ds = runoff_dataset.isel(time=[0] * 100)
    save_geo_zarr(ds, tmp_zarr_output_path, target_chunk_size="1KiB")
    array = zarr.open_group(tmp_zarr_output_path, mode="r")["land_ice_runoff_flux"]
    assert np.prod(array.chunks) * array.dtype.itemsize <= 1024
    assert array.chunks != array.shape
    np.testing.assert_array_equal(array[...], ds.land_ice_runoff_flux.values)


@pytest.mark.parametrize(
    ("name", "data", "expected_fill_value", "has_grid_mapping"),
    [
        ("unsigned", ("point", np.array([1], dtype=np.uint8)), np.iinfo(np.uint8).max, True),
        ("floating", ("point", np.array([1.0], dtype=np.float32)), -9999.0, True),
        ("signed", ("point", np.array([1], dtype=np.int16)), -9999, True),
        ("boolean", ("point", np.array([True], dtype=np.bool_)), 0, True),
        ("string", ("point", np.array(["value"], dtype=str)), "", True),
        ("scalar", xr.DataArray(1.0), None, True),
        ("spatial_ref", xr.DataArray(0), None, False),
    ],
)
def test_prepare_cf_data_variables_sets_fill_value_and_metadata(
    name,
    data,
    expected_fill_value,
    has_grid_mapping,
):
    ds = xr.Dataset({name: data})
    ds[name].encoding["scale_factor"] = 0.1
    actual = prepare_cf_data_variable_arrays(ds)

    assert actual is ds
    assert actual[name].attrs["_FillValue"] == expected_fill_value
    assert actual[name].encoding == {"coordinates": None}
    if has_grid_mapping:
        assert actual[name].attrs["grid_mapping"] == "spatial_ref"
    else:
        assert "grid_mapping" not in actual[name].attrs


def test_prepare_time_data_array_uses_numeric_epoch_seconds():
    time = xr.DataArray(
        np.array(["1969-12-31T23:59:59", "2019-08-15T00:00:00.999"], dtype="datetime64[ns]"),
        dims="time",
    )
    actual = prepare_cf_time_data_array(time)
    assert actual.dtype == np.dtype("float64")
    np.testing.assert_array_equal(actual.values, [-1.0, 1565827200.0])
    assert actual.attrs["units"] == "seconds since 1970-01-01T00:00:00"
