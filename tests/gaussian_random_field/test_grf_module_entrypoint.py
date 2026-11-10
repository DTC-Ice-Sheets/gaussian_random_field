"""Testing the entrypoint."""

import re
import sys
from pathlib import Path
from unittest.mock import patch

import cyclopts
import numpy as np
import pytest
import xarray as xr

from gaussian_random_field.grf_module.grf_module import ConstantFieldError
from gaussian_random_field.grf_module_entrypoint import app


def _evaluate_main_output(input_path: Path, output_path: Path) -> None:
    assert output_path.is_dir()
    ds = xr.open_zarr(output_path)
    assert ds.dims == {"time": 1, "x": 100, "y": 100}
    assert ds.land_ice_runoff_flux.dims == ("time", "x", "y")
    assert ds.land_ice_runoff_flux_mean.dims == ("time",)
    assert ds.land_ice_runoff_flux_uncertainty.dims == ("time", "x", "y")
    assert ds.land_ice_runoff_flux_uncertainty_mean.dims == ("time",)
    # Also check that the CRS and timestamp are preserved
    original = xr.open_zarr(input_path)
    assert original.spatial_ref == ds.spatial_ref
    assert original.time == ds.time


def _flatten(ds: xr.Dataset) -> xr.Dataset:
    """Give every point the same value, keeping the pattern of missing cells."""
    flat = ds.copy(deep=True)
    values = flat.land_ice_runoff_flux.values
    values[np.isfinite(values)] = 1.0
    return flat


_ARGV = [
    "grf_module_entrypoint",
    "--mesh_name",
    "watsonmesh",
    "--data_path",
    "https://example.com/test_preprocessed_timeseries.zarr",
]


@pytest.mark.slow
def test_main_valid_input_path(tmp_zarr_output_path, test_dataset_1_path: Path):
    input_path = test_dataset_1_path

    argv = [
        "gaussian_random_field",
        "--mesh_name",
        "watsonmesh",
        "--data_path",
        str(input_path),
        "--output_path",
        str(tmp_zarr_output_path),
    ]

    assert not tmp_zarr_output_path.is_dir()
    with patch.object(sys, "argv", argv):
        app(result_action="return_value")

    # Check output exists and has correct structure
    _evaluate_main_output(input_path, tmp_zarr_output_path)


@pytest.mark.slow
def test_main_valid_input_url(
    test_dataset_1_path: Path, test_dataset_1_loaded_zarr: xr.Dataset, tmp_zarr_output_path: Path
) -> None:
    input_zarr_url = "https://example.com/test_preprocessed_timeseries.zarr"

    with (
        # to avoid making an actual HTTP request, we patch the load_dataset function to return our fixture dataset
        patch(
            "gaussian_random_field.grf_module_entrypoint.xr.load_dataset",
            return_value=test_dataset_1_loaded_zarr,
        ),
        patch.object(
            sys,
            "argv",
            [
                "grf_module_entrypoint",
                "--mesh_name",
                "watsonmesh",
                "--data_path",
                str(input_zarr_url),
                "--output_path",
                str(tmp_zarr_output_path),
            ],
        ),
    ):
        assert not tmp_zarr_output_path.is_dir()
        app(result_action="return_value")

        # Check output exists and has correct structure
        _evaluate_main_output(test_dataset_1_path, tmp_zarr_output_path)


@pytest.mark.slow
def test_main_multiple_timesteps(test_dataset_1_loaded_zarr: xr.Dataset, tmp_zarr_output_path: Path) -> None:
    """A published dataset holds the whole record, not a single snapshot."""
    two_steps = xr.concat(
        [test_dataset_1_loaded_zarr, test_dataset_1_loaded_zarr],
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )
    two_steps = two_steps.assign_coords(time=[0, 1])

    with (
        patch("gaussian_random_field.grf_module_entrypoint.xr.load_dataset", return_value=two_steps),
        patch.object(
            sys,
            "argv",
            [
                "grf_module_entrypoint",
                "--mesh_name",
                "watsonmesh",
                "--data_path",
                "https://example.com/test_preprocessed_timeseries.zarr",
                "--output_path",
                str(tmp_zarr_output_path),
            ],
        ),
    ):
        app(result_action="return_value")

    ds = xr.open_zarr(tmp_zarr_output_path)
    assert ds.sizes["time"] == 2
    assert ds.land_ice_runoff_flux.dims == ("time", "x", "y")
    assert ds.land_ice_runoff_flux_mean.dims == ("time",)


@pytest.mark.slow
def test_main_fills_a_flat_timestep(test_dataset_1_loaded_zarr: xr.Dataset, tmp_zarr_output_path: Path) -> None:
    """A flat timestep is written as its constant, so the record comes back whole."""
    two_steps = xr.concat(
        [test_dataset_1_loaded_zarr, _flatten(test_dataset_1_loaded_zarr)],
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )
    two_steps = two_steps.assign_coords(time=[0, 1])

    with (
        patch("gaussian_random_field.grf_module_entrypoint.xr.load_dataset", return_value=two_steps),
        patch.object(sys, "argv", [*_ARGV, "--output_path", str(tmp_zarr_output_path)]),
    ):
        app(result_action="return_value")

    ds = xr.open_zarr(tmp_zarr_output_path)
    # Both timesteps come back: a gap here would leave the melt seasons
    # downstream unequal in length. The store writes its time axis as CF
    # datetimes, so the integers this fixture uses do not come back as
    # themselves, and what is checked is that neither went missing.
    assert ds.sizes["time"] == 2
    assert ds.time.values[0] < ds.time.values[1]

    filled = ds.land_ice_runoff_flux.isel(time=1).values
    present = filled[np.isfinite(filled)]
    assert present.size > 0
    np.testing.assert_allclose(present, 1.0)
    np.testing.assert_allclose(ds.land_ice_runoff_flux_mean.isel(time=1).values, 1.0)
    # No posterior was sampled for a constant, so its spread is unknown rather
    # than zero.
    assert np.isnan(ds.land_ice_runoff_flux_uncertainty.isel(time=1).values).all()
    assert np.isnan(ds.land_ice_runoff_flux_uncertainty_mean.isel(time=1).values).all()
    # The fitted timestep is untouched by any of this.
    assert np.isfinite(ds.land_ice_runoff_flux_uncertainty.isel(time=0).values).any()


@pytest.mark.slow
def test_main_fills_only_the_variable_that_is_flat(
    test_dataset_1_loaded_zarr: xr.Dataset, tmp_zarr_output_path: Path
) -> None:
    """One flat variable beside one that varies: the first is filled, the second fitted."""
    mixed = test_dataset_1_loaded_zarr.copy(deep=True)
    flat = mixed.land_ice_runoff_flux.copy(deep=True)
    values = flat.values
    values[np.isfinite(values)] = 1.0
    mixed["lake_area_fraction"] = flat

    with (
        patch("gaussian_random_field.grf_module_entrypoint.xr.load_dataset", return_value=mixed),
        patch.object(sys, "argv", [*_ARGV, "--output_path", str(tmp_zarr_output_path)]),
    ):
        app(result_action="return_value")

    ds = xr.open_zarr(tmp_zarr_output_path)
    # The varying variable is fitted as it would have been on its own, so it
    # carries a posterior spread.
    assert np.isfinite(ds.land_ice_runoff_flux_uncertainty.values).any()
    assert np.isfinite(ds.land_ice_runoff_flux_uncertainty_mean.values).all()
    # The flat one is written as its constant, with no spread, rather than
    # taking the whole request down with it.
    present = ds.lake_area_fraction.values[np.isfinite(ds.lake_area_fraction.values)]
    assert present.size > 0
    np.testing.assert_allclose(present, 1.0)
    np.testing.assert_allclose(ds.lake_area_fraction_mean.values, 1.0)
    assert np.isnan(ds.lake_area_fraction_uncertainty.values).all()
    assert np.isnan(ds.lake_area_fraction_uncertainty_mean.values).all()


def test_main_raises_when_every_timestep_is_flat(
    test_dataset_1_loaded_zarr: xr.Dataset, tmp_zarr_output_path: Path
) -> None:
    """With nothing in the record fitted, filling it all in would return a constant, not a fit."""
    with (
        patch(
            "gaussian_random_field.grf_module_entrypoint.xr.load_dataset",
            return_value=_flatten(test_dataset_1_loaded_zarr),
        ),
        patch.object(sys, "argv", [*_ARGV, "--output_path", str(tmp_zarr_output_path)]),
        pytest.raises(ConstantFieldError, match="no variable in any of the 1 timesteps"),
    ):
        app(result_action="return_value", exit_on_error=False)


@pytest.mark.parametrize(
    "test_source",
    ["invalid_file_path", "invalid_path", "invalid_url"],
)
def test_main_invalid_input_raises_validation_error(test_source: str, tmp_zarr_output_path: Path, request) -> None:
    error_string = test_data = request.getfixturevalue(test_source)
    with (
        patch.object(
            sys,
            "argv",
            [
                "grf_module_entrypoint",
                "--mesh_name",
                "watsonmesh",
                "--data_path",
                str(test_data),
                "--output_path",
                str(tmp_zarr_output_path),
            ],
        ),
        pytest.raises(
            cyclopts.ValidationError,
            match=re.escape(f"The provided path '{error_string}' is neither a valid local path nor a web URL."),
        ),
    ):
        app(result_action="return_value", exit_on_error=False)
