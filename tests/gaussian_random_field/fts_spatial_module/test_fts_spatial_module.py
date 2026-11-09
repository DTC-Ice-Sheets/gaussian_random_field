"""Testing the FTS Spatial Module."""

import logging

import numpy as np
import pytest
import xarray as xr
import zarr  # ruff:ignore[unused-import]

from gaussian_random_field.fts_spatial_module.fts_spatial_module import FtsSpatialModule
from gaussian_random_field.fts_spatial_module.hdfpca import hdfpca_factor_scores

CURVE_TIME_POINTS = 5


def encoded_dataset(n_points: int, n_times: int) -> xr.Dataset:
    """
    Dataset whose values are ``point * 1000 + time``, so any misplaced value shows where it came from.

    Parameters
    ----------
    n_points : int
        Number of spatial points. Kept different from `n_times`, so a transposed matrix cannot pass on shape.
    n_times : int
        Number of time steps.

    Returns
    -------
    xr.Dataset
        One variable, "runoff", with dimensions ("time", "point_id").
    """
    values = (np.arange(n_points)[:, None] * 1000 + np.arange(n_times)[None, :]).astype(float)
    return xr.Dataset(
        {"runoff": (("time", "point_id"), values.T)},
        coords={"time": np.arange(n_times), "point_id": np.arange(n_points)},
    )


def gridded_dataset(n_x: int, n_y: int, n_times: int, nan_border: bool = True) -> xr.Dataset:
    """
    Dataset in the form the GRF module publishes: one variable over ("time", "x", "y").

    Parameters
    ----------
    n_x : int
        Number of grid columns.
    n_y : int
        Number of grid rows. Kept different from `n_x` so a transposed grid cannot pass on shape.
    n_times : int
        Number of time steps.
    nan_border : bool, optional
        Fill the first column with NaN, as gridding does for cells outside the fitted area.

    Returns
    -------
    xr.Dataset
        One variable, "runoff", with dimensions ("time", "x", "y").
    """
    values = np.random.Generator(np.random.PCG64(7)).random((n_times, n_x, n_y))
    if nan_border:
        values[:, 0, :] = np.nan
    return xr.Dataset(
        {"runoff": (("time", "x", "y"), values)},
        coords={"time": np.arange(n_times), "x": np.arange(float(n_x)), "y": np.arange(float(n_y))},
    )


@pytest.fixture
def fts_spatial_module():
    return FtsSpatialModule()


@pytest.fixture
def test_data() -> xr.Dataset:
    num_time = 24 * 10
    num_points = 100
    # TODO Real data with multiple timesteps
    return xr.Dataset(
        data_vars={
            "some_quantity": (
                ("time", "point_id"),
                np.random.Generator(np.random.PCG64(42)).random((num_time, num_points)),
            ),
        },
        coords={"time": np.arange(num_time), "point_id": np.arange(num_points)},
    )


@pytest.fixture
def whole_curve_data() -> xr.Dataset:
    return encoded_dataset(n_points=7, n_times=20)  # 4 curves of 5


@pytest.fixture
def ragged_data() -> xr.Dataset:
    return encoded_dataset(n_points=7, n_times=23)  # 4 curves of 5, plus 3 left over


def test_fts_spatial_module_can_be_instantiated(fts_spatial_module):
    assert fts_spatial_module is not None


def test_run_spatial_analysis(fts_spatial_module, test_data):
    results = fts_spatial_module.run_spatial_analysis(test_data)

    assert isinstance(results, xr.Dataset)
    assert dict(results.dims) == {"curve_id": 10, "coef_id": 3}
    assert "some_quantity" in results.data_vars


@pytest.mark.parametrize("as_coordinate", [True, False])
def test_run_spatial_analysis_preserves_crs(fts_spatial_module, test_data, as_coordinate):
    spatial_ref = xr.DataArray(0, attrs={"crs_wkt": "example-wkt", "grid_mapping_name": "polar_stereographic"})
    test_data["spatial_ref"] = spatial_ref
    if as_coordinate:
        test_data = test_data.set_coords("spatial_ref")

    results = fts_spatial_module.run_spatial_analysis(test_data)

    assert "spatial_ref" in results.coords
    xr.testing.assert_identical(results["spatial_ref"].reset_coords(drop=True), spatial_ref.rename("spatial_ref"))
    assert results.spatial_ref.crs_wkt == test_data.spatial_ref.crs_wkt
    results["spatial_ref"].attrs["crs_wkt"] = "changed-wkt"
    assert test_data.spatial_ref.crs_wkt == "example-wkt"


def test_each_point_is_analysed_as_one_row(fts_spatial_module, test_data):
    """
    The analysis must see one row per point, whatever order the dataset's dimensions are stored in.

    The data is stored as ("time", "point_id"), so passing it through without the transpose would treat each
    time step as a point. test_hdfpca checks against R how a row is then cut into curves.
    """
    data = test_data["some_quantity"].transpose("point_id", "time").values

    results = fts_spatial_module.run_spatial_analysis(test_data)

    np.testing.assert_array_equal(results["some_quantity"].values, hdfpca_factor_scores(data, 24, 2, 3))


def test_time_axis_is_truncated_to_whole_curves(ragged_data, caplog):
    data = ragged_data["runoff"].transpose("point_id", "time").values

    with caplog.at_level(logging.WARNING):
        truncated = FtsSpatialModule.whole_curves(data, CURVE_TIME_POINTS, "runoff")

    assert truncated.shape == (7, 20)
    np.testing.assert_array_equal(truncated, data[:, :20])
    assert "not a multiple" in caplog.text


def test_whole_curves_are_left_alone(whole_curve_data, caplog):
    data = whole_curve_data["runoff"].transpose("point_id", "time").values

    with caplog.at_level(logging.WARNING):
        result = FtsSpatialModule.whole_curves(data, CURVE_TIME_POINTS, "runoff")

    np.testing.assert_array_equal(result, data)
    assert caplog.text == ""


def test_only_spatial_variables_are_analysed(whole_curve_data):
    whole_curve_data["elevation"] = ("point_id", np.arange(7, dtype=float))

    assert FtsSpatialModule.spatial_variables(whole_curve_data) == ["runoff"]


def test_a_grid_is_flattened_into_points(caplog):
    """
    The GRF module grids its fit before saving, so FTS has to accept ("time", "x", "y").

    Each kept point must still carry its own cell's time series, and cells that are NaN anywhere in the
    record are dropped, since a curve cannot be built across a gap.
    """
    gridded = gridded_dataset(n_x=5, n_y=4, n_times=20)

    with caplog.at_level(logging.WARNING):
        flattened = FtsSpatialModule.as_point_cloud(gridded)

    assert FtsSpatialModule.spatial_variables(flattened) == ["runoff"]
    assert flattened.sizes["point_id"] == 5 * 4 - 4  # the NaN column is gone
    assert np.isfinite(flattened["runoff"].values).all()
    assert "not finite throughout" in caplog.text
    expected = gridded["runoff"].stack(point_id=("x", "y")).isel(point_id=slice(4, None))
    np.testing.assert_array_equal(flattened["runoff"].values, expected.values)


def test_the_mask_is_shared_across_variables_and_timesteps(caplog):
    """
    A point that is missing from one variable at one timestep is dropped from all of them.

    Every variable is then described on the same points, so a curve fitted to one lines
    up with a curve fitted to another. Neither gap here is in the first variable and the
    first timestep together, so a mask built from either alone would keep a point that is
    not finite throughout the record.
    """
    gridded = gridded_dataset(n_x=5, n_y=4, n_times=20, nan_border=False)
    gridded["lake_area"] = gridded["runoff"] * 2
    gridded["runoff"].values[3, 1, 1] = np.nan
    gridded["lake_area"].values[7, 2, 2] = np.nan

    with caplog.at_level(logging.WARNING):
        flattened = FtsSpatialModule.as_point_cloud(gridded)

    assert sorted(FtsSpatialModule.spatial_variables(flattened)) == ["lake_area", "runoff"]
    # One cell lost to each variable, and both lost to both.
    assert flattened.sizes["point_id"] == 5 * 4 - 2
    assert "not finite throughout" in caplog.text

    stacked = gridded.stack(point_id=("x", "y"))
    keep = np.all(
        [np.isfinite(stacked[v].values) for v in ("runoff", "lake_area")],
        axis=(0, 1),
    )
    for name in ("runoff", "lake_area"):
        assert np.isfinite(flattened[name].values).all(), name
        np.testing.assert_array_equal(
            flattened[name].values,
            stacked[name].isel(point_id=keep).values,
        )


def test_a_variable_that_is_empty_at_a_timestep_is_dropped_not_the_points(caplog):
    """
    The GRF module fills a flat snapshot with its constant and writes NaN for its uncertainty.

    That leaves the uncertainty field with no values at all in that timestep. The mask here is
    shared across variables, so keeping it would drop every point in the catchment and leave
    nothing to analyse - which is what happened to the supraglacial lake record, whose lakes
    have drained by September.
    """
    gridded = gridded_dataset(n_x=5, n_y=4, n_times=20, nan_border=False)
    gridded["runoff_uncertainty"] = gridded["runoff"] * 0.1
    gridded["runoff_uncertainty"].values[9, :, :] = np.nan

    with caplog.at_level(logging.WARNING):
        flattened = FtsSpatialModule.as_point_cloud(gridded)

    assert FtsSpatialModule.spatial_variables(flattened) == ["runoff"]
    assert "runoff_uncertainty" in caplog.text
    # Every point survives: the field itself has a value everywhere.
    assert flattened.sizes["point_id"] == 5 * 4
    assert np.isfinite(flattened["runoff"].values).all()


def test_a_gap_in_one_timestep_still_drops_the_point(caplog):
    """A variable with some values at every timestep is analysed, and its gaps still cost points."""
    gridded = gridded_dataset(n_x=5, n_y=4, n_times=20, nan_border=False)
    gridded["runoff"].values[9, 2, 2] = np.nan

    with caplog.at_level(logging.WARNING):
        flattened = FtsSpatialModule.as_point_cloud(gridded)

    assert FtsSpatialModule.spatial_variables(flattened) == ["runoff"]
    assert flattened.sizes["point_id"] == 5 * 4 - 1
    assert "not finite throughout" in caplog.text


def test_a_point_cloud_is_left_alone(whole_curve_data):
    assert FtsSpatialModule.as_point_cloud(whole_curve_data) is whole_curve_data


def test_the_old_grid_coordinate_names_are_accepted():
    gridded = gridded_dataset(n_x=5, n_y=4, n_times=20, nan_border=False).rename(
        {"x": "projection_x_coordinate", "y": "projection_y_coordinate"}
    )

    flattened = FtsSpatialModule.as_point_cloud(gridded)

    assert FtsSpatialModule.spatial_variables(flattened) == ["runoff"]
    assert flattened.sizes["point_id"] == 5 * 4


def test_a_grid_gives_the_same_answer_as_the_points_it_flattens_to(fts_spatial_module):
    gridded = gridded_dataset(n_x=10, n_y=9, n_times=24 * 10)
    as_points = FtsSpatialModule.as_point_cloud(gridded)

    from_grid = fts_spatial_module.run_spatial_analysis(gridded)
    from_points = fts_spatial_module.run_spatial_analysis(as_points)

    assert "runoff" in from_grid.data_vars
    assert dict(from_grid.dims) == {"curve_id": 10, "coef_id": 3}
    xr.testing.assert_allclose(from_grid, from_points)


def test_nothing_to_analyse_is_refused_rather_than_returned_empty(fts_spatial_module):
    """
    An empty result records the run as a success and leaves the caller with a store holding no data.

    This is what a gridded input used to do, so the silence is worth a test of its own.
    """
    unusable = xr.Dataset(
        {"runoff": (("time", "level"), np.zeros((20, 3)))},
        coords={"time": np.arange(20), "level": np.arange(3)},
    )

    with pytest.raises(ValueError, match="nothing to analyse"):
        fts_spatial_module.run_spatial_analysis(unusable)
