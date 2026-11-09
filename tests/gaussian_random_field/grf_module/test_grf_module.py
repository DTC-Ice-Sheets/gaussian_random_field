"""Testing the GRF module."""

import json
from pathlib import Path

import numpy as np
import pyproj
import pytest
import xarray as xr
import zarr  # ruff:ignore[unused-import]

from gaussian_random_field.grf_module.geo_zarr_utils import sparse_grid_to_point_cloud, trim_to_mesh
from gaussian_random_field.grf_module.grf_module import ConstantFieldError, GrfModule
from gaussian_random_field.grf_module.spde_mesh import SpdeMesh


@pytest.fixture
def grf_module() -> GrfModule:
    return GrfModule()


@pytest.fixture
def spde_mesh() -> SpdeMesh:
    return SpdeMesh(
        mesh_name="watsonmesh",
        prior_sigma_1=1.0,
        # Using a tiny value here seems to make tests more stable
        prior_sigma_2=0.0001,
    )


@pytest.fixture
def test_data(test_dataset_1_path: Path) -> xr.Dataset:
    ds = xr.load_dataset(test_dataset_1_path, engine="zarr", chunks={})
    return sparse_grid_to_point_cloud(ds)


@pytest.fixture
def point_cloud_data(test_dataset_1_path: Path) -> xr.Dataset:
    """Store the same snapshot as a point cloud rather than a grid.

    RACMO's native run reaches the store in this form, because its rotated
    pole does not give a regular grid once reprojected.
    """
    gridded = xr.load_dataset(test_dataset_1_path, engine="zarr", chunks={})
    stacked = sparse_grid_to_point_cloud(gridded)
    return xr.Dataset(
        data_vars={
            "land_ice_runoff_flux": (
                ("point", "time"),
                stacked.land_ice_runoff_flux.transpose("point_id", "time").values,
            ),
            "x": (("point",), stacked.x.values),
            "y": (("point",), stacked.y.values),
        },
        coords={"time": stacked.time.values},
    )


def test_point_cloud_input_is_accepted(point_cloud_data):
    converted = sparse_grid_to_point_cloud(point_cloud_data)

    assert converted.land_ice_runoff_flux.dims == ("time", "point_id")
    assert converted.x.dims == ("point_id",)
    assert converted.y.dims == ("point_id",)
    np.testing.assert_allclose(
        converted.land_ice_runoff_flux.values,
        point_cloud_data.land_ice_runoff_flux.transpose("time", "point").values,
    )
    np.testing.assert_allclose(converted.x.values, point_cloud_data.x.values)
    np.testing.assert_allclose(converted.y.values, point_cloud_data.y.values)


def test_point_cloud_without_positions_is_rejected(point_cloud_data):
    with pytest.raises(ValueError, match="must carry x and y"):
        sparse_grid_to_point_cloud(point_cloud_data.drop_vars(["x", "y"]))


def test_dataset_without_spatial_dimensions_is_rejected():
    ds = xr.Dataset(
        data_vars={"land_ice_runoff_flux": (("time",), np.zeros(1))},
        coords={"time": [0]},
    )
    with pytest.raises(ValueError, match="expected either x and y"):
        sparse_grid_to_point_cloud(ds)


def test_mesh_covers_its_own_vertices(spde_mesh):
    x, y = spde_mesh.vertices[:, 0], spde_mesh.vertices[:, 1]
    assert spde_mesh.covers(x, y).all()


def test_mesh_does_not_cover_distant_points(spde_mesh):
    far = spde_mesh.vertices[:, 0].min() - 500_000
    assert not spde_mesh.covers(np.array([far]), np.array([0.0])).any()


def _point_cloud(x: np.ndarray, y: np.ndarray) -> xr.Dataset:
    """Build a plain point cloud, without the multi-index that stacking leaves behind."""
    return xr.Dataset(
        data_vars={"land_ice_runoff_flux": (("time", "point_id"), np.zeros((1, len(x))))},
        coords={"time": [np.datetime64("2019-08-15", "ns")], "x": ("point_id", x), "y": ("point_id", y)},
    )


def test_trim_to_mesh_drops_points_outside(spde_mesh):
    """The published inputs cover a buffer around the catchment, the mesh does not."""
    on_mesh = spde_mesh.vertices
    away = on_mesh[:, 0].min() - 500_000
    ds = _point_cloud(
        np.concatenate([on_mesh[:, 0], np.full(3, away)]),
        np.concatenate([on_mesh[:, 1], np.zeros(3)]),
    )

    trimmed = trim_to_mesh(ds, spde_mesh)

    assert trimmed.sizes["point_id"] == len(on_mesh)
    np.testing.assert_allclose(trimmed.x.values, on_mesh[:, 0])
    np.testing.assert_allclose(trimmed.y.values, on_mesh[:, 1])


def test_trim_to_mesh_keeps_everything_when_all_points_are_on_the_mesh(spde_mesh):
    ds = _point_cloud(spde_mesh.vertices[:, 0], spde_mesh.vertices[:, 1])
    assert trim_to_mesh(ds, spde_mesh).sizes["point_id"] == len(spde_mesh.vertices)


def test_trim_to_mesh_rejects_data_that_misses_the_mesh(spde_mesh):
    away = spde_mesh.vertices[:, 0].min() - 500_000
    ds = _point_cloud(np.full(5, away), np.zeros(5))
    with pytest.raises(ValueError, match="none of the"):
        trim_to_mesh(ds, spde_mesh)


def test_grf_module_can_be_instantiated(grf_module):
    assert grf_module is not None
    assert grf_module.est_gp_inla is not None


def test_spde_mesh_can_be_instantiated(spde_mesh):
    assert spde_mesh is not None
    assert spde_mesh.mesh is not None
    assert spde_mesh.spde is not None


@pytest.fixture
def log_test_data(test_data) -> xr.Dataset:
    # The fit is unstable on raw flux (mostly near zero, long tail) and sometimes
    # collapses to a flat field. It is stable on a log scale, which is also how
    # the Use Case 1 notebook fits runoff.
    test_data["land_ice_runoff_flux"] = np.log10(1 + test_data["land_ice_runoff_flux"])
    return test_data


def test_data_is_valid(test_data):
    assert test_data.dims == {"time": 1, "point_id": 5776}

    vals = test_data.land_ice_runoff_flux
    assert vals.dims == ("time", "point_id")
    np.testing.assert_allclose(vals.min(), 0.0, atol=1e-6)
    np.testing.assert_allclose(vals.max(), 1650.0, atol=1e-6)
    np.testing.assert_allclose(vals.mean(), 298.040859, atol=1e-6)

    locs_x = test_data.x
    locs_y = test_data.y
    assert locs_x.dims == ("point_id",)
    assert locs_y.dims == ("point_id",)
    np.testing.assert_allclose(locs_x.mean(), -150794.296399, atol=1e-6)
    np.testing.assert_allclose(locs_y.mean(), -2517193.81856, atol=1e-6)

    time = test_data.time
    assert time == [np.datetime64("2019-08-15")]

    spatial_ref = test_data.spatial_ref
    assert pyproj.CRS.from_wkt(spatial_ref.crs_wkt) == pyproj.CRS.from_epsg(3013)


def _summarize(obj: dict) -> dict:
    if isinstance(obj, np.ndarray):
        return {
            "shape": list(obj.shape),
            "min": np.nanmin(obj),
            "max": np.nanmax(obj),
            "mean": np.nanmean(obj),
            "nans": int(np.count_nonzero(~np.isfinite(obj))),
        }
    if isinstance(obj, dict):
        return {k: _summarize(v) for k, v in obj.items()}
    return obj


def _assert_summaries_match(
    expected: dict,
    actual: dict,
    atol: float = 0,
    rtol: float = 1e-7,
    path: str = "",
) -> None:
    for key in set(expected.keys()).union(set(actual.keys())):
        new_path = f"{path}/{key}"
        if key not in expected or key not in actual:
            raise AssertionError(f"Key {new_path} missing")
        if isinstance(expected[key], dict) and isinstance(actual[key], dict):
            _assert_summaries_match(expected[key], actual[key], atol, rtol, new_path)
            continue
        if isinstance(expected[key], float):
            if abs(expected[key] - actual[key]) > atol + rtol * abs(expected[key]):
                raise AssertionError(f"Float value mismatch for {new_path}: {expected[key]} vs {actual[key]}")
            continue
        if type(expected[key]) is not type(actual[key]):
            raise AssertionError(f"Type mismatch for {new_path}: {type(expected[key])} vs {type(actual[key])}")
        if expected[key] != actual[key]:
            raise AssertionError(f"Value mismatch for {new_path}: {expected[key]} vs {actual[key]}")


def test_run_inla_model(grf_module, spde_mesh, log_test_data):
    expected_path = Path("tests/test_data/expected_results/2019_Aug.json")

    output = grf_module.run_inla_model(
        ds=log_test_data,
        spde_mesh=spde_mesh,
        reproducible=True,
    )
    assert isinstance(output, xr.Dataset)

    summary = _summarize(
        {
            "data": output.land_ice_runoff_flux.values,
            "data_mean": output.land_ice_runoff_flux_mean.values.item(),
            "data_uncertainty": output.land_ice_runoff_flux_uncertainty.values,
            "data_uncertainty_mean": output.land_ice_runoff_flux_uncertainty_mean.values.item(),
        }
    )

    # Uncomment to update expected results
    # with expected_path.open("w") as f:
    #     f.write(json.dumps(summary, indent=2))

    with expected_path.open("r") as f:
        expected = json.loads(f.read())

    _assert_summaries_match(summary, expected, rtol=0.2)


def test_a_constant_field_is_refused_rather_than_crashing(grf_module, spde_mesh, log_test_data):
    """A flat field has no covariance to estimate, and INLA aborts the process on one.

    Supraglacial lake cover does this every September: once the lakes have
    drained the whole catchment sits at the detection floor. The module has to
    refuse before the call, because afterwards there is no process left.
    """
    flat = log_test_data.copy(deep=True)
    for name in flat.data_vars:
        if flat[name].dims == ("time", "point_id"):
            flat[name].values[:] = -4.0

    with pytest.raises(ConstantFieldError, match="no spatial variation"):
        grf_module.run_inla_model(ds=flat, spde_mesh=spde_mesh, reproducible=True)


def test_reproducible_runs_are_identical(grf_module, spde_mesh, log_test_data):
    first = grf_module.run_inla_model(ds=log_test_data, spde_mesh=spde_mesh, reproducible=True)
    second = grf_module.run_inla_model(ds=log_test_data, spde_mesh=spde_mesh, reproducible=True)

    for name in ("land_ice_runoff_flux", "land_ice_runoff_flux_uncertainty"):
        np.testing.assert_array_equal(first[name].values, second[name].values)


def test_a_constant_timestep_is_filled_in_rather_than_fitted(log_test_data, spde_mesh):
    """
    The fitted field of an observation that is constant everywhere is that constant.

    The uncertainty is NaN rather than zero: no posterior was sampled, so the spread is
    unknown, and zero would claim the value is known exactly.
    """
    flat = log_test_data.copy(deep=True)
    names = [n for n in flat.data_vars if flat[n].dims == ("time", "point_id")]
    for name in names:
        flat[name].values[:] = -4.0

    filled = GrfModule.constant_field(flat, spde_mesh=spde_mesh)

    n_vertices = len(spde_mesh.vertices)
    for name in names:
        assert filled[name].dims == ("time", "vertex_id")
        assert filled.sizes["vertex_id"] == n_vertices
        np.testing.assert_allclose(filled[name].values, -4.0)
        np.testing.assert_allclose(filled[f"{name}_mean"].values, -4.0)
        assert np.isnan(filled[f"{name}_uncertainty"].values).all()
        assert np.isnan(filled[f"{name}_uncertainty_mean"].values).all()


def test_a_varying_timestep_is_not_filled_in(log_test_data, spde_mesh):
    """Filling in a field that has something to fit would discard a result."""
    with pytest.raises(ValueError, match="has a field to fit"):
        GrfModule.constant_field(log_test_data, spde_mesh=spde_mesh)


def test_a_timestep_with_no_values_at_all_has_no_constant_to_use(log_test_data, spde_mesh):
    empty = log_test_data.copy(deep=True)
    for name in empty.data_vars:
        if empty[name].dims == ("time", "point_id"):
            empty[name].values[:] = np.nan

    with pytest.raises(ValueError, match="no values at this timestep"):
        GrfModule.constant_field(empty, spde_mesh=spde_mesh)
