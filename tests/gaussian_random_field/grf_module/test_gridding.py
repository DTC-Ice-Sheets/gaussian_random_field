"""Testing the interpolation from mesh vertices onto a grid."""

import numpy as np
import pytest
import xarray as xr

from gaussian_random_field.grf_module.gridding import grid_mesh_data
from gaussian_random_field.grf_module.spde_mesh import SpdeMesh


@pytest.fixture
def spde_mesh() -> SpdeMesh:
    return SpdeMesh(mesh_name="watsonmesh", prior_sigma_1=1.0, prior_sigma_2=0.0001)


def _field(values: np.ndarray) -> xr.Dataset:
    """Build a fitted-field dataset holding one value per mesh vertex."""
    return xr.Dataset(
        data_vars={"land_ice_runoff_flux": (("time", "vertex_id"), values)},
        coords={"time": [np.datetime64("2019-08-15", "ns")] * values.shape[0]},
    )


def test_a_constant_field_grids_to_that_constant(spde_mesh):
    """Barycentric weights sum to one, so a flat field must stay flat."""
    ds = _field(np.full((1, len(spde_mesh.vertices)), 7.0))

    gridded = grid_mesh_data(ds, mesh=spde_mesh, grid_res_x=40, grid_res_y=40)

    inside = np.isfinite(gridded.land_ice_runoff_flux.values)
    assert inside.any()
    np.testing.assert_allclose(gridded.land_ice_runoff_flux.values[inside], 7.0)


def test_a_linear_field_is_reproduced_at_the_grid_points(spde_mesh):
    """Linear interpolation is exact on a linear field, which pins the geometry.

    This is the test that catches x and y being swapped. The expected field is
    built from the dataset's own ``x`` and ``y`` coordinates, so a value placed
    under the wrong label fails here even though it would look plausible. The
    grid is deliberately not square, which a transposed write cannot survive.
    """
    x, y = spde_mesh.vertices[:, 0], spde_mesh.vertices[:, 1]
    ds = _field((3.0 * x - 2.0 * y)[None, :])

    gridded = grid_mesh_data(ds, mesh=spde_mesh, grid_res_x=40, grid_res_y=25)

    assert gridded.land_ice_runoff_flux.dims == ("time", "x", "y")
    assert gridded.sizes["x"] == 40
    assert gridded.sizes["y"] == 25

    values = gridded.land_ice_runoff_flux.values[0]
    inside = np.isfinite(values)
    grid_x, grid_y = np.meshgrid(gridded.x.values, gridded.y.values, indexing="ij")
    expected = 3.0 * grid_x - 2.0 * grid_y
    np.testing.assert_allclose(values[inside], expected[inside], rtol=1e-9)


def test_timesteps_stay_in_order(spde_mesh):
    """Each timestep must be gridded from its own values, not pooled."""
    base = np.ones(len(spde_mesh.vertices))
    ds = _field(np.stack([base, 2 * base, 3 * base]))

    gridded = grid_mesh_data(ds, mesh=spde_mesh, grid_res_x=20, grid_res_y=20)

    values = gridded.land_ice_runoff_flux.values
    inside = np.isfinite(values[0])
    for step, level in enumerate((1.0, 2.0, 3.0)):
        np.testing.assert_allclose(values[step][inside], level)
