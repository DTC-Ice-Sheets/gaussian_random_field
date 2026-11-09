"""Gridding functions for the GRF module."""

import numpy as np
import shapely
import xarray as xr

from gaussian_random_field.grf_module.spde_mesh import SpdeMesh


def grid_mesh_data(
    ds: xr.Dataset,
    mesh: SpdeMesh,
    grid_res_x: int,
    grid_res_y: int,
) -> xr.Dataset:
    """Interpolate mesh data onto a grid defined by xs and ys.

    Parameters
    ----------
    ds: xr.Dataset
        The xarray dataset containing the mesh data to be gridded.
        This must have the following dimensions:
            - time
            - vertex_id (corresponding to the mesh vertices)
        The dataset must also contain any number of spatial variables, with
        dimensions ("time", "vertex_id").
    mesh: SpdeMesh
        The SPDE mesh containing the vertices and triangles of the original data.
    grid_res_x: int
        Number of grid points in the x-direction.
    grid_res_y: int
        Number of grid points in the y-direction.

    Returns
    -------
    xr.Dataset
        A dataset containing the gridded data, with dimensions:
            - time
            - y
            - x
        Each of the spatial variables in the input dataset will be interpolated
        onto the new spatial coordinates, with the same name and attributes. Any
        other variables will be copied to the output dataset as-is.
    """
    minx = mesh.vertices[:, 0].min()
    maxx = mesh.vertices[:, 0].max()
    miny = mesh.vertices[:, 1].min()
    maxy = mesh.vertices[:, 1].max()
    xs = np.linspace(minx, maxx, grid_res_x)
    ys = np.linspace(miny, maxy, grid_res_y)

    grid_x, grid_y = np.meshgrid(xs, ys)
    numt = len(ds.time)

    spatial_vars = [v for v in ds.data_vars if ds[v].dims == ("time", "vertex_id")]
    other_vars = [v for v in ds.data_vars if v not in spatial_vars]

    # Find the containing triangle for every grid point at once. A linear scan
    # of the triangles per point is the obvious way to do this, but at a 100 x
    # 100 grid and ~550 triangles it is around five million containment tests,
    # and the cost is paid again for every timestep in the record.
    points = shapely.points(grid_x.ravel(), grid_y.ravel())
    tree = shapely.STRtree(mesh.triangles)
    point_index, triangle_index = tree.query(points, predicate="intersects")

    # A point on a shared edge hits more than one triangle; either will do, so
    # keep the first. A point outside the mesh hits none and is left as NaN.
    _, first = np.unique(point_index, return_index=True)
    point_index, triangle_index = point_index[first], triangle_index[first]

    # Barycentric weights, vectorised over the grid points.
    # reshape keeps the three columns when nothing is covered, so a grid with no
    # point on the mesh comes back all-NaN rather than raising.
    corners = np.asarray([mesh.triangle_verts[i] for i in triangle_index], dtype=int).reshape(-1, 3)
    a, b, c = (mesh.vertices[corners[:, i]] for i in range(3))
    px = np.column_stack([grid_x.ravel()[point_index], grid_y.ravel()[point_index]])
    cross = lambda u, v: u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]  # ruff:ignore[lambda-assignment]
    total = cross(b - a, c - a)
    weights = np.column_stack(
        [cross(b - px, c - px) / total, cross(px - a, c - a) / total, cross(b - a, px - a) / total]
    )

    rows, columns = np.unravel_index(point_index, grid_x.shape)
    out = {}
    for var in spatial_vars:
        # .values once, so the time axis is a single array operation rather
        # than one xarray lookup per timestep per grid cell.
        values = np.asarray(ds[var].values, dtype=float)
        gridded = np.full((numt, grid_res_x, grid_res_y), np.nan)
        interpolated = sum(weights[:, i] * values[:, corners[:, i]] for i in range(3))
        # columns is the x index and rows the y index, matching the (x, y)
        # dimension order the dataset is built with below.
        gridded[:, columns, rows] = interpolated
        out[var] = gridded

    # Collate the results into a new dataset
    out = xr.Dataset(
        data_vars={var: xr.DataArray(out[var], dims=("time", "x", "y"), attrs=ds[var].attrs) for var in spatial_vars}
        | {var: ds[var] for var in other_vars},
        coords={
            "time": ds["time"],
            "x": xs,
            "y": ys,
        },
    )

    return out
