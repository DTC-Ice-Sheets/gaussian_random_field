"""Utility functions for working with GeoZarr data within the GRF module."""

import logging
from pathlib import Path

import numpy as np
import rioxarray  # ruff:ignore[unused-import]
import xarray as xr

from gaussian_random_field.grf_module.spde_mesh import SpdeMesh

logger = logging.getLogger(__name__)


def sparse_grid_to_point_cloud(ds: xr.Dataset) -> xr.Dataset:
    """Convert a sparse grid dataset to a point cloud format.

    This function removes NaN cells in the grid, and returns the resulting cells
    as a flattened point cloud.

    Data already stored as a point cloud is accepted too. Some sources are not
    on a regular grid in the target CRS - RACMO's native run is on a rotated
    pole, so its cells do not fall on x/y axes once reprojected - and those
    datasets carry a single dimension of positions instead. They are passed
    through with the dimension renamed, rather than being gridded first.

    Parameters
    ----------
    ds: xr.Dataset
        The input GeoZarr dataset, in one of two forms.

        A sparse grid, with at least the following dimensions:
            - time (must be of length 1)
            - x (or old name: 'projection_x_coordinate')
            - y (or old name: 'projection_y_coordinate')

        Or a point cloud, with:
            - time (must be of length 1)
            - point (or 'point_id'), with x and y giving each position

    Returns
    -------
    xr.Dataset
        A dataset where the spatial X and Y dimensions have been stacked into a
        single 'point_id' dimension, and points where all variables are NaN
        across all timesteps have been removed.

    Raises
    ------
    ValueError
        If the dataset does not have exactly one time step, or has neither a
        pair of spatial dimensions nor a point dimension carrying x and y.
    """
    if len(ds.time) != 1:
        # This approach wouldn't work well with multiple timesteps
        raise ValueError("Only single-timestep datasets are supported")

    if "projection_x_coordinate" in ds.dims:
        # Old coordinate names for compatibility
        ds = ds.rename({"projection_x_coordinate": "x", "projection_y_coordinate": "y"})

    # Flatten the grid cells into a point cloud, unless the data is one already
    gridded = "x" in ds.dims and "y" in ds.dims
    ds = ds.stack({"point_id": ["x", "y"]}) if gridded else _point_cloud_as_is(ds)
    # Remove points where all variables are NaN in all timesteps
    variables = [v for v in ds.data_vars if ds[v].dims == ("time", "point_id")]
    mask = np.any([np.isfinite(ds[v]) for v in variables], axis=(0, 1))
    ds = ds.isel(point_id=mask)
    return ds


def _point_cloud_as_is(ds: xr.Dataset) -> xr.Dataset:
    """Put a dataset that is already a point cloud into the expected form.

    Parameters
    ----------
    ds: xr.Dataset
        A dataset with a point dimension and x and y variables giving the
        position of each point.

    Returns
    -------
    xr.Dataset
        The same data with the point dimension named 'point_id', x and y as
        coordinates, and spatial variables ordered ("time", "point_id").

    Raises
    ------
    ValueError
        If there is no point dimension, or x and y are not present.
    """
    point_dim = next((d for d in ("point_id", "point") if d in ds.dims), None)
    if point_dim is None:
        raise ValueError(
            f"expected either x and y dimensions or a point dimension, but found {list(ds.dims)}",
        )
    if "x" not in ds.variables or "y" not in ds.variables:
        raise ValueError("point data must carry x and y giving each position")

    if point_dim != "point_id":
        ds = ds.rename({point_dim: "point_id"})
    return ds.set_coords(["x", "y"]).transpose("time", "point_id", ...)


def trim_to_mesh(ds: xr.Dataset, mesh: SpdeMesh) -> xr.Dataset:
    """Drop points that fall outside the mesh.

    The published input datasets cover a buffer around the catchment, while a
    mesh covers the catchment alone, so a dataset normally carries points the
    mesh cannot represent. Those points give an all-zero row in the projector
    matrix and INLA refuses the fit, so they are removed before fitting rather
    than left to fail.

    Parameters
    ----------
    ds: xr.Dataset
        A point cloud, as returned by :func:`sparse_grid_to_point_cloud`.
    mesh: SpdeMesh
        The mesh the data will be fitted onto.

    Returns
    -------
    xr.Dataset
        The same dataset with only the points the mesh covers.

    Raises
    ------
    ValueError
        If no point falls on the mesh at all, which means the data and the
        mesh do not describe the same place.
    """
    covered = mesh.covers(ds["x"].values, ds["y"].values)
    if not covered.any():
        raise ValueError(
            f"none of the {covered.size} points fall on mesh '{mesh.mesh_name}' - "
            "the data and the mesh may not cover the same area, or may be in "
            "different coordinate reference systems",
        )
    if not covered.all():
        # Debug, not info: a whole record is trimmed one timestep at a time, so
        # at info this would repeat once per snapshot. The caller reports it.
        logger.debug("dropping %d of %d points outside the mesh", int((~covered).sum()), covered.size)
    return ds.isel(point_id=covered)


def save_geo_zarr(ds: xr.Dataset, zarr_path: Path | str, target_chunk_size: str = "1024MB") -> None:
    """Save dataset to GeoZarr adhering to CF conventions (https://cfconventions.org/).

    Parameters
    ----------
    ds: xr.Dataset
        The xarray dataset to save. This must follow the GeoZarr format,
        including having a `spatial_ref` attribute with a valid CRS.
    zarr_path: Path | str
        Full path of the output .zarr file to be created.
    target_chunk_size: str, optional
        The target chunk size for the GeoZarr dataset, by default "1024MB".
    """
    ds = ds.rio.write_crs(ds["spatial_ref"].attrs["crs_wkt"], grid_mapping_name="spatial_ref")
    ds = ds.reset_coords("spatial_ref")
    if "time" in ds:
        ds["time"] = prepare_cf_time_data_array(ds["time"])
    # Prepare the spatial variables with the required attributes for GeoZarr.
    for name, axis in (("x", "X"), ("y", "Y")):
        if name in ds:
            ds[name] = prepare_cf_spatial_data_array(ds[name], name=name, axis=axis)
    ds = prepare_cf_data_variable_arrays(ds)

    ds.attrs = {"_ARRAY_DIMENSIONS": ["spatial_ref", *sorted(ds.dims)]}
    ds = ds.chunk(chunks=target_chunk_size)

    ds.to_zarr(
        str(zarr_path),
        mode="w",
        zarr_format=2,
        consolidated=True,
    )


def prepare_cf_data_variable_arrays(ds: xr.Dataset) -> xr.Dataset:
    """Assign CF grid-mapping and fill-value metadata to dataset variables."""
    for name in ds.variables:
        variable = ds[name]
        # Do not reuse source packing, time units, compression, or chunk encodings.
        variable.encoding = {"coordinates": None}
        if name != "spatial_ref":
            variable.attrs["grid_mapping"] = "spatial_ref"

        dtype = variable.dtype
        fill_value = None
        if variable.ndim:
            if np.issubdtype(dtype, np.unsignedinteger):
                fill_value = np.iinfo(dtype).max
            else:
                fill_value = {"f": -9999.0, "i": -9999, "b": 0, "U": ""}.get(dtype.kind)
        # We want to preserve NaNs so set the _FillValue in .attrs, rather than .encoding.
        variable.attrs["_FillValue"] = fill_value
    return ds


def prepare_cf_spatial_data_array(
    spatial_data_array: xr.DataArray,
    name: str,
    axis: str,
) -> xr.DataArray:
    """Assign the required GeoZarr attributes to a spatial coordinate."""
    return spatial_data_array.assign_attrs(
        variable_name_override=name,
        standard_name=f"projection_{name}_coordinate",
        long_name=f"projection_{name}_coordinate",
        units="m",
        axis=axis,
    )


def prepare_cf_time_data_array(time_data_array: xr.DataArray) -> xr.DataArray:
    """Convert a time coordinate to numeric epoch seconds with legacy CF metadata."""
    logger.info("Converting type %s to seconds since epoch", time_data_array.dtype)
    time_data_array = time_data_array.astype("datetime64[s]").astype("float64")
    return time_data_array.assign_attrs(
        {
            "standard_name": "time",
            "long_name": "time",
            "axis": "T",
            "bounds": "time_bnds",
            "units": "seconds since 1970-01-01T00:00:00",
            "calendar": "standard",
        }
    )
