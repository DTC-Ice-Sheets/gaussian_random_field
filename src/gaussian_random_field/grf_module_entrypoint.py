"""Entrypoint for GRF module."""

import logging
from pathlib import Path

import cyclopts
import xarray as xr
import zarr  # ruff:ignore[unused-import]

from gaussian_random_field.grf_module.geo_zarr_utils import (
    save_geo_zarr,
    sparse_grid_to_point_cloud,
    trim_to_mesh,
)
from gaussian_random_field.grf_module.grf_module import ConstantFieldError, GrfModule
from gaussian_random_field.grf_module.gridding import grid_mesh_data
from gaussian_random_field.grf_module.spde_mesh import SpdeMesh
from gaussian_random_field.helpers import ZarrType
from gaussian_random_field.monitoring import setup_logging

logger = logging.getLogger(__name__)

app = cyclopts.App()


@app.default
def main(
    mesh_name: str,
    data_path: ZarrType,
    output_path: Path,
    prior_sigma_1: float = 1.0,
    prior_sigma_2: float = 0.01,
    grid_res_x: int = 100,
    grid_res_y: int = 100,
) -> None:
    """Run the Gaussian Random Field model.

    Parameters
    ----------
    mesh_name : str
        The name of the mesh to load and use.
    data_path : ZarrType
        Path of the input GeoZarr file.
    output_path : Path
        Path of the output GeoZarr file.
    prior_sigma_1 : float, optional
        Parameter characterizing the expected error in the data.
    prior_sigma_2 : float, optional
        Probability that error exceeds `prior_sigma_1`.
    grid_res_x : int, optional
        Number of grid points in the x direction.
    grid_res_y : int, optional
        Number of grid points in the y direction.

    Raises
    ------
    ConstantFieldError
        If no variable in any timestep has any spatial variation. A variable
        with none is filled with the constant it already holds, and any other
        variable at that timestep is still fitted; this is raised only when
        nothing in the record was fitted at all.
    """
    setup_logging()

    # Load data and setup the SPDE mesh
    grf_module = GrfModule()
    spde_mesh = SpdeMesh(
        mesh_name=mesh_name,
        prior_sigma_1=prior_sigma_1,
        prior_sigma_2=prior_sigma_2,
    )
    ds = xr.load_dataset(data_path, engine="zarr", chunks={})

    # A published dataset holds a whole record rather than a single snapshot,
    # and the fit is independent per timestep, so run them one at a time and
    # join the results. Doing it here rather than inside the model keeps both
    # of those functions single-timestep, and keeps the set of points that are
    # present decided per timestep rather than pooled across the record.
    fitted = []
    filled = 0
    fitted_anything = False
    for step in range(ds.sizes["time"]):
        snapshot = sparse_grid_to_point_cloud(ds.isel(time=[step]))
        on_mesh = trim_to_mesh(snapshot, spde_mesh)
        if step == 0:
            logger.info("%d of %d points fall on the mesh", on_mesh.sizes["point_id"], snapshot.sizes["point_id"])
        try:
            fitted.append(grf_module.run_inla_model(on_mesh, spde_mesh=spde_mesh))
            fitted_anything = True
        except ConstantFieldError as error:
            # A flat variable has no field to estimate, but the fitted field of
            # a constant observation is that constant, so write it rather than
            # dropping the timestep: a gap in the record leaves the melt
            # seasons downstream unequal in length, and the functional time
            # series module needs them whole.
            #
            # They need not be flat together - lake cover can sit at the
            # detection floor in a month when runoff still varies - and
            # run_inla_model refuses the whole timestep as soon as one of them
            # is. So the variables are split here: the flat ones are filled,
            # the rest are fitted as usual, and the two are joined.
            flat = grf_module.flat_variables(on_mesh)
            varying = [
                str(v) for v in on_mesh.data_vars if on_mesh[v].dims == ("time", "point_id") and str(v) not in flat
            ]
            logger.warning("timestep %d: filling %s with its constant: %s", step, ", ".join(flat), error)
            parts = [grf_module.constant_field(on_mesh.drop_vars(varying), spde_mesh=spde_mesh)]
            if varying:
                logger.info("timestep %d: fitting %s alongside it", step, ", ".join(varying))
                parts.append(grf_module.run_inla_model(on_mesh.drop_vars(flat), spde_mesh=spde_mesh))
                fitted_anything = True
            filled += 1
            fitted.append(xr.merge(parts))
    if not fitted_anything:
        raise ConstantFieldError(
            f"no variable in any of the {ds.sizes['time']} timesteps has any spatial "
            "variation, so nothing in the record was fitted",
        )
    if filled:
        logger.warning(
            "filled %d of %d timesteps that had a variable with no spatial variation", filled, ds.sizes["time"]
        )
    logger.info("fitted %d timestep(s) outright, %d with a variable filled as a constant", len(fitted) - filled, filled)
    ds = (
        fitted[0]
        if len(fitted) == 1
        else xr.concat(fitted, dim="time", data_vars="minimal", coords="minimal", compat="override")
    )

    # Interpolate the results from the mesh onto a grid
    ds = grid_mesh_data(
        ds,
        mesh=spde_mesh,
        grid_res_x=grid_res_x,
        grid_res_y=grid_res_y,
    )

    # Save the gridded data to a Zarr file
    save_geo_zarr(ds, zarr_path=output_path)


if __name__ == "__main__":
    app()
