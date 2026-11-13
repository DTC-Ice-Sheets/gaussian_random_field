"""Gaussian Random Field (GRF) module."""

from pathlib import Path

import numpy as np
import rpy2.robjects as ro
import xarray as xr

from gaussian_random_field.grf_module.spde_mesh import SpdeMesh


class ConstantFieldError(ValueError):
    """A field with no spatial variation, which the SPDE model cannot be fitted to."""


class GrfModule:
    """GRF module for running INLA models on spatial data."""

    est_gp_inla_path = Path(__file__).resolve().parent / "est.GP.INLA.R"

    def __init__(self) -> None:
        ro.r.source(str(self.est_gp_inla_path))  # type: ignore
        self.est_gp_inla = ro.r["est.GP.INLA"]  # type: ignore

    @staticmethod
    def flat_variables(ds: xr.Dataset) -> list[str]:
        """
        Names of the spatial variables that take one value everywhere at this timestep.

        A dataset can hold several spatial variables, and they need not be flat
        together: supraglacial lake cover can sit at the detection floor across the
        catchment in a month when runoff still varies. :meth:`run_inla_model` refuses
        the whole timestep as soon as one of them is flat, so a caller that wants to
        fill those in needs to know which ones they are.

        A variable with no values at all is not reported here. There is no constant to
        fill it with, and :meth:`run_inla_model` is left to raise on it as before.

        Parameters
        ----------
        ds : xr.Dataset
            A single-timestep point cloud, as :meth:`run_inla_model` takes.

        Returns
        -------
        list[str]
            The names of the spatial variables with no spatial variation, in the order
            they appear in the dataset.
        """
        flat = []
        for var in ds.data_vars:
            if ds[var].dims != ("time", "point_id"):
                continue
            observed = np.asarray(ds[var].squeeze().values, dtype=float)
            present = observed[np.isfinite(observed)]
            if present.size and np.ptp(present) == 0:
                flat.append(str(var))
        return flat

    @staticmethod
    def constant_field(ds: xr.Dataset, spde_mesh: SpdeMesh) -> xr.Dataset:
        """
        Build the output for a timestep with no spatial variation, without fitting it.

        A field that takes one value everywhere has no spatial covariance to estimate, so
        :meth:`run_inla_model` refuses it. The fitted field of such an observation is the
        constant itself, though, so a caller that needs a complete record can write that
        rather than dropping the timestep - supraglacial lake cover in September sits at
        the detection floor across the catchment, and dropping those months leaves the
        melt seasons downstream unequal in length.

        The uncertainty is NaN rather than zero. No posterior was sampled, so the spread
        of the fitted value is not known here; claiming zero would assert that a constant
        observation is known exactly.

        Parameters
        ----------
        ds : xr.Dataset
            A single-timestep point cloud, as :meth:`run_inla_model` takes.
        spde_mesh : SpdeMesh
            The mesh the record is being fitted onto, which sets the output vertices.

        Returns
        -------
        xr.Dataset
            The same variables and dimensions :meth:`run_inla_model` returns, with the
            constant at every vertex and NaN for both uncertainties.

        Raises
        ------
        ValueError
            If the dataset does not have exactly one time step, or if a variable has no
            values at all or does vary across its points. A varying field has something
            to fit, so filling it in as a constant would discard a result.
        """
        if len(ds.time) != 1:
            raise ValueError("Only single-timestep datasets are supported")

        spatial_vars = [v for v in ds.data_vars if ds[v].dims == ("time", "point_id")]
        output_vars = {v: ds[v] for v in ds.data_vars if v not in spatial_vars}
        n_vertices = len(spde_mesh.vertices)

        for var in spatial_vars:
            observed = np.asarray(ds[var].squeeze().values, dtype=float)
            present = observed[np.isfinite(observed)]
            if present.size == 0:
                raise ValueError(f"{var!r} has no values at this timestep, so there is no constant to use")
            if np.ptp(present) != 0:
                raise ValueError(
                    f"{var!r} varies across its points, so it has a field to fit and must "
                    "not be filled in as a constant",
                )
            constant = float(present[0])
            output_vars |= {
                f"{var}": xr.DataArray(
                    np.full((1, n_vertices), constant),
                    dims=("time", "vertex_id"),
                    attrs=ds[var].attrs,
                ),
                f"{var}_mean": xr.DataArray(
                    np.array([constant]),
                    dims=("time"),
                    attrs=ds[var].attrs,
                ),
                f"{var}_uncertainty": xr.DataArray(
                    np.full((1, n_vertices), np.nan),
                    dims=("time", "vertex_id"),
                    attrs=ds[var].attrs,
                ),
                f"{var}_uncertainty_mean": xr.DataArray(
                    np.array([np.nan]),
                    dims=("time"),
                    attrs=ds[var].attrs,
                ),
            }

        return xr.Dataset(
            data_vars=output_vars,
            coords={
                "time": ds["time"],
                "x": ("vertex_id", spde_mesh.vertices[:, 0]),
                "y": ("vertex_id", spde_mesh.vertices[:, 1]),
            },
        )

    @staticmethod
    def _uncertainty(
        model: ro.ListVector,
        field_random: ro.ListVector,
        field_fixed: ro.ListVector,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Posterior sd of (intercept + field), per node and for the spatial mean.

        Prefers the joint-posterior figures from est.GP.INLA. Falls back to
        combining the marginal sds in quadrature when an older estimator is in
        use - which still ignores the correlation between the two terms and so
        overstates, but is closer than summing them.

        Parameters
        ----------
        model : ro.ListVector
            The fitted model as returned by est.GP.INLA.
        field_random : ro.ListVector
            The summary.random$field component of the fit, used only by the
            fallback path.
        field_fixed : ro.ListVector
            The summary.fixed component of the fit, used only by the fallback
            path.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Posterior sd of the fitted value at each mesh node, and the
            posterior sd of its spatial mean as a length-one array.
        """
        joint = model.rx2("Sd.fields")
        if joint is not ro.NULL and len(joint):
            per_node = np.array(joint)
            mean_sd = model.rx2("Sd.mean")
            spatial = float(np.array(mean_sd)[0]) if mean_sd is not ro.NULL else float(np.sqrt(np.mean(per_node**2)))
            return per_node, np.array([spatial])

        random_sd = np.array(field_random.rx2("sd"))
        fixed_sd = float(field_fixed.rx2("sd")[0])
        return (np.sqrt(random_sd**2 + fixed_sd**2), np.array([fixed_sd]))

    def run_inla_model(
        self,
        ds: xr.Dataset,
        spde_mesh: SpdeMesh,
        reproducible: bool = False,
    ) -> xr.Dataset:
        """
        Run the INLA model on the specified mesh and point data.

        Parameters
        ----------
        ds: xr.Dataset
            The xarray dataset containing the point data to be modeled.
            This must have the following dimensions:

            - time (must be of length 1)
            - point_id (flattened spatial coordinates)

            The dataset must also contain the following variables:

            - x
            - y
            - Any number of spatial variables, with dimensions ("time", "point_id")

        spde_mesh: SpdeMesh
            The SPDE mesh object containing the mesh and SPDE model
        reproducible: bool, optional
            If True, run INLA single-threaded in classic mode and seed the
            posterior sampling, so that repeated runs give identical results,
            e.g. for tests. Slower, and classic mode differs slightly from the
            default mode. Default False.

        Returns
        -------
        xr.Dataset
            A dataset containing the model results, with dimensions:

            - time (length 1)
            - vertex_id (length equal to the number of vertices in the mesh)

            The dataset will contain the following variables for each input
            spatial variable:

            - {name}: Output values of dimensions ("time", "vertex_id")
            - {name}_mean: Output mean of dimensions ("time")
            - {name}_uncertainty: posterior sd of the fitted value at each
              vertex, dimensions ("time", "vertex_id")
            - {name}_uncertainty_mean: posterior sd of the spatial mean of
              the fitted field, dimensions ("time")

        Raises
        ------
        ValueError
            If the dataset does not have exactly one time step.
        ConstantFieldError
            If a variable takes the same value at every point. There is no
            spatial covariance to estimate, so the precision matrix is singular
            and INLA aborts the process rather than returning an error.
        """
        cbind = ro.r["cbind"]

        if len(ds.time) != 1:
            raise ValueError("Only single-timestep datasets are supported")

        spatial_vars = [v for v in ds.data_vars if ds[v].dims == ("time", "point_id")]
        other_vars = [v for v in ds.data_vars if v not in spatial_vars]

        # Non-spatial variables are copied to the output directly as-is
        output_vars = {v: ds[v] for v in other_vars}

        # Get point coordinates as an R matrix
        locations = cbind(  # type: ignore
            ro.FloatVector(ds["x"].values),
            ro.FloatVector(ds["y"].values),
        )

        # Run the model on each variable
        for var in spatial_vars:
            observed = np.asarray(ds[var].squeeze().values, dtype=float)
            # A field with no variation has nothing for the SPDE model to fit:
            # the precision matrix is singular and INLA aborts the whole
            # process, so this has to be caught before the call rather than
            # after it. It happens in practice - supraglacial lake cover in
            # September sits at the detection floor across the catchment.
            if np.ptp(observed[np.isfinite(observed)]) == 0:
                raise ConstantFieldError(
                    f"{var!r} takes the same value ({observed.flat[0]:g}) at every "
                    f"one of its {observed.size} points, so there is no spatial "
                    "variation to fit. Drop this timestep, or use the constant "
                    "directly - a fit would add nothing to it.",
                )
            values = ro.FloatVector(observed)
            model = self.est_gp_inla(  # type: ignore
                data=values,
                spde=spde_mesh.spde,
                locations=locations,
                seed=1 if reproducible else 0,
            )
            field_random = model.rx2("all.est").rx2("summary.random").rx2("field")
            field_fixed = model.rx2("all.est").rx2("summary.fixed")
            data = np.array(field_random.rx2("mean")) + field_fixed.rx2("mean")[0]
            data_mean = np.array([field_fixed.rx2("mean")[0]])

            # A fitted node is intercept + field. Posterior *means* add, which
            # is what the line above does. Standard deviations do not: the two
            # terms are strongly and negatively correlated, because the field
            # absorbs the overall level, so neither summing them nor combining
            # them in quadrature is right. est.GP.INLA now returns the sd of
            # the sum, taken from joint posterior draws, which is the quantity
            # this variable is supposed to hold.
            uncertainty, uncertainty_mean = self._uncertainty(model, field_random, field_fixed)

            output_vars |= {
                f"{var}": xr.DataArray(
                    data[np.newaxis, :],
                    dims=("time", "vertex_id"),
                    attrs=ds[var].attrs,
                ),
                f"{var}_mean": xr.DataArray(
                    data_mean,
                    dims=("time"),
                    attrs=ds[var].attrs,
                ),
                f"{var}_uncertainty": xr.DataArray(
                    uncertainty[np.newaxis, :],
                    dims=("time", "vertex_id"),
                    attrs=ds[var].attrs,
                ),
                f"{var}_uncertainty_mean": xr.DataArray(
                    uncertainty_mean,
                    dims=("time"),
                    attrs=ds[var].attrs,
                ),
            }

        # Collate the results into a new dataset
        return xr.Dataset(
            data_vars=output_vars,
            coords={
                "time": ds["time"],
                "x": ("vertex_id", spde_mesh.vertices[:, 0]),
                "y": ("vertex_id", spde_mesh.vertices[:, 1]),
            },
        )
