Overview
========

The Gaussian Random Field (GRF) module brings spatially distributed ice-sheet
data onto a shared support. Ice-sheet datasets rarely agree on a grid: a
regional climate model delivers monthly runoff on a 5 km mesh, an optical
satellite product maps supraglacial lakes at tens of metres, an altimeter
samples along irregular ground tracks. Anything that compares them, or uses
them together as predictors, has to put them on common ground first.

The module fits a Gaussian random field to each snapshot of each variable and
returns the fitted field, its uncertainty, and summaries of both on a regular
grid.

Scientific background
---------------------

Interpolating or regressing onto a shared grid ties the length scale of the
result to the spacing of the input points: dense observations produce
fine-scale variation and sparse ones produce smooth fields, whether or not the
underlying quantity behaves that way. A **Gaussian random field** instead
models the correlation between two locations explicitly as a function of the
distance between them, so the length scale is a property of the fitted field
rather than of the sampling. The fit is Bayesian, so every estimated value
carries a posterior spread that later steps can propagate.

The fit runs through `R-INLA <https://www.r-inla.org/>`_. A **Matérn**
covariance is specified as a stochastic partial differential equation, which
INLA represents in finite-element form over a triangular mesh and solves by
approximate Bayesian inference (Lindgren, Rue and Lindström, 2011,
`doi:10.1111/j.1467-9868.2011.00777.x
<https://doi.org/10.1111/j.1467-9868.2011.00777.x>`_). The model is fitted with
a Gaussian likelihood and penalised-complexity priors on the range and on the
marginal standard deviation of the field.

The **mesh** is not built here. It ships with the module as a stored R object -
one is included, covering the Watson catchment in west Greenland with 295
vertices - and the same mesh is used for every variable and every timestep, so
that anything fitted through this module lands on identical nodes and can be
compared, swapped or combined. Observations lying outside it are dropped before
fitting, since INLA refuses a fit otherwise.

Two properties of the output are worth drawing out. The **spatial mean** is
reported with a spread taken from joint draws of the posterior rather than by
combining per-node spreads: a fitted field is smooth, so neighbouring nodes err
together, and treating them as independent understates the uncertainty on an
average. And a timestep in which a variable takes the same value everywhere has
no spatial pattern to fit; rather than failing, the module can write that
constant across the field, so a record keeps its full length for anything
downstream that expects regular sampling.

Repository structure
--------------------

``grf_module_entrypoint.py``
    Command-line entry point. Takes a mesh name, an input GeoZarr path and an
    output path, fits every timestep in the input, and writes the result.
    Grid resolution defaults to 100 × 100.

``grf_module/grf_module.py``
    The fit itself, and the handling of timesteps with no spatial variation.

``grf_module/spde_mesh.py``
    Loads a stored mesh and exposes its vertices and triangles.

``grf_module/gridding.py``
    Interpolates fitted values from the mesh onto a regular grid, using the
    barycentric weights of the triangle each cell falls in.

Data flow
---------

1. Read the input GeoZarr and take one timestep at a time, flattening gridded
   data to a point cloud and discarding points outside the mesh.
2. Fit the Matérn SPDE model on the mesh through R-INLA, or, where a variable
   has no spatial variation at that timestep, write out its constant.
3. Interpolate the fitted field and its uncertainty onto a regular grid, and
   write all timesteps to a GeoZarr store.

For each input variable the output carries four things: the posterior mean
field, its posterior standard deviation, the posterior mean of the model's
intercept, and the posterior standard deviation of the spatial mean, estimated
from 500 joint draws.

Two limits are worth stating. The hyperparameters are held at their posterior
mode, which is what keeps a run fast enough to repeat over a long record, but
it means the reported spread reflects the field given those values rather than
the uncertainty in them. And the spatial mean is taken over the whole mesh,
which extends beyond any particular catchment outline; restricting it to a
catchment is the caller's job.

Use within DTC-IS
-----------------

The module is deployed in the DTC-IS environment and runs as a job through the
`Query API <https://query.dtc-ice-sheets.org>`_. In Use Case 1 it harmonises
RACMO and MAR surface runoff and a supraglacial lake record onto one mesh of
the Watson catchment, so that a climate model and a satellite product can drive
the same prediction of river discharge.
