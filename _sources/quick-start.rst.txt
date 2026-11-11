Quick Start
===========

The ``gaussian_random_field`` package has one command-line entrypoint,
``gaussian_random_field.grf_module_entrypoint``. It fits a Gaussian Random
Field to each timestep in an input GeoZarr dataset, interpolates the fitted
field onto a regular grid, and writes the result as GeoZarr.

Requirements
------------

Use Python 3.12 or newer, and install the project dependencies with Poetry.
The model also requires R with the ``fmesher`` and ``INLA`` packages installed.
The supplied ``create.sh`` script creates a Conda environment named
``inla_env`` with Python, R, and those R packages:

.. code-block:: bash

   ./create.sh
   conda activate inla_env
   poetry install

Run the command from the repository root. For a list of arguments and their
defaults:

.. code-block:: bash

   poetry run python -m gaussian_random_field.grf_module_entrypoint --help

Run the model
-------------

The required arguments are the mesh name, input data path, and output path.
This example uses the sample input included in the repository and the bundled
``watsonmesh`` mesh:

.. code-block:: bash

   mkdir -p output
   poetry run python -m gaussian_random_field.grf_module_entrypoint \
      --mesh_name watsonmesh \
      --data_path tests/test_data/test_inputs/2019_Aug.zarr \
      --output_path output/2019_Aug_grf.zarr

The input must be a GeoZarr dataset with a time dimension and spatial
coordinates. The model accepts gridded data or point-cloud data, and supports
local paths as well as publicly available URLs which point to GeoZarr datasets. 
The output is a local GeoZarr store.

Options
-------

The following options are optional:

``--prior_sigma_1``
   Parameter characterizing the expected error in the data (default: ``1.0``).
``--prior_sigma_2``
   Probability that error exceeds ``prior_sigma_1`` (default: ``0.01``).
``--grid_res_x``, ``--grid_res_y``
   Number of output grid points in the x and y directions (default: ``100``
   for each).

For example, to write a finer 200-by-200 output grid:

.. code-block:: bash

   poetry run python -m gaussian_random_field.grf_module_entrypoint \
      --mesh_name watsonmesh \
      --data_path tests/test_data/test_inputs/2019_Aug.zarr \
      --output_path output/2019_Aug_grf_fine.zarr \
      --grid_res_x 200 \
      --grid_res_y 200
