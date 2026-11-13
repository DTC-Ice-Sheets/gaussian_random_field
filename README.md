![Tests](https://github.com/dtc-ice-sheets/gaussian_random_field/actions/workflows/gaussian_random_field_test.yml/badge.svg)
![Deployment](https://github.com/dtc-ice-sheets/gaussian_random_field/actions/workflows/gaussian_random_field_deploy.yml/badge.svg)
[![License: GPLv3](https://img.shields.io/badge/license-GPLv3-blue.svg)](https://github.com/DTC-Ice-Sheets/gaussian_random_field/blob/main/LICENSE)

# DTC-IS _Gaussian Random Field_

The `gaussian_random_field` module is a sub-component of the
[Digital Twin Component for Ice Sheets (DTC-IS)](https://dtc-ice-sheets.org/), an ESA-funded, EO-driven digital twin project.

The `gaussian_random_field` module fits a Gaussian random field to spatially distributed ice-sheet data, bringing
variables of different resolution and sampling onto a shared grid and carrying an estimate of uncertainty with them.
It was developed to provide hydropower insights for the Watson River catchment on the Greenland ice sheet, where it
harmonises climate-model surface runoff and a satellite record of supraglacial lake cover onto one mesh.

For an in-depth overview of the gaussian_random_field module please [read the docs](https://dtc-ice-sheets.github.io/gaussian_random_field/).

The documentation includes sections on:

1. [Getting started](https://dtc-ice-sheets.github.io/gaussian_random_field/quick-start.html)
1. [Scientific overview](https://dtc-ice-sheets.github.io/gaussian_random_field/overview.html)
1. [API reference](https://dtc-ice-sheets.github.io/gaussian_random_field/api.html)

For a more interactive way of exploring the system, please visit the digital twin's [dashboard](https://dashboards.dtc-ice-sheets.org/)
and/or explore the project [notebook repository](https://github.com/DTC-Ice-Sheets/dtc_is_notebooks).

## Installation

This package has a dependency on _[R](https://www.r-project.org/)_ and requires additional R packages and python 
bindings to be installed. 
The installation is therefore relatively complex and dependent on your OS/version & python installation.
We provide 2 routes towards installation - installing with [conda](#conda-install) and installing via a 
[dev container](#dev-container-install-quickerrecommended).

### Conda install

An isolated R installation can be created via conda. Assuming you have conda installed already, a bash install script
can be used to setup the environment. From the repository root, the following command can be run:

#### Prerequisites
- conda is installed
- conda is active in your shell
- you are in your base conda environment (use `conda deactivate` to exit from a active environment)

#### Install commands

``` bash
bash create-conda-env-with-R.sh
```
> [!WARNING]
> The INLA package is built from source which is likely to take a long time. 

The above bash script will create a conda environment for you, `dtc_gaussian_random_field`. 
Once your conda env is created, you can install the package with pip:

``` bash
conda activate dtc_gaussian_random_field # activate the conda environment
pip install gaussian_random_field
```

#### Troubleshooting

If the `pip install` process raises dependency errors - it is very likely that you have additional packages installed
in your base conda environment. 
Note that `dtc_gaussian_random_field` inherits packages that are installed in the base conda environment.
Use the error messages to guide you which packages to remove before retrying the install process.

### Dev container install (quicker/recommended)

For users of VScode, the repository dev container may be a more convenient way to install the package.

#### Pre-requisites

1. Install and setup [Docker](https://docs.docker.com/engine/install/).
1. clone the repository:

```bash
git clone https://github.com/dtc-ice-sheets/gaussian_random_field.git
cd gaussian_random_field
```

#### Setup the dev container

1. Open the cloned repository folder in VSCode.
1. Click the blue box at the very bottom left of the VSCode window, and then click "Dev Containers: Reopen in Container". \
 For more guidance on this, see the [dev containers documentation](https://code.visualstudio.com/docs/devcontainers/containers).
1. Wait for the Docker setup to finish (might take a few minutes).
1. There should be a popup in the bottom right asking you to install recommended extensions, click "Yes". Otherwise go to the "Extensions" tab and install all the extensions that are marked as recommended.
1. Click on the "Testing" tab icon on the left of the VSCode window. From there, click "Refresh Tests", and then click "Run Tests". After a short while you should see that all test have passed with a green check.

And that's it!

Note that the list of VSCode projects should now show two copies of this project: one labelled as "dev container", and one not.
You should ignore the normal project and make sure to always do all development work inside the dev container project.
If you are in the correct project the blue box in the bottom left should start with "Dev Container: ...".
If it doesn't, repeat Step 2 to switch to the correct project.
