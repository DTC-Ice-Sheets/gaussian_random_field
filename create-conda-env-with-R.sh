#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="dtc_gaussian_random_field"
PYTHON_VERSION="3.13"

echo "=== Setting up environment: ${ENV_NAME} ==="
echo "=== Checking if conda environment already exists ==="
if conda info --envs | grep -q "${ENV_NAME}"; then
    echo "Environment ${ENV_NAME} already exists. Skipping creation."
    echo "If you want to recreate it, please remove it first using:"
    echo "  conda remove -n ${ENV_NAME} --all"
    exit 0
else
    echo "Environment ${ENV_NAME} does not exist. Proceeding with creation."
fi

echo "=== Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION}) ==="
conda create -y -n "${ENV_NAME}" -c conda-forge --strict-channel-priority \
    python="${PYTHON_VERSION}" \
    cffi \
    compilers \
    r-base \
    r-fmesher

echo "=== Installing INLA via Rscript - may take a while! ==="
conda run -n "${ENV_NAME}" Rscript -e '
install.packages(
  "INLA",
  repos = c(
    INLA = "https://inla.r-inla-download.org/R/stable",
    CRAN = "https://cloud.r-project.org"
  ),
  dep = TRUE
)
'

echo "=== Verifying installation ==="
conda run -n "${ENV_NAME}" python -c "import sys, cffi; print(f'Python: {sys.version.split()[0]} | CFFI: {cffi.__version__}')"

conda run -n "${ENV_NAME}" Rscript -e '
suppressPackageStartupMessages({
  library(fmesher)
  library(INLA)
})
cat("R, fmesher, and INLA loaded successfully:\n")
print(inla.version())
'

echo "=== Setup complete! ==="

conda remove -n "${ENV_NAME}" ansible-core ansible-lint ansible-compat 
conda remove -n "${ENV_NAME}" keyrings-google-artifactregistry-auth
