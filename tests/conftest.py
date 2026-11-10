"""Provides pytest fixtures for general use across all test modules in this package."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def test_dataset_1_path() -> Path:
    return Path("tests/test_data/test_inputs/2019_Aug.zarr")


@pytest.fixture
def test_dataset_1_loaded_zarr(test_dataset_1_path: Path) -> xr.Dataset:
    return xr.load_dataset(test_dataset_1_path, engine="zarr", chunks={})


@pytest.fixture
def tmp_zarr_output_path(tmp_path: Path) -> Path:
    return tmp_path / "output.zarr"


@pytest.fixture
def invalid_file_path(tmp_path: Path) -> Path:
    return tmp_path / "non_existent_file.zarr"


@pytest.fixture
def invalid_url() -> str:
    return "invalid_url/doesnt/start/with/http/or/https"


@pytest.fixture
def invalid_path() -> str:
    return "invalid_path/doesnt/exist"


def assert_equal_up_to_column_sign(actual: np.ndarray, expected: np.ndarray, atol: float = 1e-12) -> None:
    """
    Assert that two arrays agree once each column of `actual` is given the sign that best matches `expected`.

    Parameters
    ----------
    actual : np.ndarray
        The result under test.
    expected : np.ndarray
        The reference result, whose column signs are arbitrary.
    atol : float, optional
        Largest absolute difference allowed.
    """
    signs = np.sign((actual * expected).sum(axis=0))
    np.testing.assert_allclose(actual * signs, expected, rtol=0, atol=atol)
