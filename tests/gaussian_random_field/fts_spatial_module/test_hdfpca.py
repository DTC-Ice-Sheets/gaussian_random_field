"""Testing the NumPy port of hdftsa's hdfpca."""

import numpy as np
import pytest
from conftest import assert_equal_up_to_column_sign

from gaussian_random_field.fts_spatial_module.hdfpca import (
    column_space_factor_directions,
    factor_scores,
    flat_top_weight,
    hdfpca_factor_scores,
    row_space_factor_directions,
)


def random_data(seed: int, shape: tuple[int, int]) -> np.ndarray:
    """
    Uniform random data, generated the same way as the input the R reference values were computed from.

    Parameters
    ----------
    seed : int
        Seed for the PCG64 generator.
    shape : tuple[int, int]
        Shape of the array, (n_points, n_times).

    Returns
    -------
    np.ndarray
        Array of the given shape.
    """
    return np.random.Generator(np.random.PCG64(seed)).random(shape)


@pytest.fixture
def r_reference_16_curves() -> tuple[np.ndarray, np.ndarray]:
    """
    Input data and hdftsa's result for 40 points over 16 curves of 5 time steps.

    16 curves gives a bandwidth of 4, so lag index 2 lands exactly on the edge of wtflat's flat top, where R's strict
    inequality gives a weight of 0 rather than 1. The result is
    ``hdfpca(curves, order = 2, r = 3)$model$model2[[1]]$coef`` computed in R with hdftsa 1.1.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The (n_points, n_times) input and the (n_curves, 3) result.
    """
    expected = np.array(
        [
            [-0.3255701267644603, 0.3660284505442329, -0.04012935548183474],
            [-0.3504449701391944, 0.2016059302249202, 0.1031620585439197],
            [-0.2148587078394607, 0.4138042305406687, -0.4701462443952478],
            [-0.9650991212530201, 0.07347060920130018, -0.5283412022236287],
            [-1.11320048648473, 0.01261614373235415, -0.6425699753897585],
            [-0.7261947963130877, 1.0613412898066, -0.40215284658518],
            [-1.416288637026566, 0.3214423936808681, -0.1458826970269763],
            [-0.9097860857789174, 0.4073900276333902, 0.0002679068963184761],
            [-0.5742484534078445, -0.9884200941692614, 1.110484713132072],
            [-0.06422453976640327, -1.709005271308412, 1.137605631451044],
            [0.8650972465571194, -1.160105751603324, -0.3717475023502412],
            [1.206731632052069, -1.158381260348085, -0.2931058260303148],
            [1.370674225738106, -0.1591433575371878, -1.411257683500477],
            [1.20415494584707, 0.5259075184058274, 0.7962822249483915],
            [1.228183522165739, 0.7886087552059029, 0.1084104406750277],
            [0.785074352413582, 1.002840385990206, 1.049120357336885],
        ]
    )
    return random_data(4, (40, 80)), expected


def test_matches_r_where_the_kernel_edge_case_applies(r_reference_16_curves):
    """
    The port must reproduce hdftsa at 16 curves, where wtflat's edge case changes the result.

    Real data with a number of curves that does not hit the edge case is checked against R in
    test_integrated_fts_spatial_module.py.
    """
    data, expected = r_reference_16_curves

    result = hdfpca_factor_scores(data, curve_time_points=5, num_pca=2, num_factors=3)

    assert_equal_up_to_column_sign(result, expected)


def test_flat_top_weight_keeps_r_edge_case():
    assert flat_top_weight(0.0) == pytest.approx(1.0)
    assert flat_top_weight(-0.5) == pytest.approx(1.0)
    assert flat_top_weight(0.5) == pytest.approx(0.0)  # R's strict ``c < x``
    assert flat_top_weight(0.75) == pytest.approx(0.5)
    assert flat_top_weight(-0.75) == pytest.approx(0.5)
    assert flat_top_weight(1.0) == pytest.approx(0.0)
    assert flat_top_weight(-1.0) == pytest.approx(0.0)


@pytest.mark.parametrize("shape", [(12, 300), (300, 12)])
def test_both_ways_of_finding_the_factors_agree(shape):
    x = random_data(13, shape)
    centred = x - x.mean(axis=0)

    # With 300 columns the column-space eigendecomposition rounds differently by a few times 1e-12.
    assert_equal_up_to_column_sign(
        centred @ row_space_factor_directions(centred, 3),
        centred @ column_space_factor_directions(centred, 3),
        atol=1e-10,
    )


def test_float32_input_is_analysed_in_double_precision():
    """R converts its input to double, so float32 data must give the result for its exact float64 values."""
    data = random_data(3, (12, 40)).astype(np.float32)

    np.testing.assert_allclose(
        hdfpca_factor_scores(data, 5, 2, 3), hdfpca_factor_scores(data.astype(np.float64), 5, 2, 3), rtol=0, atol=1e-12
    )


def test_factor_scores_have_a_fixed_sign():
    scores = factor_scores(random_data(6, (10, 30)), 3)

    largest = scores[np.abs(scores).argmax(axis=0), np.arange(3)]
    assert (largest > 0).all()


def test_the_order_of_points_does_not_matter():
    data = random_data(7, (30, 50))
    shuffled = data[np.random.Generator(np.random.PCG64(8)).permutation(30)]

    np.testing.assert_allclose(
        hdfpca_factor_scores(shuffled, 5, 2, 3), hdfpca_factor_scores(data, 5, 2, 3), rtol=0, atol=1e-12
    )


def test_num_pca_does_not_change_the_result():
    data = random_data(9, (20, 50))

    np.testing.assert_allclose(
        hdfpca_factor_scores(data, 5, 1, 3), hdfpca_factor_scores(data, 5, 4, 3), rtol=0, atol=1e-12
    )


def test_too_few_curves_are_refused():
    with pytest.raises(ValueError, match="too few"):
        hdfpca_factor_scores(random_data(10, (20, 25)), 5, 2, 3)


def test_too_many_factors_are_refused():
    with pytest.raises(ValueError, match="cannot keep"):
        hdfpca_factor_scores(random_data(11, (2, 50)), 5, 2, 3)
