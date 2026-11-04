"""Tests of tycho.ri.discretize: thresholding rules and informative-column selection."""
from __future__ import annotations

import numpy as np
import pytest

from tycho.ri.discretize import binarize, select_informative


def test_binarize_methods_on_known_columns():
    A = np.array([[-1.0, 0.0, 5.0],
                  [0.0, 1.0, 1.0],
                  [2.0, 2.0, 2.0],
                  [3.0, 10.0, 3.0]])
    np.testing.assert_array_equal(binarize(A, "zero"), (A > 0).astype(np.int8))
    # medians: 1.0, 1.5, 2.5
    np.testing.assert_array_equal(binarize(A, "median"), [[0, 0, 1], [0, 0, 0], [1, 1, 0], [1, 1, 1]])
    # means: 1.0, 3.25, 2.75
    np.testing.assert_array_equal(binarize(A, "mean"), [[0, 0, 1], [0, 0, 0], [1, 0, 0], [1, 1, 1]])
    # 0.75-quantiles (linear interpolation): 2.25, 4.0, 3.5
    np.testing.assert_array_equal(binarize(A, "quantile", q=0.75), [[0, 0, 1], [0, 0, 0], [0, 0, 0], [1, 1, 0]])
    out = binarize(A)
    assert out.dtype == np.int8 and out.shape == A.shape


def test_binarize_median_balances_continuous_columns_and_strict_ties():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(1001, 6)) * np.arange(1, 7) + 3.0
    B = binarize(A, "median")
    np.testing.assert_array_equal(B.sum(axis=0), np.full(6, 500))  # strictly above the median
    relu = np.maximum(rng.normal(size=(1000, 1)) - 0.5, 0.0)  # > 50 % exact zeros: median 0
    assert binarize(relu, "median").mean() == pytest.approx(binarize(relu, "zero").mean())
    assert binarize(relu, "median").mean() < 0.5


def test_binarize_one_dimensional_and_empty_and_errors():
    np.testing.assert_array_equal(binarize(np.array([3.0, 1.0, 2.0])), [1, 0, 0])
    assert binarize(np.zeros((0, 3)), "median").shape == (0, 3)
    with pytest.raises(ValueError):
        binarize(np.array([[1.0, np.nan]]))
    with pytest.raises(ValueError):
        binarize(np.ones((2, 2)), "sign")
    with pytest.raises(ValueError):
        binarize(np.ones((2, 2)), "quantile", q=1.5)
    with pytest.raises(ValueError):
        binarize(np.ones((2, 2, 2)))


def test_select_informative_by_entropy():
    rng = np.random.default_rng(1)
    n = 4000
    probs = np.array([0.5, 0.1, 0.0, 0.3, 0.45, 0.02, 1.0])
    B = (rng.random((n, probs.size)) < probs).astype(np.int8)
    # entropy order: col 0 (0.5) > col 4 (0.45) > col 3 (0.3) > col 1 (0.1) > col 5 (0.02); cols 2, 6 constant
    np.testing.assert_array_equal(select_informative(B, 3), [0, 3, 4])
    np.testing.assert_array_equal(select_informative(B, 10), [0, 1, 3, 4, 5])  # constants excluded
    assert select_informative(B, 10, exclude_constant=False).size == 7
    assert select_informative(B, 0).size == 0
    tied = np.tile(np.array([[0, 1, 1], [1, 0, 0]], dtype=np.int8), (10, 1))
    np.testing.assert_array_equal(select_informative(tied, 2), [0, 1])  # ties broken by column index


def test_select_informative_multistate_and_random():
    col_a = np.repeat([0, 1, 2, 3], 25)  # entropy log 4
    col_b = np.repeat([0, 1], 50)  # log 2
    col_c = np.repeat([0, 1, 2], [80, 10, 10])
    B = np.column_stack([col_c, col_b, col_a])
    np.testing.assert_array_equal(select_informative(B, 1), [2])
    r1 = select_informative(B, 2, method="random", rng=np.random.default_rng(5))
    r2 = select_informative(B, 2, method="random", rng=np.random.default_rng(5))
    np.testing.assert_array_equal(r1, r2)
    assert r1.size == 2 and np.all(np.diff(r1) > 0)
    with pytest.raises(ValueError):
        select_informative(B, 2, method="variance")
    with pytest.raises(ValueError):
        select_informative(B[:, 0], 1)
    with pytest.raises(ValueError):
        select_informative(B, -1)
