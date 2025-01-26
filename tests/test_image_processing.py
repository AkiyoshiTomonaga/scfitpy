import pytest

import numpy as np
from scfitpy.image_processing import _calc_poly_length


def test_apply_image_filter():
    raise NotImplementedError


def test__calc_poly_length():
    polygons = np.array([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    assert _calc_poly_length(polygons) == pytest.approx(4)


def test__calc_poly_length_ng():
    polygons = np.array([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert _calc_poly_length(polygons) == pytest.approx(4)


def test_find_contours():
    raise NotImplementedError


def test_assign_contours():
    raise NotImplementedError


def test_mask_img():
    raise NotImplementedError


def test_determine_frequencies():
    raise NotImplementedError
