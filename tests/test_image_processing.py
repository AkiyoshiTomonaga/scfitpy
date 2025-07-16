from math import pi

import numpy as np
import pytest

from scfitpy.image_processing import (
    _make_enclosure,
    apply_nofilter,
    apply_sato_filter,
    assign_contours,
    determine_peak_positions,
    determine_regions,
    find_contours,
)


def test_all_singleline():
    # preparation
    prefix = "./outs/test_all_singleline_"
    np.random.seed(42)

    xs = np.linspace(-1, 1, 101)  # sweep-axis, like dc bias
    fs = np.linspace(100, 200, 201)  # frequency-axis
    pts = np.full_like(xs, xs * 10 + np.average(fs))  # dip point: y=y0 (constant)
    rtol = 0.01  # relative tolerance for analysis

    spec = np.array(
        [2 / (0.05**2 + (fs - f0) ** 2) for f0 in pts]
    ).T  # 形状固定のLorentzian
    spec += np.random.normal(size=spec.shape)
    mag = 20 * np.log10(np.abs(spec))

    # run
    apply_nofilter(mag, with_plot=True, filename_plot=prefix + "mag.png")
    mag_filtered = apply_sato_filter(
        mag,
        4,
        black_ridges=False,
        with_plot=True,
        show=False,
        filename_plot=prefix + "mag_filtered.png",
    )
    cont_list = find_contours(
        mag_filtered,
        level=0.06,
        threshold_length=1,
        with_plot=True,
        show=False,
        filename_plot=prefix + "mag_find_contours.png",
    )
    cont_dict = assign_contours(
        cont_list,
        {"a": (0, 1)},
        with_plot=True,
        show=False,
        filename_plot=prefix + "contours.png",
    )
    region_dict = determine_regions(
        mag,
        cont_dict,
        kernel=5,
        with_plot=True,
        show=False,
        filename_plot=prefix + "regions.png",
    )
    result = determine_peak_positions(
        mag,
        region_dict,
        xaxis=xs,
        yaxis=fs,
        with_plot=True,
        show=False,
        filename_plot=prefix + "peaks.png",
    )

    # check
    assert pytest.approx(xs) == [v for v, _ in result["a"]]
    assert pytest.approx(pts, rtol) == [v for _, v in result["a"]]


def test_all_twoline():
    # preparation
    prefix = "./outs/test_all_twoline_"
    np.random.seed(42)

    xs = np.linspace(-1, 1, 101)  # sweep-axis, like dc bias
    fs = np.linspace(100, 200, 201)  # frequency-axis
    pts1 = np.full_like(xs, xs * 10 + np.average(fs) + 20)  # dip point: y=y0 (constant)
    pts2 = np.full_like(xs, xs * -3 + np.average(fs) - 20)  # dip point: y=y0 (constant)
    rtol = 0.01  # relative tolerance for analysis

    spec = (
        np.array([117 / (0.05**2 + (fs - f0) ** 2) for f0 in pts1]).T
        + np.array([60 / (0.1**2 + (fs - f0) ** 2) for f0 in pts2]).T
    )
    spec += np.random.normal(size=spec.shape)
    mag = 20 * np.log10(np.abs(spec))

    # run
    apply_nofilter(mag, with_plot=True, filename_plot=prefix + "mag.png")
    mag_filtered = apply_sato_filter(
        mag,
        2,
        black_ridges=False,
        with_plot=True,
        show=False,
        filename_plot=prefix + "mag_filtered.png",
    )
    cont_list = find_contours(
        mag_filtered,
        level=0.02,
        threshold_length=2,
        with_plot=True,
        show=False,
        filename_plot=prefix + "mag_find_contours.png",
    )
    cont_dict = assign_contours(
        cont_list,
        {"upper": (15, 14), "lower": (8, 9)},
        with_plot=True,
        show=False,
        filename_plot=prefix + "contours.png",
    )
    region_dict = determine_regions(
        mag,
        cont_dict,
        kernel=5,
        with_plot=True,
        show=False,
        filename_plot=prefix + "regions.png",
    )
    result = determine_peak_positions(
        mag,
        region_dict,
        xaxis=xs,
        yaxis=fs,
        with_plot=True,
        show=False,
        filename_plot=prefix + "peaks.png",
    )

    # check
    assert pytest.approx(xs) == [v for v, _ in result["upper"]]
    assert pytest.approx(xs) == [v for v, _ in result["lower"]]
    assert pytest.approx(pts1, rtol) == [v for _, v in result["upper"]]
    assert pytest.approx(pts2, rtol) == [v for _, v in result["lower"]]


def test_all_single_curve():
    # preparation
    prefix = "./outs/test_all_single_curve_"
    np.random.seed(42)

    xs = np.linspace(-1, 1, 101)  # sweep-axis, like dc bias
    fs = np.linspace(100, 200, 201)  # frequency-axis
    pts = np.full_like(
        xs, 40 * xs**2 + np.average(fs) - 30
    )  # dip point: y=y0 (constant)
    rtol = 0.01  # relative tolerance for analysis

    amps = 20 * np.ones_like(xs)
    spec = np.array(
        [a / (0.05**2 + (fs - f0) ** 2) for a, f0 in zip(amps, pts)]
    ).T  # 形状固定のLorentzian
    spec += np.random.normal(size=spec.shape)
    mag = 20 * np.log10(np.abs(spec))

    # run
    apply_nofilter(mag, with_plot=True, filename_plot=prefix + "mag.png")
    mag_filtered = apply_sato_filter(
        mag,
        1,
        False,
        with_plot=True,
        show=False,
        filename_plot=prefix + "mag_filtered.png",
    )
    cont_list = find_contours(
        mag_filtered,
        level=0.025,
        threshold_length=20,
        with_plot=True,
        show=False,
        filename_plot=prefix + "mag_find_contours.png",
    )
    cont_dict = assign_contours(
        cont_list,
        {"a": (156, 162)},
        with_plot=True,
        show=False,
        filename_plot=prefix + "contours.png",
    )
    region_dict = determine_regions(
        mag,
        cont_dict,
        kernel=5,
        with_plot=True,
        show=False,
        filename_plot=prefix + "regions.png",
    )
    result = determine_peak_positions(
        mag,
        region_dict,
        xaxis=xs,
        yaxis=fs,
        eliminate_nan=False,
        with_plot=True,
        show=False,
        filename_plot=prefix + "peaks.png",
    )

    # check
    assert pytest.approx(xs) == [v for v, _ in result["a"]]
    assert pytest.approx(pts, rtol) == [v for _, v in result["a"]]


def test_all_steep_curve():
    # preparation
    prefix = "./outs/test_all_steep_curve_"
    np.random.seed(42)

    xs = np.linspace(-1, 1, 101)  # sweep-axis, like dc bias
    fs = np.linspace(100, 200, 201)  # frequency-axis
    pts = np.full_like(
        xs,
        200 * xs**3 + np.average(fs),  # 左右ではなく上下に突き抜ける場合
    )  # dip point: y=y0 (constant)
    rtol = 0.01  # relative tolerance for analysis

    amps = 20 * np.ones_like(xs)
    spec = np.array(
        [a / (0.05**2 + (fs - f0) ** 2) for a, f0 in zip(amps, pts)]
    ).T  # 形状固定のLorentzian
    spec += np.random.normal(size=spec.shape)
    mag = 20 * np.log10(np.abs(spec))

    # run
    apply_nofilter(mag, with_plot=True, filename_plot=prefix + "mag.png")
    mag_filtered = apply_sato_filter(
        mag,
        1,
        False,
        with_plot=True,
        show=False,
        filename_plot=prefix + "mag_filtered.png",
    )
    cont_list = find_contours(
        mag_filtered,
        level=0.025,
        threshold_length=1000,
        with_plot=True,
        show=False,
        filename_plot=prefix + "mag_find_contours.png",
    )
    cont_dict = assign_contours(
        cont_list,
        {"a": (1, 2)},
        with_plot=True,
        show=False,
        filename_plot=prefix + "contours.png",
    )
    region_dict = determine_regions(
        mag,
        cont_dict,
        kernel=5,
        with_plot=True,
        show=False,
        filename_plot=prefix + "regions.png",
    )
    result = determine_peak_positions(
        mag,
        region_dict,
        xaxis=xs,
        yaxis=fs,
        eliminate_nan=False,
        with_plot=True,
        show=False,
        filename_plot=prefix + "peaks.png",
    )

    # check
    assert pytest.approx(xs) == [v for v, _ in result["a"]]
    assert pytest.approx(pts[20:81], rtol) == [v for _, v in result["a"]][20:81]


def test_all_steep_curve2():
    # preparation
    prefix = "./outs/test_all_steep_curve2_"
    np.random.seed(42)

    xs = np.linspace(-1, 1, 101)  # sweep-axis, like dc bias
    fs = np.linspace(100, 200, 201)  # frequency-axis
    pts = np.full_like(
        xs,
        200 * xs**2 + np.average(fs),  # 上からきて上に抜ける場合
    )  # dip point: y=y0 (constant)
    rtol = 0.01  # relative tolerance for analysis

    amps = 20 * np.ones_like(xs)
    spec = np.array(
        [a / (0.05**2 + (fs - f0) ** 2) for a, f0 in zip(amps, pts)]
    ).T  # 形状固定のLorentzian
    spec += np.random.normal(size=spec.shape)
    mag = 100 * np.log10(np.abs(spec))

    # run
    apply_nofilter(mag, with_plot=True, filename_plot=prefix + "mag.png")
    mag_filtered = apply_sato_filter(
        mag,
        1,
        False,
        with_plot=True,
        show=False,
        filename_plot=prefix + "mag_filtered.png",
    )
    cont_list = find_contours(
        mag_filtered,
        level=0.03,
        threshold_length=10,
        with_plot=True,
        show=False,
        filename_plot=prefix + "mag_find_contours.png",
    )
    cont_dict = assign_contours(
        cont_list,
        {"a": (245, 255)},
        with_plot=True,
        show=False,
        filename_plot=prefix + "contours.png",
    )
    region_dict = determine_regions(
        mag,
        cont_dict,
        kernel=5,
        with_plot=True,
        show=False,
        filename_plot=prefix + "regions.png",
    )
    result = determine_peak_positions(
        mag,
        region_dict,
        xaxis=xs,
        yaxis=fs,
        eliminate_nan=False,
        with_plot=True,
        show=False,
        filename_plot=prefix + "peaks.png",
    )

    # check
    assert pytest.approx(xs) == [v for v, _ in result["a"]]
    assert pytest.approx(pts[25:76], rtol) == [v for _, v in result["a"]][25:76]


def test_all_isolated_curve():
    # preparation
    prefix = "./outs/test_all_isolated_curve_"
    np.random.seed(42)

    xs = np.linspace(-1, 1, 101)  # sweep-axis, like dc bias
    fs = np.linspace(100, 200, 201)  # frequency-axis
    pts = np.full_like(
        xs, 40 * xs**2 + np.average(fs) - 30
    )  # dip point: y=y0 (constant)
    rtol = 0.01  # relative tolerance for analysis

    amps = 101 * (1 - np.tanh(xs * 10) ** 4)
    print(amps)
    spec = np.array(
        [a / (0.1**2 + (fs - f0) ** 2) for a, f0 in zip(amps, pts)]
    ).T  # 形状固定のLorentzian
    spec += np.random.normal(size=spec.shape)
    mag = 20 * np.log10(np.abs(spec))

    # run
    apply_nofilter(mag, with_plot=True, filename_plot=prefix + "mag.png")
    mag_filtered = apply_sato_filter(
        mag,
        1,
        False,
        with_plot=True,
        show=False,
        filename_plot=prefix + "mag_filtered.png",
    )
    cont_list = find_contours(
        mag_filtered,
        level=0.05,
        threshold_length=10,
        with_plot=True,
        show=False,
        filename_plot=prefix + "mag_find_contours.png",
    )
    cont_dict = assign_contours(
        cont_list,
        {"a": (4,)},
        with_plot=True,
        show=False,
        filename_plot=prefix + "contours.png",
    )
    region_dict = determine_regions(
        mag,
        cont_dict,
        kernel=5,
        with_plot=True,
        show=False,
        filename_plot=prefix + "regions.png",
    )
    result = determine_peak_positions(
        mag,
        region_dict,
        xaxis=xs,
        yaxis=fs,
        eliminate_nan=False,
        with_plot=True,
        show=False,
        filename_plot=prefix + "peaks.png",
    )

    # check
    assert pytest.approx(xs) == [v for v, _ in result["a"]]
    assert pytest.approx(pts[35:68], rtol) == [v for _, v in result["a"]][35:68]


def test_all_single_disjointcurve():
    # preparation
    prefix = "./outs/test_all_single_disjointcurve_"
    np.random.seed(42)

    xs = np.linspace(-1, 1, 101)  # sweep-axis, like dc bias
    fs = np.linspace(100, 200, 201)  # frequency-axis
    amps = np.tanh(xs) ** 4 * 20
    pts = np.full_like(
        xs, 40 * xs**2 + np.average(fs) - 30
    )  # dip point: y=y0 (constant)
    rtol = 0.01  # relative tolerance for analysis

    spec = np.array(
        [a / (0.05**2 + (fs - f0) ** 2) for a, f0 in zip(amps, pts)]
    ).T  # 形状固定のLorentzian
    spec += np.random.normal(size=spec.shape)
    mag = 20 * np.log10(np.abs(spec))

    # run
    apply_nofilter(mag, with_plot=True, filename_plot=prefix + "mag.png")
    mag_filtered = apply_sato_filter(
        mag,
        1,
        False,
        with_plot=True,
        show=False,
        filename_plot=prefix + "mag_filtered.png",
    )
    cont_list = find_contours(
        mag_filtered,
        level=0.025,
        threshold_length=100,
        with_plot=True,
        show=False,
        filename_plot=prefix + "mag_find_contours.png",
    )
    cont_dict = assign_contours(
        cont_list,
        {"left": (257,), "right": (230,)},
        with_plot=True,
        show=False,
        filename_plot=prefix + "contours.png",
    )
    region_dict = determine_regions(
        mag,
        cont_dict,
        kernel=5,
        with_plot=True,
        show=False,
        filename_plot=prefix + "regions.png",
    )
    result = determine_peak_positions(
        mag,
        region_dict,
        xaxis=xs,
        yaxis=fs,
        eliminate_nan=False,
        with_plot=True,
        show=False,
        filename_plot=prefix + "peaks.png",
    )

    # check
    assert pytest.approx(xs) == [v for v, _ in result["left"]]
    assert pytest.approx(xs) == [v for v, _ in result["right"]]
    assert pytest.approx(pts[:35], rtol) == [v for _, v in result["left"]][:35]
    assert pytest.approx(pts[:36], rtol) != [v for _, v in result["left"]][:36]
    assert pytest.approx(pts[59:], rtol) == [v for _, v in result["right"]][59:]
    assert pytest.approx(pts[58:], rtol) != [v for _, v in result["right"]][58:]


def test_all_curves():
    # preparation
    prefix = "./outs/test_all_curves_"
    np.random.seed(42)

    xs = np.linspace(-1, 1, 101)  # sweep-axis, like dc bias
    fs = np.linspace(100, 200, 201)  # frequency-axis
    amps = np.tanh(xs) ** 4 * 20
    pts1 = np.full_like(
        xs, 40 * xs**2 + np.average(fs) - 30
    )  # dip point: y=y0 (constant)
    pts2 = pts1 + 20
    rtol = 0.01  # relative tolerance for analysis

    amps = np.tanh(xs) ** 4 * 30
    spec = (
        np.array([25 / (0.05**2 + (fs - f0) ** 2) for f0 in pts1]).T
        + np.array([a / (0.05**2 + (fs - f0) ** 2) for a, f0 in zip(amps, pts2)]).T
    )
    spec += np.random.normal(size=spec.shape)
    mag = 20 * np.log10(np.abs(spec))

    # run
    apply_nofilter(mag, with_plot=True, filename_plot=prefix + "mag.png")
    mag_filtered = apply_sato_filter(
        mag,
        1,
        False,
        with_plot=True,
        show=False,
        filename_plot=prefix + "mag_filtered.png",
    )
    cont_list = find_contours(
        mag_filtered,
        level=0.025,
        threshold_length=100,
        with_plot=True,
        show=False,
        filename_plot=prefix + "mag_find_contours.png",
    )
    cont_dict = assign_contours(
        cont_list,
        {"connected": (192, 202), "left": (361,), "right": (357,)},
        with_plot=True,
        show=False,
        filename_plot=prefix + "contours.png",
    )
    region_dict = determine_regions(
        mag,
        cont_dict,
        kernel=5,
        with_plot=True,
        show=False,
        filename_plot=prefix + "regions.png",
    )
    result = determine_peak_positions(
        mag,
        region_dict,
        xaxis=xs,
        yaxis=fs,
        eliminate_nan=False,
        with_plot=True,
        show=False,
        filename_plot=prefix + "peaks.png",
    )

    # check
    assert pytest.approx(xs) == [v for v, _ in result["left"]]
    assert pytest.approx(xs) == [v for v, _ in result["right"]]
    assert pytest.approx(xs) == [v for v, _ in result["connected"]]
    _pts1 = [v for _, v in result["connected"]]
    assert pytest.approx(pts1, rtol) == _pts1
    assert pytest.approx(pts2[:38], rtol) == [v for _, v in result["left"]][:38]
    assert pytest.approx(pts2[60:], rtol) == [v for _, v in result["right"]][60:]


# --------------------------------------------#


def test__make_enclosure():
    # 横に貫くポリゴン線が2本 → マージして端を閉じた閉ポリゴンにする
    line1_poly = np.array([(3, 0), (5, 5), (4, 10)])  # (y,x)
    line2_poly = np.array([(6, 10), (6, 5), (4, 0)])

    expected_polygon = [
        np.array(
            [
                (4, 0),
                (3, 0),
                (5, 5),
                (4, 10),
                (6, 10),
                (6, 5),
                (4, 0),
            ]
        )
    ]

    assert np.isclose(
        expected_polygon,
        _make_enclosure(polygons=[line1_poly, line2_poly], h=100, w=10 + 1),
    ).all()

    assert np.isclose(
        expected_polygon,
        _make_enclosure(polygons=[line1_poly, line2_poly[::-1]], h=100, w=10 + 1),
    ).all()


def test__make_enclosure__pass():
    # 初めから閉じているものはそのまま返す
    closed_polygon = np.array([(5, 5), (10, 10), (15, 5), (10, 1), (5, 5)])  # y,x

    assert np.isclose(
        closed_polygon,
        _make_enclosure(polygons=[closed_polygon], h=100, w=20),
    ).all()


def test__make_enclosure__single_open_curve():
    # 初めから空いているものは閉じる
    open_polygon = np.array([(1, 0), (5, 5), (10, 5), (15, 0)])  # y,x
    expect_polygon = np.array([(15, 0), (1, 0), (5, 5), (10, 5), (15, 0)])  # y,x

    polygons = _make_enclosure(polygons=[open_polygon], h=100, w=20)

    for p1, p2 in zip([expect_polygon], polygons):
        assert np.isclose(p1, p2).all()


def test__make_enclosure_disjoint():
    # 横に貫くポリゴン線が2本 → マージして端を閉じた閉ポリゴンにする
    line1_poly = np.array([(30, 0), (50, 50), (40, 100)])  # (y,x)
    line2_poly = np.array([(60, 100), (60, 50), (40, 0)])

    # 初めから閉じているものはそのまま返す
    closed_poly = np.array([(5, 5), (10, 10), (15, 5), (10, 1), (5, 5)])  # y,x

    expected_polygons = [
        closed_poly,
        np.array(
            [
                (40, 0),
                (30, 0),
                (50, 50),
                (40, 100),
                (60, 100),
                (60, 50),
                (40, 0),
            ]
        ),
    ]

    polygons = _make_enclosure(
        polygons=[line1_poly, line2_poly, closed_poly], h=1001, w=101
    )
    for p1, p2 in zip(expected_polygons, polygons):
        assert np.isclose(p1, p2).all()
