"""A module for post-processing to extract peak/dip structures from two-dimensional spectrum."""

import itertools
from math import isclose, isnan
from typing import Callable, Iterable, Mapping, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2.typing import MatLike
from numpy.typing import ArrayLike, NDArray
from skimage import filters, measure


def normalize(x) -> NDArray[np.floating]:
    """x in R --> [0..1]"""
    x = np.array(x)
    mn = x.min(axis=None, keepdims=True)
    mx = x.max(axis=None, keepdims=True)
    return (x - mn) / (mx - mn)


DEFAULT_KWARGS_SAVEFIG = dict(bbox_inches="tight", pad_inches=0.5, dpi=500)


def apply_image_filter(
    img: MatLike,
    func: Callable[[MatLike], MatLike],
    with_plot=False,
    filename_plot: str | None = None,
    show=True,
    **kwargs,
) -> MatLike:
    """Apply image processing for an arbitrary filter function.

    Args:
        img: (M, N[, P]) ndarray, An input image(color image of 2d spectrum)
        func: callable, A function implementing an image processing filter. The first argument is `img`.
        with_plot: if True, make a matplotlib plot.
        filename_plot: ...
        kwargs: will be passed to the function.

    Return:
        (M, N[, P]) ndarray
    """
    result = func(img, **kwargs)

    if with_plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        mappable = ax.pcolor(result, cmap="bwr")
        cbar_num_format = "%.2f"
        plt.colorbar(mappable, ax=ax, format=cbar_num_format)
        plt.tight_layout()
        if filename_plot is not None:
            plt.savefig(filename_plot, **DEFAULT_KWARGS_SAVEFIG)
        if show:
            plt.show()

    return result


def apply_sato_filter(
    img: MatLike,
    sigmas: float | Iterable[float] | NDArray[np.floating] = 4.0,
    black_ridges: bool = False,
    with_plot=False,
    filename_plot: str | None = None,
    show=True,
    **kwargs,
) -> MatLike:
    """Apply image processing (sato-function).

    Args:
        img: (M, N[, P]) ndarray, an input image(color image of 2d spectrum)
        sigmas: TBD
        black_ridges: see <skimage.filters.sato>
        with_plot: if True, make a matplotlib plot.
        filename_plot: ...

    Return:
        (M, N[, P]) ndarray
    """
    img = normalize(img)

    if isinstance(sigmas, (int, float)):
        sigmas = [sigmas]
    kwargs["sigmas"] = sigmas
    kwargs["black_ridges"] = black_ridges

    return apply_image_filter(
        img,
        func=filters.sato,
        with_plot=with_plot,
        filename_plot=filename_plot,
        show=show,
        **kwargs,
    )


def apply_nofilter(
    img: MatLike,
    with_plot=False,
    filename_plot: str | None = None,
    show=True,
    **kwargs,
) -> MatLike:
    """Apply image processing without any processing."""
    img = normalize(img)

    return apply_image_filter(
        img,
        func=lambda img: img,
        with_plot=with_plot,
        filename_plot=filename_plot,
        show=show,
        **kwargs,
    )


def _calc_poly_length(polygon: NDArray) -> float:
    """ポリゴンの長さを計算"""
    xs, ys = np.array(polygon).T
    dx = xs[1:] - xs[:-1]
    dy = ys[1:] - ys[:-1]
    ls = dx**2 + dy**2
    return np.sum(ls)


def find_contours(
    img: MatLike,
    level: float | None = None,
    threshold_length: float | None = None,
    with_plot=False,
    filename_plot: str | None = None,
    show=True,
    **kwargs,
) -> list[NDArray[np.floating]]:
    """Find contours from a 2d spectrum.

    1. apply `skimage.measure.find_contours`
    2. pick up some contours that have a certain length (`>= threshold_length`)
    3. (make plot if necessary)

    Args:
        img: (M, N[, P]) ndarray
        level: see <find_contours>
        threshold_length: プロットに表示する閾値
        with_plot: ...
        filename_plot: ...
        kwargs: the other arguments will be passed to find_contours.

    Return:
        list of polygons
    """

    contours = [(poly) for poly in measure.find_contours(img, level, **kwargs)]
    len_list = [_calc_poly_length(poly) for poly in contours]

    if threshold_length is None:
        threshold_length = float(np.median(len_list))
    flg_list = [L >= threshold_length for L in len_list]

    if with_plot:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        (ax, ax_L), (ax_cont, _) = axes

        ax.pcolor(img, cmap="bwr")
        for idx, (poly, flg) in enumerate(zip(contours, flg_list)):
            if not flg:
                continue

            ax.plot(poly[:, 1], poly[:, 0], "k-", linewidth=1)
            ax.plot(poly[:, 1], poly[:, 0], f"C{idx % 10}--", linewidth=2)

            ax_cont.plot(poly[:, 1], poly[:, 0], f"C{idx % 10}-", linewidth=1)
            ax_cont.annotate(
                idx,
                xy=(np.average(poly[:, 1]), np.average(poly[:, 0])),
                fontsize=10,
                color="k",
                bbox={"facecolor": f"C{idx % 10}", "edgecolor": "w", "alpha": 0.3},
            )
        for poly, flg in zip(contours, flg_list):
            if not flg:
                ax.plot(poly[:, 1], poly[:, 0], "k-", linewidth=1, alpha=0.8)

        ax_L.plot(len_list, ds="steps-mid")
        ax_L.axhline(y=threshold_length, color="k", ls="dashed")
        ax_L.grid()
        ax_L.set_ylabel("Contour length")
        ax_L.set_yscale("log")
        ax_L.set_xlabel("# of contours")

        if filename_plot is not None:
            plt.savefig(filename_plot, **DEFAULT_KWARGS_SAVEFIG)
        if show:
            plt.show()

    return contours


def assign_contours(
    cont_list: Sequence[NDArray[np.floating]],
    dict_indexes: dict[str, tuple[int, ...] | None] | None = None,
    with_plot=False,
    filename_plot: str | None = None,
    show=True,
) -> dict[str, list[NDArray[np.floating]]]:
    """Make a dict of contours. key=label, val=contour

    Example:
        >>> assign_contours(cont_list,
        >>>                 {"a": (2, 3), "b": (7, 6), "c": (4, 5)})
        >>> {"a": ..., "b": ..., "c": ...}
    """

    if dict_indexes is None:
        dict_indexes = {"all": None}

    def _extract(idexes: Sequence[int] | None):
        if idexes is None:
            return [v for v in cont_list]
        else:
            return [cont_list[i] for i in idexes]

    cont_dict = {k: _extract(ids) for k, ids in dict_indexes.items()}

    if with_plot:
        fig, ax = plt.subplots(figsize=(8, 6))

        for i, (kw, contours) in enumerate(cont_dict.items()):
            for j, poly in enumerate(contours):
                ax.plot(
                    poly[:, 1],
                    poly[:, 0],
                    f"C{i % 10}-",
                    label=(kw if j == 0 else None),  # 各ポリゴン群で一つだけラベルする
                )
            ax.legend()

        if filename_plot is not None:
            plt.savefig(filename_plot, **DEFAULT_KWARGS_SAVEFIG)
        if show:
            plt.show()

    return cont_dict


def _is_closed(polygon: NDArray[np.floating]) -> bool:
    return len(polygon) > 1 and all(np.isclose(polygon[0], polygon[-1]))


def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


def _make_enclosure(
    polygons: Sequence[NDArray[np.floating]], h: int, w: int
) -> list[NDArray[np.floating]]:
    """ポリゴンのリストを受け取って、閉ポリゴンになるように変更したリストを返す

    仮定
    - コーナーは必ず範囲外にある
    - 包含関係にはならない

    [ [(y,x), (y,x), ...],
      [(y,x), (y,x), ...],
      ...
    ]
    """
    # Note: +yが上辺. y=0, ..., h-1 / x=0, ..., w-1

    # corner case: polygonがエッジ上から始まって同じ点で終わる（未対応）

    # case: multi polygons are found.
    closed_polygons = [p for p in polygons if _is_closed(p)]
    polygons = [p for p in polygons if not _is_closed(p)]
    if len(polygons) == 0:
        return closed_polygons

    # 左回りにエッジを走査して、端点のリストを順に作っていく
    # その後、2個ずつリストに加える。角は閉包に含まれないと仮定しているので、
    # (終点,始点) のペアが順に得られる
    _xval = lambda v: v[1]  # noqa: E731
    _yval = lambda v: v[0]  # noqa: E731
    edge_points = list(itertools.chain.from_iterable([(p[0], p[-1]) for p in polygons]))
    sorted_edge_points = (
        sorted(
            [pos for pos in edge_points if isclose(pos[1], 0.0)],  # left edge
            key=_yval,
            reverse=True,
        )
        + sorted(
            [pos for pos in edge_points if isclose(pos[0], 0.0)],  # bottom
            key=_xval,
        )
        + sorted(
            [pos for pos in edge_points if isclose(pos[1], w - 1)],  # right
            key=_yval,
        )
        + sorted(
            [pos for pos in edge_points if isclose(pos[0], h - 1)],  # top
            key=_xval,
            reverse=True,
        )
    )
    edge_polygons = [
        np.array(pair) for pair in batched(sorted_edge_points, 2, strict=True)
    ]
    polygons = edge_polygons + list(polygons)

    # 順に接続するように並び替え & マージ
    _polygons = [polygons.pop(0)]
    while len(polygons) > 0:
        p_last = _polygons[-1]

        # 端点がpivotに最も近いものを選択
        dist_to_head = [np.linalg.norm(p[0] - p_last[-1]) for p in polygons]
        dist_to_tail = [np.linalg.norm(p[-1] - p_last[-1]) for p in polygons]
        dist_to = np.min(np.column_stack([dist_to_head, dist_to_tail]), axis=1)
        _idx = np.argmin(dist_to)

        p_next = polygons.pop(_idx)
        if isclose(dist_to[_idx], 0):
            if dist_to_head[_idx] > dist_to_tail[_idx]:
                p_next = np.flip(p_next, axis=0)

            _polygons[-1] = np.concatenate([p_last, p_next[1:]])

        else:
            _polygons.append(p_next)
    polygons = list(_polygons)

    return closed_polygons + polygons


def determine_regions(
    img: MatLike,
    cont_dict: Mapping[str, Sequence[NDArray[np.floating]]],
    kernel: int | NDArray = 3,
    with_plot=False,
    filename_plot: str | None = None,
    show=True,
) -> dict[str, MatLike]:
    """Extract regions to freq-determination.

    1. Dilate contours to get band-like regions.
    2. XOR operation to ensure that each band has no overlap.

    Args:
        img:
        cont_dict: key=label, value=list of polygons
        offset: size of dilation
        ...

    Return:
        A dictionary, key=label, value=list of binary images.
    """

    def make_filled_img(contours: Sequence[NDArray[np.floating]]) -> MatLike:
        _img: MatLike = np.zeros(img.shape, np.uint8)

        for poly in _make_enclosure(contours, *img.shape):
            pts = np.flip(poly, axis=1)
            pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            _img = cv2.fillPoly(_img, [pts], (1,))
        return _img

    # make regeion (binary image)
    region_img_dict = {
        k: make_filled_img(contours) for k, contours in cont_dict.items()
    }

    # dilate regions
    if isinstance(kernel, int):
        kernel = np.ones((kernel, kernel), np.uint8)
    region_dict = {k: cv2.dilate(r, kernel) for k, r in region_img_dict.items()}

    # logical op.
    # A, B, C, ... ==> A, B & not A, C & not A & not B, ...
    # 始めに来た領域を優先して、領域ごとの被りをなくす
    Rt: MatLike = np.zeros(img.shape, np.uint8)
    for k in region_dict.keys():
        # Note: keysではなくregion.itemsにすると終わらなくなる。これは辞書内を直接編集しているため。
        region_dict[k] = cv2.bitwise_and(region_dict[k], cv2.bitwise_not(Rt))
        Rt = cv2.bitwise_or(Rt, region_dict[k])

    if with_plot:
        fig, ax = plt.subplots(figsize=(8, 6))

        for k, r in region_dict.items():
            ax.pcolor(r, alpha=0.2)

        for i, (kw, contours) in enumerate(cont_dict.items()):
            for j, poly in enumerate(contours):
                ax.plot(
                    poly[:, 1],
                    poly[:, 0],
                    f"C{i % 10}-",
                    label=(
                        kw if j == 0 else None
                    ),  # 各閉ポリゴン群で一つだけラベルする
                )

        if filename_plot is not None:
            plt.savefig(filename_plot, **DEFAULT_KWARGS_SAVEFIG)
        if show:
            plt.show()

    return region_dict


def _argmax(xs, ys) -> int:
    """ピーク位置決定"""
    return int(np.nanargmax(ys))


def determine_peak_positions(
    img: MatLike,
    region_dict: Mapping[str, MatLike],
    xaxis: NDArray[np.floating] | Sequence | None = None,
    yaxis: NDArray[np.floating] | Sequence | None = None,
    method: Callable[[NDArray[np.floating], NDArray[np.floating]], int] = _argmax,
    eliminate_nan=True,
    with_plot=False,
    filename_plot: str | None = None,
    show=True,
) -> dict[str, list[tuple[float, float]]]:
    """! TODO あとで書く

    Args:
        img (np.ndarray): _description_
        region_dict (Mapping[str, np.ndarray[int]]): _description_
        xaxis (np.ndarray | Sequence | None, optional): _description_. Defaults to None.
        yaxis (np.ndarray | Sequence | None, optional): _description_. Defaults to None.
        method: ピーク位置を決定するアルゴリズム
        with_plot (bool, optional): _description_. Defaults to False.
        filename_plot (str, optional): _description_. Defaults to None.

    Returns:
        dict[str, list[tuple[float, float]]]: _description_
    """
    assert len(img.shape) == 2
    h, w = img.shape

    if xaxis is None:
        xaxis = range(w)
    xaxis = np.array(xaxis)
    if yaxis is None:
        yaxis = range(h)
    yaxis = np.array(yaxis)
    assert len(xaxis) == img.shape[1]
    assert len(yaxis) == img.shape[0]

    peak_dict: dict[str, list[tuple[float, float]]] = {
        k: [] for k in region_dict.keys()
    }
    for k, region in region_dict.items():
        _img = np.array(img, dtype=np.float128)
        _img[np.where(region == 0)] = np.nan

        for idx_x in range(w):
            try:
                idx_ypeak = method(xaxis, _img[:, idx_x])

            except ValueError:  # all-nan slice, その軸に値が見つからないケース
                peak_dict[k].append((xaxis[idx_x], np.nan))

            else:
                peak_dict[k].append((xaxis[idx_x], yaxis[idx_ypeak]))

    if eliminate_nan:
        peak_dict = {
            k: [v for v in indexes if not isnan(v[1])]
            for k, indexes in peak_dict.items()
        }

    if with_plot:
        fig, (ax, ax_orig, ax_pos) = plt.subplots(1, 3, figsize=(15, 4))
        X, Y = np.meshgrid(xaxis, yaxis)
        ax.pcolor(X, Y, img)
        ax_orig.pcolor(X, Y, img)
        for i, (kw, positions) in enumerate(peak_dict.items()):
            pos = np.array(positions)
            ax.scatter(pos[:, 0], pos[:, 1], c="red", label=kw, marker="x", alpha=0.5)
            ax_pos.scatter(pos[:, 0], pos[:, 1], c=f"C{i % 10}", label=kw, marker=".")
        ax_pos.set_xlim(ax.get_xlim())
        ax_pos.set_ylim(ax.get_ylim())
        ax_pos.legend()
        if filename_plot is not None:
            plt.savefig(filename_plot, **DEFAULT_KWARGS_SAVEFIG)
        if show:
            plt.show()

    return peak_dict
