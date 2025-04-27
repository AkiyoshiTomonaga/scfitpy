"""A module for post-processing to extract peak/dip structures from two-dimensional spectrum."""

from typing import Callable, Iterable, Mapping, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from skimage.filters import sato


def normalize(x) -> np.ndarray:
    """x in R --> [0..1]"""
    x = np.array(x)
    mn = x.min(axis=None, keepdims=True)
    mx = x.max(axis=None, keepdims=True)
    return (x - mn) / (mx - mn)


def apply_image_filter(
    img: np.ndarray,
    func: Callable[..., np.ndarray],
    with_plot=False,
    filename_plot: str = None,
    **kwargs,
) -> np.ndarray:
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
        plt.savefig(filename_plot, bbox_inches="tight", pad_inches=0.5, dpi=500)

    return result


def apply_sato_filter(
    img: np.ndarray,
    sigmas: float | Iterable[float],
    black_ridges: bool,
    with_plot=False,
    filename_plot: str = None,
    **kwargs,
) -> np.ndarray:
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
        img, func=sato, with_plot=with_plot, filename_plot=filename_plot, **kwargs
    )


def _calc_poly_length(polygon: np.ndarray) -> float:
    """閉曲線ポリゴンの周囲長を計算"""
    assert np.isclose(polygon[0], polygon[-1]).all()

    xs, ys = np.array(polygon).T
    dx = xs[1:] - xs[:-1]
    dy = ys[1:] - ys[:-1]
    ls = dx**2 + dy**2
    return np.sum(ls)


def find_contours(
    img: np.ndarray,
    level: float,
    threshold_length: float = None,
    with_plot=False,
    filename_plot: str = None,
    **kwargs,
) -> list[np.ndarray]:
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

    def _ensure_closed(polygon) -> np.ndarray:
        """polygon --> enclosed polygon"""
        assert len(polygon) >= 3

        if np.isclose(polygon[0], polygon[-1]).all():
            return polygon
        return np.concatenate((polygon, [polygon[0]]))

    contours = [
        _ensure_closed(poly) for poly in measure.find_contours(img, level, **kwargs)
    ]
    len_list = [_calc_poly_length(poly) for poly in contours]

    if threshold_length is None:
        threshold_length = np.median(len_list)
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
        ax_L.set_xlabel("# of contours")

        if filename_plot is not None:
            plt.savefig(filename_plot, bbox_inches="tight", pad_inches=0.5, dpi=500)
        plt.show()

    return contours


def assign_contours(
    cont_list: Sequence[np.ndarray],
    dict_indexes: dict[str, tuple[int, ...]],
    with_plot=False,
    filename_plot: str = None,
) -> dict[str, list[np.ndarray]]:
    """Make a dict of contours. key=label, val=contour

    Example:
        >>> assign_contours(cont_list,
        >>>                 {"a": (2, 3), "b": (7, 6), "c": (4, 5)})
        >>> {"a": ..., "b": ..., "c": ...}
    """

    cont_dict = {k: [cont_list[i] for i in ids] for k, ids in dict_indexes.items()}

    if with_plot:
        fig, ax = plt.subplots(figsize=(8, 6))

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
            ax.legend()

        if filename_plot is not None:
            plt.savefig(filename_plot, bbox_inches="tight", pad_inches=0.5, dpi=500)
        plt.show()

    return cont_dict


def determine_regions(
    img: np.ndarray,
    cont_dict: Mapping[str, Sequence[np.ndarray]],
    kernel: int | np.ndarray = 3,
    with_plot=False,
    filename_plot: str = None,
) -> dict[str, np.ndarray[int]]:
    """Extract regions to freq-determination.

    1. Dilate contours to get band-like regions.
    2. XOR operation to ensure that each band has no overlap.

    Args:
        img:
        cont_dict: key=label, value=list of enclosed-polygon
        offset: size of dilation
        ...

    Return:
        A dictionary, key=label, value=list of binary images.
    """

    def make_filled_img(contours: list[np.ndarray]):
        _img = np.zeros(img.shape, np.uint8)
        pts = [
            np.flip(
                np.array(poly, dtype=int), axis=1
            )  # converting positions to coordinates for cv2
            for poly in contours
        ]
        return cv2.fillPoly(_img, pts, 1)

    # make regeion (binary image)
    region_dict = {k: make_filled_img(contours) for k, contours in cont_dict.items()}

    # dilate regions
    if isinstance(kernel, int):
        kernel = np.ones((kernel, kernel), np.uint8)
    region_dict = {k: cv2.dilate(r, kernel) for k, r in region_dict.items()}

    # logical op.
    # A, B, C, ... ==> A, B & not A, C & not A & not B, ...
    # 始めに来た領域を優先して、領域ごとの被りをなくす
    Rt = np.zeros(img.shape, np.uint8)
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
            plt.savefig(filename_plot, bbox_inches="tight", pad_inches=0.5, dpi=500)
        plt.show()

    return region_dict


def determine_peak_positions(
    img: np.ndarray,
    region_dict: Mapping[str, np.ndarray[int]],
    xaxis: np.ndarray | Sequence | None = None,
    yaxis: np.ndarray | Sequence | None = None,
    with_plot=False,
    filename_plot: str = None,
) -> dict[str, list[np.ndarray]]:
    """! TODO あとで書く

    Args:
        img (np.ndarray): _description_
        region_dict (Mapping[str, np.ndarray[int]]): _description_
        xaxis (np.ndarray | Sequence | None, optional): _description_. Defaults to None.
        yaxis (np.ndarray | Sequence | None, optional): _description_. Defaults to None.
        with_plot (bool, optional): _description_. Defaults to False.
        filename_plot (str, optional): _description_. Defaults to None.

    Returns:
        dict[str, list[tuple[float, float]]]: _description_
    """
    assert len(img.shape) == 2
    h, w = img.shape
    xaxis = np.array(xaxis) if xaxis is not None else np.arange(w)
    yaxis = np.array(yaxis) if yaxis is not None else np.arange(h)
    assert len(xaxis) == img.shape[1]
    assert len(yaxis) == img.shape[0]

    peak_dict: dict[str, list[tuple[float, float]]] = {
        k: [] for k in region_dict.keys()
    }
    for k, region in region_dict.items():
        _img = np.copy(img)
        _img[np.where(region == 0)] = np.nan

        for idx_x in range(w):
            vs = _img[:, idx_x]
            try:
                idx_ypeak = int(np.nanargmax(vs))
            except ValueError:
                pass  # all-nan slice, その軸に値が見つからないケース
            else:
                peak_dict[k].append((xaxis[idx_x], yaxis[idx_ypeak]))
    peak_dict = {k: np.array(pos) for k, pos in peak_dict.items()}

    if with_plot:
        fig, (ax, ax_pos) = plt.subplots(1, 2, figsize=(14, 6))
        X, Y = np.meshgrid(xaxis, yaxis)
        ax.pcolor(X, Y, img)
        for i, (kw, positions) in enumerate(peak_dict.items()):
            pos = np.array(positions)
            ax.scatter(pos[:, 0], pos[:, 1], c="red", label=kw, marker="x", alpha=0.1)
            ax_pos.scatter(pos[:, 0], pos[:, 1], c=f"C{i % 10}", label=kw, marker=".")
        ax_pos.set_xlim(ax.get_xlim())
        ax_pos.set_ylim(ax.get_ylim())
        ax_pos.legend()
        if filename_plot is not None:
            plt.savefig(filename_plot, bbox_inches="tight", pad_inches=0.5, dpi=500)
        plt.show()

    return peak_dict
