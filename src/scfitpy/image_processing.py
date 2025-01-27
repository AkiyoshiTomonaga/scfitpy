"""A module for post-processing to extract peak/dip structures from two-dimensional spectrum."""

import math
from numbers import Number
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import sato
from skimage import measure


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


# def peakTrace(mag2, freq, current, cont_band, th, vwid, hwid, black_ridges):
def determine_regions(
    img: np.ndarray,
    cont_dict: Mapping[str, Sequence[np.ndarray]],
    offset: float,
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
        A dictionary, key=label, value=list of pixel positions.
    """
    h, w = img.shape
    X, Y = np.meshgrid(range(w), range(h))
    [pos for pos, v in np.ndenumerate(img)]

    for label, contours in cont_dict.items():
        min_length = np.zeros_like(img)  # あるバンドへの最短距離
        for p in contours:
            pass

    # dilate
    pass

    # logical op.
    pass

    masks: dict[str, np.ndarray[int]]  # key=label, val=list of position

    # null_vec = np.array([0 for i in range(img.shape[1])])
    # cont_bla = np.array(
    #     [[null_vec for i in range(img.shape[0])] for i in range(len(cont_band))]
    # )
    # for i, cont in enumerate(cont_band):
    #     for j in range(len(cont)):
    #         for k in range(2 * offset_y):
    #             if (
    #                 int(cont[j][0]) - offset_y + k < len(img[1])
    #                 and int(cont[j][0]) - offset_y + k > 0
    #             ):
    #                 cont_bla[i][int(cont[j][1])][int(cont[j][0]) - offset_y + k] = 1
    #         for k in range(2 * offset_x):
    #             if (
    #                 int(cont[j][1]) - offset_x + k < len(img)
    #                 and int(cont[j][1]) - offset_x + k > 0
    #             ):
    #                 cont_bla[i][int(cont[j][1]) - offset_x + k][int(cont[j][0])] = 1

    if with_plot:
        blade = null_vec
        for i in range(len(cont_bla)):
            blade = blade + cont_bla[i]
        fig, ax = plt.subplots(figsize=(8, 6))
        for n, cont in enumerate(cont_band):
            ax.plot(cont[:, 1], cont[:, 0], "+", ms=2, label=str(n))
        X, Y = np.meshgrid(
            np.linspace(0, len(blade[0]), len(blade[0])),
            np.linspace(0, len(blade), len(blade)),
        )
        mappable = ax.pcolor(Y, X, blade, cmap="bwr")
        ax.legend(loc="upper right")
        plt.savefig(filename_plot, bbox_inches="tight", pad_inches=0.5, dpi=500)
        plt.show()

    bi_sato = np.array([null_vec for i in range(len(mag2))])
    for i in range(len(mag2)):  # convert mag2 to binary matrix
        for j in range(len(mag2[1])):
            if mag2[i][j] > th:
                bi_sato[i][j] = 1

    # cont_blaとbi_satoの重なる部分の座標を抽出 かつ　他のバンドと場所を共有しないようにフィルタ。
    # Get the coordinate data from the set bi_sato ∧ cont_bla, and filtaling to band_pos[i] ∧ band_pos[j] = Φ at i≠j
    band_pos = [[] for i in range(len(cont_bla))]
    for i in range(len(cont_bla)):
        cont_bla[i] = np.logical_and(cont_bla[i], bi_sato)
    for i in range(len(cont_bla) - 1):
        band_pos[i] = (
            cont_bla[i]
            - np.logical_and(cont_bla[i], cont_bla[i + 1])
            - np.logical_and(cont_bla[i - 1], cont_bla[i])
        )
        band_pos[i + 1] = cont_bla[i + 1] - np.logical_and(cont_bla[i], cont_bla[i + 1])

    # 上で定めた領域の点をmag2の値として持ち出す。
    # Get the data from mag2 at the coordinate in band_pos
    band = np.array(
        [[null_vec for i in range(mag2.shape[0])] for i in range(len(cont_band))]
    )
    for k in range(len(band)):
        for i in range(len(mag2)):
            for j in range(len(mag2[0])):
                if band_pos[k][i][j] == 1:
                    band[k][i][j] = mag2[i][j]


def determine_frequencies(
    mag,
    masks: dict[str, np.ndarray[int]],
    freq,
    current,
    cont_band,
    th,
    vwid,
    hwid,
    black_ridges,
):
    # min/max値を取り出す。（セクション分けをしたのだから、重みづけして、幅を持たせた全部の点でフィットするとかもありか？？）
    # 横軸はcurrent
    # Get the band structure data as min or max position in mag2
    band_min = [[] for i in range(len(band))]
    cur_p = [[] for i in range(len(band))]
    for k in range(len(band)):
        for i in range(len(mag2)):
            if sum(band[k][i]) > 0:
                band_min[k].append(freq[np.argmax(band[k][i])])
                cur_p[k].append(current[i])

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(band)):
        ax.plot(cur_p[i], band_min[i], ".", label=str(i))
    ax.legend(loc="upper right")
    plt.savefig("peak.png", bbox_inches="tight", pad_inches=0.5, dpi=500)
    plt.show()

    return cur_p, band_min
