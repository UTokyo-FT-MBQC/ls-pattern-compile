from __future__ import annotations

from lspattern.tiling.base import ConnectedTiling


def plot_connected_tiling(
    ct: ConnectedTiling,
    ax=None,
    *,
    show: bool = True,
    title: str | None = None,
):
    """2D 散布図で ConnectedTiling を簡易可視化する。

    - 色分け: data=白, X=緑, Z=青
    - 軸ラベル/方眼/凡例を付加
    - `ax` 未指定なら新規に作成
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib が必要です。`pip install matplotlib` を実行してください") from e

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = ax.figure

    # Get coordinates (tuple[int,int])
    data = list(getattr(ct, "data_coords", []) or [])
    xs = list(getattr(ct, "x_coords", []) or [])
    zs = list(getattr(ct, "z_coords", []) or [])

    def _split_xy(points: list[tuple[int, int]]):
        if not points:
            return [], []
        px, py = zip(*points, strict=False)
        return list(px), list(py)

    dx, dy = _split_xy(data)
    xx, xy = _split_xy(xs)
    zx, zy = _split_xy(zs)

    # 点描画
    if dx:
        ax.scatter(dx, dy, s=28, c="white", edgecolors="black", label="data", zorder=3)
    if xx:
        ax.scatter(xx, xy, s=26, c="#2ecc71", edgecolors="#1e8449", label="X", zorder=3)
    if zx:
        ax.scatter(zx, zy, s=26, c="#3498db", edgecolors="#1f618d", label="Z", zorder=3)

    # 体裁
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if title:
        ax.set_title(title)
    ax.legend(loc="best")

    if show:
        fig.tight_layout()
        plt.show()
    return ax


def plot_layer_tiling(layer, *, anchor: str = "inner", show: bool = True, title: str | None = None):
    """TemporalLayer のブロック/パイプを 2D タイルに再構成して表示する。

    - `layer.get_connected_tiling(anchor)` を用いて ConnectedTiling を得る
    - `plot_connected_tiling` で描画
    """
    ct = layer.get_connected_tiling(anchor=anchor)
    if title is None:
        title = f"TemporalLayer z={getattr(layer, 'z', '?')} (anchor={anchor})"
    return plot_connected_tiling(ct, ax=None, show=show, title=title)
