from __future__ import annotations

"""
ConnectedTiling の最小テストユーティリティ。

pytest 等に依存せず、`python -m lspattern.tiling.base_test` で
自己検証できる簡易テストを提供する。

検証内容:
- 非衝突マージ（data/X/Z の単純結合・重複除去）
- 同種内重複検出（X座標の重複）
- 異種間重なり検出（data と X の重なり）
- RotatedPlanarTemplate を用いた実地マージ（左右結合の件）
"""

from collections.abc import Iterable

from lspattern.tiling.base import ConnectedTiling, Tiling
from lspattern.tiling.template import RotatedPlanarTemplate, merge_pair_spatial


def _mk_tiling(
    *,
    data: Iterable[tuple[int, int]] = (),
    xs: Iterable[tuple[int, int]] = (),
    zs: Iterable[tuple[int, int]] = (),
) -> Tiling:
    data_l = list(data)
    x_l = list(xs)
    z_l = list(zs)
    return Tiling(
        data_coords=data_l,
        x_coords=x_l,
        z_coords=z_l,
    )


def test_no_collision_merge() -> None:
    a = _mk_tiling(data=[(0, 0), (2, 0)], xs=[(1, 1)], zs=[(3, 1)])
    b = _mk_tiling(data=[(0, 2), (2, 2)], xs=[(1, 3)], zs=[(3, 3)])
    ct = ConnectedTiling([a, b], check_collisions=True)

    assert set(ct.data_coords) == {(0, 0), (2, 0), (0, 2), (2, 2)}
    assert set(ct.x_coords) == {(1, 1), (1, 3)}
    assert set(ct.z_coords) == {(3, 1), (3, 3)}
    # coord2qubitindex は全座標（data+X+Z）をキーに持つ
    assert len(ct.coord2qubitindex) == (
        len(ct.data_coords) + len(ct.x_coords) + len(ct.z_coords)
    )


def test_within_type_duplicate_detected() -> None:
    a = _mk_tiling(xs=[(1, 1)])
    b = _mk_tiling(xs=[(1, 1)])  # 同一 X 座標が重複
    try:
        _ = ConnectedTiling([a, b], check_collisions=True)
        raise AssertionError("expected collision was not raised")
    except ValueError as e:
        msg = str(e)
        assert "duplicate X coords" in msg


def test_across_type_overlap_detected() -> None:
    a = _mk_tiling(data=[(0, 0)])
    b = _mk_tiling(xs=[(0, 0)])  # data と X が重なる
    try:
        _ = ConnectedTiling([a, b], check_collisions=True)
        raise AssertionError("expected overlap was not raised")
    except ValueError as e:
        msg = str(e)
        assert "data/X overlap" in msg


def test_pair_merge_with_templates() -> None:
    edgespec = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    d = 3

    # 2つのテンプレートを用意し、to_tiling() で 2D 座標を作る
    t1 = RotatedPlanarTemplate(d=d, edgespec=edgespec)
    t1.to_tiling()
    t2 = RotatedPlanarTemplate(d=d, edgespec=edgespec)
    t2.to_tiling()

    # 右方向に隣接させる（境界は自動 trim、オフセットは 2*d）
    connected = merge_pair_spatial(t1, t2, direction="X+")
    # merge_pair_spatial の返り値は ConnectedTiling
    assert isinstance(connected, ConnectedTiling)

    # data は d*d + d*d 個（重複がなければ）
    assert len(connected.data_coords) == d * d * 2
    # X/Z も双方の分が（境界 trim 分は減少しうるが）少なくとも片側相当は残る
    assert len(connected.x_coords) > 0
    assert len(connected.z_coords) > 0


def run_all() -> None:
    test_no_collision_merge()
    test_within_type_duplicate_detected()
    test_across_type_overlap_detected()
    test_pair_merge_with_templates()
    print("ConnectedTiling tests: OK")


if __name__ == "__main__":
    run_all()
