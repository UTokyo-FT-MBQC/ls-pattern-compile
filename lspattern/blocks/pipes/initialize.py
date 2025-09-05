from __future__ import annotations

from dataclasses import dataclass

from graphix_zx.common import Plane, PlannerMeasBasis
from graphix_zx.graphstate import GraphState

from lspattern.consts.consts import DIRECTIONS3D, PIPEDIRECTION
from lspattern.mytype import (
    FlowLocal,
    NodeIdLocal,
    NodeSetLocal,
    PatchCoordGlobal3D,
    PhysCoordLocal2D,
    PhysCoordLocal3D,
    ScheduleTuplesLocal,
    SpatialEdgeSpec,
)
from lspattern.tiling.template import RotatedPlanarPipetemplate
from lspattern.utils import get_direction

from .base import RHGPipe, RHGPipeSkeleton


@dataclass
class InitPlusPipeSkeleton(RHGPipeSkeleton):
    """InitPlus 相当の Pipe Skeleton。

    edgespec を省略した場合、方向に応じた既定値を用いる：
        - RIGHT/LEFT（水平）: {TOP:'O', BOTTOM:'O', LEFT:'X', RIGHT:'Z'}
        - TOP/BOTTOM（垂直）: {LEFT:'O', RIGHT:'O', TOP:'X', BOTTOM:'Z'}
    """

    edgespec: SpatialEdgeSpec | None = None

    def to_pipe(self, source: PatchCoordGlobal3D, sink: PatchCoordGlobal3D) -> InitPlusPipe:
        direction = get_direction(source, sink)
        spec = self.edgespec
        if spec is None:
            if direction in (PIPEDIRECTION.RIGHT, PIPEDIRECTION.LEFT):
                spec = {"TOP": "O", "BOTTOM": "O", "LEFT": "X", "RIGHT": "Z"}
            elif direction in (PIPEDIRECTION.TOP, PIPEDIRECTION.BOTTOM):
                spec = {"LEFT": "O", "RIGHT": "O", "TOP": "X", "BOTTOM": "Z"}
            else:
                # 時間方向は未対応
                raise NotImplementedError("Temporal pipe (UP/DOWN) is not supported")
        return InitPlusPipe(d=self.d, edgespec=spec, direction=direction)


class InitPlusPipe(RHGPipe):
    def __init__(
        self,
        d: int,
        edgespec: SpatialEdgeSpec,
        direction: PIPEDIRECTION,
        **kwargs,
    ):
        super().__init__(d=d, edgespec=edgespec, direction=direction, **kwargs)
        self.template = RotatedPlanarPipetemplate(d=d, edgespec=edgespec)
        self.materialize()

    def materialize(self) -> None:
        # 既に materialize 済みならスキップ
        if getattr(self, "graph_local", None) and getattr(self, "node2coord", None):
            if self.graph_local is not None and self.node2coord:
                return

        # 2D テンプレートを取得
        tiling = self.template.to_tiling() if self.template else {"data": [], "X": [], "Z": []}
        data_xy = tiling.get("data", [])
        x_xy = tiling.get("X", [])
        z_xy = tiling.get("Z", [])

        g = GraphState()
        coord2node: dict[PhysCoordLocal3D, NodeIdLocal] = {}
        node2coord: dict[NodeIdLocal, PhysCoordLocal3D] = {}
        node2role: dict[NodeIdLocal, str] = {}

        # z スライス 0..2d
        max_t = 2 * getattr(self, "d", 0)
        nodes_by_z: dict[int, dict[PhysCoordLocal2D, NodeIdLocal]] = {}
        for z in range(max_t + 1):
            cur: dict[PhysCoordLocal2D, NodeIdLocal] = {}

            # data は全スライス
            for x, y in data_xy:
                n = g.add_physical_node()
                if z != max_t:
                    g.assign_meas_basis(n, PlannerMeasBasis(Plane.XY, 0.0))
                coord = (x, y, z)
                coord2node[coord] = n
                node2coord[n] = coord
                node2role[n] = "data"
                cur[x, y] = n

            # ancilla は最終スライス以外、偶数: X / 奇数: Z
            if z != max_t:
                if z % 2 == 0:
                    for x, y in x_xy:
                        n = g.add_physical_node()
                        g.assign_meas_basis(n, PlannerMeasBasis(Plane.XY, 0.0))
                        coord = (x, y, z)
                        coord2node[coord] = n
                        node2coord[n] = coord
                        node2role[n] = "ancilla_x"
                        cur[x, y] = n
                else:
                    for x, y in z_xy:
                        n = g.add_physical_node()
                        g.assign_meas_basis(n, PlannerMeasBasis(Plane.XY, 0.0))
                        coord = (x, y, z)
                        coord2node[coord] = n
                        node2coord[n] = coord
                        node2role[n] = "ancilla_z"
                        cur[x, y] = n

            nodes_by_z[z] = cur

        # 同一スライス内の斜め隣接（xy 平面）
        for z, cur in nodes_by_z.items():
            for (x, y), src in cur.items():
                for dx, dy, dz in DIRECTIONS3D:
                    if dz != 0:
                        continue
                    tgt = cur.get((x + dx, y + dy))
                    if tgt is not None and tgt > src:
                        g.add_physical_edge(src, tgt)

        # 縦方向（時間）エッジと最小 X-flow
        flow_local: FlowLocal = {}
        for z in range(1, max_t + 1):
            cur = nodes_by_z[z]
            prev = nodes_by_z[z - 1]
            for xy, u in cur.items():
                v = prev.get(xy)
                if v is not None:
                    g.add_physical_edge(u, v)
                    flow_local[v] = {u}

        # パリティ（同一種 ancilla の z 間隔2）
        x_checks: list[NodeSetLocal] = []
        z_checks: list[NodeSetLocal] = []
        for n, (x, y, z) in node2coord.items():
            role = node2role[n]
            if role == "ancilla_x":
                nxt = coord2node.get((x, y, z + 2))
                if nxt is not None and node2role.get(nxt) == "ancilla_x":
                    x_checks.append({n, nxt})
            elif role == "ancilla_z":
                nxt = coord2node.get((x, y, z + 2))
                if nxt is not None and node2role.get(nxt) == "ancilla_z":
                    z_checks.append({n, nxt})

        # スケジュール: 偶数タイムスロット=ancilla, 奇数=data（最終は除く）
        schedule_local: ScheduleTuplesLocal = []
        for z, cur in nodes_by_z.items():
            anc: set[int] = set()
            dat: set[int] = set()
            for n in cur.values():
                r = node2role[n]
                if r.startswith("ancilla"):
                    anc.add(n)
                elif r == "data":
                    dat.add(n)
            if anc:
                schedule_local.append((2 * z, anc))
            if z != max_t and dat:
                schedule_local.append((2 * z + 1, dat))

        # seam 上の data 端点を in/out に割り当て
        in_ports: NodeSetLocal = set()
        out_ports: NodeSetLocal = set()
        if data_xy:
            xs = [x for x, _ in data_xy]
            ys = [y for _, y in data_xy]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            if self.direction in (PIPEDIRECTION.RIGHT, PIPEDIRECTION.LEFT):
                left_x, right_x = x_min, x_max
                in_x = left_x if self.direction == PIPEDIRECTION.RIGHT else right_x
                out_x = right_x if self.direction == PIPEDIRECTION.RIGHT else left_x
                for (x, y, z), n in coord2node.items():
                    if node2role[n] == "data":
                        if x == in_x:
                            in_ports.add(n)
                        elif x == out_x:
                            out_ports.add(n)
            elif self.direction in (PIPEDIRECTION.TOP, PIPEDIRECTION.BOTTOM):
                bot_y, top_y = y_min, y_max
                in_y = bot_y if self.direction == PIPEDIRECTION.TOP else top_y
                out_y = top_y if self.direction == PIPEDIRECTION.TOP else bot_y
                for (x, y, z), n in coord2node.items():
                    if node2role[n] == "data":
                        if y == in_y:
                            in_ports.add(n)
                        elif y == out_y:
                            out_ports.add(n)

        # finalize
        self.graph_local = g
        self.node2coord = node2coord
        self.coord2node = coord2node
        self.node2role = node2role
        self.schedule_local = schedule_local
        self.flow_local = flow_local
        self.x_checks = x_checks
        self.z_checks = z_checks
        self.in_ports = in_ports
        self.out_ports = out_ports
