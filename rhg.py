from graphix_zx.common import Plane, PlannerMeasBasis
from graphix_zx.graphstate import GraphState

allowed_parities = [(0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 0), (0, 0, 1), (1, 1, 1)]

# ────────────────────────────────────────────────────────────────────────────
def create_rhg(
    Lx: int,
    Ly: int,
    Lz: int,
    allowed_parities: list[tuple[int, int, int]] = allowed_parities
) -> tuple[GraphState, dict[tuple[int,int,int], int]]:
    """
    Places a node only if the parity pattern (x % 2, y % 2, z % 2) of the integer coordinates (x, y, z)
    is included in `allowed_parities`, and returns the corresponding GraphState and a coordinate-to-node-index mapping.

    Returns:
    - graphstate: GraphState
        RHG graphstate
    - coord2node: dict[tuple[int,int,int], int]
        { (x, y, z): node_index }
    """

    gs = GraphState()
    coord2node: dict[tuple[int,int,int], int] = {}

    for x in range(Lx):
        for y in range(Ly):
            for z in range(Lz):
                parity = (x % 2, y % 2, z % 2)
                if parity not in allowed_parities:
                    continue

                node_idx = gs.add_physical_node()
                coord2node[(x, y, z)] = node_idx

                if (parity[0] + parity[1] + parity[2]) % 2 == 0:
                    mb = PlannerMeasBasis(Plane.XY, 0.0)  # primal
                else:
                    mb = PlannerMeasBasis(Plane.YZ, 0.0)  # dual
                gs.assign_meas_basis(node_idx, mb)

    # add edges
    for (x, y, z), u in coord2node.items():
        for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            nx, ny, nz = x + dx, y + dy, z + dz
            if (nx, ny, nz) in coord2node:
                v = coord2node[(nx, ny, nz)]
                try:
                    gs.add_physical_edge(u, v)
                except ValueError:
                    pass  

    return gs, coord2node


"""
Usage

Lx, Ly, Lz = 6, 6, 6

lattice_state, coord2node = create_raussendorf_by_parity(Lx, Ly, Lz)


node2coord: dict[int, tuple[int,int,int]] = {node: coord for coord, node in coord2node.items()}

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect((1,1,1))  # アスペクト比を等しくする
ax.grid(False)
ax.set_axis_off()

xs, ys, zs = [], [], []
colors = []
for node, (x,y,z) in node2coord.items():
    xs.append(x)
    ys.append(y)
    zs.append(z)

    parity = (x % 2, y % 2, z % 2)
    if parity in allowed_parities[:3]:
        colors.append('white') 
    else:
        colors.append('red') 

ax.scatter(xs, ys, zs, c=colors, edgecolors='black', s=50, depthshade=True, label='nodes')

for (u, v) in lattice_state.physical_edges:
    # node2coord から座標を取り出し
    x1, y1, z1 = node2coord[u]
    x2, y2, z2 = node2coord[v]
    ax.plot([x1, x2], [y1, y2], [z1, z2], c='gray', linewidth=1, alpha=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Raussendorf lattice (allowed parity nodes)')

plt.legend()
plt.tight_layout()
plt.show()


"""