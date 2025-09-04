from lspattern.blocks import InitPlus
from lspattern.template.base import RotatedPlanarTemplate
from lspattern.canvas2 import RHGCanvas2
from lspattern.mytype import PatchCoordGlobal3D
from lspattern.visualizers.temporallayer import visualize_temporal_layer

# Build block and canvas
canvas = RHGCanvas2("Memory X")
tmpl = RotatedPlanarTemplate(d=3, kind=("X","X","Z"))
_ = tmpl.to_tiling()
blk = InitPlus(d=3, kind=("X","X","Z"), template=tmpl)
canvas.add_block(PatchCoordGlobal3D((0,0,0)), blk)

layers = canvas.to_temporal_layers()
layer0 = layers[0]

# Visualize (headless)
fig, ax = visualize_temporal_layer(layer0, show=False)
print('OK', len(layer0.node2coord))
