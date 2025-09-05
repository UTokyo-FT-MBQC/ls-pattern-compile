from lspattern.tiling.template import RotatedPlanarPipetemplate, RotatedPlanarTemplate

spec_x = {"LEFT": "O", "RIGHT": "O", "TOP": "X", "BOTTOM": "Z"}
spec_y = {"TOP": "O", "BOTTOM": "O", "LEFT": "X", "RIGHT": "Z"}

px = RotatedPlanarPipetemplate(d=5, edgespec=spec_x)
py = RotatedPlanarPipetemplate(d=5, edgespec=spec_y)

print('Pipe-X lens:', {k: len(v) for k,v in px.to_tiling().items()})
print('Pipe-Y lens:', {k: len(v) for k,v in py.to_tiling().items()})

bt = RotatedPlanarTemplate(d=3, edgespec={"LEFT":"X","RIGHT":"Z","TOP":"X","BOTTOM":"Z"})
print('Block lens:', {k: len(v) for k,v in bt.to_tiling().items()})
