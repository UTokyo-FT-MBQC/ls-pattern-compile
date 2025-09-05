"""Base classes for cube-shaped RHG blocks."""

from lspattern.blocks.base import RHGBlock, RHGBlockSkeleton


class RHGCubeSkeleton(RHGBlockSkeleton):
    """Skeleton for a cube-shaped RHG block."""


class RHGCube(RHGBlock):
    """A cube-shaped RHG block."""
