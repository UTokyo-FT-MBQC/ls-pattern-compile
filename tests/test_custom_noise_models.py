"""Tests for custom graphqomb noise model support."""

from __future__ import annotations

from pathlib import Path

import pytest
from graphqomb.noise_model import NoiseModel, PrepareEvent, RawStimOp

from lspattern.canvas_loader import load_canvas
from lspattern.compiler import compile_canvas_to_stim

_MEMORY_CANVAS_PATH = Path(__file__).resolve().parent.parent / "examples" / "memory_canvas.yml"


class _PrepareXErrorNoise(NoiseModel):  # type: ignore[misc]
    def __init__(self, p: float) -> None:
        self._p = p

    def on_prepare(self, event: PrepareEvent) -> tuple[RawStimOp]:
        return (RawStimOp(f"X_ERROR({self._p}) {event.node.id}"),)


def test_compile_canvas_to_stim_accepts_custom_noise_models() -> None:
    canvas, _ = load_canvas(_MEMORY_CANVAS_PATH, code_distance=3)

    circuit_str = compile_canvas_to_stim(
        canvas,
        noise_models=[_PrepareXErrorNoise(0.123)],
    )

    assert "X_ERROR(0.123)" in circuit_str


def test_compile_canvas_to_stim_rejects_mixed_legacy_and_noise_models() -> None:
    canvas, _ = load_canvas(_MEMORY_CANVAS_PATH, code_distance=3)

    with pytest.raises(ValueError, match="cannot be used together with noise_models"):
        compile_canvas_to_stim(
            canvas,
            p_before_meas_flip=0.001,
            noise_models=[_PrepareXErrorNoise(0.123)],
        )
