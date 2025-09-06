"""Local shim of typing_extensions for repo-local execution.

This minimal shim provides only the `override` decorator used by
src/graphix_zx. If the real `typing_extensions` package is installed,
it will shadow this file; otherwise this keeps local examples working.
"""

from __future__ import annotations

from typing import TypeVar, Callable, Any

F = TypeVar("F", bound=Callable[..., Any])


def override(func: F) -> F:  # type: ignore[reportUnusedFunction]
    return func

