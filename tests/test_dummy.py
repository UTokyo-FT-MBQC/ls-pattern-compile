from __future__ import annotations

import pytest


@pytest.mark.skip(
    reason="Placeholder test with no actual testing logic. "
    "This test was created in the initial commit (db5de9a 'dummy test') "
    "as a minimal test file to satisfy test infrastructure requirements. "
    "It only contains 'assert True' and provides no actual test coverage. "
    "Should be removed or replaced with meaningful tests if needed."
)
def test_dummy() -> None:
    assert True
