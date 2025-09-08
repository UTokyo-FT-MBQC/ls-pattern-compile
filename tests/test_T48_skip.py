import pytest


@pytest.mark.skip(reason="T48 requires external CLIs (ruff/mypy) and network; skip in unit tests.")
def test_T48_external_linters_skipped():
    pass
