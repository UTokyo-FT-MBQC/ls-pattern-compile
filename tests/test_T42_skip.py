import pytest


@pytest.mark.skip(reason="T42 debug script depended on a gating env toggle now removed (T46/design).")
def test_T42_gating_toggle_deprecated() -> None:
    # Placeholder to document that T42's ON/OFF env toggle semantics are obsolete.
    # The production code always gates using allowed pairs; behavior verified in T46 tests.
    pass
