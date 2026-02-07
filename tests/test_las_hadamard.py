"""Unit tests for Hadamard detection and invert_ancilla_order in las.py."""

from __future__ import annotations

from lspattern.importer.las import (
    _color_kp_to_boundary,
    _compute_invert_ancilla_order,
    _is_hadamard_cube,
    _resolve_color_km,
    _resolve_color_kp,
)


class TestResolveColorKp:
    """Test _resolve_color_kp function."""

    def test_returns_direct_value_when_not_minus_one(self) -> None:
        """ColorKP = 0 or 1 should return directly."""
        color_kp = [[[0, 1, 0]]]
        assert _resolve_color_kp(0, 0, 0, color_kp, 1, 1, 3) == 0
        assert _resolve_color_kp(0, 0, 1, color_kp, 1, 1, 3) == 1
        assert _resolve_color_kp(0, 0, 2, color_kp, 1, 1, 3) == 0

    def test_follows_minus_one_to_previous_k(self) -> None:
        """ColorKP = -1 should refer to k-1."""
        color_kp = [[[1, -1, -1]]]
        assert _resolve_color_kp(0, 0, 0, color_kp, 1, 1, 3) == 1
        assert _resolve_color_kp(0, 0, 1, color_kp, 1, 1, 3) == 1  # follows k-1
        assert _resolve_color_kp(0, 0, 2, color_kp, 1, 1, 3) == 1  # follows k-1 twice

    def test_returns_zero_when_all_minus_one_from_k_zero(self) -> None:
        """When chaining -1 reaches k=0 with -1, return 0 as default.

        -1 at k=0 can't reference k-1, so it returns 0 (default).
        """
        color_kp = [[[-1, -1]]]
        # _get_color_kp returns -1, but _resolve checks != -1: False
        # Then k > 0 check: k=0 is not > 0, so return 0 (default)
        assert _resolve_color_kp(0, 0, 0, color_kp, 1, 1, 2) == 0
        assert _resolve_color_kp(0, 0, 1, color_kp, 1, 1, 2) == 0  # follows to k=0 which is 0


class TestResolveColorKm:
    """Test _resolve_color_km function."""

    def test_returns_direct_value_when_not_minus_one(self) -> None:
        """ColorKM = 0 or 1 should return directly."""
        color_km = [[[0, 1, 0]]]
        assert _resolve_color_km(0, 0, 0, color_km, 1, 1, 3) == 0
        assert _resolve_color_km(0, 0, 1, color_km, 1, 1, 3) == 1
        assert _resolve_color_km(0, 0, 2, color_km, 1, 1, 3) == 0

    def test_follows_minus_one_to_previous_k(self) -> None:
        """ColorKM = -1 should refer to k-1."""
        color_km = [[[1, -1, -1]]]
        assert _resolve_color_km(0, 0, 0, color_km, 1, 1, 3) == 1
        assert _resolve_color_km(0, 0, 1, color_km, 1, 1, 3) == 1  # follows k-1
        assert _resolve_color_km(0, 0, 2, color_km, 1, 1, 3) == 1  # follows k-1 twice


class TestIsHadamardCube:
    """Test _is_hadamard_cube function."""

    def test_k_zero_never_hadamard(self) -> None:
        """k=0 cubes can never be Hadamard."""
        color_km = [[[0]]]
        color_kp = [[[1]]]
        assert _is_hadamard_cube(0, 0, 0, color_km, color_kp, 1, 1, 1) is False

    def test_hadamard_when_kp_differs_from_km_below(self) -> None:
        """Hadamard when ColorKP[k] != ColorKM[k-1]."""
        # ColorKM[0] = 0, ColorKP[1] = 1 -> different -> Hadamard
        color_km = [[[0, 0]]]
        color_kp = [[[0, 1]]]
        assert _is_hadamard_cube(0, 0, 1, color_km, color_kp, 1, 1, 2) is True

    def test_no_hadamard_when_kp_equals_km_below(self) -> None:
        """No Hadamard when ColorKP[k] == ColorKM[k-1]."""
        # ColorKM[0] = 0, ColorKP[1] = 0 -> same -> no Hadamard
        color_km = [[[0, 0]]]
        color_kp = [[[0, 0]]]
        assert _is_hadamard_cube(0, 0, 1, color_km, color_kp, 1, 1, 2) is False

    def test_hadamard_with_resolved_minus_one(self) -> None:
        """Hadamard detection works with resolved -1 values."""
        # ColorKM[0] = 0, ColorKM[1] = -1 (resolves to 0)
        # ColorKP[1] = 1, ColorKP[2] = -1 (resolves to 1)
        # At k=2: ColorKP[2] resolved = 1, ColorKM[1] resolved = 0 -> different -> Hadamard
        color_km = [[[0, -1, 0]]]
        color_kp = [[[0, 1, -1]]]
        assert _is_hadamard_cube(0, 0, 2, color_km, color_kp, 1, 1, 3) is True


class TestComputeInvertAncillaOrder:
    """Test _compute_invert_ancilla_order function."""

    def test_no_hadamard_all_false(self) -> None:
        """When no Hadamard, all cubes have invert=False."""
        cubes = {(0, 0, 0), (0, 0, 1), (0, 0, 2)}
        # ColorKP[k] == ColorKM[k-1] for all k -> no Hadamard
        color_km = [[[0, 0, 0]]]
        color_kp = [[[0, 0, 0]]]
        hadamard_cubes, invert_map = _compute_invert_ancilla_order(
            cubes, color_km, color_kp, 1, 1, 3
        )
        assert hadamard_cubes == set()
        assert invert_map[0, 0, 0] is False
        assert invert_map[0, 0, 1] is False
        assert invert_map[0, 0, 2] is False

    def test_single_hadamard_inverts_after(self) -> None:
        """Single Hadamard at k=1 inverts cubes k>=1."""
        cubes = {(0, 0, 0), (0, 0, 1), (0, 0, 2)}
        # ColorKM[0] = 0, ColorKP[1] = 1 -> Hadamard at k=1
        # ColorKM[1] = 1, ColorKP[2] = 1 -> no Hadamard at k=2
        color_km = [[[0, 1, 1]]]
        color_kp = [[[0, 1, 1]]]
        hadamard_cubes, invert_map = _compute_invert_ancilla_order(
            cubes, color_km, color_kp, 1, 1, 3
        )
        assert hadamard_cubes == {(0, 0, 1)}
        assert invert_map[0, 0, 0] is False
        assert invert_map[0, 0, 1] is True  # Hadamard cube, inverted after
        assert invert_map[0, 0, 2] is True  # after Hadamard, still inverted

    def test_double_hadamard_toggles_back(self) -> None:
        """Two Hadamards toggle invert back to False."""
        cubes = {(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)}
        # ColorKM[0] = 0, ColorKP[1] = 1 -> Hadamard at k=1
        # ColorKM[1] = 1, ColorKP[2] = 1 -> no Hadamard at k=2
        # ColorKM[2] = 1, ColorKP[3] = 0 -> Hadamard at k=3
        color_km = [[[0, 1, 1, 0]]]
        color_kp = [[[0, 1, 1, 0]]]
        hadamard_cubes, invert_map = _compute_invert_ancilla_order(
            cubes, color_km, color_kp, 1, 1, 4
        )
        assert hadamard_cubes == {(0, 0, 1), (0, 0, 3)}
        assert invert_map[0, 0, 0] is False
        assert invert_map[0, 0, 1] is True  # first Hadamard
        assert invert_map[0, 0, 2] is True  # between Hadamards
        assert invert_map[0, 0, 3] is False  # second Hadamard toggles back

    def test_multiple_ij_columns_independent(self) -> None:
        """Different (i,j) columns have independent invert tracking."""
        cubes = {(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)}
        # ColorKM[0][0][0] = 0, ColorKP[0][0][1] = 1 -> Hadamard at (0,0,1)
        # ColorKM[1][0][0] = 0, ColorKP[1][0][1] = 0 -> no Hadamard at (1,0,1)
        color_km = [[[0, 1], [0, 0]], [[0, 0], [0, 0]]]
        color_kp = [[[0, 1], [0, 0]], [[0, 0], [0, 0]]]
        hadamard_cubes, invert_map = _compute_invert_ancilla_order(
            cubes, color_km, color_kp, 2, 2, 2
        )
        assert (0, 0, 1) in hadamard_cubes
        assert (1, 0, 1) not in hadamard_cubes
        assert invert_map[0, 0, 0] is False
        assert invert_map[0, 0, 1] is True
        assert invert_map[1, 0, 0] is False
        assert invert_map[1, 0, 1] is False


class TestColorKpToBoundary:
    """Test _color_kp_to_boundary with inverted parameter."""

    def test_normal_mapping(self) -> None:
        """Normal mapping: 0->ZZXX, 1->XXZZ."""
        assert _color_kp_to_boundary(0) == list("ZZXX")
        assert _color_kp_to_boundary(1) == list("XXZZ")
        assert _color_kp_to_boundary(-1) is None

    def test_inverted_mapping(self) -> None:
        """Inverted mapping: 0->XXZZ, 1->ZZXX."""
        assert _color_kp_to_boundary(0, inverted=True) == list("XXZZ")
        assert _color_kp_to_boundary(1, inverted=True) == list("ZZXX")
        assert _color_kp_to_boundary(-1, inverted=True) is None

    def test_explicit_false_same_as_default(self) -> None:
        """Explicit inverted=False is same as default."""
        assert _color_kp_to_boundary(0, inverted=False) == list("ZZXX")
        assert _color_kp_to_boundary(1, inverted=False) == list("XXZZ")
