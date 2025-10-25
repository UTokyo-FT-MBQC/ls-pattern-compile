from __future__ import annotations

import dataclasses as _dc
import hashlib as _hashlib
import json as _json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    import stim


@_dc.dataclass(frozen=True)
class CircuitFingerprint:
    """Stable fingerprint and metadata for a Stim circuit.

    Fields are intentionally minimal and stable across runs for regression checks.
    """

    name: str
    sha256: str
    num_qubits: int
    num_detectors: int
    num_observables: int

    @staticmethod
    def from_circuit(name: str, circuit: stim.Circuit) -> CircuitFingerprint:
        """Create a fingerprint from a `stim.Circuit`.

        Parameters
        ----------
        name
            Identifier used as the registry key.
        circuit
            Stim circuit to fingerprint.
        """
        # Use the canonical text form for stable hashing
        text = str(circuit)
        digest = _hashlib.sha256(text.encode("utf-8")).hexdigest()
        return CircuitFingerprint(
            name=name,
            sha256=digest,
            num_qubits=int(circuit.num_qubits),
            num_detectors=int(circuit.num_detectors),
            num_observables=int(circuit.num_observables),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "num_qubits": self.num_qubits,
            "num_detectors": self.num_detectors,
            "num_observables": self.num_observables,
        }

    @staticmethod
    def from_dict(name: str, data: Mapping[str, Any]) -> CircuitFingerprint:
        return CircuitFingerprint(
            name=name,
            sha256=str(data["sha256"]),
            num_qubits=int(data["num_qubits"]),
            num_detectors=int(data["num_detectors"]),
            num_observables=int(data["num_observables"]),
        )


class FingerprintRegistry:
    """Manages loading, saving, and verifying circuit fingerprints.

    JSON schema stored on disk:

    {
      "<name>": {
        "sha256": "...",
        "num_qubits": 0,
        "num_detectors": 0,
        "num_observables": 0
      },
      ...
    }
    """

    def __init__(self, path: Path, auto_update: bool = False) -> None:
        self.path = Path(path)
        self._items: dict[str, CircuitFingerprint] = {}
        self.auto_update = auto_update
        self._updated = False

    def load(self) -> None:
        if not self.path.exists():
            self._items = {}
            return
        data = _json.loads(self.path.read_text(encoding="utf-8"))
        self._items = {name: CircuitFingerprint.from_dict(name, payload) for name, payload in dict(data).items()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {name: fp.to_dict() for name, fp in sorted(self._items.items())}
        self.path.write_text(_json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def set(self, fp: CircuitFingerprint) -> None:
        self._items[fp.name] = fp

    def get(self, name: str) -> CircuitFingerprint | None:
        return self._items.get(name)

    def verify(self, fp: CircuitFingerprint) -> tuple[bool, str | None]:
        """Verify a fingerprint against the registry.

        Returns (ok, error_message). When the name is missing, returns (False, ...).
        If auto_update is enabled, automatically updates mismatched fingerprints.
        """
        golden = self._items.get(fp.name)
        if golden is None:
            if self.auto_update:
                self.set(fp)
                self._updated = True
                return True, f"[AUTO-UPDATE] Added new fingerprint for {fp.name}"
            return False, f"Missing golden for {fp.name}"
        if golden.sha256 != fp.sha256:
            if self.auto_update:
                self.set(fp)
                self._updated = True
                return True, f"[AUTO-UPDATE] Updated fingerprint for {fp.name}: {golden.sha256} -> {fp.sha256}"
            return (
                False,
                f"Fingerprint mismatch for {fp.name}: expected {golden.sha256}, got {fp.sha256}",
            )
        # Also check metadata stability
        if (
            golden.num_qubits != fp.num_qubits
            or golden.num_detectors != fp.num_detectors
            or golden.num_observables != fp.num_observables
        ):
            msg = (
                f"Metadata changed for {fp.name} "
                f"(q:{golden.num_qubits}->{fp.num_qubits}, "
                f"d:{golden.num_detectors}->{fp.num_detectors}, "
                f"o:{golden.num_observables}->{fp.num_observables})"
            )
            if self.auto_update:
                self.set(fp)
                self._updated = True
                return True, f"[AUTO-UPDATE] {msg}"
            return False, msg
        return True, None

    def save_if_updated(self) -> bool:
        """Save the registry if it was updated during auto-update mode.

        Returns True if the registry was saved, False otherwise.
        """
        if self._updated:
            self.save()
            return True
        return False
