# Testing Guide

- Unit tests: `pytest -q`.
- Integration tests: `tests/integration/` validate end-to-end compilation and guard against regressions using circuit fingerprints.
- Slow tests: marked with `slow` and include quick error simulations via sinter/pymatching.

## Circuit Fingerprinting

Fingerprints are stored at `tests/integration/fixtures/circuit_fingerprints.json` as a map of test name to metadata and a SHA256 of the circuit text form.

To (re)generate goldens locally for d=3 examples:

```bash
source ./.venv/Scripts/activate
python .local/tests/task7_generate_golden.py
```

Then run integration tests:

```bash
pytest tests/integration -v
pytest tests/integration -v -m slow  # include simulations
```

