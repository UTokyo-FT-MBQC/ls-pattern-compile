Id: 2
Status: closed
Title: Generate CNOT debug txt files
Author: gpt-5-codex
Date: 2025-11-13
Labels: debugging, enhancement

## Summary
Produce standalone text exports of the CNOT mockup's compiled global graph physical node list and coord2node mapping so they can be reviewed on mobile devices.

## Motivation
The existing debug script prints the structures to stdout, which is inconvenient to review on an iPhone. Having committed `.txt` artifacts lets the requester inspect the data without running Python locally.

## Steps to Reproduce
1. Run `python .local/test/debug_coord2node.py`.
2. Observe that only console output is produced.

## Expected Behavior
Two readable `.txt` files containing the global graph physical nodes and the coord2node mapping are available in the repository.

## Actual Behavior
No persistent text files exist for these structures; they only print to stdout.

## Screenshots/Logs
N/A

## Environment
- Repository: ls-pattern-compile
- Python: 3.12
- Commit: current work branch

## Acceptance Criteria
- [x] Debug utility writes physical node and coord2node snapshots to deterministic `.txt` files.
- [x] The generated files are committed for remote/mobile inspection.
- [x] Existing coord2node regression tests continue to pass.

## Checklist
- [x] Implement feature
- [x] Update tests / add new tests if needed
- [x] Document in changelog

## References
- Request from user conversation thread
