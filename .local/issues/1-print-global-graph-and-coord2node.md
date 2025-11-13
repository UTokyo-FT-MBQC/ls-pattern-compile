Id: 1
Status: open
Title: Print global graph and coord2node for CNOT mockup
Author: user
Date: 2025-11-13
Labels: debugging, verification

## Summary
The CNOT mockup canvas should expose its compiled global graph structure and coord2node mapping so we can manually verify that node identifiers are unique and incrementing. Add a lightweight script or tooling to print these data structures for inspection.

## Motivation
Recent fixes to the coord2node merge logic need additional verification. Having an easy way to print the compiled graph data lets us confirm there are no duplicate node identifiers or missing coordinates.

## Steps to Reproduce
1. Compile the CNOT mockup canvas via the example script.
2. Observe that there is no straightforward way to print the compiled global graph or coord2node mapping.

## Expected Behavior
A debug utility should compile the same canvas and print the global graph summary together with the coord2node mapping, ideally ordered by node id for clarity.

## Actual Behavior
No dedicated helper exists to emit these structures, making manual inspection tedious.

## Screenshots/Logs
_Not provided._

## Environment
- OS: container runtime
- Python: 3.12
- Commit: current `work` branch

## Acceptance Criteria
- [x] Running a documented command prints the global graph and coord2node contents for the CNOT mockup canvas.
- [x] Output makes it easy to verify node ids.

## Checklist
- [x] Implementation created
- [x] Tests or scripts demonstrated
- [x] Documentation/notes added if needed

## References
- User request in ongoing review
