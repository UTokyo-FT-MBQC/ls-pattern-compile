Id: 1
Title: Add debug script for coord2node verification
Date: 2025-11-13
Author: gpt-5-codex
Description: Added a debug utility that compiles the CNOT mockup canvas and prints both the global graph summary and coord2node mapping to support verification of the recent coord2node merge fix. Documented the request as a local issue so future debugging steps remain traceable and reproducible.
Affected Files:
- .local/
  - changelog.md
  - issues/
    - 1-print-global-graph-and-coord2node.md
  - test/
    - debug_coord2node.py
Future tasks if any: []
Tags: debugging, tooling
---
Id: 2
Title: Export CNOT debug data to txt files
Date: 2025-11-13
Author: gpt-5-codex
Description: Extended the coord2node debug utility with CLI options that persist the global graph's physical nodes and the coord2node mapping to committed text files for mobile review. Generated the artifacts and documented the request as issue 2 to keep the workflow traceable.
Affected Files:
- .local/
  - changelog.md
  - issues/
    - closed/
      - 2-generate-cnot-debug-txt-files.md
  - test/
    - cnot_coord2node.txt
    - cnot_physical_nodes.txt
    - debug_coord2node.py
Future tasks if any: []
Tags: debugging, tooling
---
