# Versions

One folder per version of the notebook: `SPEC.md` and the built notebook exactly as they were.
The newest folder is always the notebook on `main`. Make one with `just version "what changed"`
before merging to main; `just version-check` fails if `main` has moved on without one.
Never edit a version folder.

| Version | Date | What changed | Commit |
|---|---|---|---|
| v01 | 2026-10-20 | Night run 2026-10-19: all issues built (PR #15) | 1bcac93 |
| v02 | 2026-10-21 | Transfer pass: your EDA review applied here, bugs #16 #17 fixed (PR #20) | 5c7c9f9 |
| v03 | 2026-10-26 | rules sync N-050–N-052 + lint checks (PR #22) | d391250 |
| v04 | 2026-10-26 | cuts 2026-10-26 (round 1 + 2 decisions) | c990866 |
| v05 | 2026-10-26 | reorders: Q12, Q16, Q17 before their cells (N-052) | 2893a93 |
