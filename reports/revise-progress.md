# Transfer pass progress — 2026-10-21

Orchestrator checklist (reviews/2026-10-21-transfer.md). Status: todo · doing · done · partly · skipped + one line.
Resumed runs continue from the first unfinished phase.

## Phases
- Phase 0 · Callout system (N-034, Opus): **done** — all 11 sections + setup_check converted; lint 0 problems; `just check` green (run 2)
- Phase 1 · Transfer T-01…T-12 across all sections: **done** — brief in reports/phase1-brief.md; owners A 00+01+setup_check · B 02+03 · C 04+05 · D 06+07 · E 08+09+10 (Sonnet). Lint 0, smoke 11 passed. Net +1400 source lines (new visuals) → cuts reviewer
- Phase 2 · Bugs #16 (Drive cache) and #17 (ASK_UPLOAD): **done** — #16 owner A (setup_check, 00) + orchestrator 3-line SAM 3 loader change in 07; #17 owner E (09). `just check` green; tests/test_setup_check.py + test_structure.py: 2 passed (plain-file folders load in timm/transformers); Drive branch of 00 needs a real-Colab run
- Phase 3 · Three reviewers (facilitator, participant, cuts): **doing** — launched in parallel
- Phase 4 · Apply non-removal suggestions, write cut-proposals.md: todo
- Phase 5 · Report reports/revise-2026-10-21.md: todo

## T-items
| Item | Status | Note |
|---|---|---|
| T-01 Cut boxes and banner | done | Phase 0: callout markers → hidden form cells with saved HTML output |
| T-02 Collapse run-only cells | todo | |
| T-03 Objective / Takeaway colours, When → do | todo | |
| T-04 No walls of text | todo | |
| T-05 Show before asking | todo | |
| T-06 Concrete example per theory point | todo | |
| T-07 Metrics defined + shown min/median/max | todo | |
| T-08 Visual + formula for mechanisms | todo | |
| T-09 Cryptic sentences, simulations, violins, depth secondary | todo | |
| T-10 Licence text (open question, L-004) | todo | |
| T-11 Hidden reference solutions | todo | |
| T-12 Length and interaction | todo | |
| Bug #16 Drive cache survives new session | done | snapshot_download(local_dir=<cache>/models/org--name) + .fm-complete.json marker; 00/07 load from folder else hub id; flush_and_unmount at end (gh blocked: worked from summary) |
| Bug #17 Upload opt-in (ASK_UPLOAD) | done | ASK_UPLOAD=False @param; files.upload() only when ticked in Colab |

## Notes
- Run 2 (Opus orchestrator, resumed): build.py callouts were done; lint helpers existed but were not wired — wired them
  (N-003 accepts objective callout, table-box FAIL, setup_check.py checked). SPEC L-001/L-003 updated. 01–05, 07, 08
  converted in run 1; run 2 converts 00+setup_check, 06, 09+10 (06 had lost its Objective line mid-run).
- Baseline `just check` started at the beginning of the run (build/check-baseline.log).
