# Facilitation guide — When Foundation Models Fall Short: From Off-the-Shelf AI to Problem-Specific Adaptation

For whoever presents this notebook. It is generated from `SPEC.md` (N-033); regenerate it when the SPEC changes.

**Session:** <venue / course> · <total minutes> · <blocks and breaks>
**Before the session:** <setup check sent? data cached? room/projector?>
**Core principle:** predict → run → reflect. Ask learners to say or write what they expect before each interactive cell.

## Section-by-section

| Section | You present (what to say) | They do | Pause / interaction | Min |
|---|---|---|---|---|
| 0 · Setup & the question | Read the question on the banner aloud: "what can modern AI solve out of the box, when do its limitations start to matter, and what do we do next?" Point at the three outcomes and the level legend (🟡 is the default path, 🔴 optional). Say that nothing downloads in the room: `setup_check.ipynb` filled their Drive last week. Explain `use()` in one sentence: "you write the code, the notebook checks it against ours and continues either way, so nothing you skip can break it." | Choose the T4 runtime, run the setup cell (Drive pop-up: click Allow), the theme cell, the data cell and the model cell. Look at the two tasks side by side and write their prediction (which task will the foundation model handle?). Q1 🟢 in their head. | **Show of hands** after the data figure: "who thinks the model will manage the specialised task out of the box?" Count it, you will come back to it in section 2. **Where they get stuck:** no `fm-workshop-cache` folder → they skipped the setup check, point them to it (5 min, they can catch up during section 1); "no GPU" message → Runtime → Change runtime type → T4 and re-run the first cell. | 5 |

## Where people get stuck
<!-- Known failure points, with the one-line fix to give them. -->

## If you are short on time
<!-- Which sections or 🔴 parts to skip, in order. -->
