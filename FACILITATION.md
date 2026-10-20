# Facilitation guide — When Foundation Models Fall Short: From Off-the-Shelf AI to Problem-Specific Adaptation

For whoever presents this notebook. It is generated from `SPEC.md` (N-033); regenerate it when the SPEC changes.

**Session:** <venue / course> · <total minutes> · <blocks and breaks>
**Before the session:** <setup check sent? data cached? room/projector?>
**Core principle:** predict → run → reflect. Ask learners to say or write what they expect before each interactive cell.

## Section-by-section

| Section | You present (what to say) | They do | Pause / interaction | Min |
|---|---|---|---|---|
| 0 · Setup & the question | Read the question on the banner aloud: "what can modern AI solve out of the box, when do its limitations start to matter, and what do we do next?" Point at the three outcomes and the level legend (🟡 is the default path, 🔴 optional). Say that nothing downloads in the room: `setup_check.ipynb` filled their Drive last week. Explain `use()` in one sentence: "you write the code, the notebook checks it against ours and continues either way, so nothing you skip can break it." | Choose the T4 runtime, run the setup cell (Drive pop-up: click Allow), the theme cell, the data cell and the model cell. Look at the two tasks side by side and write their prediction (which task will the foundation model handle?). Q1 🟢 in their head. | **Show of hands** after the data figure: "who thinks the model will manage the specialised task out of the box?" Count it, you will come back to it in section 2. **Where they get stuck:** no `fm-workshop-cache` folder → they skipped the setup check, point them to it (5 min, they can catch up during section 1); "no GPU" message → Runtime → Change runtime type → T4 and re-run the first cell. | 5 |
| 1 · Out of the box: pets | One sentence on each tool: "SAM segments what you point at, no training; DINOv3 describes every 16 px patch; a linear head on those features learns a class from five masks in seconds." Say that the three numbers from this section come back in section 2 on the specialised task, so they should note them. On the SAM figure, point out the part-vs-whole ambiguity when it appears (one click on a leg can give the leg) and the box fixing it: that is normal SAM behaviour, not a bug. On the PCA figure: "colours are not classes, same colour = the model describes these patches alike; see the 16 px blocks." | Predict, then run the one-click + box cell (Q2 🟡, try other indices). Optionally fill `my_iou` (✍️, 2 min, `use()` checks it). Run the zero-shot scoring loop, the PCA cell (Q3 🟡), predict a number for the head, train it (seconds; `TRAIN_YOURSELF = False` loads the checkpoint), score it (Q4 🟡). | **Show of hands** before the head cell: "who expects the five-pet head to beat one click?" Note the count, then show the two numbers. **👥 optional, 3 min:** compare the demo pet with a neighbour: did one click fail on either? **Where they get stuck:** the Q2 index out of range on a CPU session (only 12 eval pets); "no precomputed head" → they skipped the setup check. **If short on time:** skip `my_iou` (the reference is used) and the 👥 moment; keep Q2 and the head's number. | 15 |

## Where people get stuck
<!-- Known failure points, with the one-line fix to give them. -->

## If you are short on time
<!-- Which sections or 🔴 parts to skip, in order. -->
