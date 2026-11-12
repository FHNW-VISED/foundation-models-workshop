# One entry point for humans, agents and CI. `just check` is the acceptance gate.

# Assemble sections/*.py into foundation-models-workshop.ipynb
build:
    uv run python scripts/build.py

# Rule checks that a script can decide + ruff on the section sources
lint:
    uv run python scripts/lint_notebook.py
    uv run ruff check --quiet sections scripts tests
    uv run pytest -q -p no:warnings tests/test_version.py tests/test_tool_*.py
    python3 scripts/rules_review.py --check

# Execute the whole notebook headless on tiny data (NB_SMOKE=1)
smoke: build
    uv run pytest -q -x -p no:warnings tests/test_smoke.py tests/test_content.py

# Lint + smoke: what the night run and CI require before a commit counts
check: lint smoke

# Every function/class defined anywhere in the notebook, with its file:line — read this instead of all sections
api:
    @grep -nE "^(def|class) |^    def " sections/*.py || true

# Open the notebook in Jupyter with sections paired (edit in the browser, sync back with `just sync`)
open: build
    uv run jupyter lab foundation-models-workshop.ipynb

# After editing foundation-models-workshop.ipynb by hand in Jupyter/Colab: split it back into sections (review the diff!)
sync:
    uv run python scripts/split.py

# Committed notebook must equal a fresh build (catches hand edits that skipped `just sync`)
drift: build
    git diff --exit-code -- foundation-models-workshop.ipynb

# Compare runtime deps with what Google Colab preinstalls (gpu runtime by default)
colab-pins runtime="gpu":
    uv run python scripts/colab_pins.py {{runtime}}

# Snapshot SPEC.md + the built notebooks as the next version in versions/ (do this before merging to main)
version label: build
    uv run python scripts/version.py new "{{label}}"

# The newest version in versions/ must equal SPEC.md and the built notebooks (true on main, always)
version-check: build
    uv run python scripts/version.py check

# Does the SAM 3 workshop copy download, load and segment? (~3.4 GB, temporary). anonymous="--anonymous" = as a participant
sam3-copy anonymous="":
    uv run python scripts/check_sam3_copy.py {{anonymous}}

# Start a phase handoff: reports/handoff.md with the six sections (fill it in, then check it)
handoff-new phase workflow model:
    python3 scripts/handoff.py new --phase {{phase}} --workflow {{workflow}} --model {{model}}

# Validate reports/handoff.md: no placeholders, no quotes, real paths, a start command
handoff-check:
    python3 scripts/handoff.py check

# Print the command and first sentence that start the next session, and what it will read
handoff-start:
    python3 scripts/handoff.py start

# Port her edited copy into sections/*.py: unambiguous edits are written, the rest is listed in reviews/<date>-her-copy/
port-her her base:
    python3 scripts/port_her.py --her {{her}} --base {{base}}

# Same, but write nothing: print what would change and what would be a conflict
port-her-dry her base:
    python3 scripts/port_her.py --her {{her}} --base {{base}} --dry-run

# Run one headless helper agent from inside this repo (Sonnet, turn cap always set): see scripts/helper.sh
helper name brief:
    bash scripts/helper.sh {{name}} {{brief}}

# The cells of one section: position k (as in the built cell id), md/code, bytes, flags, first line
cells nn:
    python3 scripts/sections_info.py cells {{nn}}

# Visible words per section and the total against the Word budget in SPEC.md
words:
    uv run python scripts/sections_info.py words

# Regenerate RULES-review.md (only the rules a script cannot check) after RULES.md changes
rules-review:
    python3 scripts/rules_review.py

# Compare this repo's scripts with the vault template (night.sh runs this first)
template-diff:
    python3 scripts/template_diff.py

# Prove every difference between her copy and this build is an approved item (reviews/<date>-her-copy/)
her-verify folder her:
    python3 scripts/verify_her.py --folder {{folder}} --her {{her}} --built foundation-models-workshop.ipynb
