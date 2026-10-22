# One entry point for humans, agents and CI. `just check` is the acceptance gate.

# Assemble sections/*.py into foundation-models-workshop.ipynb
build:
    uv run python scripts/build.py

# Rule checks that a script can decide + ruff on the section sources
lint:
    uv run python scripts/lint_notebook.py
    uv run ruff check --quiet sections scripts tests
    uv run pytest -q -p no:warnings tests/test_version.py

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
