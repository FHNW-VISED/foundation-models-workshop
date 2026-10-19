# %% [markdown]
# # When Foundation Models Fall Short: From Off-the-Shelf AI to Problem-Specific Adaptation
#
# **Objective:** get the notebook running and know what you will be able to do at the end.
#
# At the end of this notebook you will be able to:
# 1. …
# 2. …
# 3. …
#
# Questions are marked by level: 🟢 beginner · 🟡 intermediate · 🔴 advanced (optional).

# %% [markdown]
# ## 0. Setup
#
# Run this cell once. It installs what Colab does not already have and detects where you are running.

# %%
# @title Setup (run me first) { display-mode: "form" }
import os
import sys

IN_COLAB = "google.colab" in sys.modules
SMOKE = os.environ.get("NB_SMOKE") == "1"  # automated tests: tiny data, CPU, seconds — never set this by hand
SEED = 42

if IN_COLAB:
    pass  # e.g. %pip install -q <package-not-preinstalled-in-colab>

print(f"Colab: {IN_COLAB} · smoke mode: {SMOKE}")
