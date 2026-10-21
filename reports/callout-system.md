# Callout system (N-034) — the contract every agent follows in this pass

Why: Colab sets `td { white-space: nowrap; overflow: hidden }`, so every `<table bgcolor>` box and the banner are cut.
Boxes become the HTML **output** of a hidden form code cell, saved into the `.ipynb` at build time.

## Source shape (what you write in `sections/NN_*.py` and `setup_check.py`)

A callout is a **markdown cell** whose first line is a marker, followed by normal markdown:

```
# %% [markdown]
# <!-- callout: pitfall | The first slow step -->
# The first SAM call compiles kernels and takes **20–40 s** on a T4.
#
# - After that every call is well under a second.
# - On a CPU session it is slower; `TRAIN_YOURSELF = False` loads our checkpoint.
```

- One callout per cell. The marker is the **first line**; nothing before it. The cell holds nothing but the callout.
- `kind` is one of the kinds below (lower-case, hyphen). `title` is plain text, no markdown, no HTML.
  The build prepends the kind's icon. If the title itself starts with an emoji (e.g. `👥 Optional, 3 min`),
  that emoji replaces the kind's icon (used for the 👥 pair moments, which keep the `question` colour).
- Body: markdown (bold, lists, `code`, links, line breaks). Keep HTML out of it, except `<details>` for a folded part.
- Blank body is allowed only for `objective` (the title carries the text) — prefer a body anyway.

## Kinds (L-001 after this pass), colours bg / bar+text, violet → orange theme (L-003)

| kind | icon | title default | bg | bar / text |
|---|---|---|---|---|
| `banner` | – | (title used as the big heading) | `#4A148C` | white text, no bar |
| `objective` | 🎯 | Objective | `#E8F5E9` | `#1B5E20` |
| `takeaway` | ✅ | Takeaway | `#E0F2F1` | `#004D40` |
| `theory` | 📘 | Theory | `#EDE7F6` | `#311B92` |
| `your-turn` | ✏️ | Your turn | `#FFE8D6` | `#7A2E00` |
| `question` | ❓ | Question | `#F3E5F5` | `#4A148C` |
| `across` | 🔀 | Across approaches | `#E8EAF6` | `#1A237E` |
| `pitfall` | ⚠️ | Pitfall | `#FDECEA` | `#8B1A10` |
| `note` | 🧾 | Note | `#FFF8E1` | `#5D4037` |
| `go-deeper` | 🔍 | Go deeper | `#F5F5F5` | `#616161` (muted; body wrapped in `<details>` collapsed by default) |

## What the build does (`scripts/build.py`)

For every marked markdown cell:
1. Render the body markdown → HTML (package `markdown`, extensions `extra` + `sane_lists`).
2. Wrap it in the box: `<div style="background:<bg>;border-left:6px solid <bar>;border-radius:6px;padding:12px 14px;
   margin:6px 0;color:<text>;font-family:inherit;line-height:1.5"><b><icon> <title></b><br>…body…</div>`.
   Banner: full width, bg `#4A148C`, white text, the title in a large bold line, then the body.
   Go deeper: the box holds `<details><summary><b>🔍 title</b></summary>…</details>`.
3. Emit a **code cell** with `metadata.cellView = "form"`, source:
   ```
   # @title
   from IPython.display import HTML, display
   display(HTML(r"""<the same box HTML>"""))
   ```
   plus one saved output of type `display_data` with `data["text/html"]` = the same HTML (so the box shows before
   anything runs; Run all redraws it).
4. Objective callouts: identical mechanism, kind `objective`.
5. Everything stays deterministic: same sources → byte-identical `.ipynb`.

## Lint (`scripts/lint_notebook.py`), Phase 0 additions

- FAIL on any markdown cell containing `<table` with `bgcolor` (N-034).
- FAIL on an unknown callout kind, a marker that is not the first line, or a title containing `<` or `*`.
- N-003 now accepts the `**Objective:**` line either in the first (heading) markdown cell **or** in an `objective`
  callout cell that is the second cell of the section (the heading cell stays first).
- The `style=` check (N-025) keeps applying to ordinary markdown cells; callout cells are exempt (the build owns the HTML).

## Conversion rules for existing boxes

- `<table bgcolor="#4A148C">` at the top of a file → `banner`.
- `#EDE7F6` → `theory`, `#FFE8D6` → `your-turn`, `#F3E5F5` → `question` (👥 boxes: title starts with `👥`),
  `#E8EAF6` → `across`, `#FDECEA` → `pitfall`, `#FFF8E1` → `note`.
- The old `<b>Title.</b>` becomes the marker title (drop the trailing dot); the rest becomes the markdown body,
  HTML tags (`<b>`, `<code>`, `<br>`, `<ol>`, `<li>`, `<font>`) rewritten as markdown.
- A callout that today shares a markdown cell with other text is split into its own cell (markdown before it stays
  a markdown cell; markdown after it becomes a new markdown cell).
- The `**Objective:** …` line of each section moves out of the heading cell into an `objective` callout cell placed
  right after the heading cell (second cell). The heading cell keeps the `# N. Title` heading and any intro prose.
