"""Phase handoff file: create, validate and announce `reports/handoff.md`.

A phase is a bounded job done in one fresh session. It ends by writing this short file, so the next
session starts from it and the few files it names instead of from a 300k-token conversation.
Standard library only; every subcommand takes --root (default: the repo root above scripts/).

    handoff.py new --phase P --workflow W --model M [--force]
    handoff.py check [file]
    handoff.py start [file]
    handoff.py night <report>
"""

import argparse
import datetime
import os
import re
import subprocess
import sys
from pathlib import Path

HANDOFF = "reports/handoff.md"
KEYS = ["phase", "workflow", "written", "branch", "worktree", "model_next"]
SECTIONS = [
    "Done",
    "Left",
    "Files that matter",
    "Decisions taken",
    "Open questions for Simone",
    "Start the next session",
]
MAX_LINES = 60
MAX_QUOTE_WORDS = 8  # ADR-0004 point 5: no pasted meeting quotes
FILL = "<fill in>"
DEFAULT_VAULT = "/Users/simone.bonato/Desktop/second-brain"
QUOTE_RE = re.compile(r'"([^"]*)"|“([^”]*)”')


def vault_path():
    return os.path.abspath(os.path.expanduser(os.environ.get("NB_VAULT") or DEFAULT_VAULT))


def git(root, *args):
    """Return (ok, stdout) of a git command run in root."""
    try:
        done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    except OSError:
        return False, ""
    return done.returncode == 0, done.stdout.strip()


def render(root, phase, workflow, model, bodies, start_command, start_sentence):
    """Build the file text. bodies maps a section name to its lines."""
    ok, branch = git(root, "rev-parse", "--abbrev-ref", "HEAD")
    worktree = os.path.abspath(root)
    written = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    head = [
        "---",
        f"phase: {phase}",
        f"workflow: {workflow}",
        f"written: {written}",
        f"branch: {branch if ok and branch else 'unknown'}",
        f"worktree: {worktree}",
        f"model_next: {model}",
        "---",
        "",
        f"# Handoff — {os.path.basename(worktree)} — after {phase}",
    ]
    out = list(head)
    for name in SECTIONS:
        out += ["", f"## {name}"]
        if name == "Start the next session":
            out += ["```bash", start_command, "```", "```text", start_sentence, "```"]
        else:
            out += bodies.get(name) or [f"- {FILL}"]
    return "\n".join(out) + "\n"


def start_lines(root, workflow, model, phase_file):
    vault = vault_path()
    command = f"cd {os.path.abspath(root)} && claude --model {model} --add-dir {vault}"
    sentence = f"Read reports/handoff.md and {phase_file} and do that phase only."
    return command, sentence


def write_file(root, text, force):
    path = Path(root) / HANDOFF
    if path.exists() and not force:
        ok, status = git(root, "status", "--porcelain", "--", HANDOFF)
        if not ok or status:
            print(f"refusing to overwrite {path}: uncommitted changes (use --force)", file=sys.stderr)
            return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}")
    return 0


def cmd_new(args):
    phase_dir = f"{vault_path()}/.claude/skills/notebook/phases/{args.workflow}"
    command, sentence = start_lines(args.root, args.workflow, args.model, f"{phase_dir}/{FILL}.md")
    text = render(args.root, args.phase, args.workflow, args.model, {}, command, sentence)
    return write_file(args.root, text, args.force)


def parse(text):
    """Return (frontmatter dict, frontmatter end index, {section: (start_idx, [lines])}, heading counts)."""
    lines = text.splitlines()
    front, end = {}, 0
    if lines and lines[0].strip() == "---":
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                end = i + 1
                break
            if ":" in line:
                key, _, value = line.partition(":")
                front[key.strip()] = value.strip()
    sections, counts, current = {}, {}, None
    for i, line in enumerate(lines):
        if line.startswith("## "):
            current = line[3:].strip()
            counts.setdefault(current, []).append(i)
            sections.setdefault(current, (i, []))
        elif current is not None:
            sections[current][1].append(line)
    return front, end, sections, counts


def fences(lines):
    """Yield (language, [content lines]) for each fenced block."""
    lang, buf = None, []
    for line in lines:
        if line.startswith("```"):
            if lang is None:
                lang, buf = line[3:].strip(), []
            else:
                yield lang, buf
                lang = None
        elif lang is not None:
            buf.append(line)


def named_path(line, root):
    """The path a Files-that-matter line names: text after '- ', before ' — ', backticks removed."""
    item = line[2:].split(" — ")[0].replace("`", "").strip()
    return item if os.path.isabs(item) else str(Path(root) / item)


def problems(path, root):
    """Return a list of (line number, message)."""
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    found = []
    front, _, sections, counts = parse(text)
    for key in KEYS:
        if key not in front:
            found.append((1, f"frontmatter key missing: {key}"))
    for name in SECTIONS:
        n = len(counts.get(name, []))
        if n != 1:
            found.append((1, f"section '## {name}' must appear exactly once (found {n})"))
    order = [counts[n][0] for n in SECTIONS if len(counts.get(n, [])) == 1]
    if order != sorted(order):
        found.append((1, "sections are not in the required order: " + ", ".join(SECTIONS)))
    if len(lines) > MAX_LINES:
        found.append((len(lines), f"file has {len(lines)} lines, at most {MAX_LINES}"))
    for i, line in enumerate(lines, 1):
        if FILL in line:
            found.append((i, f"{FILL} not replaced"))
        if line.startswith(">"):
            found.append((i, "block-quote line: paraphrase, never quote (ADR-0004 point 5)"))
        for match in QUOTE_RE.finditer(line):
            span = match.group(1) if match.group(1) is not None else match.group(2)
            if len(span.split()) >= MAX_QUOTE_WORDS:
                found.append((i, f"quoted span of {len(span.split())} words: paraphrase it"))
    if "Files that matter" in sections:
        start, body = sections["Files that matter"]
        for k, line in enumerate(body, start + 2):
            if line.startswith("- ") and FILL not in line and not Path(named_path(line, root)).exists():
                found.append((k, f"path does not exist: {line[2:].split(' — ')[0].strip()}"))
    if "Start the next session" in sections:
        start, body = sections["Start the next session"]
        blocks = list(fences(body))
        bash = [b for lang, b in blocks if lang == "bash"]
        texts = [b for lang, b in blocks if lang == "text"]
        if not any("cd " in ln and "claude " in ln for b in bash for ln in b):
            found.append((start + 1, "Start block has no bash fence with a 'cd … && claude …' command"))
        if not texts or len([ln for ln in texts[0] if ln.strip()]) != 1:
            found.append((start + 1, "Start block needs a text fence holding exactly one line"))
    return sorted(found)


def cmd_check(args):
    path = Path(args.file) if args.file else Path(args.root) / HANDOFF
    if not path.exists():
        print(f"{path}: no such file", file=sys.stderr)
        return 1
    found = problems(path, args.root)
    for number, message in found:
        print(f"{path}:{number}: {message}")
    if not found:
        print(f"{path}: ok")
    return 1 if found else 0


def cmd_start(args):
    path = Path(args.file) if args.file else Path(args.root) / HANDOFF
    text = path.read_text(encoding="utf-8")
    front, _, sections, _ = parse(text)
    body = sections.get("Start the next session", (0, []))[1]
    blocks = list(fences(body))
    command = next((ln for lang, b in blocks if lang == "bash" for ln in b if "claude " in ln), "")
    sentence = next((ln for lang, b in blocks if lang == "text" for ln in b if ln.strip()), "")
    if not command or not sentence:
        print("handoff has no start command or first sentence; run handoff.py check", file=sys.stderr)
        return 1
    total = len(text.encode("utf-8"))
    for line in sections.get("Files that matter", (0, []))[1]:
        if line.startswith("- ") and FILL not in line:
            named = Path(named_path(line, args.root))
            if named.is_file():
                total += named.stat().st_size
    print(f"Phase {front.get('phase', '?')} is done. Start a new session for the next phase:")
    print()
    print(f"  {command}")
    print()
    print("Then type:")
    print()
    print(f"  {sentence}")
    print()
    print(
        f"The next session reads about {total / 4000:.1f}k tokens first: "
        f"{total} bytes in handoff.md plus the files it names."
    )
    return 0


def cmd_night(args):
    report = Path(args.report)
    done, left = [], []
    for line in report.read_text(encoding="utf-8").splitlines():
        match = re.match(r"## (✅|❌|⏸)\s*#?(\d+)", line)
        if match:
            (done if match.group(1) == "✅" else left).append(f"#{match.group(2)}")
    root = os.path.abspath(args.root)
    ok, branch = git(root, "rev-parse", "--abbrev-ref", "HEAD")
    branch = branch if ok and branch else "BRANCH"
    try:
        shown = str(report.resolve().relative_to(Path(root).resolve()))
    except ValueError:
        shown = str(report.resolve())
    bodies = {
        "Done": [f"- issues done: {', '.join(done)}" if done else "- no done issue found in the report"],
        "Left": (
            [f"- issues stopped or paused: {', '.join(left)}; review the report, decide merge or fix"]
            if left
            else ["- no stopped or paused issue found; review the report, decide merge or fix"]
        ),
        "Files that matter": [
            f"- {shown}",
            f"Command, not a file: `git diff main...{branch} --stat`",
        ],
        "Decisions taken": ["- none recorded by the night run beyond the report"],
        "Open questions for Simone": ["- merge the night branch or fix the stopped issues first?"],
    }
    phase_file = f"{vault_path()}/.claude/skills/notebook/phases/new-notebook/03-morning.md"
    command, sentence = start_lines(root, "new-notebook", "sonnet", phase_file)
    text = render(root, "new-notebook/night", "new-notebook", "sonnet", bodies, command, sentence)
    return write_file(root, text, args.force)


def main():
    default_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add(name, func, **kw):
        p = sub.add_parser(name, **kw)
        p.add_argument("--root", default=str(default_root))
        p.set_defaults(func=func)
        return p

    p = add("new", cmd_new)
    p.add_argument("--phase", required=True)
    p.add_argument("--workflow", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--force", action="store_true")
    p = add("check", cmd_check)
    p.add_argument("file", nargs="?")
    p = add("start", cmd_start)
    p.add_argument("file", nargs="?")
    p = add("night", cmd_night)
    p.add_argument("report")
    p.add_argument("--force", action="store_true")
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
