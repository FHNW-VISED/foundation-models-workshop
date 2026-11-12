#!/usr/bin/env bash
# One headless helper agent, started from inside this repo so it loads this repo's small AGENTS.md
# and not the vault's large instruction prefix. Sonnet by default, a turn cap that is always set,
# and only a short result printed. Usage:
#   scripts/helper.sh <name> <brief-file> [--model <alias-or-id>] [--max-turns N] [--dry-run]
# Written for bash 3.2, the macOS default: only features that old shell has.
set -euo pipefail

ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

usage() { echo "usage: scripts/helper.sh <name> <brief-file> [--model <alias-or-id>] [--max-turns N] [--dry-run]" >&2; exit 2; }
fail() { echo "helper.sh: $1" >&2; exit 2; }

NAME=""; BRIEF=""; DRY=0
MODEL="${HELPER_MODEL:-sonnet}"
MAX_TURNS=25
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
POS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --model) [ $# -ge 2 ] || usage; MODEL="$2"; shift 2 ;;
    --max-turns) [ $# -ge 2 ] || usage; MAX_TURNS="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -*) usage ;;
    *)
      POS=$((POS + 1))
      if [ "$POS" -eq 1 ]; then NAME="$1"; elif [ "$POS" -eq 2 ]; then BRIEF="$1"; else usage; fi
      shift ;;
  esac
done
[ "$POS" -eq 2 ] || usage

# Validate before anything runs.
case "$NAME" in
  ""|-*|*[!a-z0-9-]*) fail "name '$NAME' must match [a-z0-9][a-z0-9-]*" ;;
esac
[ -f "$BRIEF" ] || fail "brief file '$BRIEF' does not exist"
SIZE=$(wc -c < "$BRIEF" | tr -d ' ')
[ "$SIZE" -gt 0 ] || fail "brief file '$BRIEF' is empty"
[ "$SIZE" -le 20000 ] || fail "brief file '$BRIEF' is $SIZE bytes, the limit is 20000"
MODEL_LC="$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]')"
case "$MODEL_LC" in
  *fable*) fail "model '$MODEL' is refused: helpers never run on Fable" ;;
esac
case "$MAX_TURNS" in
  ""|*[!0-9]*) fail "--max-turns '$MAX_TURNS' must be an integer from 1 to 60" ;;
esac
[ "$MAX_TURNS" -ge 1 ] && [ "$MAX_TURNS" -le 60 ] || fail "--max-turns $MAX_TURNS must be from 1 to 60"

# Same allow-list as scripts/night.sh, minus Bash(uv add:*): a helper never adds a dependency.
TOOLS="Read,Edit,Write,Glob,Grep,Bash(just:*),Bash(uv run:*),Bash(ls:*),Bash(wc:*),Bash(head:*),Bash(tail:*),Bash(grep:*)"
CLAUDE_ARGS=(-p --model "$MODEL" --max-turns "$MAX_TURNS" --permission-mode acceptEdits --allowedTools "$TOOLS" --strict-mcp-config --mcp-config '{"mcpServers":{}}' --output-format json)
# Subagents are never Fable: Sonnet unless the agent names haiku/opus; the hook blocks "fable" and forks.
export CLAUDE_CODE_SUBAGENT_MODEL=sonnet
GUARD="python3 $ROOT/scripts/subagent_guard.py"
SETTINGS="{\"hooks\":{\"PreToolUse\":[{\"matcher\":\"Agent|Task\",\"hooks\":[{\"type\":\"command\",\"command\":\"$GUARD\"}]}]}}"
CLAUDE_ARGS+=(--settings "$SETTINGS")

# The cap is added here so no brief can omit it.
PROMPT="$(cat "$BRIEF")

Cap: at most $MAX_TURNS turns. Finish with a result of at most 15 lines. Never open an .ipynb or anything under build/. Do not run git commit, push or branch. Any Agent call must name model haiku or sonnet."

if [ "$DRY" -eq 1 ]; then
  echo "helper $NAME (dry run, nothing called)"
  echo "binary: $CLAUDE_BIN"
  echo "cwd: $ROOT"
  echo "model: $MODEL"
  echo "max-turns: $MAX_TURNS"
  echo "tools: $TOOLS"
  echo "settings: $SETTINGS"
  echo "--- prompt (first 10 lines) ---"
  printf '%s\n' "$PROMPT" | head -n 10
  exit 0
fi

mkdir -p build/helpers
RAW="build/helpers/$NAME.json"
OUT="build/helpers/$NAME.md"
RC=0
# stdin from /dev/null: claude -p otherwise waits 3 s for piped stdin and writes a warning into $RAW,
# which breaks the JSON (claude 2.1.322, seen 2026-11-12)
"$CLAUDE_BIN" "${CLAUDE_ARGS[@]}" "$PROMPT" < /dev/null > "$RAW" 2>&1 || RC=$?

python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1])).get("result") or "")' "$RAW" > "$OUT" 2>/dev/null \
  || { echo "helper $NAME: output is not JSON, first lines of $RAW:"; head -n 20 "$RAW" > "$OUT"; }
tail -n 40 "$OUT"
python3 -I -c '
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    d = {}
u = d.get("usage") or {}
g = lambda v: "unknown" if v is None else v
print("helper %s: model %s, turns %s, cache_read %s, output %s" % (
    sys.argv[2], sys.argv[3], g(d.get("num_turns")), g(u.get("cache_read_input_tokens")), g(u.get("output_tokens"))))
st = d.get("subtype")
if st and st != "success":
    # error_max_turns: the cap stopped it and there is no result text (claude 2.1.322, checked 2026-11-12)
    print("helper %s: stopped: %s" % (sys.argv[2], st))
' "$RAW" "$NAME" "$MODEL"
exit "$RC"
