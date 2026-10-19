#!/usr/bin/env bash
# Night run: implement the open issues labelled `night`, one at a time in issue-number order,
# with headless Claude Code. After each issue `just check` must pass, or the run stops there
# (later sections depend on earlier ones). Everything lands on branch night/<date> as one commit
# per issue, plus a draft PR whose body is the morning report. Never touches main.
#
#   scripts/night.sh                 # run all open `night` issues
#   scripts/night.sh --max 3         # at most 3 issues
#   scripts/night.sh --dry-run       # show what would run, call nothing
#   scripts/night.sh --model opus    # pass a model to claude
set -euo pipefail

LABEL="night"; MAX=99; DRY=0; MODEL=""; RETRIES=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --label) LABEL="$2"; shift 2 ;;
    --max) MAX="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --retries) RETRIES="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
for bin in gh claude just uv git; do command -v "$bin" >/dev/null || { echo "missing: $bin" >&2; exit 1; }; done
[[ -z "$(git status --porcelain)" ]] || { echo "working tree not clean — commit or stash first" >&2; exit 1; }

DATE="$(date +%Y-%m-%d)"
BRANCH="night/$DATE"
LOGDIR="build/night/$DATE"            # raw logs: gitignored
REPORT="reports/night-$DATE.md"       # summary: committed, becomes the PR body
mkdir -p "$LOGDIR" reports

ISSUES=$(gh issue list --label "$LABEL" --state open --limit 100 --json number --jq 'map(.number) | sort | .[]' | sort -n | head -n "$MAX")
[[ -n "$ISSUES" ]] || { echo "no open issues labelled '$LABEL'"; exit 0; }
echo "issues to run: $(echo $ISSUES | tr '\n' ' ')"

TOOLS="Read,Edit,Write,Glob,Grep,Bash(just:*),Bash(uv run:*),Bash(uv add:*),Bash(ls:*),Bash(wc:*),Bash(head:*),Bash(tail:*),Bash(grep:*)"
CLAUDE_ARGS=(-p --permission-mode acceptEdits --allowedTools "$TOOLS" --strict-mcp-config --mcp-config '{"mcpServers":{}}' --output-format json)
[[ -n "$MODEL" ]] && CLAUDE_ARGS+=(--model "$MODEL")

# The gate must not be weakened by the agent it gates: undo edits to these paths unless the
# issue's FILES line names them explicitly.
PROTECTED=(tests scripts RULES.md justfile .github AGENTS.md CLAUDE.md)
revert_protected() {  # $1 issue number, $2 issue body
  local files_line; files_line=$(grep -m1 '^FILES:' <<<"$2" || true)
  for p in "${PROTECTED[@]}"; do
    [[ "$files_line" == *"$p"* ]] && continue
    if [[ -n "$(git status --porcelain -- "$p")" ]]; then
      git checkout --quiet HEAD -- "$p" 2>/dev/null || true
      git clean -fdq -- "$p"
      echo "- reverted agent edits to protected \`$p\` (#$1)" >> "$LOGDIR/protected.txt"
    fi
  done
}

# Progressive gate (ADR-0002): content tests only switch on for sections marked built, so an issue
# that builds section NN (body line "SECTION: NN") must leave that section's status at "done".
section_marked_done() {  # $1 issue body; succeeds if no SECTION line or the row says done
  local nn; nn=$(grep -m1 -oE '^SECTION: [0-9]{2}' <<<"$1" | awk '{print $2}' || true)
  [[ -z "$nn" ]] && return 0
  grep -qE "^\|[[:space:]]*[0-9]+[[:space:]]*\|[[:space:]]*sections/${nn}_[^|]*\|.*\|[[:space:]]*done[[:space:]]*\|[[:space:]]*$" SPEC.md
}

prompt_for() {  # $1 issue number, $2 title, $3 body, $4 optional failure tail
  cat <<EOF
You are implementing GitHub issue #$1 "$2" in this repository, unattended (nobody can answer questions).
Follow AGENTS.md exactly: read SPEC.md, RULES.md, this issue, and only the files the issue names; use \`just api\` for earlier definitions.
Finish only when \`just check\` passes, then update the section's Status in SPEC.md and write reports/latest-task.md.

--- ISSUE #$1 ---
$3
--- END ISSUE ---
EOF
  if [[ -n "${4:-}" ]]; then
    printf '\nA previous attempt left `just check` failing. Fix this, keeping the issue scope:\n%s\n' "$4"
  fi
}

if [[ "$DRY" == 1 ]]; then
  first=$(echo "$ISSUES" | head -1)
  echo "--- prompt for #$first ---"
  prompt_for "$first" "$(gh issue view "$first" --json title --jq .title)" "$(gh issue view "$first" --json body --jq .body)"
  echo "--- claude ${CLAUDE_ARGS[*]} ---"
  exit 0
fi

git fetch --quiet origin
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then git switch --quiet "$BRANCH"; else git switch --quiet -c "$BRANCH" origin/main; fi
caffeinate -dimsu -w $$ &              # keep the Mac awake while this script lives
uv sync --quiet

{ echo "# Night run $DATE"; echo; echo "Branch \`$BRANCH\` · label \`$LABEL\` · started $(date +%H:%M)"; echo; } > "$REPORT"
DONE=0; STOPPED=""

for n in $ISSUES; do
  title=$(gh issue view "$n" --json title --jq .title)
  body=$(gh issue view "$n" --json body --jq .body)
  echo "== #$n $title"
  tail_msg=""; ok=0
  for attempt in $(seq 0 "$RETRIES"); do
    claude "${CLAUDE_ARGS[@]}" "$(prompt_for "$n" "$title" "$body" "$tail_msg")" > "$LOGDIR/issue-$n-try$attempt.json" 2>&1 || true
    revert_protected "$n" "$body"
    if ! section_marked_done "$body"; then
      echo "SECTION status in SPEC.md was not set to done — the content tests for it never ran." > "$LOGDIR/issue-$n-check$attempt.log"
    elif just check > "$LOGDIR/issue-$n-check$attempt.log" 2>&1; then ok=1; break; fi
    tail_msg=$(tail -40 "$LOGDIR/issue-$n-check$attempt.log")
  done

  if [[ "$ok" == 1 ]]; then
    { echo "## ✅ #$n — $title"; echo; cat reports/latest-task.md 2>/dev/null || echo "(no task note written)"; echo; } >> "$REPORT"
    rm -f reports/latest-task.md
    git add -A
    git commit --quiet -m "night: #$n $title" -m "Refs #$n"
    gh issue comment "$n" --body "Implemented on \`$BRANCH\` ($(git rev-parse --short HEAD)); \`just check\` passed. Waiting for morning review." >/dev/null
    DONE=$((DONE + 1))
  else
    note=$(cat reports/latest-task.md 2>/dev/null || true)
    rm -f reports/latest-task.md
    git reset --quiet --hard HEAD && git clean -fdq -e reports/   # drop only this issue's uncommitted work
    { echo "## ❌ #$n — $title (stopped here)"; echo; echo '```'; tail -40 "$LOGDIR/issue-$n-check$RETRIES.log"; echo '```'; echo
      echo "$note"; echo; } >> "$REPORT"                          # after the reset, which would undo it
    gh issue edit "$n" --add-label night-failed >/dev/null 2>&1 || true
    gh issue comment "$n" --body "Night run $DATE could not make \`just check\` pass; see \`$REPORT\` on \`$BRANCH\`." >/dev/null
    STOPPED="#$n"
    break
  fi
done

[[ -s "$LOGDIR/protected.txt" ]] && { echo "## ⚠️ Protected files the agent tried to change (reverted)"; cat "$LOGDIR/protected.txt"; echo; } >> "$REPORT"
{ echo "---"; echo "Finished $(date +%H:%M): $DONE issue(s) done${STOPPED:+, stopped at $STOPPED}. Raw logs (local only): \`$LOGDIR\`."; } >> "$REPORT"
git add "$REPORT" && git commit --quiet -m "night: report $DATE"
git push --quiet -u origin "$BRANCH"
if ! gh pr view "$BRANCH" >/dev/null 2>&1; then
  gh pr create --draft --base main --head "$BRANCH" --title "Night run $DATE" --body-file "$REPORT" >/dev/null
else
  gh pr edit "$BRANCH" --body-file "$REPORT" >/dev/null
fi
echo "done: $DONE issue(s)${STOPPED:+, stopped at $STOPPED}. Report: $REPORT"
