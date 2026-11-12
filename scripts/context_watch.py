"""UserPromptSubmit hook: tell the agent to stop and hand off once the conversation passes ~150k tokens.

Why: the audit measured nine long interactive sessions re-sending a 261k to 345k token context on
every turn. A phase should end in `reports/handoff.md` and a fresh session, not in a 300k conversation.

Facts this hook depends on, confirmed on 2026-11-12 against https://code.claude.com/docs/en/hooks :
1. A UserPromptSubmit command hook receives JSON on standard input that includes `session_id` and
   `transcript_path` (plus `cwd`, `permission_mode`, `hook_event_name`, `prompt`).
2. On exit 0, a command hook that prints a JSON object
   {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": "..."}}
   has that string added to Claude's context alongside the prompt (wrapped in a system reminder, not
   shown in the chat). Plain stdout is also added as context; we print JSON only (one approach per
   hook). Each injected string is capped at 10,000 characters.
3. `$CLAUDE_PROJECT_DIR` (the project root where the session started) is exported to hook commands.

Reads the last 300 KB of the transcript, takes the last line with message.usage, and sums
input_tokens + cache_creation_input_tokens + cache_read_input_tokens. Limit: env CONTEXT_WATCH_LIMIT
(default 150000); a further notice only after another 50000. Notices already given are remembered per
session in build/context-watch.json. Any error prints nothing and exits 0.
"""

import json
import os
import sys
from pathlib import Path

LIMIT_DEFAULT = 150_000
STEP = 50_000
TAIL_BYTES = 300 * 1024
STATE = Path(__file__).resolve().parent.parent / "build" / "context-watch.json"

NOTICE = (
    "[context-watch] This conversation is about {k}k tokens. Tell Simone in plain words: "
    '"This conversation is past {t}k tokens. I will finish the step I am on, write '
    'reports/handoff.md, and then you start a new session." Do not begin a new step. '
    "Run just handoff-new, fill it in, run just handoff-check, commit it, then run "
    "just handoff-start and show him its output."
)


def context_tokens(transcript):
    """Context size of the last assistant message that has usage, or None."""
    with open(transcript, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - TAIL_BYTES))
        data = f.read()
    for raw in reversed(data.splitlines()):
        try:
            usage = (json.loads(raw).get("message") or {}).get("usage")
        except (ValueError, AttributeError):
            continue
        if isinstance(usage, dict):
            return sum(
                int(usage.get(k) or 0)
                for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
            )
    return None


def main():
    hook = json.load(sys.stdin)
    session = str(hook["session_id"])
    tokens = context_tokens(hook["transcript_path"])
    if tokens is None:
        return
    limit = int(os.environ.get("CONTEXT_WATCH_LIMIT") or LIMIT_DEFAULT)
    try:
        state = json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    given = state.get(session)  # highest threshold already announced
    threshold = limit if given is None else int(given) + STEP
    if tokens < threshold:
        return
    # the highest threshold now reached (a jump from 100k to 260k counts as one notice)
    reached = limit + ((tokens - limit) // STEP) * STEP
    state[session] = reached
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state), encoding="utf-8")
    context = NOTICE.format(k=round(tokens / 1000), t=round(reached / 1000))
    print(
        json.dumps(
            {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": context}}
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:  # a watch must never break a prompt
        pass
    sys.exit(0)
