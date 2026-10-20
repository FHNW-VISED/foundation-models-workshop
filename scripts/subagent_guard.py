"""PreToolUse hook for night runs: subagents are never Fable (Simone's rule, 2026-10-20).

Allowed: haiku or sonnet (the default, via CLAUDE_CODE_SUBAGENT_MODEL), opus only for the heaviest
tasks. Blocked: model "fable", and subagent_type "fork", which always inherits the parent's model.
Exit code 2 blocks the call and shows stderr to the agent.
"""

import json
import sys

call = json.load(sys.stdin)
if call.get("tool_name") not in ("Agent", "Task"):
    sys.exit(0)
args = call.get("tool_input") or {}
model = str(args.get("model") or "").lower()
if "fable" in model or args.get("subagent_type") == "fork":
    print(
        "Blocked: subagents must never run on Fable (and a fork inherits Fable). "
        "Use model 'haiku' or 'sonnet'; 'opus' only for the heaviest tasks.",
        file=sys.stderr,
    )
    sys.exit(2)
