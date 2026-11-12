"""PreToolUse hook for subagent calls: subagents are never Fable (Simone's rule, 2026-10-20).

Allowed: an explicit model, haiku or sonnet; opus only for the heaviest tasks. Blocked: model "fable",
subagent_type "fork" (it always inherits the parent's model), and a call that names no model at all
(Simone, 2026-11-11: the agent must choose the tier on purpose).
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
if not model.strip():
    print(
        "Blocked: this subagent call names no model. Pass model 'haiku' or 'sonnet'; "
        "'opus' only for the heaviest tasks.",
        file=sys.stderr,
    )
    sys.exit(2)
