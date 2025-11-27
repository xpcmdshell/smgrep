#!/bin/bash

payload=$(cat 2>/dev/null || echo '{}')
cwd=$(echo "$payload" | jq -r '.cwd // empty' 2>/dev/null)
cwd="${cwd:-$(pwd)}"

rsgrep serve >> /tmp/rsgrep.log 2>&1 &
disown

cat <<'EOF'
{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"rsgrep serve started; prefer `rsgrep \"<complete question>\"` over grep (plain output is agent-friendly)."}}
EOF
