#!/bin/bash

payload=$(cat 2>/dev/null || echo '{}')
cwd=$(echo "$payload" | jq -r '.cwd // empty' 2>/dev/null)
cwd="${cwd:-$(pwd)}"

lockPath="$cwd/.rsgrep/server.json"
killed=false

if [[ -f "$lockPath" ]]; then
    pid=$(jq -r '.pid // empty' "$lockPath" 2>/dev/null)
    if [[ -n "$pid" && "$pid" =~ ^[0-9]+$ ]]; then
        kill -TERM "$pid" 2>/dev/null && killed=true
    fi
    rm -f "$lockPath" 2>/dev/null
fi

if [[ "$killed" != "true" ]]; then
    pkill -f "rsgrep serve" 2>/dev/null
fi
