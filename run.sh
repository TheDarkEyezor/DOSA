#!/bin/bash
#
# Run Samantha
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Start Ollama if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 2
fi

# Run Samantha
cd "$SCRIPT_DIR"
./target/release/samantha "$@"
