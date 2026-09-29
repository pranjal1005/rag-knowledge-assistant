#!/usr/bin/env bash
# Usage: ./scripts/build_index.sh "retrieval augmented generation" 40
set -euo pipefail

QUERY="${1:?Usage: ./scripts/build_index.sh \"search query\" [max_results]}"
MAX_RESULTS="${2:-40}"

python -m src.build_index --source arxiv --query "$QUERY" --max-results "$MAX_RESULTS"
