#!/usr/bin/env bash
# Thin wrapper to keep historical commands working.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/scripts/ops/run.sh" "$@"
