#!/usr/bin/env bash
# Redirect to the unified master installer setup.sh
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/setup.sh" "$@"
