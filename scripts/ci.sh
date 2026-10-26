#!/usr/bin/env bash
# The gate: ruff (lint and format, settings in pyproject.toml), then the
# offline selftest. Its last line, "selftest: N/N checks passed", is the merge
# queue's GATE_OK, so the selftest runs last.
#
# Run: ./scripts/ci.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== ruff"
ruff check src
ruff format --check src

echo "== selftest"
uv run ocr-bakeoff selftest
