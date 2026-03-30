#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TECTONIC_BIN="${TECTONIC_BIN:-$HOME/.local/bin/tectonic}"

if [[ ! -x "$TECTONIC_BIN" ]]; then
  echo "tectonic not found at $TECTONIC_BIN" >&2
  exit 1
fi

cd "$ROOT"
if [[ "${REBUILD_DATA:-0}" == "1" ]]; then
  python3 scripts/build_causal_package.py
else
  python3 scripts/build_causal_package.py --reuse-derived
fi
python3 - <<'PY'
import sys
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root / "scripts"))
import redesign_figures as rf

rf.main()
PY

export TEXINPUTS="./template:"
export BIBINPUTS=".:./template:"
export BSTINPUTS=".:./template:"
"$TECTONIC_BIN" --keep-logs --keep-intermediates --outdir "$ROOT" main.tex
