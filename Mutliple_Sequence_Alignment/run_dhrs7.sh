#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_dhrs7.sh -- one-command DHRS7 alignment + species-snapshot figure
# reproduction (macOS / Linux).
#
# Prerequisites:
#   1. conda env create -f phylo.yml
#   2. conda activate phylo
#   3. MUSCLE on PATH (conda install -c bioconda muscle), or set MUSCLE_EXE.
#
# Then:  ./run_dhrs7.sh
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use a local CA bundle if you are behind a TLS-inspecting proxy (optional).
if [[ -f "$SCRIPT_DIR/ca_bundle.pem" ]]; then
  export SSL_CERT_FILE="$SCRIPT_DIR/ca_bundle.pem"
  export REQUESTS_CA_BUNDLE="$SCRIPT_DIR/ca_bundle.pem"
fi

# With no arguments open the window; with arguments run the build directly.
if [[ $# -eq 0 ]]; then
  python "$SCRIPT_DIR/MSA_GUI.py"
else
  python "$SCRIPT_DIR/dhrs7_alignment.py" "$@"
fi
