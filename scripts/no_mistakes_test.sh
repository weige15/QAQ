#!/usr/bin/env bash
set -euo pipefail

source "$HOME/.venv/bin/activate"
which python
python --version

case "$(command -v python)" in
  "$HOME/.venv/"*) ;;
  *)
    echo "Refusing: python is not inside $HOME/.venv" >&2
    exit 1
    ;;
esac

config_file="${XDG_CONFIG_HOME:-$HOME/.config}/qaq/s03-artifact-source"

if [[ ! -s "$config_file" ]]; then
  echo "QAQ artifact source configuration is missing: $config_file" >&2
  exit 1
fi

artifact_source="$(<"$config_file")"

# A fresh No-Mistakes worktree may have an uninitialized pinned backend.
git submodule update --init --recursive -- third_party/any-precision-llm

# Creates only an ignored, hash-verified worktree-local symlink.
python scripts/provision_s03_artifact.py \
  --source "$artifact_source"

# Small permanent identity check. The evidence agent will select any additional
# tests needed for the particular change.
PYTHONPATH=src:third_party/any-precision-llm:. \
  python -m pytest -q \
  tests/unit/test_s10h_broader_validation.py::test_frozen_protocol_and_pre_execution_identity_are_fail_closed
