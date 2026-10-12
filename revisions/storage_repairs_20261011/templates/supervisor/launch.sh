#!/bin/bash
# Supported production entry: timeout starts before Python imports/preflight.
# Invoke by absolute path. Only the new diagnostic process tree is owned.
case "$0" in
  /*) ;;
  *) printf '%s\n' 'Use an absolute path to launch.sh' >&2; exit 64 ;;
esac
exec /usr/bin/timeout --signal=TERM --kill-after=40s 4300s @PROJECT_ROOT@/.venv_h155_20260830/bin/python -B "${0%/*}/supervisor.py" "$@"
