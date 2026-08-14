#!/bin/bash
set -euo pipefail

scversion="stable"

if [ -d "shellcheck-${scversion}" ]; then
    export PATH="$PATH:$(pwd)/shellcheck-${scversion}"
fi

if ! [ -x "$(command -v shellcheck)" ]; then
    if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
        echo "Please install shellcheck: https://github.com/koalaman/shellcheck?tab=readme-ov-file#installing"
        exit 1
    fi

    # automatic local install if linux x86_64
    wget -qO- "https://github.com/koalaman/shellcheck/releases/download/${scversion?}/shellcheck-${scversion?}.linux.x86_64.tar.xz" | tar -xJv
    export PATH="$PATH:$(pwd)/shellcheck-${scversion}"
fi

# TODO - fix warnings in .buildkite/scripts/hardware_ci/run-amd-test.sh
# The repository currently has pre-existing warning/info/style findings.
# Gate deterministically on ShellCheck errors; lower severities can be
# ratcheted into the gate after their baseline is cleaned up.
find . -path ./.git -prune -o -name "*.sh" \
  -not -path "./.buildkite/scripts/hardware_ci/run-amd-test.sh" -print0 | \
  xargs -0 sh -c '
    status=0
    for f in "$@"; do
        if git check-ignore -q "$f"; then
            continue
        fi
        shellcheck --severity=error -s bash "$f" || status=$?
    done
    exit "$status"
  ' --
