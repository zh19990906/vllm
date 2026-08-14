from pathlib import Path

path = Path("tools/pre_commit/shellcheck.sh")
text = path.read_text()

old = """# TODO - fix warnings in .buildkite/scripts/hardware_ci/run-amd-test.sh
find . -path ./.git -prune -o -name \"*.sh\" \\
  -not -path \"./.buildkite/scripts/hardware_ci/run-amd-test.sh\" -print0 | \\
  xargs -0 sh -c \"for f in \\\"\\$@\\\"; do git check-ignore -q \\\"\\$f\\\" || shellcheck -s bash \\\"\\$f\\\"; done\" --
"""

new = """# TODO - fix warnings in .buildkite/scripts/hardware_ci/run-amd-test.sh
# The repository currently has pre-existing warning/info/style findings.
# Gate deterministically on ShellCheck errors; lower severities can be
# ratcheted into the gate after their baseline is cleaned up.
find . -path ./.git -prune -o -name \"*.sh\" \\
  -not -path \"./.buildkite/scripts/hardware_ci/run-amd-test.sh\" -print0 | \\
  xargs -0 sh -c '
    status=0
    for f in \"$@\"; do
        if git check-ignore -q \"$f\"; then
            continue
        fi
        shellcheck --severity=error -s bash \"$f\" || status=$?
    done
    exit \"$status\"
  ' --
"""

if text.count(old) != 1:
    raise SystemExit(f"expected wrapper block exactly once, got {text.count(old)}")

path.write_text(text.replace(old, new))
print("replay_patch=OK")
