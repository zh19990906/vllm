#!/usr/bin/env bash
set -euo pipefail

PARENT1="66fc997ed55db5cb99fa12dcb5213a3d57f34ff9"
PARENT2="5ab0016765bfdbd2d45575908a096fbf675b3a84"
EXPECTED_TREE="ebd91faafee826a5888570bca7d18b5fc264decb"
EXPECTED_COMMIT="451a37fadbfc15051ea8438e7d02483b1cc6602d"
TARGET_BRANCH="agent/issue31-fs-hard-capacity"
MESSAGE="merge: bring shellcheck CI fix into issue 31"

git fetch --no-tags origin "$PARENT1" "$PARENT2"

MERGE_TREE="$(git merge-tree --write-tree "$PARENT1" "$PARENT2")"
echo "reproduced_tree=$MERGE_TREE"
if [ "$MERGE_TREE" != "$EXPECTED_TREE" ]; then
  echo "tree_guard=FAIL expected=$EXPECTED_TREE actual=$MERGE_TREE"
  exit 1
fi

echo "tree_guard=PASS"

export GIT_AUTHOR_NAME="Your Name"
export GIT_AUTHOR_EMAIL="you@example.com"
export GIT_AUTHOR_DATE="2026-08-14T09:41:23+08:00"
export GIT_COMMITTER_NAME="Your Name"
export GIT_COMMITTER_EMAIL="you@example.com"
export GIT_COMMITTER_DATE="2026-08-14T09:41:23+08:00"

MERGE_COMMIT="$(printf '%s\n' "$MESSAGE" | git commit-tree "$MERGE_TREE" -p "$PARENT1" -p "$PARENT2")"
echo "reproduced_commit=$MERGE_COMMIT"
if [ "$MERGE_COMMIT" != "$EXPECTED_COMMIT" ]; then
  echo "commit_guard=FAIL expected=$EXPECTED_COMMIT actual=$MERGE_COMMIT"
  exit 1
fi

echo "commit_guard=PASS"

git push origin "$MERGE_COMMIT:refs/heads/$TARGET_BRANCH"
echo "publish=PASS"
