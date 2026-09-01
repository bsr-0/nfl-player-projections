#!/usr/bin/env bash
# Install the repo's git hooks.
#
# .git/hooks is not version controlled, so the hook lives in scripts/hooks/ and
# is symlinked into place. A symlink rather than a copy, so edits to the tracked
# file take effect without reinstalling and the two cannot silently diverge.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
src="$root/scripts/hooks/pre-commit"
dest="$root/.git/hooks/pre-commit"

[ -f "$src" ] || { echo "missing $src"; exit 1; }
chmod +x "$src"

if [ -e "$dest" ] && [ ! -L "$dest" ]; then
  cp "$dest" "$dest.backup.$(date +%Y%m%d%H%M%S)"
  echo "existing hook backed up"
fi

ln -sf "$src" "$dest"
echo "installed: $dest -> scripts/hooks/pre-commit"
echo "bypass a false positive with: git commit --no-verify"
