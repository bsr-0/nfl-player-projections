#!/usr/bin/env python3
"""
Scan for secret patterns in staged or working files.
Run before push or in CI to avoid committing keys, passwords, or .env.

Usage:
  python scripts/scan_secrets.py              # scan staged files (for pre-push)
  python scripts/scan_secrets.py --all        # scan all tracked source files
  python scripts/scan_secrets.py --staged     # same as default
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path


# Paths that must never be committed (even if someone force-adds)
FORBIDDEN_PATHS = [
    r"\.env$",
    r"\.env\.local$",
    r"\.env\.[^.]*\.local$",
    r"^secrets\.py$",
    r"credentials.*\.json$",
    r"\.pem$",
    r"\.key$",
]
FORBIDDEN_PATH_RE = re.compile("|".join(f"({p})" for p in FORBIDDEN_PATHS))

# Content patterns: (regex, description). Match = potential secret.
# Avoid matching empty strings, placeholders, and env var references.
CONTENT_PATTERNS = [
    # API keys / tokens
    (r"(?i)(api[_-]?key|apikey)\s*=\s*['\"][^'\"]{20,}['\"]", "API key literal"),
    (r"sk-[a-zA-Z0-9]{20,}", "OpenAI-style secret key"),
    (r"ghp_[a-zA-Z0-9]{36}", "GitHub personal access token"),
    (r"gho_[a-zA-Z0-9]{36}", "GitHub OAuth token"),
    (r"AKIA[0-9A-Z]{16}", "AWS access key ID"),
    (r"(?i)bearer\s+[a-zA-Z0-9_\-\.]{20,}", "Bearer token"),
    # Connection strings with embedded credentials
    (r"postgresql://[^:]+:[^@\s]+@", "PostgreSQL URL with password"),
    (r"mysql://[^:]+:[^@\s]+@", "MySQL URL with password"),
    (r"mongodb(\+srv)?://[^:]+:[^@\s]+@", "MongoDB URL with password"),
    (r"redis://[^:]*:[^@\s]+@", "Redis URL with password"),
    # SMTP / password literals (allow empty, os.getenv, or placeholder)
    (r"(?i)(smtp_)?password\s*=\s*['\"][^'\"]{8,}['\"]", "Password literal (8+ chars)"),
    (r"(?i)secret\s*=\s*['\"][^'\"]{12,}['\"]", "Secret literal"),
]
CONTENT_REGEXES = [(re.compile(p), desc) for p, desc in CONTENT_PATTERNS]

# Files to skip for content scanning (they define or document these patterns)
CONTENT_SCAN_SKIP = ["scripts/scan_secrets.py", ".env.example"]

# Placeholder values we allow (substring match in the captured value)
ALLOWED_PLACEHOLDERS = [
    "<your-app-password>",
    "<app-password-from-env>",
    "your_app_password",
    "YOUR_APP_PASSWORD",
    "os.getenv",
    "getenv(",
    "xxx",
    "password",
    "PASSWORD",
    "***",
    "secret",
]


def get_staged_files(repo_root: Path) -> list[str]:
    """Return list of staged file paths (relative to repo root)."""
    r = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return []
    return [p for p in r.stdout.strip().splitlines() if p.strip()]


def get_tracked_source_files(repo_root: Path) -> list[str]:
    """Return tracked files that are likely source (no node_modules, etc.)."""
    r = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return []
    paths = []
    for p in r.stdout.strip().splitlines():
        p = p.strip()
        if not p:
            continue
        if "node_modules/" in p or "/dist/" in p or p.endswith(".pyc"):
            continue
        if any(
            p.endswith(ext)
            for ext in (".py", ".ts", ".tsx", ".js", ".jsx", ".env", ".json", ".yaml", ".yml", ".md", ".sh")
        ) or ".env" in p:
            paths.append(p)
    return paths


def is_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
    except OSError:
        return True
    return b"\0" in chunk


def check_path(path: str) -> list[str]:
    """Return list of violations for this path (e.g. .env committed)."""
    violations = []
    base = Path(path).name
    if FORBIDDEN_PATH_RE.search(path) or FORBIDDEN_PATH_RE.search(base):
        violations.append(f"Forbidden path (secrets/env): {path}")
    return violations


def check_content(path: Path, text: str) -> list[str]:
    """Return list of (line_no, description) for potential secrets in content."""
    violations = []
    for regex, desc in CONTENT_REGEXES:
        for m in regex.finditer(text):
            value = m.group(0)
            if any(allow in value for allow in ALLOWED_PLACEHOLDERS):
                continue
            line_no = text[: m.start()].count("\n") + 1
            violations.append((line_no, desc, value[:50] + "..." if len(value) > 50 else value))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan for secret patterns before push.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scan all tracked source files (default: only staged)",
    )
    parser.add_argument("--staged", action="store_true", help="Scan only staged files (default).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    if args.all:
        files = get_tracked_source_files(repo_root)
    else:
        files = get_staged_files(repo_root)

    all_violations = []
    for rel in files:
        full = repo_root / rel
        path_issues = check_path(rel)
        all_violations.extend((rel, 0, msg, None) for msg in path_issues)

        if path_issues:
            continue
        if rel in CONTENT_SCAN_SKIP:
            continue
        if not full.is_file() or is_binary(full):
            continue
        try:
            content = full.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for line_no, desc, snippet in check_content(full, content):
            all_violations.append((rel, line_no, desc, snippet))

    if not all_violations:
        print("OK – no secret patterns found.")
        return 0

    print("Secret scan found potential secrets. Do not commit.\n", file=sys.stderr)
    for path, line_no, msg, snippet in all_violations:
        if line_no:
            print(f"  {path}:{line_no} – {msg}", file=sys.stderr)
            if snippet:
                print(f"    snippet: {snippet}", file=sys.stderr)
        else:
            print(f"  {path} – {msg}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
