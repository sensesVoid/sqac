"""Bug reporting: collect diagnostics and file a GitHub issue."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional


def collect_diagnostics() -> dict:
    """Collect system and SQAC diagnostics for a bug report."""
    diag = {
        "sqac_version": None,
        "python_version": sys.version,
        "platform": platform.platform(),
        "os": platform.system(),
        "arch": platform.machine(),
    }
    try:
        from sqac._version import __version__
        diag["sqac_version"] = __version__
    except Exception:
        pass

    # Installed packages
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            pkgs = json.loads(result.stdout)
            diag["packages"] = {p["name"]: p["version"] for p in pkgs}
    except Exception:
        pass

    # SIMD status
    try:
        from sqac.simd import HAS_SIMD, backend
        diag["simd"] = HAS_SIMD
        if backend is not None:
            diag["simd_version"] = getattr(backend, "__version__", "unknown")
    except Exception:
        diag["simd"] = "unknown"

    # Cartridge stats
    try:
        from pathlib import Path as _Path
        diag["cartridge_count"] = 0
        # Count .sqac files in default dir
        mem_dir = _Path(os.environ.get("SQAC_MEM_DIR", str(_Path.home() / ".sqacm")))
        if mem_dir.exists():
            diag["cartridge_count"] = len(list(mem_dir.rglob("*.sqac")))
    except Exception:
        pass

    # Rust toolchain
    try:
        result = subprocess.run(
            ["cargo", "--version"], capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            diag["cargo"] = result.stdout.strip()
    except Exception:
        diag["cargo"] = "not found"

    # Git info (if in a repo)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            diag["git_head"] = result.stdout.strip()
    except Exception:
        pass

    return diag


def format_diagnostics(diag: dict) -> str:
    """Format diagnostics as markdown for a GitHub issue."""
    lines = ["## Environment\n"]
    lines.append(f"- **SQAC version:** {diag.get('sqac_version', 'unknown')}")
    lines.append(f"- **Python:** {diag.get('python_version', 'unknown').split(chr(10))[0]}")
    lines.append(f"- **Platform:** {diag.get('platform', 'unknown')}")
    lines.append(f"- **Architecture:** {diag.get('arch', 'unknown')}")
    lines.append(f"- **SIMD:** {diag.get('simd', 'unknown')}")
    if diag.get("cargo"):
        lines.append(f"- **Cargo:** {diag['cargo']}")
    if diag.get("git_head"):
        lines.append(f"- **Git HEAD:** {diag['git_head']}")
    lines.append(f"- **Cartridges:** {diag.get('cartridge_count', 0)}")

    # Key packages
    pkgs = diag.get("packages", {})
    key_pkgs = ["numpy", "mcp", "cryptography", "fastapi", "uvicorn", "boto3", "lz4", "keyring", "argon2-cffi", "tree-sitter", "tree-sitter-language-pack"]
    if pkgs:
        lines.append("\n## Key Dependencies\n")
        for name in key_pkgs:
            if name in pkgs:
                lines.append(f"- {name}: {pkgs[name]}")

    return "\n".join(lines)


def create_issue(title: str, body: str, labels: Optional[list[str]] = None) -> str:
    """Create a GitHub issue using gh CLI. Returns the issue URL."""
    cmd = ["gh", "issue", "create", "--repo", "sensesVoid/sqac", "--title", title, "--body", body]
    if labels:
        cmd.extend(["--label", ",".join(labels)])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"gh issue create failed: {result.stderr}")
    return result.stdout.strip()
