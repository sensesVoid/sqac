"""Per-CLI wiring snippets for the SQAC MCP server.

The core trick for seamless CLI switching: every harness registers the SAME
`sqac-mcp --dir <MEM_DIR>` command, so all CLIs share one memory directory and
one continuity sidecar. This module prints/installs the exact snippet per host.

`--hook-project` appends an AGENTS.md/CLAUDE.md hook to the given project so
every agent auto-bootstraps shared memory. Without it, the snippets are printed
so the user (or CI) can take them elsewhere.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

DEFAULT_MEM_DIR = "~/.sqacm"

_HOOK_LINE = (
    "> SQAC memory: this project shares memory with every CLI. If the `sqac-mcp` "
    "server is mounted (mem_* tools available), call `mem_bootstrap` once at session "
    "start to load working context, and `mem_checkpoint` at task boundaries."
)


def _mem_dir(arg: str | None) -> str:
    return (arg or DEFAULT_MEM_DIR).strip()


def _cmd(dir: str) -> list[str]:
    return ["sqac-mcp", "--dir", dir]


def _cmd_sh(dir: str) -> str:
    return " ".join(shlex.quote(c) for c in _cmd(dir))


def _opencode_json(dir: str) -> dict:
    return {
        "$schema": "https://opencode.ai/config.json",
        "mcp": {
            "sqac": {
                "type": "local",
                "command": _cmd_sh(dir),
                "enabled": True,
            }
        },
    }


def opencode(dir: str) -> str:
    return (
        "### opencode  (opencode.json — project root or ~/.config/opencode/opencode.json)\n"
        + json.dumps(_opencode_json(dir), indent=2)
        + "\n"
    )


def claude_code(dir: str) -> str:
    return (
        "### Claude Code  (run in the project, or --global)\n"
        f"claude mcp add sqac --stdio -- {shlex.quote('sqac-mcp')} {shlex.quote('--dir')} {shlex.quote(dir)}\n"
    )


def claude_desktop(dir: str) -> str:
    return (
        "### Claude Desktop  (~/Library/Application Support/Claude/claude_desktop_config.json or %APPDATA%\\Claude\\claude_desktop_config.json)\n"
        + json.dumps(
            {
                "mcpServers": {
                    "sqac": {
                        "command": "sqac-mcp",
                        "args": ["--dir", dir],
                    }
                }
            },
            indent=2,
        )
        + "\n"
    )


def codex(dir: str) -> str:
    return (
        "### Codex  (~/.codex/config.toml — append)\n"
        + "\n".join(
            [
                "[mcp_servers.sqac]",
                f'command = "sqac-mcp"',
                f'args = ["--dir", "{dir}"]',
            ]
        )
        + "\n"
    )


def cursor(dir: str) -> str:
    return (
        "### Cursor  (.cursor/mcp.json at project root)\n"
        + json.dumps(
            {
                "mcpServers": {
                    "sqac": {
                        "command": "sqac-mcp",
                        "args": ["--dir", dir],
                    }
                }
            },
            indent=2,
        )
        + "\n"
    )


def zed(dir: str) -> str:
    return (
        "### Zed  (~/.config/zed/settings.json — \"mcp\" key)\n"
        + json.dumps(
            {
                "mcp": {
                    "sqac": {
                        "command": "sqac-mcp",
                        "args": ["--dir", dir],
                        "enabled": True,
                    }
                }
            },
            indent=2,
        )
        + "\n"
    )


def hook(dirs: list[str]) -> str:
    return "\n".join([f"> SQAC memory dir(s): {', '.join(map(str, dirs))}", _HOOK_LINE, ""])


GENERATORS = {
    "opencode": opencode,
    "claude-code": claude_code,
    "claude-desktop": claude_desktop,
    "codex": codex,
    "cursor": cursor,
    "zed": zed,
}


def all_snippets(dir: str) -> str:
    return "\n".join(g(dir) for g in GENERATORS.values()) + "\n" + hook([dir])


# ── install targets (best-effort, idempotent) ───────────────────────────────

_OPCODE = ("opencode.json", "~/.config/opencode/opencode.json")
_CLAUDE_DESKTOP = (
    "~/Library/Application Support/Claude/claude_desktop_config.json",
    "%APPDATA%/Claude/claude_desktop_config.json",
)
_CLAUDE_HOME = "~/.claude"


def _expand(p: str) -> Path:
    return Path(p).expanduser()


def _merge_json(path: Path, fragment: dict) -> None:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {}
    data.update(fragment)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def install_opencode(dir: str) -> tuple[Path, str]:
    p = _expand(_OPCODE[0])
    _merge_json(p, _opencode_json(dir))
    return p, opencode(dir)


def install_claude_desktop(dir: str) -> tuple[Path, str] | None:
    for cand in _CLAUDE_DESKTOP:
        p = _expand(cand)
        if "%" in cand or not p.parent.exists():
            continue
        _merge_json(
            p,
            {
                "mcpServers": {
                    "sqac": {"command": "sqac-mcp", "args": ["--dir", dir]}
                }
            },
        )
        return p, claude_desktop(dir)
    return None


def install_claude_code(dir: str) -> None:
    # Config lives in ~/.claude.json; `claude mcp add` is the supported path.
    # We print the command; programmatic edit of ~/.claude.json is fragile.
    print(claude_code(dir))


def main(argv: list[str] | None = None) -> int:
    """`python -m sqac.mcp_setup [--dir DIR] [--host HOST|all] [--hook-project PATH]`.

    Prints the wiring snippet(s). `--hook-project` also appends the memory hook
    to that project's AGENTS.md / CLAUDE.md so agents auto-bootstrap.
    """
    import argparse

    ap = argparse.ArgumentParser(prog="sqac mcp setup", description="SQAC MCP wiring snippets")
    ap.add_argument("--dir", default=DEFAULT_MEM_DIR, help=f"shared memory dir (default: {DEFAULT_MEM_DIR})")
    ap.add_argument("--host", default="all", help="opencode, claude-code, claude-desktop, codex, cursor, zed (default: all)")
    ap.add_argument("--hook-project", default=None, help="path to a project to install the AGENTS.md hook into")
    args = ap.parse_args(argv)

    dir = _mem_dir(args.dir)
    hosts = [h.strip() for h in args.host.split(",") if h.strip()]
    out: list[str] = []
    for h in hosts:
        if h == "all":
            out.append(all_snippets(dir))
            break
        if h not in GENERATORS:
            print(f"unknown host {h!r}; choose one of {sorted(GENERATORS)}", file=sys.stderr)
            return 2
        out.append(GENERATORS[h](dir))
    text = "\n".join(out)
    if args.hook_project:
        proj = Path(args.hook_project).expanduser().resolve()
        proj.mkdir(parents=True, exist_ok=True)
        target = proj / "AGENTS.md"
        hook_text = hook([dir])
        if target.exists():
            existing = target.read_text(encoding="utf-8")
            if _HOOK_LINE in existing:
                print(f"hook already present: {target}")
            else:
                target.write_text(existing.rstrip() + "\n\n" + hook_text, encoding="utf-8")
                print(f"appended hook: {target}")
        else:
            target.write_text("# Agent memory\n\n" + hook_text, encoding="utf-8")
            print(f"wrote hook: {target}")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())