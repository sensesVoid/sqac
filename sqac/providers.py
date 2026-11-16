"""External secret providers — fetch vault master keys from secret managers.

Providers are thin CLI wrappers: they shell out to the provider's CLI
(`op` for 1Password, `bw` for Bitwarden) and return the raw secret bytes.
No Python SDK dependencies — keeps the runtime dependency surface at zero.

Cloud HTTP APIs (AWS/GCP/Azure/HashiCorp) are deliberately deferred to a
later release; they would require SDK dependencies or bespoke HTTP clients.
"""
from __future__ import annotations

import shutil
import subprocess
from typing import Optional

__all__ = [
    "ProviderNotFoundError",
    "ProviderLockedError",
    "ProviderCLIError",
    "available_providers",
    "fetch",
]

_CLI_MAP: dict[str, str] = {
    "1password": "op",
    "bitwarden": "bw",
}

_FETCH_CMD: dict[str, list[str]] = {
    "1password": ["op", "read"],
    "bitwarden": ["bw", "get", "password"],
}

_LOCKED_HINTS = ("not signed in", "locked", "session expired", "authentication")


class ProviderNotFoundError(ValueError):
    """Raised when the provider's CLI is not installed."""


class ProviderLockedError(ValueError):
    """Raised when the provider CLI is installed but locked/unauthenticated."""


class ProviderCLIError(ValueError):
    """Raised when the provider CLI fails for another reason."""


def available_providers() -> list[str]:
    """Return the list of providers whose CLI is installed."""
    return [p for p, cli in _CLI_MAP.items() if shutil.which(cli) is not None]


def fetch(provider: str, ref: str, timeout: int = 30) -> bytes:
    """Fetch a secret from an external provider.

    Args:
        provider: "1password" or "bitwarden"
        ref: provider-specific reference (e.g. "op://vault/item/field")
        timeout: subprocess timeout in seconds

    Returns:
        Raw secret bytes (trailing newline stripped).

    Raises:
        ProviderNotFoundError: CLI not installed
        ProviderLockedError: CLI installed but locked/unauthenticated
        ProviderCLIError: other CLI failure
    """
    if provider not in _CLI_MAP:
        raise ProviderNotFoundError(
            f"unknown provider: {provider!r}. "
            f"Supported: {', '.join(_CLI_MAP)}"
        )

    cli = _CLI_MAP[provider]
    if shutil.which(cli) is None:
        install_hints = {
            "1password": "Install the 1Password CLI: https://developer.1password.com/docs/cli/get-started/",
            "bitwarden": "Install the Bitwarden CLI: https://bitwarden.com/help/cli/",
        }
        raise ProviderNotFoundError(
            f"{provider} CLI ({cli}) not found on PATH. "
            f"{install_hints.get(provider, '')}"
        )

    cmd = _FETCH_CMD[provider] + [ref]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise ProviderCLIError(
            f"{provider} CLI timed out after {timeout}s"
        )
    except FileNotFoundError:
        raise ProviderNotFoundError(
            f"{provider} CLI ({cli}) not found on PATH"
        )

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").lower()
        if any(hint in stderr for hint in _LOCKED_HINTS):
            raise ProviderLockedError(
                f"{provider} CLI is locked or not authenticated. "
                f"Run '{cli} signin' first. stderr: {result.stderr.decode('utf-8', errors='replace').strip()}"
            )
        raise ProviderCLIError(
            f"{provider} CLI failed (exit {result.returncode}): "
            f"{result.stderr.decode('utf-8', errors='replace').strip()}"
        )

    # Strip trailing newline
    out = result.stdout
    if out.endswith(b"\n"):
        out = out[:-1]
    if out.endswith(b"\r"):
        out = out[:-1]
    return out
