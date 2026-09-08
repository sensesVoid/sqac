"""Credential detection — rejects content matching common secret patterns.

SQAC is a memory layer for LLMs, not a secrets manager. Storing API keys,
passwords, tokens, or other credentials in .sqac files is a security risk:
the files are plaintext, have no encryption at rest, and could be accidentally
committed to version control.

This module detects credential-like content and raises CredentialError
before it reaches the store. Detection is pattern-based and conservative —
it flags high-confidence matches to avoid false positives on legitimate
knowledge content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


class CredentialError(ValueError):
    """Raised when content appears to contain credentials."""
    pass


@dataclass
class CredentialMatch:
    """A detected credential pattern."""
    pattern_name: str
    matched_text: str
    severity: str  # "critical" or "warning"
    recommendation: str


# Pattern definitions: (name, regex, severity, recommendation)
_PATTERNS: list[tuple[str, re.Pattern, str, str]] = [
    # API Keys — high-confidence patterns
    (
        "AWS Access Key",
        re.compile(r'\b(AKIA[0-9A-Z]{16})\b'),
        "critical",
        "Use environment variables or AWS Secrets Manager instead.",
    ),
    (
        "AWS Secret Key",
        re.compile(r'(?i)(aws[_\-]?secret[_\-]?access[_\-]?key)\s*[=:]\s*["\']?([A-Za-z0-9/+=]{40})["\']?'),
        "critical",
        "Use environment variables or AWS Secrets Manager instead.",
    ),
    (
        "GitHub Token",
        re.compile(r'\b(ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{82})\b'),
        "critical",
        "Use environment variables. Never commit tokens to version control.",
    ),
    (
        "GitLab Token",
        re.compile(r'\b(glpat-[A-Za-z0-9\-_]{20,})\b'),
        "critical",
        "Use environment variables. Never commit tokens to version control.",
    ),
    (
        "OpenAI API Key",
        re.compile(r'\b(sk-[A-Za-z0-9]{48,})\b'),
        "critical",
        "Use OPENAI_API_KEY environment variable.",
    ),
    (
        "Anthropic API Key",
        re.compile(r'\b(sk-ant-[A-Za-z0-9\-_]{40,})\b'),
        "critical",
        "Use ANTHROPIC_API_KEY environment variable.",
    ),
    (
        "Google API Key",
        re.compile(r'\b(AIza[A-Za-z0-9_\-]{35})\b'),
        "critical",
        "Use environment variables or Google Secret Manager.",
    ),
    (
        "Slack Token",
        re.compile(r'\b(xox[bpsar]-[A-Za-z0-9\-]{10,})\b'),
        "critical",
        "Use environment variables. Revoke and rotate if exposed.",
    ),
    (
        "Stripe Key",
        re.compile(r'\b(sk_live_[A-Za-z0-9]{24,}|pk_live_[A-Za-z0-9]{24,}|rk_live_[A-Za-z0-9]{24,})\b'),
        "critical",
        "Use STRIPE_SECRET_KEY environment variable.",
    ),
    (
        "Private Key Block",
        re.compile(r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----'),
        "critical",
        "Store private keys in a secrets manager, not in SQAC.",
    ),
    (
        "SSH Private Key",
        re.compile(r'-----BEGIN\s+OPENSSH\s+PRIVATE\s+KEY-----'),
        "critical",
        "Store SSH keys in ~/.ssh/, not in SQAC.",
    ),
    # Passwords and tokens — pattern-based
    (
        "Password Assignment",
        re.compile(r'(?i)(password|passwd|pwd)\s*[=:]\s*["\']?(\S{8,})["\']?'),
        "critical",
        "Use environment variables or a secrets manager.",
    ),
    (
        "Token Assignment",
        re.compile(r'(?i)(token|secret|api[_\-]?key|access[_\-]?key)\s*[=:]\s*["\']?([A-Za-z0-9\-_.]{20,})["\']?'),
        "warning",
        "This may be a credential. Use environment variables instead.",
    ),
    (
        "Bearer Token",
        re.compile(r'(?i)(bearer)\s+[A-Za-z0-9\-_.=]{20,}'),
        "warning",
        "Do not store auth tokens in SQAC memory.",
    ),
    (
        "Connection String",
        re.compile(r'(?i)(mongodb|postgres|mysql|redis|amqp)://[^\s]{20,}'),
        "critical",
        "Use environment variables for connection strings.",
    ),
    (
        "JWT Token",
        re.compile(r'\beyJ[A-Za-z0-9\-_]{10,}\.[A-Za-z0-9\-_]{10,}\.[A-Za-z0-9\-_]{10,}\b'),
        "critical",
        "Do not store JWT tokens in SQAC memory.",
    ),
    # Generic high-entropy strings that look like secrets
    (
        "Base64 Secret",
        re.compile(r'(?i)(secret|key|token|password)\s*[=:]\s*["\']?([A-Za-z0-9+/]{40,}={0,2})["\']?'),
        "warning",
        "This may be a base64-encoded secret. Use environment variables.",
    ),
]


def detect_credentials(content: str) -> list[CredentialMatch]:
    """Scan content for credential-like patterns.

    Returns a list of matches. Empty list means no credentials detected.
    """
    matches: list[CredentialMatch] = []
    for name, pattern, severity, recommendation in _PATTERNS:
        for m in pattern.finditer(content):
            matched = m.group(0)
            # Truncate long matches for display
            display = matched[:60] + "..." if len(matched) > 60 else matched
            matches.append(CredentialMatch(
                pattern_name=name,
                matched_text=display,
                severity=severity,
                recommendation=recommendation,
            ))
    return matches


def validate_content(content: str, strict: bool = True) -> list[CredentialMatch]:
    """Validate content for credentials.

    Args:
        content: The text content to validate.
        strict: If True, raise CredentialError on critical matches.
                If False, return matches without raising.

    Returns:
        List of credential matches (empty if clean).

    Raises:
        CredentialError: If strict=True and critical credentials are found.
    """
    matches = detect_credentials(content)
    critical = [m for m in matches if m.severity == "critical"]
    if strict and critical:
        patterns = ", ".join(m.pattern_name for m in critical)
        raise CredentialError(
            f"Content appears to contain credentials ({patterns}). "
            f"SQAC is not a secrets manager — use environment variables "
            f"or a dedicated secrets manager for credentials. "
            f"To store this anyway, set strict=False."
        )
    return matches
