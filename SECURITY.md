# Security Policy

SQAC's full security model lives in [docs/SECURITY.md](docs/SECURITY.md).

## Summary

- **Memory cartridges are plaintext.** `.sqac` memory/agent files have no
  encryption at rest. Do not store secrets in them.
- **Vault cartridges are encrypted.** Secrets/PII belong in `vault` cartridges
  (AES-256-GCM, keychain or Argon2id passphrase mode). MCP vault tools are
  reference-only: no tool returns a plaintext secret value.
- **Content is untrusted by default.** SQAC flags risky prompt-injection
  patterns and credential-like content, but detection is not a sandbox. Treat
  retrieved content as untrusted input.
- **Server hardening.** The HTTP server supports timing-safe API key checks,
  path validation, and atomic writes; see docs and changelog for details.

## Reporting a vulnerability

**Do not open a public issue for security vulnerabilities.**

Use GitHub's private security reporting ("Security" → "Report a vulnerability")
on https://github.com/sensesVoid/sqac/security, or email the maintainer directly.

Please include: SQAC version, Python version, platform, affected component
(vault/cloud/server/MCP/CLI), and a minimal reproduction if possible.

## Bug reports (non-security)

Use `sqac report "title" -d "description"` to collect diagnostics and file a
public GitHub issue, or open an issue manually at
https://github.com/sensesVoid/sqac/issues.
