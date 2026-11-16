"""Multi-vault registry — named vault files under a directory.

A user is not limited to one vault. CartridgeRack manages many .sqac
cartridges; VaultRegistry does the same for vault cartridges, adding an
active-vault pointer and name-keyed file resolution.

Vault files live at ``<vaults_dir>/<name>.sqac``. The active vault is
tracked by a ``vault_active`` pointer file (mirrors ``agent_active``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .vault import Vault, DEFAULT_VAULT_PATH, VAULT_DIR


class VaultNotFoundError(ValueError):
    """Raised when a named vault does not exist."""


class VaultRegistry:
    """Manage named vault cartridges in a directory."""

    def __init__(self, vaults_dir: Optional[Path] = None):
        self.vaults_dir = Path(vaults_dir) if vaults_dir else VAULT_DIR
        self._active_path = self.vaults_dir.parent / "vault_active"
        self.vaults_dir.mkdir(parents=True, exist_ok=True)

    # -- listing ----------------------------------------------------------

    def list_vaults(self) -> list[str]:
        return sorted(p.stem for p in self.vaults_dir.glob("*.sqac"))

    # -- active pointer ---------------------------------------------------

    def active(self) -> str:
        if self._active_path.exists():
            name = self._active_path.read_text(encoding="utf-8").strip()
            if name:
                return name
        return "default"

    def set_active(self, name: str) -> None:
        if name != "default" and name not in self.list_vaults():
            raise VaultNotFoundError(f"vault not found: {name!r}")
        self._active_path.parent.mkdir(parents=True, exist_ok=True)
        self._active_path.write_text(name, encoding="utf-8")

    # -- path resolution --------------------------------------------------

    def path_for(self, name: str) -> Path:
        if name == "default":
            # Backward compat: default vault lives at the legacy path
            if DEFAULT_VAULT_PATH.exists():
                return DEFAULT_VAULT_PATH
            return self.vaults_dir / "default.sqac"
        p = self.vaults_dir / f"{name}.sqac"
        if not p.exists():
            raise VaultNotFoundError(f"vault not found: {name!r}")
        return p

    # -- init / open ------------------------------------------------------

    def init(self, name: str, mode: str = "keychain",
             passphrase: Optional[str] = None, master_key=None) -> "Vault":
        if name == "default":
            path = DEFAULT_VAULT_PATH
        else:
            path = self.vaults_dir / f"{name}.sqac"
        path.parent.mkdir(parents=True, exist_ok=True)
        vault = Vault.create(path, mode=mode, passphrase=passphrase,
                            master_key=master_key)
        if name != "default" and name not in self.list_vaults():
            # ensure it shows up in listing
            pass
        return vault

    def open(self, name: str) -> "Vault":
        path = self.path_for(name)
        return Vault.open(path)

    def resolve(self, secret_name: str, vault: Optional[str] = None) -> str:
        """Resolve a secret from the named (or active) vault."""
        vname = vault or self.active()
        v = self.open(vname)
        return v.resolve(secret_name)
