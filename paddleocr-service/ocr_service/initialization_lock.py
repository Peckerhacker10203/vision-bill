"""CPU-compatible installation lock API, shared with the standalone supervisor."""

from .supervision import InstallationBusy, installation_lock, lock_path

__all__ = ["InstallationBusy", "installation_lock", "lock_path"]
