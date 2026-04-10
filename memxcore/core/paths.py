"""Resolve on-disk project root (directory that contains storage/ and config.yaml)."""

import logging
import os
import warnings

logger = logging.getLogger("memxcore")


def _is_site_packages(path: str) -> bool:
    """Check if a path is inside a Python site-packages directory."""
    normalized = os.path.normpath(os.path.abspath(path))
    return "site-packages" in normalized.split(os.sep)


def resolve_workspace(fallback: str) -> str:
    """Return the workspace root directory.

    Resolution order:
      1. MEMXCORE_WORKSPACE env var (preferred for pip-installed usage)
      2. MEMNEST_WORKSPACE / CLAWDMEMORY_WORKSPACE (legacy compat)
      3. *fallback* — typically computed from __file__ by the caller
    """
    ws = (
        os.environ.get("MEMXCORE_WORKSPACE")
        or os.environ.get("MEMX_WORKSPACE")
        or os.environ.get("MEMNEST_WORKSPACE")
        or os.environ.get("CLAWDMEMORY_WORKSPACE")
        or os.path.abspath(fallback)
    )
    if _is_site_packages(ws):
        warnings.warn(
            f"memxcore: workspace resolves to site-packages ({ws}). "
            "Memories will be lost on pip upgrade. "
            "Set MEMXCORE_WORKSPACE=/path/to/your/project to fix this.",
            stacklevel=2,
        )
    return ws


def resolve_install_dir(workspace_path: str) -> str:
    """
    Return absolute path to the MemX project directory.

    Prefers ``<workspace>/memxcore`` when ``storage`` exists there.
    Then legacy ``<workspace>/memx`` (previous name).
    Then legacy ``<workspace>/memnest``.
    Then legacy ``<workspace>/ClawdMemory``.
    """
    workspace_path = os.path.abspath(workspace_path)
    new_root = os.path.join(workspace_path, "memxcore")
    memx_root = os.path.join(workspace_path, "memx")
    mid_root = os.path.join(workspace_path, "memnest")
    old_root = os.path.join(workspace_path, "ClawdMemory")
    if os.path.isdir(os.path.join(new_root, "storage")):
        return new_root
    if os.path.isdir(os.path.join(memx_root, "storage")):
        logger.warning(
            "Using legacy memx/ install directory; rename folder to memxcore/ when convenient."
        )
        return memx_root
    if os.path.isdir(os.path.join(mid_root, "storage")):
        logger.warning(
            "Using legacy memnest/ install directory; rename folder to memxcore/ when convenient."
        )
        return mid_root
    if os.path.isdir(os.path.join(old_root, "storage")):
        logger.warning(
            "Using legacy ClawdMemory/ install directory; rename folder to memxcore/ when convenient."
        )
        return old_root
    return new_root
