from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


class PathOutsideRepoError(ValueError):
    """Raised when a resolved path escapes the repository root."""


def get_repo_root() -> Path:
    return REPO_ROOT


def ensure_within_repo(path: str | Path) -> Path:
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise PathOutsideRepoError(f"Refusing to access path outside repo: {resolved}") from exc
    return resolved


def repo_path(*parts: str | Path) -> Path:
    path = REPO_ROOT
    for part in parts:
        path = path / Path(part)
    return ensure_within_repo(path)


def ensure_parent_dir(path: str | Path) -> Path:
    resolved = ensure_within_repo(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
