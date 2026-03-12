from pathlib import Path

import pytest

from watersic.utils.path_guard import PathOutsideRepoError, ensure_parent_dir, ensure_within_repo, get_repo_root


def test_repo_relative_path_is_allowed(tmp_path: Path) -> None:
    target = ensure_parent_dir("outputs/reports/test_path_guard.json")
    assert target.is_absolute()
    assert get_repo_root() in target.parents


def test_absolute_path_outside_repo_is_rejected(tmp_path: Path) -> None:
    outside = Path("/tmp/watersic_outside.txt")
    with pytest.raises(PathOutsideRepoError):
        ensure_within_repo(outside)
