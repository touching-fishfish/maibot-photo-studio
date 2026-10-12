from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from maitu_photo.gallery import (
    GalleryError,
    ReferenceGallery,
    ReferenceNotSelectableError,
    ReferenceVersionConflictError,
)
from maitu_photo.gallery_management import GalleryManagement
from maitu_photo.models import AssetStatus, ReferenceAsset
from maitu_photo.reference_service import ReferenceTagValidationError, validate_reference_tags
from maitu_photo.storage import SQLiteStorage


def _tags(category: str) -> dict[str, Any]:
    return {
        "person": {"appearance_summary": "short dark hair", "confidence": 0.9},
        "outfit": {
            "type": "dress",
            "wearing_scenes": ["home"],
            "seasons": ["summer"],
            "styles": ["minimal"],
            "confidence": 0.9,
        },
        "scene": {
            "room_type": "bedroom",
            "privacy_eligible": True,
            "scene_signature": "window-left",
            "confidence": 0.9,
        },
    }[category]


def _management(tmp_path: Path) -> tuple[SQLiteStorage, ReferenceGallery, GalleryManagement]:
    storage = SQLiteStorage(tmp_path / "db" / "gallery.sqlite3")
    gallery = ReferenceGallery(storage)
    return storage, gallery, GalleryManagement(gallery, tmp_path / "plugin-data")


def _add(gallery: ReferenceGallery, root: Path, category: str = "outfit", **kwargs: Any) -> ReferenceAsset:
    digest = kwargs.pop("sha256", "a" * 64)
    source = root / "sources" / category / f"{digest}.jpg"
    reference = root / "references" / category / f"{digest}.jpg"
    for path, data in ((source, b"source"), (reference, b"reference")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return gallery.add_reference(
        category=category,
        name="original",
        source_path=source,
        reference_path=reference,
        sha256=digest,
        tags=kwargs.pop("tags", _tags(category)),
        **kwargs,
    )


@pytest.mark.parametrize(
    ("category", "manual_tags"),
    [
        ("person", {"accessories": ["hat"]}),
        ("person", {"appearance_summary": " "}),
        ("person", {"appearance_summary": 1}),
        ("person", {"confidence": True}),
        ("outfit", {"season": "summer"}),
        ("outfit", {"styles": "minimal"}),
        ("outfit", {"styles": [""]}),
        ("outfit", {"styles": [1]}),
        ("outfit", {"confidence": math.nan}),
        ("outfit", {"confidence": 1.1}),
        ("outfit", {"confidence": 10**1000}),
        ("outfit", {"confidence": "0.9"}),
        ("scene", {"light": "morning"}),
        ("scene", {"privacy_eligible": 1}),
        ("scene", {"scene_signature": None}),
        ("scene", {1: "bad key"}),
        ("scene", []),
    ],
)
def test_edit_rejects_unknown_fields_and_wrong_types_without_changes(
    tmp_path: Path, category: str, manual_tags: Any
) -> None:
    storage, gallery, management = _management(tmp_path)
    asset = _add(gallery, management.data_dir, category)

    with pytest.raises(ReferenceTagValidationError, match="手动标签"):
        management.edit(asset.id, name="must not persist", manual_tags=manual_tags)

    assert gallery.require(asset.id) == asset
    storage.close()


def test_partial_tags_allow_omissions_but_do_not_accept_retired_keys() -> None:
    partial = validate_reference_tags("outfit", {"styles": [" minimal ", "minimal"]}, partial=True)
    assert partial.schema_valid
    assert partial.tags == {"styles": ["minimal"]}
    assert validate_reference_tags("scene", {}, partial=True).schema_valid
    assert not validate_reference_tags("scene", {"light": "morning"}, partial=True).schema_valid
    assert not validate_reference_tags("person", {}, partial=False).schema_valid


@pytest.mark.parametrize("status", [AssetStatus.ACTIVE, AssetStatus.DISABLED, AssetStatus.NEEDS_REVIEW])
def test_edit_keeps_valid_status_and_normalizes_partial_manual_tags(tmp_path: Path, status: AssetStatus) -> None:
    storage, gallery, management = _management(tmp_path)
    asset = _add(gallery, management.data_dir, status=status, manual_tags={"seasons": ["winter"]})

    edited = management.edit(
        asset.id,
        name="  new name  ",
        manual_tags={"styles": [" calm ", "calm"]},
        expected_version=asset.updated_at.isoformat(),
    )

    assert edited.name == "new name"
    assert edited.manual_tags == {"seasons": ["winter"], "styles": ["calm"]}
    assert edited.tags == asset.tags
    assert edited.status == status
    assert edited.updated_at != asset.updated_at
    storage.close()


@pytest.mark.parametrize("status", [AssetStatus.ACTIVE, AssetStatus.DISABLED])
def test_edit_disqualified_effective_tags_becomes_review(tmp_path: Path, status: AssetStatus) -> None:
    storage, gallery, management = _management(tmp_path)
    asset = _add(gallery, management.data_dir, "scene", status=status)

    edited = management.edit(asset.id, manual_tags={"privacy_eligible": False})

    assert edited.status == AssetStatus.NEEDS_REVIEW
    assert edited.tags["privacy_eligible"] is True
    assert edited.effective_tags["privacy_eligible"] is False
    storage.close()


def test_replace_and_clear_manual_tags_revalidate_effective_tags(tmp_path: Path) -> None:
    storage, gallery, management = _management(tmp_path)
    asset = _add(
        gallery,
        management.data_dir,
        "scene",
        tags={**_tags("scene"), "privacy_eligible": False},
        manual_tags={"privacy_eligible": True, "room_type": "bathroom"},
    )

    edited = management.edit(asset.id, manual_tags={"privacy_eligible": True}, merge_manual_tags=False)
    assert edited.manual_tags == {"privacy_eligible": True}
    assert edited.status == AssetStatus.ACTIVE
    cleared = management.edit(asset.id, manual_tags={}, merge_manual_tags=False)
    assert cleared.manual_tags == {}
    assert cleared.status == AssetStatus.NEEDS_REVIEW
    storage.close()


def test_enable_validates_effective_tags_and_disable_keeps_metadata(tmp_path: Path) -> None:
    storage, gallery, management = _management(tmp_path)
    asset = _add(
        gallery,
        management.data_dir,
        "scene",
        tags={**_tags("scene"), "privacy_eligible": False},
        manual_tags={"privacy_eligible": True},
        status=AssetStatus.NEEDS_REVIEW,
        selection_metadata={"note": "keep"},
    )

    enabled = management.enable(asset.id, asset.updated_at.isoformat())
    assert enabled.status == AssetStatus.ACTIVE
    disabled = management.disable(asset.id, enabled.updated_at.isoformat())
    assert disabled.status == AssetStatus.DISABLED
    assert disabled.tags == asset.tags
    assert disabled.manual_tags == asset.manual_tags
    assert disabled.selection_metadata == {"note": "keep"}
    storage.close()


@pytest.mark.parametrize(
    ("category", "tags"),
    [("person", {}), ("outfit", {"type": "dress"}), ("scene", {**_tags("scene"), "privacy_eligible": False})],
)
def test_enable_rejects_incomplete_or_disqualified_tags(tmp_path: Path, category: str, tags: dict[str, Any]) -> None:
    storage, gallery, management = _management(tmp_path)
    asset = _add(gallery, management.data_dir, category, tags=tags, status=AssetStatus.DISABLED)

    with pytest.raises(ReferenceNotSelectableError):
        management.enable(asset.id)

    assert gallery.require(asset.id) == asset
    storage.close()


@pytest.mark.parametrize("operation", ["edit", "enable", "disable", "delete"])
def test_all_mutations_reject_stale_updated_at_versions(tmp_path: Path, operation: str) -> None:
    storage, gallery, management = _management(tmp_path)
    asset = _add(gallery, management.data_dir)
    edited = management.edit(asset.id, name="new")

    with pytest.raises(ReferenceVersionConflictError):
        getattr(management, operation)(asset.id, expected_version=asset.updated_at.isoformat())

    assert gallery.require(asset.id) == edited
    assert asset.source_path.exists() and asset.reference_path.exists()
    storage.close()


def test_non_string_version_and_empty_names_are_rejected(tmp_path: Path) -> None:
    storage, gallery, management = _management(tmp_path)
    asset = _add(gallery, management.data_dir)
    for name in (" ", 1):
        with pytest.raises(ValueError):
            management.edit(asset.id, name=name)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        management.disable(asset.id, expected_version=1)  # type: ignore[arg-type]
    assert gallery.require(asset.id) == asset
    storage.close()


def test_delete_commits_tombstone_before_unlink_and_leaves_history_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, gallery, management = _management(tmp_path)
    historical = _add(gallery, management.data_dir, sha256="b" * 64)
    gallery.soft_delete(historical.id)
    asset = _add(gallery, management.data_dir)
    original_unlink = Path.unlink
    unlinked: list[Path] = []

    def check_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        with sqlite3.connect(storage.database_path) as observer:
            row = observer.execute(
                "SELECT status, deleted_at FROM reference_assets WHERE id = ?", (asset.id,)
            ).fetchone()
        assert row is not None and row[0] == "deleted" and row[1]
        unlinked.append(path)
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", check_unlink)
    deleted = management.delete(asset.id, expected_version=asset.updated_at.isoformat())

    assert deleted.status == AssetStatus.DELETED and deleted.deleted_at is not None
    assert gallery.get(asset.id) is None
    assert set(unlinked) == {asset.source_path, asset.reference_path}
    assert not asset.source_path.exists() and not asset.reference_path.exists()
    assert historical.source_path.exists() and historical.reference_path.exists()
    assert "cleanup_error" not in deleted.selection_metadata
    storage.close()


def test_unlink_failure_records_safe_cleanup_error_on_tombstone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, gallery, management = _management(tmp_path)
    asset = _add(gallery, management.data_dir)
    original_unlink = Path.unlink

    def fail_source_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        if path == asset.source_path:
            raise PermissionError(f"sensitive path: {path}")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_source_unlink)
    deleted = management.delete(asset.id)

    assert deleted.status == AssetStatus.DELETED
    assert deleted.selection_metadata["cleanup_error"] == "source_path:PermissionError"
    assert str(management.data_dir) not in str(deleted.selection_metadata)
    assert storage.get_reference_asset(asset.id, include_deleted=True) == deleted
    assert asset.source_path.exists() and not asset.reference_path.exists()
    storage.close()


@pytest.mark.parametrize(
    "unsafe", ["outside", "traversal", "file_symlink", "directory", "parent_symlink", "root_symlink"]
)
def test_delete_rejects_unsafe_paths_before_any_mutation(tmp_path: Path, unsafe: str) -> None:
    storage, gallery, management = _management(tmp_path)
    asset = _add(gallery, management.data_dir)
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"do not remove")
    if unsafe == "outside":
        asset.source_path = outside
    elif unsafe == "traversal":
        asset.source_path = asset.source_path.parent / ".." / "outfit" / asset.source_path.name
    elif unsafe == "file_symlink":
        asset.source_path.unlink()
        asset.source_path.symlink_to(outside)
    elif unsafe == "directory":
        asset.source_path.unlink()
        asset.source_path.mkdir()
    elif unsafe == "parent_symlink":
        parent = asset.source_path.parent
        moved = parent.with_name("moved")
        parent.rename(moved)
        parent.symlink_to(moved, target_is_directory=True)
    else:
        root = management.data_dir
        moved = tmp_path / "moved-data"
        root.rename(moved)
        root.symlink_to(moved, target_is_directory=True)
    asset = storage.update_reference_asset(asset)

    with pytest.raises(GalleryError, match="不能删除"):
        management.delete(asset.id)

    assert gallery.require(asset.id) == asset
    assert asset.reference_path.exists()
    assert outside.read_bytes() == b"do not remove"
    storage.close()


@pytest.mark.parametrize("shared", ["source", "cross_role", "hardlink", "disabled", "needs_review"])
def test_delete_rejects_paths_shared_with_any_live_asset(tmp_path: Path, shared: str) -> None:
    storage, gallery, management = _management(tmp_path)
    asset = _add(gallery, management.data_dir)
    other = _add(gallery, management.data_dir, "scene", sha256="b" * 64)
    if shared == "hardlink":
        other.source_path.unlink()
        other.source_path.hardlink_to(asset.source_path)
    elif shared == "cross_role":
        other.reference_path = asset.source_path
    else:
        other.source_path = asset.source_path
        if shared in ("disabled", "needs_review"):
            other.status = AssetStatus(shared)
    storage.update_reference_asset(other)

    with pytest.raises(GalleryError, match="共用"):
        management.delete(asset.id)

    assert gallery.require(asset.id) == asset
    assert asset.source_path.exists() and asset.reference_path.exists()
    storage.close()


def test_delete_rejects_outer_transaction_and_accepts_already_missing_files(tmp_path: Path) -> None:
    storage, gallery, management = _management(tmp_path)
    asset = _add(gallery, management.data_dir)
    with storage.transaction(), pytest.raises(GalleryError, match="外部事务"):
        management.delete(asset.id)
    assert gallery.require(asset.id) == asset
    asset.source_path.unlink()
    asset.reference_path.unlink()

    deleted = management.delete(asset.id)

    assert deleted.status == AssetStatus.DELETED
    assert "cleanup_error" not in deleted.selection_metadata
    storage.close()
