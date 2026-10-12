"""Shared, model-free gallery management for chat commands and Web APIs."""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .gallery import (
    GalleryError,
    ReferenceGallery,
    ReferenceNotSelectableError,
    ReferenceVersionConflictError,
)
from .models import AssetStatus, ReferenceAsset
from .reference_service import ReferenceTagValidationError, validate_reference_tags


class GalleryManagement:
    """Apply validated administrator changes without calling image or tag models."""

    def __init__(self, gallery: ReferenceGallery, data_dir: Path | str) -> None:
        self.gallery = gallery
        # Preserve the lexical root so a symlink cannot silently become trusted.
        self.data_dir = Path(os.path.abspath(data_dir))

    def edit(
        self,
        asset_id: str,
        name: str | None = None,
        manual_tags: Mapping[str, Any] | None = None,
        merge_manual_tags: bool = True,
        expected_version: str = "",
    ) -> ReferenceAsset:
        with self.gallery.storage.transaction():
            asset = self._require_version(asset_id, expected_version)
            if name is not None:
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("参考图名称不能为空")
                asset.name = name.strip()
            if manual_tags is not None:
                validation = validate_reference_tags(asset.category, manual_tags, partial=True)
                if not validation.schema_valid:
                    raise ReferenceTagValidationError(("手动标签字段或类型不合法", *validation.errors))
                combined = dict(asset.manual_tags) if merge_manual_tags else {}
                combined.update(validation.tags)
                # Validate retained overrides as well as the incoming partial object.
                validation = validate_reference_tags(asset.category, combined, partial=True)
                if not validation.schema_valid:
                    raise ReferenceTagValidationError(("手动标签字段或类型不合法", *validation.errors))
                asset.manual_tags = validation.tags
            validation = validate_reference_tags(asset.category, asset.effective_tags)
            if not validation.selectable:
                asset.status = AssetStatus.NEEDS_REVIEW
            return self.gallery.storage.update_reference_asset(asset)

    def enable(self, asset_id: str, expected_version: str = "") -> ReferenceAsset:
        with self.gallery.storage.transaction():
            asset = self._require_version(asset_id, expected_version)
            validation = validate_reference_tags(asset.category, asset.effective_tags)
            if not validation.selectable:
                raise ReferenceNotSelectableError("参考图生效标签不合格，不能启用")
            asset.status = AssetStatus.ACTIVE
            return self.gallery.storage.update_reference_asset(asset)

    def disable(self, asset_id: str, expected_version: str = "") -> ReferenceAsset:
        with self.gallery.storage.transaction():
            asset = self._require_version(asset_id, expected_version)
            asset.status = AssetStatus.DISABLED
            return self.gallery.storage.update_reference_asset(asset)

    def delete(self, asset_id: str, expected_version: str = "") -> ReferenceAsset:
        """Tombstone one asset, then remove only its safely managed artifacts.

        Unsafe or shared paths reject the entire request before the tombstone.
        An unlink failure leaves the tombstone in place and records a safe error
        code, so a later startup scan cannot treat the leftover file as new.
        """

        if self.gallery.storage.connection.in_transaction:
            raise GalleryError("删除参考图不能嵌套在外部事务中")
        with self.gallery.storage.transaction():
            asset = self._require_version(asset_id, expected_version)
            try:
                paths = self._delete_paths(asset)
            except (OSError, RuntimeError) as exc:
                if isinstance(exc, GalleryError):
                    raise
                raise GalleryError("无法验证参考图文件安全性，不能删除") from None
            asset = self.gallery.soft_delete(asset.id)

        # Keep the tombstone durable even if filesystem cleanup fails.  The
        # second transaction prevents live records sharing these paths from
        # appearing between the final safety check and unlink.
        with self.gallery.storage.transaction():
            asset = self.gallery.require(asset.id, include_deleted=True)
            cleanup_errors: list[str] = []
            try:
                paths = self._delete_paths(asset)
            except (GalleryError, OSError, RuntimeError) as exc:
                cleanup_errors.append(f"paths:{type(exc).__name__}")
            else:
                for field_name, path in paths:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError as exc:
                        cleanup_errors.append(f"{field_name}:{type(exc).__name__}")
            if cleanup_errors:
                asset.selection_metadata["cleanup_error"] = "; ".join(cleanup_errors)
                asset = self.gallery.storage.update_reference_asset(asset)
            return asset

    def _require_version(self, asset_id: str, expected_version: str) -> ReferenceAsset:
        asset = self.gallery.require(asset_id)
        if not isinstance(expected_version, str):
            raise ValueError("参考图版本必须为 updated_at ISO 字符串")
        if expected_version and expected_version != asset.updated_at.isoformat():
            raise ReferenceVersionConflictError("参考图已变更，请刷新后重试")
        return asset

    def _delete_paths(self, asset: ReferenceAsset) -> list[tuple[str, Path]]:
        paths: list[tuple[str, Path]] = []
        seen: set[Path] = set()
        for field_name, kind in (("source_path", "sources"), ("reference_path", "references")):
            path = self._managed_path(getattr(asset, field_name), kind, asset)
            if path not in seen:
                paths.append((field_name, path))
                seen.add(path)

        offset = 0
        while True:
            assets = self.gallery.list_assets(limit=10_000, offset=offset)
            for other in assets:
                if other.id == asset.id:
                    continue
                for other_path in (other.source_path, other.reference_path):
                    candidate = Path(other_path)
                    if not candidate.is_absolute():
                        candidate = self.data_dir / candidate
                    candidate = candidate.resolve()
                    if any(candidate == path or self._same_file(candidate, path) for _, path in paths):
                        raise GalleryError("参考图文件被其他存活素材共用，不能删除")
            if len(assets) < 10_000:
                break
            offset += len(assets)
        return paths

    def _managed_path(self, value: Path | str, kind: str, asset: ReferenceAsset) -> Path:
        raw = Path(value)
        if ".." in raw.parts:
            raise GalleryError("参考图文件路径不受管理，不能删除")
        path = raw if raw.is_absolute() else self.data_dir / raw
        path = Path(os.path.abspath(path))
        root = self.data_dir / kind / asset.category.value
        if not path.is_relative_to(root) or path == root:
            raise GalleryError("参考图文件不在受管数据目录内，不能删除")

        # Inspect every component before resolving it.  Resolving first would
        # erase symlink information and could authorize an external target.
        for component in (*reversed(path.parents), path):
            try:
                info = component.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(info.st_mode):
                raise GalleryError("参考图文件路径包含符号链接，不能删除")
            if component == path and not stat.S_ISREG(info.st_mode):
                raise GalleryError("参考图文件不是普通文件，不能删除")
            if component != path and not stat.S_ISDIR(info.st_mode):
                raise GalleryError("参考图父路径不是目录，不能删除")
        return path

    @staticmethod
    def _same_file(first: Path, second: Path) -> bool:
        try:
            return first.samefile(second)
        except FileNotFoundError:
            return False
