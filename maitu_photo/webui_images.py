"""有预算的 WebUI 缩略图缓存，不暴露图库文件路径。"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .compression import CompressionConfig, compress_image
from .gallery import ReferenceGallery


class WebUIImages:
    def __init__(self, gallery: ReferenceGallery, data_dir: Path, runtime_dir: Path) -> None:
        self.gallery = gallery
        self.data_dir = data_dir.resolve()
        self.root = runtime_dir.resolve() / "webui-images"
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, asset_id: str, variant: str) -> tuple[Path, Path]:
        if variant not in {"thumbnail", "source", "reference"}:
            raise ValueError("不支持的预览类型")
        asset = self.gallery.require(asset_id)
        if asset.deleted_at is not None:
            raise ValueError("已删除的参考图不能预览")
        raw_path = asset.source_path if variant == "source" else asset.reference_path
        source = Path(raw_path)
        resolved = source.resolve()
        resolved.relative_to(self.data_dir)
        relative = source.absolute().relative_to(self.data_dir)
        current = self.data_dir
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("不能预览符号链接图片")
        if not resolved.is_file():
            raise ValueError("参考图片文件不存在")
        stat = resolved.stat()
        key = hashlib.sha256(
            f"{asset.id}:{variant}:{asset.updated_at.isoformat()}:{stat.st_mtime_ns}:{stat.st_size}".encode()
        ).hexdigest()
        return resolved, self.root / f"{asset.id}-{variant}-{key}.jpg"

    def prepare(self, asset_id: str, variant: str) -> None:
        source, target = self._paths(asset_id, variant)
        if target.is_file():
            return
        result = compress_image(
            source,
            config=CompressionConfig(
                target_bytes=24_000 if variant == "thumbnail" else 250_000,
                max_edge=320 if variant == "thumbnail" else 1600,
            ),
        )
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.root, suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(result.data)
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        for old in self.root.glob(f"{asset_id}-{variant}-*.jpg"):
            if old != target:
                old.unlink(missing_ok=True)

    def invalidate(self, asset_id: str) -> None:
        if not re.fullmatch(r"[0-9a-f]{32}", asset_id):
            raise ValueError("参考图 ID 无效")
        for path in self.root.glob(f"{asset_id}-*.jpg"):
            path.unlink(missing_ok=True)

    def query(self, asset_ids: list[str], variant: str = "thumbnail") -> dict[str, Any]:
        """只读缓存；缺失缓存由 prepare action 建立，不在 query 中写入。"""
        if not asset_ids or len(asset_ids) > (6 if variant == "thumbnail" else 1):
            raise ValueError("缩略图每次最多 6 张，大图每次只加载 1 张")
        images = []
        for asset_id in asset_ids:
            _, target = self._paths(asset_id, variant)
            item: dict[str, Any] = {"asset_id": asset_id, "variant": variant, "ready": target.is_file()}
            if item["ready"]:
                data = target.read_bytes()
                if len(data) > (24_000 if variant == "thumbnail" else 250_000):
                    raise ValueError("预览缓存超过响应预算")
                item["data_url"] = "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
            images.append(item)
        response = {"images": images}
        if len(json.dumps(response, ensure_ascii=False).encode()) > 480_000:
            raise ValueError("预览响应超过预算，请减少图片数量")
        return response
