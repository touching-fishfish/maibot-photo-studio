from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from maitu_photo.compression import compress_image
from maitu_photo.gallery import ReferenceGallery
from maitu_photo.models import ReferenceAsset, ReferenceCategory
from maitu_photo.storage import SQLiteStorage
from maitu_photo.webui_images import WebUIImages


def _asset(gallery: ReferenceGallery, root: Path, colour: str = "red") -> ReferenceAsset:
    output = io.BytesIO()
    Image.new("RGB", (1800, 1400), colour).save(output, "PNG")
    image = compress_image(output.getvalue())
    asset = ReferenceAsset(
        category=ReferenceCategory.OUTFIT,
        name=colour,
        source_path=root / "sources" / "outfit" / f"{colour}.jpg",
        reference_path=root / "references" / "outfit" / f"{colour}.jpg",
        sha256=image.sha256,
    )
    for path in (asset.source_path, asset.reference_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(image.data)
    return gallery.storage.create_reference_asset(asset)


def test_query_is_read_only_until_explicit_prepare(tmp_path: Path) -> None:
    with SQLiteStorage(tmp_path / "data" / "maitu.sqlite3") as storage:
        gallery = ReferenceGallery(storage)
        asset = _asset(gallery, tmp_path / "data")
        images = WebUIImages(gallery, tmp_path / "data", tmp_path / "runtime")
        before = storage._connection.total_changes
        assert images.query([asset.id]) == {"images": [{"asset_id": asset.id, "variant": "thumbnail", "ready": False}]}
        assert not list(images.root.iterdir())
        assert storage._connection.total_changes == before
        original = asset.reference_path.read_bytes()
        images.prepare(asset.id, "thumbnail")
        response = images.query([asset.id])
        assert response["images"][0]["ready"] is True
        encoded = response["images"][0]["data_url"].split(",", 1)[1]
        data = base64.b64decode(encoded)
        assert len(data) <= 24_000
        with Image.open(io.BytesIO(data)) as preview:
            assert max(preview.size) <= 320
        assert asset.reference_path.read_bytes() == original
        assert storage._connection.total_changes == before
        assert str(tmp_path) not in json.dumps(response)


def test_full_preview_budget_and_cache_invalidation(tmp_path: Path) -> None:
    with SQLiteStorage(tmp_path / "data" / "maitu.sqlite3") as storage:
        gallery = ReferenceGallery(storage)
        asset = _asset(gallery, tmp_path / "data")
        images = WebUIImages(gallery, tmp_path / "data", tmp_path / "runtime")
        for variant in ("source", "reference"):
            images.prepare(asset.id, variant)
            response = images.query([asset.id], variant)
            assert len(json.dumps(response).encode()) < 480_000
            data = base64.b64decode(response["images"][0]["data_url"].split(",", 1)[1])
            assert len(data) <= 250_000
        old_files = set(images.root.glob(f"{asset.id}-reference-*.jpg"))
        assert len(old_files) == 1
        gallery.edit(asset.id, name="renamed")
        assert not images.query([asset.id], "reference")["images"][0]["ready"]
        images.prepare(asset.id, "reference")
        assert len(list(images.root.glob(f"{asset.id}-reference-*.jpg"))) == 1
        assert not all(path.exists() for path in old_files)


@pytest.mark.parametrize("variant", ["thumbnail", "source", "reference"])
def test_deleted_missing_escape_and_symlink_images_rejected(tmp_path: Path, variant: str) -> None:
    with SQLiteStorage(tmp_path / "data" / "maitu.sqlite3") as storage:
        gallery = ReferenceGallery(storage)
        asset = _asset(gallery, tmp_path / "data")
        images = WebUIImages(gallery, tmp_path / "data", tmp_path / "runtime")
        source = asset.source_path if variant == "source" else asset.reference_path
        outside = tmp_path / "outside.jpg"
        outside.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(outside)
        with pytest.raises(ValueError):
            images.prepare(asset.id, variant)
        source.unlink()
        with pytest.raises(ValueError):
            images.query([asset.id], variant)
        source.write_bytes(outside.read_bytes())
        gallery.soft_delete(asset.id)
        with pytest.raises(RuntimeError):
            images.query([asset.id], variant)
        assert outside.exists()


def test_delete_invalidates_only_target_asset_previews(tmp_path: Path) -> None:
    with SQLiteStorage(tmp_path / "data" / "maitu.sqlite3") as storage:
        gallery = ReferenceGallery(storage)
        asset = _asset(gallery, tmp_path / "data")
        other = _asset(gallery, tmp_path / "data", "blue")
        images = WebUIImages(gallery, tmp_path / "data", tmp_path / "runtime")
        for variant in ("thumbnail", "source", "reference"):
            images.prepare(asset.id, variant)
        images.prepare(other.id, "thumbnail")
        gallery.soft_delete(asset.id)
        images.invalidate(asset.id)
        assert list(images.root.glob(f"{asset.id}-*.jpg")) == []
        assert images.query([other.id])["images"][0]["ready"] is True
        assert other.source_path.is_file() and other.reference_path.is_file()
        with pytest.raises(ValueError):
            images.invalidate("../*")


def test_preview_batch_limits(tmp_path: Path) -> None:
    with SQLiteStorage(tmp_path / "data" / "maitu.sqlite3") as storage:
        gallery = ReferenceGallery(storage)
        asset = _asset(gallery, tmp_path / "data")
        images = WebUIImages(gallery, tmp_path / "data", tmp_path / "runtime")
        with pytest.raises(ValueError):
            images.query([asset.id] * 7)
        with pytest.raises(ValueError):
            images.query([asset.id] * 2, "reference")
        with pytest.raises(ValueError):
            images.query([], "thumbnail")
        with pytest.raises(ValueError):
            images.prepare(asset.id, "raw")


def test_six_thumbnails_stay_under_result_limit(tmp_path: Path) -> None:
    with SQLiteStorage(tmp_path / "data" / "maitu.sqlite3") as storage:
        gallery = ReferenceGallery(storage)
        images = WebUIImages(gallery, tmp_path / "data", tmp_path / "runtime")
        assets = [
            _asset(gallery, tmp_path / "data", colour)
            for colour in ("red", "blue", "green", "orange", "black", "white")
        ]
        for asset in assets:
            images.prepare(asset.id, "thumbnail")
        response = images.query([asset.id for asset in assets])
        assert len(response["images"]) == 6
        assert len(json.dumps(response).encode()) < 480_000
