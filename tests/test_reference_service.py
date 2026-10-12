from __future__ import annotations

import asyncio
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from maitu_photo.compression import CompressionConfig, CompressionResult, compress_image
from maitu_photo.gallery import DuplicateReferenceError, ReferenceGallery
from maitu_photo.models import AssetStatus, ImageTask, ReferenceAsset, ReferenceCategory, TaskStatus
from maitu_photo.provider import GeneratedImage, OpenAICompatibleProvider, ProviderImageDecodeError
from maitu_photo.reference_service import (
    ReferenceGenerationError,
    ReferencePrompts,
    ReferenceService,
    ReferenceServiceConfig,
    ReferenceTagValidationError,
    require_valid_reference_tags,
    validate_reference_tags,
)
from maitu_photo.storage import SQLiteStorage


def _image(
    colour: tuple[int, int, int, int] = (180, 80, 40, 180),
    *,
    size: tuple[int, int] = (1400, 900),
) -> bytes:
    output = io.BytesIO()
    Image.new("RGBA", size, colour).save(
        output,
        "PNG",
        pnginfo=None,
    )
    return output.getvalue()


def _tags(category: ReferenceCategory) -> dict[str, Any]:
    if category == ReferenceCategory.PERSON:
        return {
            "appearance_summary": "dark shoulder-length hair",
            "confidence": 0.92,
        }
    if category == ReferenceCategory.OUTFIT:
        return {
            "type": "dress",
            "wearing_scenes": ["home", "casual"],
            "seasons": ["spring", "summer"],
            "styles": ["minimal"],
            "confidence": 0.88,
        }
    return {
        "room_type": "bedroom",
        "privacy_eligible": True,
        "scene_signature": "bedroom-window-left",
        "confidence": 0.9,
    }


class FakeProvider:
    def __init__(self, outputs: list[bytes] | None = None) -> None:
        self.outputs = list(outputs or [_image((20, 150, 90, 255))])
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def generate(self, prompt: str, **kwargs: Any) -> GeneratedImage:
        self.calls.append((prompt, kwargs))
        data = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        return GeneratedImage(data=data, media_type="image/png")


class FakeLLM:
    def __init__(self, replies: list[dict[str, Any] | Exception]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def generate_json(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"prompt": prompt, **kwargs})
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


class FailingProvider:
    async def generate(self, prompt: str, **kwargs: Any) -> GeneratedImage:
        raise RuntimeError("Bearer sk-sensitive-compatible-provider-key")


class ProviderDecodeFailingProvider:
    async def generate(self, prompt: str, **kwargs: Any) -> GeneratedImage:
        raise ProviderImageDecodeError("provider returned a protected image URL")


def _run(coro):
    return asyncio.run(coro)


def _service(
    tmp_path: Path,
    *,
    provider: FakeProvider,
    llm: FakeLLM,
    auto_enable: bool = True,
    before_provider_request: Callable[[], None] | None = None,
    before_tag_request: Callable[[], None] | None = None,
    after_persist: Callable[[ReferenceAsset], None] | None = None,
    check_mutation: Callable[[], None] | None = None,
    mutation_lock: asyncio.Lock | None = None,
) -> tuple[SQLiteStorage, ReferenceGallery, ReferenceService]:
    storage = SQLiteStorage(tmp_path / "db" / "maitu.sqlite3")
    gallery = ReferenceGallery(storage)
    config = ReferenceServiceConfig(
        compression=CompressionConfig(target_bytes=120_000, max_edge=1024),
        auto_enable_generated_references=auto_enable,
        prompts=ReferencePrompts(),
        reference_model="reference-model",
        reference_mode="images_api",
        prompt_version="tests-v1",
    )
    service = ReferenceService(
        gallery=gallery,
        provider=provider,  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
        data_dir=tmp_path / "plugin-data",
        config=config,
        before_provider_request=before_provider_request,
        before_tag_request=before_tag_request,
        after_persist=after_persist,
        check_mutation=check_mutation,
        mutation_lock=mutation_lock,
    )
    return storage, gallery, service


def test_extract_outfit_uses_sheet_prompt_and_compresses_both_artifacts(
    tmp_path: Path,
) -> None:
    provider = FakeProvider()
    llm = FakeLLM([_tags(ReferenceCategory.OUTFIT)])
    storage, _, service = _service(tmp_path, provider=provider, llm=llm)

    asset = _run(
        service.extract_reference(
            "outfit",
            _image(),
            name="summer dress",
            source_task_id="task-1",
        )
    )

    assert asset.status == AssetStatus.ACTIVE
    assert asset.tags["type"] == "dress"
    assert asset.selection_metadata["layout"] == "2x2_outfit_sheet"
    assert "2x2" in provider.calls[0][0]
    assert provider.calls[0][1]["extraction"] is True
    assert provider.calls[0][1]["model"] == "reference-model"
    assert provider.calls[0][1]["mode"] == "images_api"
    provider_input = provider.calls[0][1]["images"][0]
    assert provider_input.startswith(b"\xff\xd8\xff")
    assert len(provider_input) <= 120_000

    for path in (asset.source_path, asset.reference_path):
        assert path.resolve().is_relative_to((tmp_path / "plugin-data").resolve())
        assert path.stat().st_size <= 120_000
        with Image.open(path) as image:
            image.load()
            assert image.format == "JPEG"
            assert image.mode == "RGB"
            assert image.getexif() == {}
            assert "icc_profile" not in image.info
    assert llm.calls[0]["image_bytes"].startswith(b"\xff\xd8\xff")
    storage.close()


def test_reference_generation_options_and_explicit_size_are_forwarded(tmp_path: Path) -> None:
    provider = FakeProvider()
    llm = FakeLLM([_tags(ReferenceCategory.OUTFIT), _tags(ReferenceCategory.SCENE)])
    storage = SQLiteStorage(tmp_path / "db" / "maitu.sqlite3")
    service = ReferenceService(
        gallery=ReferenceGallery(storage),
        provider=provider,  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
        data_dir=tmp_path / "plugin-data",
        config=ReferenceServiceConfig(
            extraction_size="1536x1024",
            extraction_quality="high",
            extraction_output_format="webp",
            extraction_moderation="low",
            extraction_extra_params={"background": "opaque", "metadata": {"source": "test"}},
        ),
    )

    _run(service.extract_reference("outfit", _image(), name="default size"))
    _run(service.extract_reference("scene", _image(), name="explicit size", size="1024x1024"))

    assert provider.calls[0][1]["size"] == "1536x1024"
    assert provider.calls[1][1]["size"] == "1024x1024"
    for _, kwargs in provider.calls:
        assert kwargs["quality"] == "high"
        assert kwargs["output_format"] == "webp"
        assert kwargs["moderation"] == "low"
        assert kwargs["extra"] == {"background": "opaque", "metadata": {"source": "test"}}
    storage.close()


def test_tag_schema_is_exact_and_public_scene_references_are_not_selectable() -> None:
    valid = validate_reference_tags("outfit", _tags(ReferenceCategory.OUTFIT))
    assert valid.schema_valid is True
    assert valid.selectable is True

    with_extra = {**_tags(ReferenceCategory.OUTFIT), "unexpected": "field"}
    invalid = validate_reference_tags("outfit", with_extra)
    assert invalid.schema_valid is False
    assert invalid.error_code == "tag_schema_invalid"
    with pytest.raises(ReferenceTagValidationError):
        require_valid_reference_tags("outfit", with_extra)

    scene = validate_reference_tags(
        "scene",
        {
            **_tags(ReferenceCategory.SCENE),
            "light": "window light",
            "time_of_day": "morning",
            "privacy_eligible": False,
        },
    )
    assert scene.schema_valid is True
    assert scene.selectable is False
    assert scene.error_code == "scene_not_private"
    assert set(scene.tags) == {"room_type", "privacy_eligible", "scene_signature", "confidence"}

    person = validate_reference_tags("person", _tags(ReferenceCategory.PERSON))
    assert person.schema_valid is True
    assert "accessories" not in person.tags


def test_generate_person_from_personality_does_not_send_source_image(tmp_path: Path) -> None:
    provider = FakeProvider()
    llm = FakeLLM([_tags(ReferenceCategory.PERSON)])
    storage, gallery, service = _service(tmp_path, provider=provider, llm=llm)

    asset = _run(
        service.generate_person_from_personality(
            name="persona",
            personality="大二女生，短发，圆脸",
            nickname="bot",
            appearance_hint="自然刘海",
        )
    )

    assert asset.status.value == "active"
    assert gallery.get_person() is not None
    assert "images" not in provider.calls[0][1]
    assert "大二女生" in provider.calls[0][0]
    assert "自然刘海" in provider.calls[0][0]
    storage.close()


def test_scene_generation_attributes_are_not_persisted_and_public_scene_is_saved_for_review(
    tmp_path: Path,
) -> None:
    legacy_scene_tags = {
        **_tags(ReferenceCategory.SCENE),
        "light": "window light",
        "time_of_day": "morning",
        "privacy_eligible": False,
    }
    provider = FakeProvider()
    llm = FakeLLM([legacy_scene_tags, RuntimeError("Bearer secret-value-should-not-persist")])
    storage, gallery, service = _service(tmp_path, provider=provider, llm=llm)

    scene = _run(service.import_reference("scene", _image(), name="cafe"))
    assert scene.status == AssetStatus.NEEDS_REVIEW
    assert scene.tags == {**_tags(ReferenceCategory.SCENE), "privacy_eligible": False}
    assert "light" not in scene.effective_tags
    assert "time_of_day" not in scene.effective_tags
    assert scene.effective_tags["privacy_eligible"] is False
    assert gallery.candidates("scene") == []

    outfit = _run(service.import_reference("outfit", _image((30, 40, 50, 255)), name="unknown"))
    assert outfit.status == AssetStatus.NEEDS_REVIEW
    assert outfit.tags == {}
    metadata_json = json.dumps(outfit.selection_metadata)
    assert "secret-value" not in metadata_json
    assert outfit.selection_metadata["tag_error_code"] == "tagging_failed"
    storage.close()


def test_malformed_non_json_tags_still_create_review_asset(tmp_path: Path) -> None:
    provider = FakeProvider()
    llm = FakeLLM([{1: object()}])  # type: ignore[dict-item, list-item]
    storage, _, service = _service(tmp_path, provider=provider, llm=llm)

    asset = _run(service.import_reference("outfit", _image(), name="review me"))

    assert asset.status == AssetStatus.NEEDS_REVIEW
    assert asset.tags == {}
    assert asset.selection_metadata["tag_error_code"] == "tag_schema_invalid"
    storage.close()


def test_person_requires_explicit_replace_and_gallery_deduplicates_imports(
    tmp_path: Path,
) -> None:
    provider = FakeProvider()
    llm = FakeLLM(
        [
            _tags(ReferenceCategory.PERSON),
            _tags(ReferenceCategory.PERSON),
            _tags(ReferenceCategory.OUTFIT),
            _tags(ReferenceCategory.OUTFIT),
        ]
    )
    storage, gallery, service = _service(tmp_path, provider=provider, llm=llm)
    first = _run(service.import_reference("person", _image(), name="first"))

    duplicate_person = _run(service.import_reference("person", _image(), name="second"))
    assert duplicate_person == first
    assert len(llm.calls) == 1

    with pytest.raises(DuplicateReferenceError) as conflict:
        _run(service.import_reference("person", _image((2, 3, 4, 255)), name="second"))
    assert conflict.value.existing_id == first.id
    assert len(llm.calls) == 1

    second = _run(
        service.import_reference(
            "person",
            _image((10, 20, 30, 255)),
            name="second",
            replace_person=True,
        )
    )
    assert gallery.get_person() is not None
    assert gallery.get_person().id == second.id  # type: ignore[union-attr]
    assert storage.get_reference_asset(first.id, include_deleted=True).status == AssetStatus.DELETED  # type: ignore[union-attr]

    source = _image((70, 80, 90, 255))
    outfit = _run(service.import_reference("outfit", source, name="first outfit"))
    duplicate = _run(service.import_reference("outfit", source, name="duplicate outfit"))
    assert duplicate == outfit
    assert len(llm.calls) == 3
    assert len(list((tmp_path / "plugin-data" / "sources" / "outfit").glob("*.jpg"))) == 1
    assert len(list((tmp_path / "plugin-data" / "references" / "outfit").glob("*.jpg"))) == 1
    storage.close()


def test_retag_failure_marks_existing_asset_needs_review(tmp_path: Path) -> None:
    provider = FakeProvider()
    llm = FakeLLM([_tags(ReferenceCategory.OUTFIT), RuntimeError("upstream unavailable")])
    storage, _, service = _service(tmp_path, provider=provider, llm=llm)
    asset = _run(service.import_reference("outfit", _image(), name="dress"))
    assert asset.status == AssetStatus.ACTIVE

    retagged = _run(service.retag_reference(asset.id))
    assert retagged.status == AssetStatus.NEEDS_REVIEW
    assert retagged.selection_metadata["tag_error_code"] == "tagging_failed"
    storage.close()


def test_regenerate_creates_auditable_replacement_and_deletes_old_asset(
    tmp_path: Path,
) -> None:
    provider = FakeProvider([_image((15, 160, 210, 255))])
    llm = FakeLLM([_tags(ReferenceCategory.OUTFIT), _tags(ReferenceCategory.OUTFIT)])
    storage, gallery, service = _service(tmp_path, provider=provider, llm=llm)
    original = _run(
        service.import_reference(
            "outfit",
            _image((210, 40, 50, 255)),
            name="dress",
            manual_tags={"styles": ["admin-edited"]},
        )
    )

    replacement = _run(service.regenerate_reference(original.id))

    assert replacement.id != original.id
    assert replacement.manual_tags == {"styles": ["admin-edited"]}
    assert replacement.status == AssetStatus.ACTIVE
    assert replacement.reference_path.stat().st_size <= 120_000
    deleted = storage.get_reference_asset(original.id, include_deleted=True)
    assert deleted is not None and deleted.status == AssetStatus.DELETED
    assert gallery.get(original.id) is None
    assert provider.calls[0][1]["images"][0].startswith(b"\xff\xd8\xff")
    storage.close()


def test_valid_import_respects_auto_enable_setting(tmp_path: Path) -> None:
    provider = FakeProvider()
    llm = FakeLLM([_tags(ReferenceCategory.OUTFIT)])
    storage, _, service = _service(
        tmp_path,
        provider=provider,
        llm=llm,
        auto_enable=False,
    )
    asset = _run(service.import_reference("outfit", _image(), name="dress"))
    assert asset.status == AssetStatus.DISABLED
    storage.close()


def test_extraction_error_does_not_expose_provider_response(tmp_path: Path) -> None:
    storage, _, service = _service(
        tmp_path,
        provider=FailingProvider(),  # type: ignore[arg-type]
        llm=FakeLLM([_tags(ReferenceCategory.OUTFIT)]),
    )

    with pytest.raises(ReferenceGenerationError) as caught:
        _run(service.extract_reference("outfit", _image(), name="dress"))

    assert "sensitive-compatible-provider-key" not in str(caught.value)
    assert list((tmp_path / "plugin-data").rglob("*.jpg")) == []
    storage.close()


def test_extraction_error_preserves_safe_provider_category(tmp_path: Path) -> None:
    storage, _, service = _service(
        tmp_path,
        provider=ProviderDecodeFailingProvider(),  # type: ignore[arg-type]
        llm=FakeLLM([_tags(ReferenceCategory.OUTFIT)]),
    )

    with pytest.raises(ReferenceGenerationError, match=r"provider_image_decode"):
        _run(service.extract_reference("outfit", _image(), name="dress"))

    storage.close()


def test_compression_target_above_480000_is_rejected() -> None:
    with pytest.raises(ValueError, match="480000"):
        ReferenceServiceConfig(compression=CompressionConfig(target_bytes=480_001))


def test_import_duplicate_returns_original_unchanged_without_models_or_new_files(tmp_path: Path) -> None:
    provider = FakeProvider()
    llm = FakeLLM([_tags(ReferenceCategory.OUTFIT)])
    persisted: list[ReferenceAsset] = []
    storage, gallery, service = _service(tmp_path, provider=provider, llm=llm, after_persist=persisted.append)
    source = _image()
    original = _run(service.import_reference("outfit", source, name="first", source_task_id="first-task"))
    original = gallery.edit(original.id, manual_tags={"styles": ["manual"]}, selection_metadata={"custom": "keep"})
    original = gallery.disable(original.id)
    files_before = {path: path.read_bytes() for path in service.data_dir.rglob("*.jpg")}

    duplicate = _run(
        service.import_reference(
            "outfit", source, name="not applied", manual_tags={"invalid": object()}, source_task_id="new-task"
        )
    )

    assert duplicate == original
    assert gallery.require(original.id) == original
    assert len(llm.calls) == 1 and provider.calls == []
    assert persisted[-1] == original and len(persisted) == 2
    assert {path: path.read_bytes() for path in service.data_dir.rglob("*.jpg")} == files_before
    storage.close()


def test_processed_hash_dedup_is_category_specific_and_ignores_deleted_rows(tmp_path: Path) -> None:
    storage, gallery, service = _service(
        tmp_path,
        provider=FakeProvider(),
        llm=FakeLLM([_tags(ReferenceCategory.OUTFIT), _tags(ReferenceCategory.SCENE)]),
    )
    source = _image()
    outfit = _run(service.import_reference("outfit", source, name="outfit"))
    scene = _run(service.import_reference("scene", source, name="scene"))
    assert outfit.sha256 == scene.sha256 and outfit.id != scene.id
    gallery.soft_delete(outfit.id)
    service.llm = FakeLLM([_tags(ReferenceCategory.OUTFIT)])  # type: ignore[assignment]

    replacement = _run(service.import_reference("outfit", source, name="re-import"))

    assert replacement.id != outfit.id
    assert len(service.llm.calls) == 1  # type: ignore[attr-defined]
    storage.close()


@pytest.mark.parametrize("operation", ["import", "extract"])
def test_prepared_source_skips_first_compression_and_preserves_reference_bytes(tmp_path: Path, operation: str) -> None:
    storage, _, service = _service(tmp_path, provider=FakeProvider(), llm=FakeLLM([_tags(ReferenceCategory.OUTFIT)]))
    prepared = compress_image(_image(), config=service.config.compression)
    expected = compress_image(prepared.data, config=service.config.compression)
    if operation == "import":
        asset = _run(
            service.import_reference("outfit", b"unused invalid source", name="prepared", prepared_source=prepared)
        )
        assert asset.reference_path.read_bytes() == expected.data
    else:
        asset = _run(
            service.extract_reference("outfit", b"unused invalid source", name="prepared", prepared_source=prepared)
        )
        assert service.provider.calls[0][1]["images"] == [prepared.data]  # type: ignore[union-attr]
    assert asset.source_path.read_bytes() == prepared.data
    assert asset.selection_metadata["source_compression"] == prepared.to_dict()
    storage.close()


def test_final_transaction_rechecks_concurrent_duplicate_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, gallery, service = _service(
        tmp_path, provider=FakeProvider(), llm=FakeLLM([_tags(ReferenceCategory.OUTFIT)])
    )
    original_compress = service._compress
    reference_result: CompressionResult | None = None
    existing: ReferenceAsset | None = None
    write_calls: list[Path] = []

    async def track_compress(source: Any) -> CompressionResult:
        nonlocal reference_result
        reference_result = await original_compress(source)
        return reference_result

    class RacingLLM(FakeLLM):
        async def generate_json(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
            nonlocal existing
            assert reference_result is not None
            existing = gallery.add_reference(
                category="outfit",
                name="race winner",
                source_path=service.data_dir / "sources" / "outfit" / "race.jpg",
                reference_path=service.data_dir / "references" / "outfit" / "race.jpg",
                sha256=reference_result.sha256,
                tags=_tags(ReferenceCategory.OUTFIT),
                status=AssetStatus.DISABLED,
            )
            return await super().generate_json(prompt, **kwargs)

    monkeypatch.setattr(service, "_compress", track_compress)
    monkeypatch.setattr(service, "_atomic_write", lambda path, data: write_calls.append(path))
    service.llm = RacingLLM([_tags(ReferenceCategory.OUTFIT)])  # type: ignore[assignment]

    result = _run(service.import_reference("outfit", _image(), name="race loser"))

    assert existing is not None and result == existing
    assert write_calls == []
    assert len(gallery.list_assets()) == 1
    storage.close()


def test_shared_mutation_lock_serializes_final_duplicate_commit(tmp_path: Path) -> None:
    async def scenario() -> None:
        lock = asyncio.Lock()
        storage, gallery, service = _service(
            tmp_path, provider=FakeProvider(), llm=FakeLLM([_tags(ReferenceCategory.OUTFIT)]), mutation_lock=lock
        )
        entered = 0
        release = asyncio.Event()
        callbacks: list[str] = []

        class WaitingLLM(FakeLLM):
            async def generate_json(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
                nonlocal entered
                entered += 1
                if entered == 2:
                    release.set()
                await release.wait()
                return await super().generate_json(prompt, **kwargs)

        llm = WaitingLLM([_tags(ReferenceCategory.OUTFIT)])

        def after_persist(asset: ReferenceAsset) -> None:
            assert lock.locked()
            assert storage.connection.in_transaction
            callbacks.append(asset.id)

        services = [
            service,
            ReferenceService(
                gallery=gallery,
                provider=service.provider,
                llm=llm,
                data_dir=service.data_dir,
                config=service.config,
                mutation_lock=lock,
                after_persist=after_persist,
            ),
        ]
        service.llm = llm  # type: ignore[assignment]
        service._after_persist = after_persist
        first, second = await asyncio.gather(
            services[0].import_reference("outfit", _image(), name="first"),
            services[1].import_reference("outfit", _image(), name="second"),
        )
        assert first == second
        assert len(gallery.list_assets()) == 1
        assert len(list(service.data_dir.rglob("*.jpg"))) == 2
        assert callbacks == [first.id, first.id]
        storage.close()

    _run(scenario())


@pytest.mark.parametrize("case", ["same", "other_duplicate"])
def test_regenerate_duplicate_does_not_retag_mutate_or_delete_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    provider = FakeProvider()
    llm = FakeLLM([_tags(ReferenceCategory.OUTFIT)])
    storage, gallery, service = _service(tmp_path, provider=provider, llm=llm)
    original = _run(service.import_reference("outfit", _image(), name="target", manual_tags={"styles": ["manual"]}))
    existing = original
    if case == "other_duplicate":
        existing = _run(service.import_reference("outfit", _image((7, 8, 9, 255)), name="existing"))
        existing = gallery.disable(existing.id)
    persisted: list[ReferenceAsset] = []
    service._after_persist = persisted.append
    artifact = compress_image(existing.reference_path, config=service.config.compression)
    # Force a byte-identical processed provider result, avoiding encoder-version
    # differences: these tests concern the hash/commit path, not JPEG quality.
    artifact = CompressionResult(
        data=existing.reference_path.read_bytes(), sha256=existing.sha256, width=artifact.width, height=artifact.height
    )
    original_compress = service._compress
    compression_calls = 0

    async def regenerated_compress(source: Any) -> CompressionResult:
        nonlocal compression_calls
        compression_calls += 1
        return artifact if compression_calls == 2 else await original_compress(source)

    monkeypatch.setattr(service, "_compress", regenerated_compress)
    llm_calls_before = len(llm.calls)
    files_before = {path: path.read_bytes() for path in service.data_dir.rglob("*.jpg")}

    result = _run(service.regenerate_reference(original.id, name="not applied", prompt_version="not applied"))

    assert result == existing
    assert len(llm.calls) == llm_calls_before
    assert len(provider.calls) == 1
    assert gallery.require(original.id) == original
    assert gallery.require(existing.id) == existing
    assert persisted == [existing]
    assert {path: path.read_bytes() for path in service.data_dir.rglob("*.jpg")} == files_before
    storage.close()


@pytest.mark.parametrize("initial_status", [AssetStatus.ACTIVE, AssetStatus.DISABLED, AssetStatus.NEEDS_REVIEW])
@pytest.mark.parametrize("auto_enable", [True, False])
def test_retag_valid_preserves_manual_and_explicit_status(
    tmp_path: Path, initial_status: AssetStatus, auto_enable: bool
) -> None:
    llm = FakeLLM([_tags(ReferenceCategory.OUTFIT), {**_tags(ReferenceCategory.OUTFIT), "type": "coat"}])
    storage, gallery, service = _service(tmp_path, provider=FakeProvider(), llm=llm, auto_enable=auto_enable)
    asset = _run(service.import_reference("outfit", _image(), name="target", manual_tags={"styles": ["manual"]}))
    asset = gallery._set_status(asset.id, initial_status)
    persisted: list[ReferenceAsset] = []
    service._after_persist = persisted.append

    retagged = _run(service.retag_reference(asset.id))

    expected = initial_status
    if initial_status == AssetStatus.NEEDS_REVIEW:
        expected = AssetStatus.ACTIVE if auto_enable else AssetStatus.DISABLED
    assert retagged.status == expected
    assert retagged.tags["type"] == "coat"
    assert retagged.manual_tags == {"styles": ["manual"]}
    assert retagged.selection_metadata["tag_selectable"] is True
    assert persisted == [retagged]
    storage.close()


@pytest.mark.parametrize(
    ("ai_private", "manual_private", "expected_status"),
    [(True, False, AssetStatus.NEEDS_REVIEW), (False, True, AssetStatus.ACTIVE)],
)
def test_new_and_retagged_scene_use_final_effective_tags(
    tmp_path: Path, ai_private: bool, manual_private: bool, expected_status: AssetStatus
) -> None:
    llm = FakeLLM([{**_tags(ReferenceCategory.SCENE), "privacy_eligible": ai_private}])
    storage, _, service = _service(tmp_path, provider=FakeProvider(), llm=llm)
    asset = _run(
        service.import_reference("scene", _image(), name="scene", manual_tags={"privacy_eligible": manual_private})
    )
    assert asset.status == expected_status
    retagged = _run(service.retag_reference(asset.id))
    assert retagged.status == expected_status
    assert retagged.tags["privacy_eligible"] is ai_private
    assert retagged.manual_tags == {"privacy_eligible": manual_private}
    assert retagged.selection_metadata["tag_selectable"] is (expected_status == AssetStatus.ACTIVE)
    storage.close()


@pytest.mark.parametrize("reply", [RuntimeError("secret"), {"unexpected": "invalid"}])
def test_retag_failed_ai_keeps_old_ai_and_manual_but_requires_review(tmp_path: Path, reply: Any) -> None:
    llm = FakeLLM([_tags(ReferenceCategory.PERSON), reply])
    storage, gallery, service = _service(tmp_path, provider=FakeProvider(), llm=llm)
    asset = _run(
        service.import_reference("person", _image(), name="person", manual_tags=_tags(ReferenceCategory.PERSON))
    )

    result = _run(service.retag_reference(asset.id))

    assert result.tags == asset.tags and result.manual_tags == asset.manual_tags
    assert result.status == AssetStatus.NEEDS_REVIEW
    assert gallery.get_person().id == asset.id  # type: ignore[union-attr]
    storage.close()


def test_failed_new_ai_cannot_be_enabled_only_by_complete_manual_tags(tmp_path: Path) -> None:
    storage, _, service = _service(tmp_path, provider=FakeProvider(), llm=FakeLLM([RuntimeError("failed")]))

    asset = _run(
        service.import_reference("outfit", _image(), name="review", manual_tags=_tags(ReferenceCategory.OUTFIT))
    )

    assert asset.tags == {} and asset.status == AssetStatus.NEEDS_REVIEW
    assert asset.manual_tags == _tags(ReferenceCategory.OUTFIT)
    storage.close()


@pytest.mark.parametrize("auto_enable", [True, False])
def test_retag_uses_manual_edits_made_while_tag_model_was_running(tmp_path: Path, auto_enable: bool) -> None:
    storage, gallery, service = _service(
        tmp_path, provider=FakeProvider(), llm=FakeLLM([_tags(ReferenceCategory.SCENE)]), auto_enable=auto_enable
    )
    asset = _run(service.import_reference("scene", _image(), name="scene"))

    class EditingLLM(FakeLLM):
        async def generate_json(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
            gallery.edit(asset.id, manual_tags={"privacy_eligible": False})
            return await super().generate_json(prompt, **kwargs)

    service.llm = EditingLLM([_tags(ReferenceCategory.SCENE)])  # type: ignore[assignment]
    retagged = _run(service.retag_reference(asset.id))

    assert retagged.manual_tags == {"privacy_eligible": False}
    assert retagged.status == AssetStatus.NEEDS_REVIEW
    storage.close()


def test_new_review_person_replaces_old_person_only_on_successful_commit(tmp_path: Path) -> None:
    storage, gallery, service = _service(
        tmp_path,
        provider=FakeProvider(),
        llm=FakeLLM([_tags(ReferenceCategory.PERSON), RuntimeError("tag model failed")]),
    )
    original = _run(service.import_reference("person", _image(), name="first"))

    review = _run(service.import_reference("person", _image((9, 7, 5, 255)), name="review", replace_person=True))

    assert review.status == AssetStatus.NEEDS_REVIEW
    assert gallery.get_person().id == review.id  # type: ignore[union-attr]
    assert gallery.require(original.id, include_deleted=True).status == AssetStatus.DELETED
    storage.close()


@pytest.mark.parametrize("operation", ["import", "extract"])
def test_replaces_asset_is_deleted_in_same_transaction_and_duplicate_does_not_delete_target(
    tmp_path: Path, operation: str
) -> None:
    storage, gallery, service = _service(
        tmp_path, provider=FakeProvider(), llm=FakeLLM([_tags(ReferenceCategory.OUTFIT)])
    )
    old = _run(service.import_reference("outfit", _image(), name="old"))
    other_source = _image((7, 9, 11, 255))
    existing = _run(service.import_reference("outfit", other_source, name="existing"))
    callbacks: list[str] = []

    def after_persist(asset: ReferenceAsset) -> None:
        assert storage.connection.in_transaction
        if asset.id != existing.id:
            assert gallery.require(old.id, include_deleted=True).status == AssetStatus.DELETED
        else:
            assert gallery.require(old.id) == old
        callbacks.append(asset.id)

    service._after_persist = after_persist
    duplicate = _run(service.import_reference("outfit", other_source, name="duplicate", replaces_asset_id=old.id))
    assert duplicate == existing and gallery.require(old.id) == old
    if operation == "import":
        result = _run(
            service.import_reference("outfit", _image((31, 41, 51, 255)), name="new", replaces_asset_id=old.id)
        )
    else:
        result = _run(
            service.extract_reference("outfit", _image((31, 41, 51, 255)), name="new", replaces_asset_id=old.id)
        )
    assert callbacks == [existing.id, result.id]
    assert gallery.require(old.id, include_deleted=True).status == AssetStatus.DELETED
    storage.close()


@pytest.mark.parametrize("cancel_at", ["before_ai", "before_commit"])
def test_check_mutation_cancellation_prevents_paid_request_or_commit(tmp_path: Path, cancel_at: str) -> None:
    provider = FakeProvider()
    llm = FakeLLM([_tags(ReferenceCategory.OUTFIT)])
    tag_calls: list[str] = []
    checks = 0
    cancelled = False

    def check_mutation() -> None:
        nonlocal checks
        checks += 1
        if cancelled and (cancel_at == "before_ai" or checks == 3):
            raise asyncio.CancelledError

    storage, gallery, service = _service(
        tmp_path,
        provider=provider,
        llm=llm,
        check_mutation=check_mutation,
        before_tag_request=lambda: tag_calls.append("paid"),
    )
    original = _run(service.import_reference("person", _image(), name="keep"))
    checks = 0
    cancelled = True
    with pytest.raises(asyncio.CancelledError):
        _run(service.import_reference("person", _image((1, 5, 9, 255)), name="cancelled", replace_person=True))

    assert gallery.get_person() == original
    assert len(list(service.data_dir.rglob("*.jpg"))) == 2
    assert not storage.connection.in_transaction
    expected_calls = 1 if cancel_at == "before_ai" else 2
    assert len(llm.calls) == expected_calls and len(tag_calls) == expected_calls
    assert provider.calls == []
    storage.close()


def test_provider_and_tag_hooks_follow_cancellation_checks(tmp_path: Path) -> None:
    events: list[str] = []
    storage, _, service = _service(
        tmp_path,
        provider=FakeProvider(),
        llm=FakeLLM([_tags(ReferenceCategory.OUTFIT)]),
        check_mutation=lambda: events.append("check"),
        before_provider_request=lambda: events.append("provider_paid"),
        before_tag_request=lambda: events.append("tag_paid"),
        after_persist=lambda asset: events.append("persisted"),
    )

    _run(service.extract_reference("outfit", _image(), name="hooks"))

    assert events == ["check", "provider_paid", "check", "check", "tag_paid", "check", "persisted"]
    storage.close()


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_after_persist_failure_rolls_back_person_replacement_and_removes_new_artifacts(
    tmp_path: Path, failure: type[BaseException]
) -> None:
    storage, gallery, service = _service(
        tmp_path, provider=FakeProvider(), llm=FakeLLM([_tags(ReferenceCategory.PERSON)])
    )
    old = _run(service.import_reference("person", _image(), name="keep"))
    task = storage.create_task(ImageTask(kind="reference", scope_key="test"))
    existing_files = set(service.data_dir.rglob("*.jpg"))

    def fail_completion(asset: ReferenceAsset) -> None:
        storage.set_task_status(task.id, TaskStatus.SENT)
        raise failure("completion failed")

    service._after_persist = fail_completion
    with pytest.raises(failure):
        _run(service.import_reference("person", _image((2, 4, 6, 255)), name="replace", replace_person=True))

    assert gallery.get_person() == old
    assert set(service.data_dir.rglob("*.jpg")) == existing_files
    assert storage.get_task(task.id).status == TaskStatus.QUEUED  # type: ignore[union-attr]
    assert not storage.connection.in_transaction
    storage.close()


def test_retag_completion_failure_rolls_back_tags_status_and_task(tmp_path: Path) -> None:
    storage, gallery, service = _service(
        tmp_path,
        provider=FakeProvider(),
        llm=FakeLLM([_tags(ReferenceCategory.OUTFIT), {**_tags(ReferenceCategory.OUTFIT), "type": "coat"}]),
    )
    original = _run(service.import_reference("outfit", _image(), name="keep"))
    task = storage.create_task(ImageTask(kind="reference", scope_key="test"))

    def fail_completion(asset: ReferenceAsset) -> None:
        storage.set_task_status(task.id, TaskStatus.SENT)
        raise RuntimeError("completion failed")

    service._after_persist = fail_completion
    with pytest.raises(RuntimeError, match="completion failed"):
        _run(service.retag_reference(original.id))

    assert gallery.require(original.id) == original
    assert storage.get_task(task.id).status == TaskStatus.QUEUED  # type: ignore[union-attr]
    storage.close()


def test_write_failure_after_atomic_replace_leaves_no_new_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, gallery, service = _service(
        tmp_path, provider=FakeProvider(), llm=FakeLLM([_tags(ReferenceCategory.PERSON)])
    )
    old = _run(service.import_reference("person", _image(), name="keep"))
    existing_files = set(service.data_dir.rglob("*.jpg"))
    atomic_write = service._atomic_write

    def fail_write(path: Path, data: bytes) -> None:
        atomic_write(path, data)
        raise OSError("disk unavailable after replace")

    monkeypatch.setattr(service, "_atomic_write", fail_write)
    with pytest.raises(OSError):
        _run(service.import_reference("person", _image((4, 8, 12, 255)), name="replace", replace_person=True))

    assert gallery.get_person() == old
    assert set(service.data_dir.rglob("*.jpg")) == existing_files
    storage.close()


def test_retry_limit_is_only_passed_to_openai_compatible_provider(tmp_path: Path) -> None:
    class RecordingCompatibleProvider(OpenAICompatibleProvider):
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def generate(self, prompt: str, **kwargs: Any) -> GeneratedImage:
            self.calls.append(kwargs)
            return GeneratedImage(data=_image((20, 150, 90, 255)), media_type="image/png")

    provider = RecordingCompatibleProvider()
    storage, _, service = _service(
        tmp_path,
        provider=provider,
        llm=FakeLLM([_tags(ReferenceCategory.OUTFIT)]),  # type: ignore[arg-type]
    )
    _run(service.extract_reference("outfit", _image(), name="no retry"))
    assert provider.calls[0]["retry_limit"] == 0
    fake = FakeProvider()
    service.provider = fake  # type: ignore[assignment]
    _run(service.extract_reference("scene", _image(), name="fake"))
    assert "retry_limit" not in fake.calls[0][1]
    storage.close()


def test_personality_generation_keeps_low_level_explicit_replace_contract(tmp_path: Path) -> None:
    storage, gallery, service = _service(
        tmp_path, provider=FakeProvider(), llm=FakeLLM([_tags(ReferenceCategory.PERSON)])
    )
    old = _run(service.import_reference("person", _image(), name="old"))
    with pytest.raises(DuplicateReferenceError):
        _run(service.generate_person_from_personality(name="new", personality="short hair"))
    assert service.provider.calls == []  # type: ignore[union-attr]

    generated = _run(
        service.generate_person_from_personality(name="new", personality="short hair", replace_person=True)
    )

    assert gallery.get_person().id == generated.id  # type: ignore[union-attr]
    assert gallery.require(old.id, include_deleted=True).status == AssetStatus.DELETED
    storage.close()
