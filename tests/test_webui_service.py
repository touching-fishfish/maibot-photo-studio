"""Offline service lifecycle checks; every model and delivery capability is fake."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest
from test_service import _LLM, _BlockingProvider, _config, _Context, _invocation, _png, _Provider

from maitu_photo.compression import compress_image
from maitu_photo.models import ImageTask, ReferenceAsset, TaskReference, TaskStatus
from maitu_photo.runtime import InvocationError, ReferenceJobContext, invocation_context
from maitu_photo.selection import SelectionResult
from maitu_photo.service import PhotoStudioError, PhotoStudioService
from maitu_photo.storage import StorageError


def _tags() -> dict:
    return {
        "type": "shirt",
        "wearing_scenes": ["home"],
        "seasons": ["summer"],
        "styles": ["casual"],
        "confidence": 0.9,
    }


def _assert_no_delivery(ctx: _Context) -> None:
    assert ctx.send.images == []
    assert ctx.send.texts == []
    assert ctx.maisaka.context.append_calls == []
    assert ctx.maisaka.proactive.trigger_calls == []


def _receipt(service: PhotoStudioService, task: ImageTask) -> str:
    request_id = task.request.get("request_id") or uuid4().hex
    service.storage.begin_request(request_id, "submit", "a" * 64)
    service.storage.update_request(
        request_id,
        state="accepted",
        task_id=task.id,
        response={"success": True, "task_id": task.id, "status": "queued"},
    )
    return request_id


def _submit(service: PhotoStudioService, *, operation: str = "import") -> ImageTask:
    source = compress_image(_png("blue"))
    task = service.submit_webui_reference_job(
        operation=operation,
        category="outfit",
        name="WebUI shirt",
        image=source.data,
        source_metadata=source.to_dict(),
        batch_id="batch-one",
        request_id=uuid4().hex,
    )
    _receipt(service, task)
    return task


async def _run_next(service: PhotoStudioService) -> None:
    claimed = service.storage.claim_next_task()
    assert claimed is not None
    await service._handle_task(claimed)


class _BlockingTagger(_LLM):
    def __init__(self) -> None:
        super().__init__([_tags()])
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, prompt, **kwargs):
        self.started.set()
        await self.release.wait()
        return await super().generate(prompt, **kwargs)


def test_reference_job_context_does_not_allow_a_chat_or_weaken_stream_guard() -> None:
    context = ReferenceJobContext()
    assert context.stream_id == context.user_id == context.message_id == ""
    assert context.group_id is None
    assert context.scope_key == "webui:gallery"
    assert (context.platform, context.origin, context.delivery) == ("webui", "webui", "none")
    with pytest.raises(TypeError):
        ReferenceJobContext(stream_id="chat")
    with pytest.raises(InvocationError, match="stream_id"):
        invocation_context({"platform": "webui", "message": {}})


@pytest.mark.asyncio
async def test_webui_import_duplicate_and_retag_finish_without_chat_or_planner(tmp_path: Path) -> None:
    ctx = _Context([_tags(), _tags()])
    service = PhotoStudioService(ctx, _config(), tmp_path)
    try:
        task = _submit(service)
        payload = service.payloads.get(task.id)
        assert task.scope_key == "webui:gallery" and not task.stream_id and task.user_id is None
        for key in ("origin", "delivery", "batch_id", "request_id", "asset_id", "expected_version", "person_version"):
            assert task.request[key] == payload[key]
        assert task.request["origin"] == "webui" and task.request["delivery"] == "none"
        await _run_next(service)
        saved = service.storage.get_task(task.id)
        assert saved.status == TaskStatus.SENT and saved.paid_request_started
        assert saved.result_metadata["duplicate"] is False
        asset = service.gallery.require(saved.result_metadata["asset_id"])
        assert asset.source_path.read_bytes() == compress_image(_png("blue")).data
        assert not (service.payloads.root / f"{task.id}.json").exists()
        assert not (service.payloads.upload_root / f"{task.id}.bin").exists()
        with pytest.raises(StorageError, match="已经完成"):
            service.retry_task(task.id)

        duplicate = _submit(service)
        await _run_next(service)
        duplicated = service.storage.get_task(duplicate.id)
        assert duplicated.status == TaskStatus.SENT and not duplicated.paid_request_started
        assert duplicated.result_metadata["duplicate"] is True
        assert service.gallery.require(asset.id) == asset
        assert len(ctx.llm.calls) == 1

        retag = service.submit_webui_reference_job(
            operation="retag",
            asset_id=asset.id,
            expected_version=asset.updated_at.isoformat(),
        )
        _receipt(service, retag)
        await _run_next(service)
        retagged = service.storage.get_task(retag.id)
        assert retagged.status == TaskStatus.SENT and retagged.paid_request_started
        assert retagged.result_metadata["duplicate"] is False
        assert service.storage.get_request(task.request["request_id"])["state"] == "completed"
        assert len(ctx.llm.calls) == 2
        _assert_no_delivery(ctx)
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_failed_reference_keeps_upload_and_manual_retry_preserves_webui_delivery(tmp_path: Path) -> None:
    class _FailOnceProvider(_Provider):
        async def generate(self, prompt, **kwargs):
            if not self.calls:
                self.calls.append((prompt, kwargs))
                raise RuntimeError("fake upstream error with sensitive details")
            return await super().generate(prompt, **kwargs)

    ctx = _Context([_tags()])
    service = PhotoStudioService(ctx, _config(), tmp_path)
    provider = _FailOnceProvider([_png("red")])
    service._provider = provider
    try:
        task = _submit(service, operation="extract")
        request = dict(task.request)
        await _run_next(service)
        failed = service.storage.get_task(task.id)
        assert failed.status == TaskStatus.FAILED and failed.paid_request_started
        assert service.payloads.get_upload(task.id)
        original_receipt = service.storage.get_request(task.request["request_id"])
        assert original_receipt["state"] == "failed"
        assert "sensitive" not in original_receipt["response"]["error"]
        _assert_no_delivery(ctx)

        queued = service.retry_task(task.id)
        assert queued.id == task.id and queued.request == request and not queued.paid_request_started
        retry_id = uuid4().hex
        service.storage.begin_request(retry_id, "retry", "b" * 64)
        service.storage.update_request(retry_id, state="accepted", task_id=task.id)
        await _run_next(service)
        saved = service.storage.get_task(task.id)
        assert saved.status == TaskStatus.SENT and saved.request == request
        assert len(provider.calls) == 2 and len(ctx.llm.calls) == 1
        assert service.storage.get_request(retry_id)["state"] == "completed"
        assert service.storage.get_request(task.request["request_id"]) == original_receipt
        assert not (service.payloads.upload_root / f"{task.id}.bin").exists()
        _assert_no_delivery(ctx)
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["queued", "provider", "tagging"])
async def test_cancel_reference_cannot_commit_send_or_claim_missing_upload_retry(tmp_path: Path, boundary: str) -> None:
    ctx = _Context([_tags()])
    service = PhotoStudioService(ctx, _config(), tmp_path)
    provider = _BlockingProvider(_png("red"))
    service._provider = provider
    if boundary == "tagging":
        ctx.llm = _BlockingTagger()
    worker = None
    try:
        task = _submit(service, operation="extract" if boundary == "provider" else "import")
        if boundary != "queued":
            worker = asyncio.create_task(_run_next(service))
            blocked = provider if boundary == "provider" else ctx.llm
            await asyncio.wait_for(blocked.started.wait(), timeout=3)
            assert service.storage.get_task(task.id).paid_request_started
        service.cancel_task(task.id)
        if worker is not None:
            blocked.release.set()
            await asyncio.wait_for(worker, timeout=3)
        saved = service.storage.get_task(task.id)
        assert saved.status == TaskStatus.CANCELLED
        assert service.storage.get_request(task.request["request_id"])["response"]["status"] == "cancelled"
        assert service.storage.list_reference_assets() == []
        assert not (service.payloads.root / f"{task.id}.json").exists()
        assert not (service.payloads.upload_root / f"{task.id}.bin").exists()
        with pytest.raises(PhotoStudioError, match="载荷不存在"):
            service.retry_task(task.id)
        _assert_no_delivery(ctx)
    finally:
        if worker is not None and not worker.done():
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupted", ["queued", "before_paid", "after_paid"])
async def test_restart_recovers_unpaid_webui_jobs_but_quarantines_paid_jobs(tmp_path: Path, interrupted: str) -> None:
    first = PhotoStudioService(_Context([]), _config(), tmp_path)
    task = _submit(first)
    if interrupted != "queued":
        task.status = TaskStatus.RUNNING
        task.paid_request_started = interrupted == "after_paid"
        first.storage.update_task(task)
    await first.close()

    ctx = _Context([_tags()])
    reopened = PhotoStudioService(ctx, _config(), tmp_path)
    try:
        await reopened.start()
        assert await reopened.tasks.drain(timeout=3)
        await reopened.wait_for_startup_reference_scan()
        recovered = reopened.storage.get_task(task.id)
        receipt = reopened.storage.get_request(task.request["request_id"])
        assert recovered.request == task.request
        if interrupted == "after_paid":
            assert recovered.status == TaskStatus.FAILED
            assert receipt["state"] == "failed" and receipt["response"]["success"] is False
            assert ctx.llm.calls == [] and reopened.payloads.get_upload(task.id)
        else:
            assert recovered.status == TaskStatus.SENT
            assert receipt["state"] == "completed" and len(ctx.llm.calls) == 1
        _assert_no_delivery(ctx)
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_restart_cleans_completed_queue_residue_without_any_model_or_delivery(
    tmp_path: Path, monkeypatch
) -> None:
    first = PhotoStudioService(_Context([_tags()]), _config(), tmp_path)
    task = _submit(first)
    monkeypatch.setattr(first, "_cleanup_reference_payload", lambda task_id: None)
    await _run_next(first)
    saved = first.storage.get_task(task.id)
    assert saved.status == TaskStatus.SENT and first.payloads.get_upload(task.id)
    await first.close()

    ctx = _Context([])
    reopened = PhotoStudioService(ctx, _config(), tmp_path)
    try:
        await reopened.start()
        await reopened.wait_for_startup_reference_scan()
        assert reopened.storage.get_task(task.id) == saved
        assert not (reopened.payloads.root / f"{task.id}.json").exists()
        assert not (reopened.payloads.upload_root / f"{task.id}.bin").exists()
        assert ctx.llm.calls == []
        _assert_no_delivery(ctx)
    finally:
        await reopened.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("guard", ["origin", "delivery", "scope"])
async def test_notification_retry_and_direct_delivery_fail_closed_even_with_a_stream(
    tmp_path: Path, guard: str
) -> None:
    ctx = _Context([])
    service = PhotoStudioService(ctx, _config(), tmp_path)
    result = tmp_path / "result.png"
    result.write_bytes(_png("green"))
    task = service.storage.create_task(
        ImageTask(
            kind="notification_retry",
            scope_key="webui:gallery" if guard == "scope" else "stream:chat",
            stream_id="chat",
            request={guard: "webui" if guard == "origin" else "none"} if guard != "scope" else {},
            status=TaskStatus.RUNNING,
            result_path=result,
            result_metadata={"notification_retry": True},
        )
    )
    try:
        await service._deliver(task)
        await service._notify_success(task)
        await service._notify_failure(task.id, "fake failure")
        await service._handle_task(task)
        saved = service.storage.get_task(task.id)
        assert saved.status == TaskStatus.SENT and saved.planner_notified_at is None
        assert "notification_retry" not in saved.result_metadata
        _assert_no_delivery(ctx)
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_legacy_completed_reference_notification_retry_never_becomes_planner_work(tmp_path: Path) -> None:
    ctx = _Context([])
    service = PhotoStudioService(ctx, _config(), tmp_path)
    task = service.storage.create_task(
        ImageTask(
            kind="reference_import",
            scope_key="webui:gallery",
            stream_id="accidental-chat",
            request={"origin": "webui", "delivery": "none"},
            status=TaskStatus.RUNNING,
            result_metadata={"completed": True, "notification_retry": True, "asset_id": "f" * 32},
        )
    )
    try:
        await service._handle_task(task)
        saved = service.storage.get_task(task.id)
        assert saved.status == TaskStatus.SENT and saved.planner_notified_at is None
        assert "notification_retry" not in saved.result_metadata
        with pytest.raises(StorageError, match="已经完成"):
            service.retry_task(task.id)
        assert ctx.llm.calls == []
        _assert_no_delivery(ctx)
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("automatic", [False, True])
async def test_existing_reference_qq_and_automatic_delivery_behavior_is_unchanged(
    tmp_path: Path, automatic: bool
) -> None:
    ctx = _Context([_tags()])
    service = PhotoStudioService(ctx, _config(), tmp_path)
    try:
        task = service.submit_reference_job(
            _invocation(),
            operation="import",
            category="outfit",
            name="QQ shirt",
            image=_png("blue"),
            automatic=automatic,
        )
        assert task.request["origin"] == ("automatic" if automatic else "qq")
        assert task.request["delivery"] == "qq"
        await _run_next(service)
        saved = service.storage.get_task(task.id)
        assert saved.status == TaskStatus.SENT
        assert len(ctx.send.texts) == (0 if automatic else 1)
        assert ctx.send.images == []
        assert ctx.maisaka.context.append_calls == ctx.maisaka.proactive.trigger_calls == []
    finally:
        await service.close()


def _guard_asset(service: PhotoStudioService, category: str, colour: str = "blue") -> ReferenceAsset:
    path = service.data_dir / f"guard-{category}-{uuid4().hex}.png"
    path.write_bytes(_png(colour))
    tags = {"privacy_eligible": True} if category == "scene" else {}
    return service.gallery.add_reference(
        category=category,
        name=f"guard {category}",
        source_path=path,
        reference_path=path,
        tags=tags,
    )


def _guard_task(
    service: PhotoStudioService,
    *,
    kind: str = "photo",
    status: TaskStatus = TaskStatus.QUEUED,
    payload: dict | None = None,
    request: dict | None = None,
) -> ImageTask:
    task = service.storage.create_task(
        ImageTask(
            kind=kind,
            scope_key="stream:guard",
            stream_id="guard-stream",
            status=status,
            request=request or {},
        )
    )
    if payload is not None:
        service.payloads.put(task.id, payload)
    return task


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["photo", "scene_photo"])
async def test_shared_busy_guard_covers_references_held_by_selector_across_await(tmp_path: Path, kind: str) -> None:
    ctx = _Context([])
    config = _config()
    config.references.auto_extract_missing = False
    service = PhotoStudioService(ctx, config, tmp_path)
    person = _guard_asset(service, "person")
    outfit = _guard_asset(service, "outfit")
    scene = _guard_asset(service, "scene")
    unselected_outfit = _guard_asset(service, "outfit", "red")
    unselected_scene = _guard_asset(service, "scene", "red")
    started, release = asyncio.Event(), asyncio.Event()
    provider = _BlockingProvider(_png("green"))
    service._provider = provider

    class HoldingSelector:
        async def select(self, **kwargs):
            # Model the real selector's continuity/candidate objects, captured
            # before its auxiliary model call yields control to administration.
            held_outfit = service.gallery.require(outfit.id) if kwargs["allow_outfit"] else None
            held_scene = service.gallery.require(scene.id) if kwargs["allow_scene"] else None
            started.set()
            await release.wait()
            return SelectionResult(
                outfit=held_outfit,
                scene=held_scene,
                scene_signature="guard-room",
                scene_eligible=True,
                reasons={},
            )

    service.selector = HoldingSelector()
    task = _guard_task(service, kind=kind, payload={"description": "fake selection"})
    worker = asyncio.create_task(_run_next(service))
    try:
        await asyncio.wait_for(started.wait(), timeout=3)
        assert service.storage.list_task_references(task.id) == []
        protected = [scene, unselected_scene] + ([person, outfit, unselected_outfit] if kind == "photo" else [])
        for asset in protected:
            with pytest.raises(ValueError, match="生图任务使用"):
                service.ensure_reference_idle(asset.id)
            assert asset.reference_path.is_file()
        if kind != "photo":
            service.ensure_reference_idle(person.id)
            service.ensure_reference_idle(outfit.id)
        release.set()
        await asyncio.wait_for(provider.started.wait(), timeout=3)
        assert {item.role for item in service.storage.list_task_references(task.id)} == {"person", "outfit", "scene"}
        # Once committed role records exist, unselected candidates are no longer
        # locked, even though the image provider itself is still running.
        service.ensure_reference_idle(unselected_outfit.id)
        service.ensure_reference_idle(unselected_scene.id)
        with pytest.raises(ValueError, match="生图任务使用"):
            service.ensure_reference_idle(scene.id)
        provider.release.set()
        await asyncio.wait_for(worker, timeout=3)
        assert service.storage.get_task(task.id).status == TaskStatus.SENT
        for asset in (person, outfit, scene):
            service.ensure_reference_idle(asset.id)
    finally:
        release.set()
        provider.release.set()
        if not worker.done():
            worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        await service.close()


@pytest.mark.asyncio
async def test_queued_photo_protects_explicit_refs_and_implicit_person_but_not_unselected_candidates(
    tmp_path: Path,
) -> None:
    service = PhotoStudioService(_Context([]), _config(), tmp_path)
    person = _guard_asset(service, "person")
    outfit = _guard_asset(service, "outfit")
    scene = _guard_asset(service, "scene")
    unrelated = _guard_asset(service, "outfit", "red")
    try:
        service.submit_photo(_invocation(), description="queued", outfit_id=outfit.id, scene_id=scene.id)
        for asset in (person, outfit, scene):
            with pytest.raises(ValueError, match="生图任务使用"):
                service.ensure_reference_idle(asset.id)
        service.ensure_reference_idle(unrelated.id)
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["import", "extract"])
@pytest.mark.parametrize("origin", ["webui", "qq", "startup", "unknown"])
async def test_implicit_person_reference_jobs_protect_singleton_versions_and_legacy_category(
    tmp_path: Path,
    operation: str,
    origin: str,
) -> None:
    service = PhotoStudioService(_Context([]), _config(), tmp_path)
    person = _guard_asset(service, "person")
    unrelated = _guard_asset(service, "outfit")
    request = {"operation": operation, "category": "person", "origin": origin}
    if origin == "webui":
        request["person_version"] = f"{person.id}:{person.updated_at.isoformat()}"
    try:
        _guard_task(service, kind=f"reference_{operation}", request=request)
        with pytest.raises(ValueError, match="未完成的作业"):
            service.ensure_reference_idle(person.id)
        service.ensure_reference_idle(unrelated.id)
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_shared_guard_reads_legacy_reference_targets_and_person_versions_from_payload(tmp_path: Path) -> None:
    service = PhotoStudioService(_Context([]), _config(), tmp_path)
    person = _guard_asset(service, "person")
    outfit = _guard_asset(service, "outfit")
    unrelated = _guard_asset(service, "scene")
    try:
        _guard_task(service, kind="reference_retag", payload={"asset_id": outfit.id})
        _guard_task(
            service,
            kind="reference_import",
            request={"origin": "webui"},
            payload={
                "category": "person",
                "person_version": f"{person.id}:{person.updated_at.isoformat()}",
            },
        )
        for asset in (person, outfit):
            with pytest.raises(ValueError, match="未完成的作业"):
                service.ensure_reference_idle(asset.id)
        service.ensure_reference_idle(unrelated.id)
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["photo", "scene_photo", "image"])
@pytest.mark.parametrize("status", [TaskStatus.QUEUED, TaskStatus.RUNNING])
async def test_shared_guard_respects_optouts_and_configuration(tmp_path: Path, kind: str, status: TaskStatus) -> None:
    config = _config()
    config.references.require_person_reference = False
    service = PhotoStudioService(_Context([]), config, tmp_path)
    assets = [_guard_asset(service, category) for category in ("person", "outfit", "scene")]
    try:
        task = _guard_task(
            service,
            kind=kind,
            status=status,
            payload={
                "use_person_reference": False,
                "use_outfit_reference": False,
                "use_scene_reference": False,
            },
        )
        for asset in assets:
            service.ensure_reference_idle(asset.id)
        service.storage.set_task_status(task.id, TaskStatus.CANCELLED)
        config.references.person_reference_enabled = False
        config.references.outfit_reference_enabled = False
        config.references.scene_reference_enabled = False
        _guard_task(service, kind=kind, status=status, payload={})
        for asset in assets:
            service.ensure_reference_idle(asset.id)
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [TaskStatus.SENT, TaskStatus.FAILED, TaskStatus.CANCELLED])
async def test_terminal_tasks_do_not_block_reference_mutation_even_with_corrupt_payloads(
    tmp_path: Path, status: TaskStatus
) -> None:
    service = PhotoStudioService(_Context([]), _config(), tmp_path)
    asset = _guard_asset(service, "outfit")
    try:
        task = _guard_task(service, status=status, request={"asset_id": asset.id})
        service.payloads.put(task.id, {"outfit_id": asset.id})
        service.storage.record_task_reference(TaskReference(task.id, "outfit", asset_id=asset.id))
        (service.payloads.root / f"{task.id}.json").write_text("not json")
        service.ensure_reference_idle(asset.id)
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_registered_final_roles_protect_only_actual_refs_not_payload_or_other_categories(tmp_path: Path) -> None:
    service = PhotoStudioService(_Context([]), _config(), tmp_path)
    person = _guard_asset(service, "person")
    outfit = _guard_asset(service, "outfit")
    scene = _guard_asset(service, "scene")
    try:
        task = _guard_task(service, status=TaskStatus.RUNNING)
        for role in ("person", "outfit", "scene"):
            service.storage.record_task_reference(
                TaskReference(
                    task.id,
                    role,
                    asset_id=outfit.id if role == "outfit" else None,
                )
            )
        with pytest.raises(ValueError, match="生图任务使用"):
            service.ensure_reference_idle(outfit.id)
        service.ensure_reference_idle(person.id)
        service.ensure_reference_idle(scene.id)
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["photo", "reference_retag", "reference_import"])
@pytest.mark.parametrize("fault", ["missing", "corrupt", "bad_shape"])
async def test_unknown_active_payload_fails_closed_without_hiding_unexpected_errors(
    tmp_path: Path,
    monkeypatch,
    kind: str,
    fault: str,
) -> None:
    service = PhotoStudioService(_Context([]), _config(), tmp_path)
    asset = _guard_asset(service, "outfit")
    try:
        task = _guard_task(service, kind=kind, status=TaskStatus.RUNNING)
        path = service.payloads.root / f"{task.id}.json"
        if fault != "missing":
            path.write_text("[1]" if fault == "bad_shape" else "not json")
        with pytest.raises(ValueError, match="未完成的作业"):
            service.ensure_reference_idle(asset.id)

        def unexpected(task_id):
            raise RuntimeError("fake unexpected bug")

        monkeypatch.setattr(service.payloads, "get", unexpected)
        with pytest.raises(RuntimeError, match="unexpected bug"):
            service.ensure_reference_idle(asset.id)
    finally:
        await service.close()
