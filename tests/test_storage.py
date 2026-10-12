from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from maitu_photo.gallery import DuplicateReferenceError, ReferenceGallery
from maitu_photo.models import (
    AssetStatus,
    ImageTask,
    ReferenceAsset,
    TaskReference,
    TaskStatus,
)
from maitu_photo.storage import SQLiteStorage, StorageError


def _gallery(tmp_path: Path) -> tuple[SQLiteStorage, ReferenceGallery]:
    storage = SQLiteStorage(tmp_path / "data" / "maitu.sqlite3")
    return storage, ReferenceGallery(storage)


def _add(
    gallery: ReferenceGallery,
    category: str,
    name: str,
    digest_char: str,
    **kwargs: object,
):
    return gallery.add_reference(
        category=category,
        name=name,
        source_path=f"source/{name}.jpg",
        reference_path=f"references/{name}.jpg",
        sha256=digest_char * 64,
        **kwargs,
    )


def test_storage_uses_wal_and_persists_assets(tmp_path: Path) -> None:
    storage, gallery = _gallery(tmp_path)
    assert storage.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

    outfit = _add(
        gallery,
        "outfit",
        "linen dress",
        "a",
        tags={"type": "dress", "season": ["spring", "summer"]},
        manual_tags={"style": "quiet"},
        selection_metadata={"source": "admin_extract"},
    )
    storage.close()

    with SQLiteStorage(tmp_path / "data" / "maitu.sqlite3") as reopened:
        loaded = reopened.get_reference_asset(outfit.id)
        assert loaded is not None
        assert loaded.effective_tags == {
            "type": "dress",
            "season": ["spring", "summer"],
            "style": "quiet",
        }
        assert loaded.as_selection_metadata()["source"] == "admin_extract"


def test_gallery_enforces_person_singleton_and_category_hash_dedup(
    tmp_path: Path,
) -> None:
    storage, gallery = _gallery(tmp_path)
    first = _add(gallery, "person", "first", "1")

    with pytest.raises(DuplicateReferenceError) as conflict:
        _add(gallery, "person", "second", "2")
    assert conflict.value.existing_id == first.id

    second = _add(gallery, "person", "second", "2", replace_person=True)
    assert gallery.get_person().id == second.id  # type: ignore[union-attr]
    replaced = storage.get_reference_asset(first.id, include_deleted=True)
    assert replaced is not None
    assert replaced.status == AssetStatus.DELETED
    assert replaced.deleted_at is not None

    outfit = _add(gallery, "outfit", "dress", "3")
    with pytest.raises(DuplicateReferenceError) as duplicate:
        _add(gallery, "outfit", "same bytes", "3")
    assert duplicate.value.existing_id == outfit.id

    gallery.soft_delete(outfit.id)
    replacement = _add(gallery, "outfit", "replacement", "3")
    assert replacement.id != outfit.id
    storage.close()


def test_gallery_tags_status_and_selection_priority(tmp_path: Path) -> None:
    storage, gallery = _gallery(tmp_path)
    summer = _add(
        gallery,
        "outfit",
        "summer dress",
        "4",
        tags={"type": "dress", "season": ["summer"]},
    )
    winter = _add(
        gallery,
        "outfit",
        "winter coat",
        "5",
        tags={"type": "coat", "season": ["winter"]},
    )

    explicit = gallery.select_reference("outfit", explicit_id=winter.id)
    assert explicit.asset.id == winter.id  # type: ignore[union-attr]
    assert explicit.source == "explicit"

    continuity = gallery.select_reference("outfit", continuity_id=summer.id)
    assert continuity.asset.id == summer.id  # type: ignore[union-attr]
    assert continuity.source == "continuity"

    filtered = gallery.select_reference("outfit", tag_filters={"season": "winter"}, hints=("winter",))
    assert filtered.asset.id == winter.id  # type: ignore[union-attr]

    gallery.apply_generated_tags(winter.id, None, valid=False)
    assert gallery.get(winter.id).status == AssetStatus.NEEDS_REVIEW  # type: ignore[union-attr]
    fallback = gallery.select_reference("outfit", tag_filters={"season": "winter"})
    assert fallback.asset is None
    assert fallback.fallback_reason == "no_matching_reference"

    gallery.edit(summer.id, manual_tags={"season": "autumn"})
    assert gallery.get(summer.id).effective_tags["season"] == "autumn"  # type: ignore[union-attr]
    storage.close()


def test_task_queue_reference_audit_and_interruption_recovery(tmp_path: Path) -> None:
    storage, gallery = _gallery(tmp_path)
    outfit = _add(gallery, "outfit", "task outfit", "6")
    first = storage.create_task(
        ImageTask(
            kind="photo",
            scope_key="group:10",
            prompt_summary="portrait",
            prompt_hash="f" * 64,
            request={"size": "1024x1024"},
        )
    )
    storage.record_task_reference(
        TaskReference(
            task_id=first.id,
            role="outfit",
            asset_id=outfit.id,
            selection_source="explicit",
            selection_metadata={"id": outfit.id},
        )
    )
    storage.record_task_reference(
        TaskReference(
            task_id=first.id,
            role="scene",
            selection_source="text_fallback",
            fallback_reason="no_matching_reference",
        )
    )
    claimed = storage.claim_next_task()
    assert claimed is not None and claimed.id == first.id
    assert claimed.status == TaskStatus.RUNNING
    storage.mark_task_request_started(first.id)

    second = storage.create_task(ImageTask(kind="image", scope_key="group:10"))
    second_claimed = storage.claim_next_task()
    assert second_claimed is not None and second_claimed.id == second.id

    generated = storage.create_task(
        ImageTask(
            kind="image",
            scope_key="group:10",
            status=TaskStatus.GENERATED,
            result_path="results/generated.png",
        )
    )
    requeued, failed = storage.recover_interrupted_tasks()
    assert set(requeued) == {second.id, generated.id}
    assert failed == [first.id]
    assert storage.get_task(first.id).status == TaskStatus.FAILED  # type: ignore[union-attr]
    assert storage.get_task(second.id).status == TaskStatus.QUEUED  # type: ignore[union-attr]
    assert storage.get_task(generated.id).status == TaskStatus.QUEUED  # type: ignore[union-attr]

    refs = storage.list_task_references(first.id)
    assert [(ref.role, ref.asset_id) for ref in refs] == [
        ("outfit", outfit.id),
        ("scene", None),
    ]
    storage.close()


def test_running_paid_task_with_saved_result_resumes_delivery(tmp_path: Path) -> None:
    storage, _ = _gallery(tmp_path)
    task = storage.create_task(ImageTask(kind="image", scope_key="group:10"))
    claimed = storage.claim_next_task()
    assert claimed is not None and claimed.id == task.id
    storage.mark_task_request_started(task.id)
    storage.set_task_status(
        task.id,
        TaskStatus.RUNNING,
        result_path="results/already-generated.png",
    )

    requeued, failed = storage.recover_interrupted_tasks()

    assert requeued == [task.id]
    assert failed == []
    resumed = storage.get_task(task.id)
    assert resumed is not None
    assert resumed.status == TaskStatus.QUEUED
    assert resumed.result_path == Path("results/already-generated.png")
    storage.close()


def test_usage_updates_are_deduplicated(tmp_path: Path) -> None:
    storage, gallery = _gallery(tmp_path)
    asset = _add(gallery, "scene", "bedroom", "7")
    used_at = datetime(2026, 8, 11, 3, 0, tzinfo=timezone.utc)
    gallery.record_usage([asset.id, asset.id], used_at=used_at)
    loaded = gallery.get(asset.id)
    assert loaded is not None
    assert loaded.use_count == 1
    assert loaded.last_used_at == used_at
    storage.close()


def _legacy_database(path: Path, version: int) -> tuple[dict[str, list[tuple]], list[tuple]]:
    """Create a frozen v1 fixture without using the v2 initializer."""

    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE reference_assets (
                id TEXT PRIMARY KEY,
                category TEXT NOT NULL CHECK (category IN ('person', 'outfit', 'scene')),
                name TEXT NOT NULL,
                source_path TEXT NOT NULL,
                reference_path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                tags_json TEXT NOT NULL DEFAULT '{}',
                manual_tags_json TEXT NOT NULL DEFAULT '{}',
                prompt_version TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL CHECK (status IN ('active', 'disabled', 'needs_review', 'deleted')),
                source_task_id TEXT,
                selection_metadata_json TEXT NOT NULL DEFAULT '{}',
                use_count INTEGER NOT NULL DEFAULT 0 CHECK (use_count >= 0),
                last_used_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT
            );
            CREATE TABLE image_tasks (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                scope_key TEXT NOT NULL,
                user_id TEXT,
                stream_id TEXT,
                group_id TEXT,
                status TEXT NOT NULL CHECK (
                    status IN ('queued', 'running', 'generated', 'sent', 'failed', 'cancelled')
                ),
                prompt_summary TEXT NOT NULL DEFAULT '',
                prompt_hash TEXT NOT NULL DEFAULT '',
                prompt_version TEXT NOT NULL DEFAULT '',
                request_json TEXT NOT NULL DEFAULT '{}',
                result_path TEXT,
                result_metadata_json TEXT NOT NULL DEFAULT '{}',
                error_message TEXT,
                parent_task_id TEXT,
                paid_request_started INTEGER NOT NULL DEFAULT 0,
                sent_at TEXT,
                planner_notified_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(parent_task_id) REFERENCES image_tasks(id) ON DELETE SET NULL
            );
            CREATE TABLE task_references (
                task_id TEXT NOT NULL,
                role TEXT NOT NULL,
                asset_id TEXT,
                selection_source TEXT NOT NULL,
                fallback_reason TEXT,
                selection_metadata_json TEXT NOT NULL DEFAULT '{}',
                recorded_at TEXT NOT NULL,
                PRIMARY KEY(task_id, role),
                FOREIGN KEY(task_id) REFERENCES image_tasks(id) ON DELETE CASCADE,
                FOREIGN KEY(asset_id) REFERENCES reference_assets(id) ON DELETE SET NULL
            );
            CREATE TABLE group_continuity (
                scope_key TEXT PRIMARY KEY,
                local_date TEXT NOT NULL,
                scene_signature TEXT NOT NULL,
                last_photo_at TEXT NOT NULL,
                outfit_id TEXT,
                scene_id TEXT,
                pinned_outfit_id TEXT,
                pinned_scene_id TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL,
                FOREIGN KEY(outfit_id) REFERENCES reference_assets(id) ON DELETE SET NULL,
                FOREIGN KEY(scene_id) REFERENCES reference_assets(id) ON DELETE SET NULL,
                FOREIGN KEY(pinned_outfit_id) REFERENCES reference_assets(id) ON DELETE SET NULL,
                FOREIGN KEY(pinned_scene_id) REFERENCES reference_assets(id) ON DELETE SET NULL
            );
            CREATE INDEX legacy_asset_marker ON reference_assets(name);
            """
        )
        stamp = "2026-08-11T03:00:00.000000+00:00"
        connection.execute(
            "INSERT INTO reference_assets VALUES (" + ",".join("?" for _ in range(17)) + ")",
            (
                "legacy-asset",
                "outfit",
                "旧衣服",
                "source/旧 名称.jpg",
                "references/原图.png",
                "a" * 64,
                '{"season":["summer"]}',
                '{"style":"quiet"}',
                "v1",
                "disabled",
                "legacy-task",
                '{"source":"import"}',
                7,
                stamp,
                stamp,
                stamp,
                None,
            ),
        )
        connection.execute(
            "INSERT INTO image_tasks VALUES (" + ",".join("?" for _ in range(20)) + ")",
            (
                "legacy-task",
                "reference_retag",
                "group:10",
                "legacy-user",
                "legacy-stream",
                "10",
                "sent",
                "旧任务",
                "b" * 64,
                "v1",
                '{"automatic":false,"asset_id":"legacy-asset"}',
                "results/旧.png",
                '{"asset_id":"legacy-asset"}',
                None,
                None,
                1,
                stamp,
                stamp,
                stamp,
                stamp,
            ),
        )
        connection.execute(
            "INSERT INTO task_references VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("legacy-task", "outfit", "legacy-asset", "explicit", None, '{"legacy":true}', stamp),
        )
        connection.execute(
            "INSERT INTO group_continuity VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("group:10", "2026-08-11", "room", stamp, "legacy-asset", None, "legacy-asset", None, "{}", stamp),
        )
        connection.execute(f"PRAGMA user_version = {version}")
        rows = {
            table: connection.execute(f"SELECT * FROM {table}").fetchall()
            for table in ("reference_assets", "image_tasks", "task_references", "group_continuity")
        }
        schema = connection.execute(
            "SELECT name, rootpage, sql FROM sqlite_master WHERE type = 'table' OR name = 'legacy_asset_marker'"
        ).fetchall()
    return rows, schema


@pytest.mark.parametrize("version", [0, 1])
def test_legacy_migration_preserves_all_rows_and_table_layout(tmp_path: Path, version: int) -> None:
    database = tmp_path / "legacy.sqlite3"
    old_rows, old_schema = _legacy_database(database, version)
    with SQLiteStorage(database) as storage:
        assert storage.SCHEMA_VERSION == 2
        assert storage.connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert storage.connection.execute("PRAGMA foreign_key_check").fetchall() == []
        for table, expected in old_rows.items():
            assert [tuple(row) for row in storage.connection.execute(f"SELECT * FROM {table}")] == expected
        for name, rootpage, sql in old_schema:
            actual = storage.connection.execute(
                "SELECT rootpage, sql FROM sqlite_master WHERE name = ?", (name,)
            ).fetchone()
            assert tuple(actual) == (rootpage, sql)
        asset = storage.get_reference_asset("legacy-asset")
        task = storage.get_task("legacy-task")
        assert asset is not None and task is not None
        assert asset.id == "legacy-asset"
        assert asset.source_path == Path("source/旧 名称.jpg")
        assert asset.reference_path == Path("references/原图.png")
        assert task.result_path == Path("results/旧.png")
        assert storage.list_task_references(task.id)[0].asset_id == asset.id
        assert storage.get_continuity("group:10").outfit_id == asset.id  # type: ignore[union-attr]
        storage.begin_request("1" * 32, "retag", "a" * 64)

    with SQLiteStorage(database) as reopened:
        assert reopened.connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert reopened.get_request("1" * 32)["state"] == "preparing"  # type: ignore[index]
        for table, expected in old_rows.items():
            assert [tuple(row) for row in reopened.connection.execute(f"SELECT * FROM {table}")] == expected


def test_new_v0_database_and_v2_reopen_are_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "new.sqlite3"
    with SQLiteStorage(database) as storage:
        assert storage.connection.execute("PRAGMA user_version").fetchone()[0] == 2
        expected_columns = (
            "request_id",
            "parameter_hash",
            "operation",
            "state",
            "task_id",
            "response_json",
            "created_at",
            "updated_at",
        )
        assert (
            tuple(row[1] for row in storage.connection.execute("PRAGMA table_info(webui_requests)")) == expected_columns
        )
        assert storage.connection.execute("PRAGMA foreign_key_list(webui_requests)").fetchall() == []
        storage.begin_request("2" * 32, "import", "b" * 64)
        receipt = storage.update_request("2" * 32, state="completed", response={"success": True, "name": "测试"})
        schema = [tuple(row) for row in storage.connection.execute("SELECT * FROM sqlite_master ORDER BY name")]

    for _ in range(2):
        with SQLiteStorage(database) as reopened:
            assert reopened.connection.execute("PRAGMA user_version").fetchone()[0] == 2
            assert reopened.get_request("2" * 32) == receipt
            assert [
                tuple(row) for row in reopened.connection.execute("SELECT * FROM sqlite_master ORDER BY name")
            ] == schema


@pytest.mark.parametrize("version", [3, 99])
def test_future_schema_is_rejected_without_downgrading_or_mutation(tmp_path: Path, version: int) -> None:
    database = tmp_path / "future.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE future_data (value TEXT)")
        connection.execute("INSERT INTO future_data VALUES ('do not touch')")
        connection.execute(f"PRAGMA user_version = {version}")
        old_schema = connection.execute("SELECT * FROM sqlite_master").fetchall()
        old_journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]

    with pytest.raises(StorageError, match="unsupported.*schema version"):
        SQLiteStorage(database)

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == version
        assert connection.execute("SELECT * FROM sqlite_master").fetchall() == old_schema
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == old_journal_mode
        assert connection.execute("SELECT * FROM future_data").fetchall() == [("do not touch",)]


@pytest.mark.parametrize("version", [0, 1])
def test_failed_migration_rolls_back_schema_and_version(tmp_path: Path, version: int) -> None:
    database = tmp_path / "failed.sqlite3"
    if version == 1:
        _legacy_database(database, version)
    with sqlite3.connect(database) as connection:
        # An incompatible pre-existing table forces failure after base DDL.
        connection.execute("CREATE TABLE webui_requests (request_id TEXT PRIMARY KEY)")
        old_schema = connection.execute("SELECT * FROM sqlite_master ORDER BY name").fetchall()

    with pytest.raises(sqlite3.OperationalError, match="task_id"):
        SQLiteStorage(database)

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == version
        assert connection.execute("SELECT * FROM sqlite_master ORDER BY name").fetchall() == old_schema
        if version == 1:
            assert connection.execute("SELECT id FROM reference_assets").fetchall() == [("legacy-asset",)]


def test_query_reference_assets_filters_count_and_stable_pagination(tmp_path: Path) -> None:
    with SQLiteStorage(tmp_path / "query.sqlite3") as storage:
        gallery = ReferenceGallery(storage)
        first = _add(gallery, "outfit", "linen dress", "a", tags={"season": "summer"})
        second = _add(gallery, "outfit", "wool coat", "b", status=AssetStatus.DISABLED)
        scene = _add(gallery, "scene", "quiet room", "c", status=AssetStatus.NEEDS_REVIEW)
        deleted = _add(gallery, "outfit", "gone dress", "d")
        gallery.soft_delete(deleted.id)
        tie = datetime(2026, 8, 11, 3, 0, tzinfo=timezone.utc)
        for asset in (first, second, scene):
            asset.created_at = tie
            storage.connection.execute(
                "UPDATE reference_assets SET created_at = ? WHERE id = ?",
                (tie.isoformat(timespec="microseconds"), asset.id),
            )
        expected = sorted([first.id, second.id, scene.id])
        page, total = storage.query_reference_assets(limit=2)
        assert [asset.id for asset in page] == expected[:2]
        assert total == 3
        page, total = storage.query_reference_assets(limit=2, offset=2)
        assert [asset.id for asset in page] == expected[2:]
        assert total == 3
        assert storage.query_reference_assets(offset=100) == ([], 3)
        assert storage.query_reference_assets(category="outfit")[1] == 2
        assert storage.query_reference_assets(category="outfit", status="active")[0][0].id == first.id
        assert storage.query_reference_assets(status="disabled")[0][0].id == second.id
        assert storage.query_reference_assets(category="scene", status="needs_review")[0][0].id == scene.id
        assert storage.query_reference_assets(status="deleted")[0][0].id == deleted.id
        assert storage.query_reference_assets(category="scene", status="active") == ([], 0)
        assert storage.query_reference_assets(search="summer")[0][0].id == first.id
        assert storage.query_reference_assets(search=scene.id)[0][0].id == scene.id
        assert storage.query_reference_assets(category="", status="", search=" ")[1] == 3


@pytest.mark.parametrize("literal", ["%", "_", "\\", "' OR 1=1 --"])
def test_query_asset_search_escapes_like_metacharacters(tmp_path: Path, literal: str) -> None:
    with SQLiteStorage(tmp_path / "search.sqlite3") as storage:
        gallery = ReferenceGallery(storage)
        expected = _add(gallery, "outfit", f"dress {literal} literal", "a")
        _add(gallery, "outfit", "dress ordinary literal", "b")
        page, total = storage.query_reference_assets(search=literal)
        assert [asset.id for asset in page] == [expected.id]
        assert total == 1
        assert storage.query_reference_assets(search="DRESS")[1] == 2


@pytest.mark.parametrize("method", ["query_reference_assets", "query_reference_tasks"])
@pytest.mark.parametrize(
    "kwargs",
    [{"limit": 0}, {"limit": 101}, {"limit": True}, {"limit": 1.5}, {"offset": -1}, {"offset": True}, {"offset": 1.5}],
)
def test_queries_validate_pagination(tmp_path: Path, method: str, kwargs: dict[str, object]) -> None:
    with SQLiteStorage(tmp_path / "limits.sqlite3") as storage:
        query = getattr(storage, method)
        with pytest.raises(ValueError):
            query(**kwargs)
        assert query(limit=100) == ([], 0)


def test_query_reference_tasks_origin_fallback_status_and_pagination(tmp_path: Path) -> None:
    with SQLiteStorage(tmp_path / "tasks.sqlite3") as storage:
        specs = [
            ("webui", {"origin": "webui", "automatic": True}, "stream", "sent"),
            ("automatic", {"automatic": True}, "stream", "running"),
            ("qq", {"automatic": False}, "stream", "queued"),
            ("unknown", {}, None, "failed"),
            ("empty-origin", {"origin": "", "automatic": True}, None, "queued"),
            ("null-origin", {"origin": None}, "stream", "generated"),
            ("text-bool", {"automatic": "true"}, None, "cancelled"),
        ]
        tie = datetime(2026, 8, 11, 3, 0, tzinfo=timezone.utc)
        tasks = {}
        for name, request, stream_id, status in specs:
            tasks[name] = storage.create_task(
                ImageTask(
                    kind="reference_retag",
                    scope_key="test",
                    request=request,
                    stream_id=stream_id,
                    status=status,
                    created_at=tie,
                )
            )
        storage.create_task(ImageTask(kind="photo", scope_key="test", request={"origin": "webui"}))
        storage.create_task(ImageTask(kind="referenceXretag", scope_key="test", request={"origin": "webui"}))
        expected_ids = sorted(task.id for task in tasks.values())
        page, total = storage.query_reference_tasks(limit=3)
        assert [task.id for task in page] == expected_ids[:3]
        assert total == len(tasks)
        assert [task.id for task in storage.query_reference_tasks(limit=3, offset=3)[0]] == expected_ids[3:6]
        assert storage.query_reference_tasks(offset=100) == ([], len(tasks))
        for origin, names in (
            ("webui", {"webui"}),
            ("automatic", {"automatic", "empty-origin"}),
            ("qq", {"qq", "null-origin"}),
            ("unknown", {"unknown", "text-bool"}),
        ):
            page, total = storage.query_reference_tasks(origin=origin)
            assert {task.id for task in page} == {tasks[name].id for name in names}
            assert total == len(names)
        assert storage.query_reference_tasks(origin="webui", status="completed")[0][0].id == tasks["webui"].id
        assert storage.query_reference_tasks(status=TaskStatus.SENT)[1] == 1
        assert storage.query_reference_tasks(status="queued")[1] == 2
        assert storage.query_reference_tasks(origin="unknown", status="completed") == ([], 0)
        assert storage.query_reference_tasks(origin="no-such-origin") == ([], 0)


def test_file_is_referenced_checks_shared_live_paths_not_deleted_history(tmp_path: Path) -> None:
    with SQLiteStorage(tmp_path / "paths.sqlite3") as storage:
        first = storage.create_reference_asset(
            ReferenceAsset("outfit", "first", "source/shared.jpg", "references/first.jpg", "a" * 64)
        )
        second = storage.create_reference_asset(
            ReferenceAsset("scene", "second", "source/second.jpg", "source/shared.jpg", "b" * 64, status="disabled")
        )
        assert storage.file_is_referenced("source/shared.jpg", exclude_asset_id=first.id)
        assert storage.file_is_referenced(Path("source/second.jpg"))
        assert not storage.file_is_referenced("references/first.jpg", exclude_asset_id=first.id)
        assert not storage.file_is_referenced("missing.jpg")
        storage.soft_delete_reference_asset(second.id)
        assert not storage.file_is_referenced("source/shared.jpg", exclude_asset_id=first.id)
        assert not storage.file_is_referenced("source/second.jpg")
        storage.set_reference_status(first.id, AssetStatus.NEEDS_REVIEW)
        assert storage.file_is_referenced("references/first.jpg")
        storage.soft_delete_reference_asset(first.id)
        assert not storage.file_is_referenced("references/first.jpg")


@pytest.mark.parametrize("status", ["queued", "running", "generated", "sent", "failed", "cancelled"])
def test_file_is_referenced_protects_active_reference_task_target(tmp_path: Path, status: str) -> None:
    with SQLiteStorage(tmp_path / "target.sqlite3") as storage:
        asset = storage.create_reference_asset(
            ReferenceAsset("outfit", "target", "source/target.jpg", "references/target.jpg", "a" * 64)
        )
        storage.soft_delete_reference_asset(asset.id)
        task = storage.create_task(
            ImageTask(kind="reference_replace", scope_key="test", status=status, request={"asset_id": asset.id})
        )
        active = status in {"queued", "running", "generated"}
        assert storage.file_is_referenced("source/target.jpg", exclude_asset_id=asset.id) is active
        assert storage.file_is_referenced("references/target.jpg") is active
        storage.delete_task(task.id)
        storage.create_task(ImageTask(kind="photo", scope_key="test", request={"asset_id": asset.id}))
        assert not storage.file_is_referenced("source/target.jpg")
