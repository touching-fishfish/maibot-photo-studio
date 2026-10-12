"""Offline request-receipt tests; no API, provider, or production database."""

from __future__ import annotations

import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Barrier

import pytest

from maitu_photo.models import ImageTask, TaskStatus
from maitu_photo.storage import RecordNotFoundError, SQLiteStorage, StorageError

REQUEST_ID = "0123456789abcdef0123456789abcdef"
PARAMETER_HASH = "a" * 64


def test_begin_request_returns_dto_and_repeated_submit_is_unchanged(tmp_path: Path) -> None:
    database = tmp_path / "receipts.sqlite3"
    with SQLiteStorage(database) as storage:
        assert storage.get_request(REQUEST_ID) is None
        first = storage.begin_request(REQUEST_ID, "import", PARAMETER_HASH)
        assert set(first) == {
            "request_id",
            "state",
            "operation",
            "parameter_hash",
            "task_id",
            "response",
            "created_at",
            "updated_at",
        }
        assert first["request_id"] == REQUEST_ID
        assert first["state"] == "preparing"
        assert first["operation"] == "import"
        assert first["parameter_hash"] == PARAMETER_HASH
        assert first["task_id"] is None and first["response"] is None
        assert first["created_at"] == first["updated_at"]
        assert datetime.fromisoformat(first["created_at"]).utcoffset().total_seconds() == 0
        assert storage.begin_request(REQUEST_ID, "import", PARAMETER_HASH) == first
        accepted = storage.update_request(
            REQUEST_ID, state="accepted", task_id="retained-legacy-task", response={"success": True, "status": "queued"}
        )
        assert accepted["created_at"] == first["created_at"]
        assert accepted["updated_at"] >= first["updated_at"]
        assert storage.begin_request(REQUEST_ID, "import", PARAMETER_HASH) == accepted
        assert storage.connection.execute("SELECT COUNT(*) FROM webui_requests").fetchone()[0] == 1

    with SQLiteStorage(database) as reopened:
        assert reopened.get_request(REQUEST_ID) == accepted
        assert reopened.begin_request(REQUEST_ID, "import", PARAMETER_HASH) == accepted


@pytest.mark.parametrize(("operation", "digest"), [("replace", PARAMETER_HASH), ("import", "b" * 64)])
def test_same_request_id_rejects_operation_or_parameter_conflict(tmp_path: Path, operation: str, digest: str) -> None:
    with SQLiteStorage(tmp_path / "conflict.sqlite3") as storage:
        first = storage.begin_request(REQUEST_ID, "import", PARAMETER_HASH)
        with pytest.raises(StorageError, match="different operation or parameters"):
            storage.begin_request(REQUEST_ID, operation, digest)
        assert storage.get_request(REQUEST_ID) == first
        assert storage.connection.execute("SELECT COUNT(*) FROM webui_requests").fetchone()[0] == 1


@pytest.mark.parametrize("separate_connections", [False, True])
@pytest.mark.parametrize("conflicting_hash", [False, True])
def test_concurrent_double_submit_is_serialized_without_claim_token(
    tmp_path: Path, separate_connections: bool, conflicting_hash: bool
) -> None:
    database = tmp_path / "concurrent.sqlite3"
    first = SQLiteStorage(database)
    second = SQLiteStorage(database) if separate_connections else first
    barrier = Barrier(2)

    def submit(storage: SQLiteStorage, digest: str) -> dict | StorageError:
        barrier.wait(timeout=5)
        try:
            return storage.begin_request(REQUEST_ID, "import", digest)
        except StorageError as exc:
            return exc

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(submit, first, PARAMETER_HASH),
                pool.submit(submit, second, "b" * 64 if conflicting_hash else PARAMETER_HASH),
            ]
            results = [future.result(timeout=10) for future in futures]
        assert first.connection.execute("SELECT COUNT(*) FROM webui_requests").fetchone()[0] == 1
        errors = [result for result in results if isinstance(result, StorageError)]
        receipts = [result for result in results if isinstance(result, dict)]
        if conflicting_hash:
            assert len(errors) == 1 and len(receipts) == 1
        else:
            assert not errors and receipts[0] == receipts[1]
        assert all("claimed" not in receipt and "token" not in receipt for receipt in receipts)
        assert first.get_request(REQUEST_ID) == receipts[0]
    finally:
        first.close()
        if second is not first:
            second.close()


@pytest.mark.parametrize(
    "request_id",
    [
        "",
        "short",
        "A" * 32,
        "g" * 32,
        "a" * 31,
        "a" * 33,
        "a" * 32 + "\n",
        " " + REQUEST_ID,
        "{01234567-89ab-cdef-0123-456789abcdef}",
        "01234567-89AB-cdef-0123-456789abcdef",
        None,
    ],
)
def test_request_ids_are_strict_for_all_receipt_methods(tmp_path: Path, request_id: object) -> None:
    with SQLiteStorage(tmp_path / "id.sqlite3") as storage:
        with pytest.raises(StorageError, match="request_id"):
            storage.begin_request(request_id, "import", PARAMETER_HASH)  # type: ignore[arg-type]
        with pytest.raises(StorageError, match="request_id"):
            storage.get_request(request_id)  # type: ignore[arg-type]
        with pytest.raises(StorageError, match="request_id"):
            storage.update_request(request_id, state="accepted")  # type: ignore[arg-type]
        assert storage.connection.execute("SELECT COUNT(*) FROM webui_requests").fetchone()[0] == 0


def test_canonical_uuid_request_id_is_preserved_not_rewritten(tmp_path: Path) -> None:
    request_id = "01234567-89ab-cdef-0123-456789abcdef"
    with SQLiteStorage(tmp_path / "uuid.sqlite3") as storage:
        receipt = storage.begin_request(request_id, "retag", PARAMETER_HASH)
        assert receipt["request_id"] == request_id
        assert storage.get_request(request_id) == receipt


@pytest.mark.parametrize("digest", ["", "a" * 63, "a" * 65, "A" * 64, "g" * 64, "a" * 64 + "\n", None])
def test_parameter_hash_is_strict_lowercase_sha256(tmp_path: Path, digest: object) -> None:
    with SQLiteStorage(tmp_path / "hash.sqlite3") as storage:
        with pytest.raises(StorageError, match="parameter_hash"):
            storage.begin_request(REQUEST_ID, "import", digest)  # type: ignore[arg-type]
        assert storage.get_request(REQUEST_ID) is None


@pytest.mark.parametrize("operation", ["", " ", None])
def test_empty_operation_is_rejected(tmp_path: Path, operation: object) -> None:
    with SQLiteStorage(tmp_path / "operation.sqlite3") as storage:
        with pytest.raises(StorageError, match="operation"):
            storage.begin_request(REQUEST_ID, operation, PARAMETER_HASH)  # type: ignore[arg-type]


def test_update_preserves_task_and_response_and_returns_independent_json(tmp_path: Path) -> None:
    with SQLiteStorage(tmp_path / "updates.sqlite3") as storage:
        storage.begin_request(REQUEST_ID, "replace", PARAMETER_HASH)
        saved_response = {"success": True, "asset": {"name": "旧名字", "tags": ["夏季"]}}
        first = storage.update_request(REQUEST_ID, state="accepted", response=saved_response, task_id="old-task-id")
        saved_response["asset"]["name"] = "caller changed"  # type: ignore[index]
        first["response"]["asset"]["name"] = "dto changed"
        second = storage.update_request(REQUEST_ID, state="completed")
        assert second["task_id"] == "old-task-id"
        assert second["response"]["asset"]["name"] == "旧名字"
        third = storage.update_request(REQUEST_ID, state="completed", response={})
        assert third["response"] == {}
        assert third["task_id"] == "old-task-id"
        with pytest.raises(RecordNotFoundError):
            storage.update_request("f" * 32, state="failed")
        with pytest.raises(StorageError, match="invalid request state"):
            storage.update_request(REQUEST_ID, state="sent")
        with pytest.raises(StorageError, match="task_id"):
            storage.update_request(REQUEST_ID, state="accepted", task_id="")
        assert storage.get_request(REQUEST_ID) == third


@pytest.mark.parametrize("response", [[1], {"unserializable": object()}])
def test_invalid_response_does_not_modify_receipt(tmp_path: Path, response: object) -> None:
    with SQLiteStorage(tmp_path / "json.sqlite3") as storage:
        first = storage.begin_request(REQUEST_ID, "import", PARAMETER_HASH)
        with pytest.raises(StorageError, match="response"):
            storage.update_request(REQUEST_ID, state="completed", response=response)  # type: ignore[arg-type]
        with pytest.raises(StorageError, match="response"):
            storage.complete_requests_for_task("task", response)  # type: ignore[arg-type]
        assert storage.get_request(REQUEST_ID) == first


@pytest.mark.parametrize("success", [True, False])
def test_completion_receipts_participate_in_outer_task_transaction(tmp_path: Path, success: bool) -> None:
    database = tmp_path / "completion.sqlite3"
    with SQLiteStorage(database) as storage:
        task = storage.create_task(ImageTask(kind="reference_import", scope_key="webui"))
        other = storage.create_task(ImageTask(kind="reference_retag", scope_key="webui"))
        for request_id, state in [(REQUEST_ID, "accepted"), ("b" * 32, "preparing"), ("c" * 32, "failed")]:
            storage.begin_request(request_id, "import", PARAMETER_HASH)
            storage.update_request(request_id, state=state, response={"previous": True}, task_id=task.id)
        storage.begin_request("d" * 32, "retag", PARAMETER_HASH)
        other_receipt = storage.update_request("d" * 32, state="accepted", task_id=other.id)
        storage.begin_request("e" * 32, "import", PARAMETER_HASH)
        preparing_receipt = storage.get_request("e" * 32)
        terminal_receipt = storage.get_request("c" * 32)
        initial_receipt = storage.get_request(REQUEST_ID)
        response = {"success": success, "task_id": task.id, "asset_id": "asset" if success else None}
        task_status = TaskStatus.SENT if success else TaskStatus.FAILED
        receipt_state = "completed" if success else "failed"

        with pytest.raises(RuntimeError, match="abort"):
            with storage.transaction():
                storage.set_task_status(task.id, task_status)
                assert storage.complete_requests_for_task(task.id, response) is None
                assert storage.get_request(REQUEST_ID)["state"] == receipt_state  # type: ignore[index]
                raise RuntimeError("abort")
        assert storage.get_task(task.id).status == TaskStatus.QUEUED  # type: ignore[union-attr]
        assert storage.get_request(REQUEST_ID) == initial_receipt

        with storage.transaction():
            storage.set_task_status(task.id, task_status)
            storage.complete_requests_for_task(task.id, response)
        completed = storage.get_request(REQUEST_ID)
        assert completed["state"] == receipt_state  # type: ignore[index]
        assert completed["response"] == response  # type: ignore[index]
        assert storage.get_request("b" * 32)["state"] == receipt_state  # type: ignore[index]
        assert storage.get_request("c" * 32) == terminal_receipt
        assert storage.get_request("d" * 32) == other_receipt
        assert storage.get_request("e" * 32) == preparing_receipt
        storage.complete_requests_for_task(task.id, {"success": not success, "late": True})
        assert storage.get_request(REQUEST_ID) == completed
        storage.complete_requests_for_task("expired-or-unknown-task", {"success": True})
        assert storage.delete_task(task.id)
        assert storage.get_request(REQUEST_ID) == completed
        assert storage.connection.execute("PRAGMA foreign_key_check").fetchall() == []

    with SQLiteStorage(database) as reopened:
        assert reopened.get_request(REQUEST_ID) == completed
        assert reopened.get_task(task.id) is None


@pytest.mark.parametrize("nested", [False, True])
def test_cancelled_transaction_rolls_back_and_restores_depth(tmp_path: Path, nested: bool) -> None:
    with SQLiteStorage(tmp_path / "cancel.sqlite3") as storage:
        if nested:
            with storage.transaction():
                outer = storage.begin_request("f" * 32, "import", PARAMETER_HASH)
                with pytest.raises(asyncio.CancelledError):
                    with storage.transaction():
                        storage.begin_request(REQUEST_ID, "import", PARAMETER_HASH)
                        raise asyncio.CancelledError()
                assert storage.get_request(REQUEST_ID) is None
                assert storage.get_request("f" * 32) == outer
                assert storage._transaction_depth == 1
        else:
            with pytest.raises(asyncio.CancelledError):
                with storage.transaction():
                    storage.begin_request(REQUEST_ID, "import", PARAMETER_HASH)
                    raise asyncio.CancelledError()
            assert storage.get_request(REQUEST_ID) is None
        assert storage._transaction_depth == 0
        assert not storage.connection.in_transaction
        storage.begin_request(REQUEST_ID, "import", PARAMETER_HASH)
        assert storage.get_request(REQUEST_ID)["state"] == "preparing"  # type: ignore[index]


def test_failed_begin_does_not_corrupt_transaction_depth(tmp_path: Path) -> None:
    with SQLiteStorage(tmp_path / "begin.sqlite3") as storage:
        storage.connection.execute("BEGIN")
        try:
            with pytest.raises(sqlite3.OperationalError, match="within a transaction"):
                with storage.transaction():
                    pytest.fail("nested unmanaged BEGIN must not run body")
            assert storage._transaction_depth == 0
        finally:
            storage.connection.rollback()
        storage.begin_request(REQUEST_ID, "import", PARAMETER_HASH)
        assert storage.get_request(REQUEST_ID) is not None


def test_receipt_methods_raise_storage_error_after_close(tmp_path: Path) -> None:
    storage = SQLiteStorage(tmp_path / "closed.sqlite3")
    storage.close()
    with pytest.raises(StorageError, match="closed"):
        storage.begin_request(REQUEST_ID, "import", PARAMETER_HASH)
    with pytest.raises(StorageError, match="closed"):
        storage.get_request(REQUEST_ID)
    with pytest.raises(StorageError, match="closed"):
        storage.update_request(REQUEST_ID, state="accepted")
    with pytest.raises(StorageError, match="closed"):
        storage.complete_requests_for_task("task", {"success": True})
