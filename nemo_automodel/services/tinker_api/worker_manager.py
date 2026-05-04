# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local process supervision for the Tinker API prototype."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import queue
import time
from dataclasses import dataclass, field
from typing import Any, Optional


def _default_start_method() -> str:
    """Choose a multiprocessing start method that works well for local service workers."""
    if "fork" in mp.get_all_start_methods():
        return "fork"
    return "spawn"


def _rpc_worker(stop_event, command_queue, result_queue) -> None:
    """Serve simple worker-management RPC commands until shutdown."""
    while not stop_event.is_set():
        try:
            request = command_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        request_id = request.get("request_id")
        command = request.get("command")
        try:
            if command == "ping":
                response = {
                    "worker_pid": os.getpid(),
                    "time": time.time(),
                }
            elif command == "stop":
                stop_event.set()
                response = {"stopping": True}
            else:
                raise ValueError(f"Unknown worker command: {command!r}")
            result_queue.put({"request_id": request_id, "ok": True, "result": response})
        except Exception as exc:
            result_queue.put(
                {
                    "request_id": request_id,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )


@dataclass
class WorkerProcessRecord:
    """Serializable status for one managed worker process."""

    worker_id: str
    pid: Optional[int]
    status: str
    restarts: int = 0
    started_at: float = field(default_factory=time.time)
    stopped_at: Optional[float] = None
    last_exitcode: Optional[int] = None


@dataclass
class _WorkerSlot:
    record: WorkerProcessRecord
    process: mp.Process
    stop_event: mp.Event
    command_queue: mp.Queue
    result_queue: mp.Queue


class ProcessWorkerManager:
    """Supervise a fixed set of local worker processes.

    The current prototype uses this as a lifecycle and placement primitive. A
    future step can replace `_idle_worker` with a subprocess RPC server while
    keeping the assignment and health model stable.
    """

    def __init__(
        self,
        *,
        num_workers: int,
        start_method: Optional[str] = None,
        worker_prefix: str = "tinker-worker",
        stop_timeout_seconds: float = 5.0,
    ):
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if stop_timeout_seconds <= 0:
            raise ValueError("stop_timeout_seconds must be positive")
        self.num_workers = num_workers
        self.worker_prefix = worker_prefix
        self.stop_timeout_seconds = stop_timeout_seconds
        self._ctx = mp.get_context(start_method or _default_start_method())
        self._slots: dict[str, _WorkerSlot] = {}

    def start(self) -> None:
        """Start all configured worker processes."""
        for index in range(self.num_workers):
            worker_id = f"{self.worker_prefix}-{index}"
            if worker_id in self._slots and self._slots[worker_id].process.is_alive():
                continue
            self._slots[worker_id] = self._start_slot(worker_id, restarts=self._restart_count(worker_id))

    def stop(self) -> None:
        """Stop all managed worker processes."""
        for slot in list(self._slots.values()):
            self._submit_to_slot(slot, "stop", timeout_seconds=0.25, raise_on_error=False)
            slot.stop_event.set()
        deadline = time.monotonic() + self.stop_timeout_seconds
        for slot in list(self._slots.values()):
            remaining = max(0.0, deadline - time.monotonic())
            slot.process.join(timeout=remaining)
            if slot.process.is_alive():
                slot.process.terminate()
                slot.process.join(timeout=1.0)
            slot.record.status = "stopped"
            slot.record.stopped_at = time.time()
            slot.record.last_exitcode = slot.process.exitcode

    def restart_dead(self) -> list[WorkerProcessRecord]:
        """Restart workers that exited unexpectedly and return their new records."""
        restarted = []
        for worker_id, slot in list(self._slots.items()):
            if slot.process.is_alive():
                continue
            old_record = slot.record
            old_record.status = "exited"
            old_record.stopped_at = time.time()
            old_record.last_exitcode = slot.process.exitcode
            new_slot = self._start_slot(worker_id, restarts=old_record.restarts + 1)
            self._slots[worker_id] = new_slot
            restarted.append(new_slot.record)
        return restarted

    def snapshot(self) -> list[WorkerProcessRecord]:
        """Return current worker process records."""
        records = []
        for slot in self._slots.values():
            record = slot.record
            if record.status == "running" and not slot.process.is_alive():
                record.status = "exited"
                record.stopped_at = time.time()
                record.last_exitcode = slot.process.exitcode
            records.append(record)
        return records

    def assign(self, key: str) -> Optional[WorkerProcessRecord]:
        """Assign a stable key to one currently running worker."""
        running = [record for record in self.snapshot() if record.status == "running"]
        if not running:
            return None
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], byteorder="big") % len(running)
        return sorted(running, key=lambda record: record.worker_id)[index]

    def submit(self, worker_id: str, command: str, *, timeout_seconds: float = 30.0) -> dict[str, Any]:
        """Submit a management command to a worker process and return its result."""
        slot = self._slots.get(worker_id)
        if slot is None:
            raise KeyError(f"Unknown worker_id: {worker_id}")
        if not slot.process.is_alive():
            raise RuntimeError(f"Worker {worker_id!r} is not running")
        return self._submit_to_slot(slot, command, timeout_seconds=timeout_seconds, raise_on_error=True)

    def _restart_count(self, worker_id: str) -> int:
        slot = self._slots.get(worker_id)
        return 0 if slot is None else slot.record.restarts

    def _start_slot(self, worker_id: str, *, restarts: int) -> _WorkerSlot:
        stop_event = self._ctx.Event()
        command_queue = self._ctx.Queue()
        result_queue = self._ctx.Queue()
        process = self._ctx.Process(
            target=_rpc_worker,
            args=(stop_event, command_queue, result_queue),
            name=worker_id,
            daemon=True,
        )
        process.start()
        record = WorkerProcessRecord(
            worker_id=worker_id,
            pid=process.pid or os.getpid(),
            status="running",
            restarts=restarts,
        )
        return _WorkerSlot(
            record=record,
            process=process,
            stop_event=stop_event,
            command_queue=command_queue,
            result_queue=result_queue,
        )

    def _submit_to_slot(
        self,
        slot: _WorkerSlot,
        command: str,
        *,
        timeout_seconds: float,
        raise_on_error: bool,
    ) -> dict[str, Any]:
        request_id = f"req_{time.monotonic_ns()}"
        slot.command_queue.put({"request_id": request_id, "command": command})
        deadline = time.monotonic() + timeout_seconds
        parked_results = []
        while time.monotonic() < deadline:
            try:
                response = slot.result_queue.get(timeout=max(0.01, min(0.1, deadline - time.monotonic())))
            except queue.Empty:
                continue
            if response.get("request_id") != request_id:
                parked_results.append(response)
                continue
            for parked in parked_results:
                slot.result_queue.put(parked)
            if response.get("ok"):
                return response.get("result", {})
            if raise_on_error:
                raise RuntimeError(response.get("error", "worker command failed"))
            return {"error": response.get("error")}
        for parked in parked_results:
            slot.result_queue.put(parked)
        if raise_on_error:
            raise TimeoutError(f"Worker {slot.record.worker_id!r} did not answer command {command!r}")
        return {"error": "timeout"}
