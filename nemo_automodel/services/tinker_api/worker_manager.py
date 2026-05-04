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
import time
from dataclasses import dataclass, field
from typing import Optional


def _idle_worker(stop_event) -> None:
    """Keep a managed worker process alive until the supervisor asks it to stop."""
    while not stop_event.is_set():
        time.sleep(0.1)


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
        start_method: str = "spawn",
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
        self._ctx = mp.get_context(start_method)
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

    def _restart_count(self, worker_id: str) -> int:
        slot = self._slots.get(worker_id)
        return 0 if slot is None else slot.record.restarts

    def _start_slot(self, worker_id: str, *, restarts: int) -> _WorkerSlot:
        stop_event = self._ctx.Event()
        process = self._ctx.Process(
            target=_idle_worker,
            args=(stop_event,),
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
        return _WorkerSlot(record=record, process=process, stop_event=stop_event)
