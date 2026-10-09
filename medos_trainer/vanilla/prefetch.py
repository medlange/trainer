# SPDX-License-Identifier: Apache-2.0
"""One daemon thread keeping a small queue of ready batches between the
training loop's appetite and a batch factory's cost.

WHY THIS EXISTS: on GPU the training loop spends ~0.3-0.5 s per step in the
main process on patch sampling + mirror/rotate augmentation + batch stacking,
and the GPU waits idle through all of it (measured on the PulmoAI benchmark,
see trainer/docs/benchmark-pulmo-2026-10-07.md). nnU-Net hides that latency
behind worker processes; this is the same overlap without multiprocessing.

WHY A THREAD AND NOT PROCESSES: a case array is 100-500 MB, and a process
worker would need it pickled per epoch per worker — the copy would cost more
than the 0.3-0.5 s it saves (and Windows makes process spawning a fresh
import-and-pickle ceremony per worker). The numpy operations that dominate
the producer — fancy indexing, flip/rot90, zoom, stacking — release the GIL,
so a producer thread genuinely runs while the consumer's CUDA kernels do.
That is the same reasoning torch's own DataLoader uses for its pin_memory
thread, and the same trade nnU-Net's workers make with more machinery.

THREAD-SAFETY CONTRACT: the factory runs on THIS module's daemon thread, the
consumer runs on the caller's thread, and the two communicate ONLY through
the queue. A factory must therefore be a pure reader of its inputs — for the
trainer, only the case list and the plan; it must never touch the net, the
optimizer, or any other training-thread state (see the factory's own
docstring in trainer.py).

FAILURE SEMANTICS: a factory exception is captured and re-raised in the
consumer on the next ``__next__`` — never swallowed, and never a deadlock:
the sentinel or the failure is always put on the queue eventually. The
thread is a daemon, so process exit is safe even without ``close()``;
``close()`` exists so long-lived processes (tests, services) can drop the
queue and let the thread wind down promptly.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterator

import numpy as np

#: One training batch: stacked images (B,C,*), labels (B,*), masks (B,C,*)
#: or None when the cases carry no labelled-mask.
Batch = tuple[np.ndarray, np.ndarray, np.ndarray | None]

#: Queue marker for a cleanly exhausted factory. A plain object() identity is
#: enough — batches are tuples, markers are not.
_END = object()

#: __next__'s "no terminal state yet" — None would be ambiguous with the
#: sentinel's meaning, so a dedicated identity.
_PENDING = object()


class _Failure:
    """Queue marker carrying a factory exception to the consumer."""

    __slots__ = ("error",)

    def __init__(self, error: BaseException) -> None:
        self.error = error


class BatchPrefetcher:
    """Pull batches from ``iter_factory`` on one daemon thread into a bounded
    queue; the consumer reads them through the iterator protocol.

    ``iter_factory`` is called ONCE, on the producer thread; it is a factory
    (not an iterator) so the thread owns the iterator end to end. The factory
    may be endless — the fit loop simply stops calling ``__next__`` — or
    finite: on exhaustion the thread enqueues a sentinel and the iterator
    raises ``StopIteration`` on this and every later call. If the factory
    raises, the exception is re-raised in the consumer at the next
    ``__next__`` with its type and message intact, and again on any later
    call.

    ``queue_size`` bounds how far the producer may run ahead; each queued
    entry is one stacked batch (already-allocated numpy arrays), so the memory
    cost is ``queue_size`` batches, not ``queue_size`` cases.
    """

    def __init__(
        self,
        iter_factory: Callable[[], Iterator[Batch]],
        queue_size: int = 4,
    ) -> None:
        if queue_size < 1:
            raise ValueError(f"queue_size is a positive queue depth, got {queue_size}")
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        # Terminal state, remembered so a next() after the end — sentinel,
        # failure, or close() — answers from memory: the producer thread is
        # done by then and a fresh _queue.get() would park the consumer
        # forever. _PENDING = not ended; None = clean StopIteration;
        # a BaseException = the factory's error, re-raised as-is.
        self._ended: object = _PENDING
        self._thread = threading.Thread(
            target=self._produce,
            args=(iter_factory,),
            name="batch-prefetcher",
            daemon=True,
        )
        self._thread.start()

    def _put(self, item: object) -> bool:
        """Enqueue, honouring ``close()``; False means the prefetcher was
        closed and the item should be dropped."""
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                # A full queue means the consumer is busy, not gone — retry,
                # unless close() landed while we waited.
                continue
        return False

    def _produce(self, iter_factory: Callable[[], Iterator[Batch]]) -> None:
        try:
            for item in iter_factory():
                if not self._put(item):
                    return
            self._put(_END)
        except Exception as exc:  # noqa: BLE001 — forwarded to the consumer
            self._put(_Failure(exc))

    def __iter__(self) -> BatchPrefetcher:
        return self

    def __next__(self) -> Batch:
        if self._stop.is_set():
            raise StopIteration
        if self._ended is not _PENDING:
            # The producer is finished; answer the terminal state from memory
            # (StopIteration for clean exhaustion or close(), the factory's
            # error otherwise) instead of parking on an empty queue.
            if isinstance(self._ended, BaseException):
                raise self._ended
            raise StopIteration
        item = self._queue.get()
        if item is _END:
            self._ended = None
            raise StopIteration
        if isinstance(item, _Failure):
            self._ended = item.error
            raise item.error
        return item  # type: ignore[return-value]

    def close(self) -> None:
        """Stop the producer thread at its next put (the thread is a daemon,
        so this is a courtesy for long-lived processes, not a correctness
        requirement)."""
        self._stop.set()
