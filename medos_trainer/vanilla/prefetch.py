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

THREAD-SAFETY CONTRACT: the factory runs on THIS module's daemon threads
(one per `n_threads`), the consumer runs on the caller's thread, and the
two communicate ONLY through the queue. A factory must therefore be a
pure reader of its inputs — for the trainer, only the case list and the
plan; it must never touch the net, the optimizer, or any other
training-thread state (see the factory's own docstring in trainer.py).
`iter_factory` is called ONCE PER THREAD, so each producer owns a private
iterator end to end; give each thread its own numpy generator and the
streams stay independent (the same changed-sample-stream honesty
single-producer prefetch already carries — see `VanillaTrainer.fit`).

FAILURE SEMANTICS: a factory exception is captured and re-raised in the
consumer on the next ``__next__`` — never swallowed, and never a deadlock:
the sentinel or the failure is always put on the queue eventually. The
threads are daemons, so process exit is safe even without ``close()``;
``close()`` exists so long-lived processes (tests, services) can drop the
queue and let the threads wind down promptly.
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
    """Pull batches from ``iter_factory`` on `n_threads` daemon threads into
    a bounded queue; the consumer reads them through the iterator protocol.

    ``iter_factory`` is called ONCE PER PRODUCER THREAD, on that thread; it
    is a factory (not an iterator) so each thread owns its iterator end to
    end. The factory may be endless — the fit loop simply stops calling
    ``__next__`` — or finite: on exhaustion a thread enqueues a sentinel,
    and the iterator raises ``StopIteration`` on this and every later call
    once ALL producers have reported (the last sentinel wins, earlier ones
    are absorbed). If a factory raises, the exception is re-raised in the
    consumer at the next ``__next__`` with its type and message intact, and
    again on any later call.

    ``queue_size`` bounds how far the producers may run ahead; each queued
    entry is one stacked batch (already-allocated numpy arrays), so the
    memory cost is ``queue_size`` batches, not ``queue_size`` cases.

    WHY MORE THAN ONE THREAD (W23): the W22 micro-benchmark showed the
    fp32 train step at ~426 ms while one producer spends ~1-2 s per batch
    on sampling + augmentation — the GPU starves waiting on ONE CPU thread.
    Producers are pure readers of shared numpy case arrays (writes happen
    only on per-patch copies), so threads scale where processes would not;
    three producers roughly triple the pipeline's throughput ceiling.

    DETERMINISM IS PRESERVED, not traded away: every producer owns its own
    iterator and queue, and the consumer pulls from the queues in a FIXED
    ROTATION (batch k comes from queue k mod N, with exhausted producers
    dropped from the ring). Threads change only WHEN a batch is ready,
    never WHICH batch sits at position k — so a same-seed fit replays
    exactly even with N producers, and the reproducibility promise `fit`
    documents survives the speedup. Same-seed runs with different producer
    counts see different streams (each count defines its own rotation), as
    documented on FitPlan.prefetch_threads — same honesty as prefetch
    itself."""

    def __init__(
        self,
        iter_factory: Callable[[], Iterator[Batch]],
        queue_size: int = 4,
        n_threads: int = 1,
    ) -> None:
        if queue_size < 1:
            raise ValueError(f"queue_size is a positive queue depth, got {queue_size}")
        if n_threads < 1:
            raise ValueError(f"n_threads is a positive producer count, got {n_threads}")
        self._stop = threading.Event()
        # ONE QUEUE PER PRODUCER, and the consumer walks them in a fixed
        # rotation: batch k is drawn from queue (k mod N), exhausted
        # producers drop out of the ring. Timing is threaded; ASSIGNMENT
        # is deterministic — the batch sitting at position k never depends
        # on thread scheduling, so a same-seed fit replays exactly (see the
        # class docstring).
        self._queues: list[queue.Queue] = [
            queue.Queue(maxsize=queue_size) for _ in range(n_threads)
        ]
        self._ring: list[int] = list(range(n_threads))
        self._ring_pos = 0
        # Terminal state, remembered so a next() after the end — all
        # producers exhausted, failure, or close() — answers from memory
        # instead of parking on an empty queue. _PENDING = not ended; None
        # = clean StopIteration; a BaseException = a factory's error,
        # re-raised as-is.
        self._ended: object = _PENDING
        self._threads = [
            threading.Thread(
                target=self._produce,
                args=(iter_factory, self._queues[i]),
                name=f"batch-prefetcher-{i}",
                daemon=True,
            )
            for i in range(n_threads)
        ]
        for thread in self._threads:
            thread.start()

    def _put(self, q: queue.Queue, item: object) -> bool:
        """Enqueue on one producer's queue, honouring ``close()``; False
        means the prefetcher was closed and the item should be dropped."""
        while not self._stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                # A full queue means the consumer is busy, not gone —
                # retry, unless close() landed while we waited.
                continue
        return False

    def _produce(self, iter_factory: Callable[[], Iterator[Batch]],
                 q: queue.Queue) -> None:
        try:
            for item in iter_factory():
                if not self._put(q, item):
                    return
            self._put(q, _END)
        except Exception as exc:  # noqa: BLE001 — forwarded to the consumer
            self._put(q, _Failure(exc))

    def __iter__(self) -> BatchPrefetcher:
        return self

    def __next__(self) -> Batch:
        if self._stop.is_set():
            raise StopIteration
        if self._ended is not _PENDING:
            # Terminal state from memory (StopIteration for clean
            # exhaustion or close(), a factory's error otherwise) instead
            # of parking on an empty queue.
            if isinstance(self._ended, BaseException):
                raise self._ended
            raise StopIteration
        while self._ring:
            idx = self._ring[self._ring_pos]
            item = self._queues[idx].get()
            if item is _END:
                # This producer is done: drop it from the rotation WITHOUT
                # advancing, so the next batch takes this slot's successor
                # deterministically.
                self._ring.pop(self._ring_pos)
                if self._ring:
                    self._ring_pos %= len(self._ring)
                continue
            if isinstance(item, _Failure):
                self._ended = item.error
                raise item.error
            self._ring_pos = (self._ring_pos + 1) % len(self._ring)
            return item  # type: ignore[return-value]
        self._ended = None
        raise StopIteration

    def close(self) -> None:
        """Stop the producer threads at their next put (the threads are
        daemons, so this is a courtesy for long-lived processes, not a
        correctness requirement)."""
        self._stop.set()
