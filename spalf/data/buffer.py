import threading
import torch

from spalf.data.store import ActivationStore


class ActivationBuffer:
    def __init__(self, store: ActivationStore, buffer_size: int) -> None:
        self.store = store
        self.buffer_size = buffer_size
        self._ptr = 0
        self._total_served = 0
        self._buffer = self._fill(buffer_size)

        self._refill_stream = torch.cuda.Stream()
        self._refill_event: torch.cuda.Event | None = None
        self._refill_thread: threading.Thread | None = None

    def _fill(self, n: int) -> torch.Tensor:
        chunks, total = [], 0
        while total < n:
            batch = self.store.next_batch()
            chunks.append(batch)
            total += batch.shape[0]
        return torch.cat(chunks, dim=0)[:n]

    def _refill_half(self) -> None:
        half = self.buffer_size // 2
        fresh = self._fill(half)
        with torch.cuda.stream(self._refill_stream):
            end = self._ptr + half
            if end <= self.buffer_size:
                self._buffer[self._ptr:end] = fresh
            else:
                first = self.buffer_size - self._ptr
                self._buffer[self._ptr:] = fresh[:first]
                self._buffer[:half - first] = fresh[first:]
        self._ptr = end % self.buffer_size
        self._refill_event = self._refill_stream.record_event()

    def next_batch(self, batch_size: int) -> torch.Tensor:
        if self._refill_thread is not None:
            self._refill_thread.join()
            self._refill_thread = None
        if self._refill_event is not None:
            self._refill_event.synchronize()
            self._refill_event = None

        indices = torch.randint(0, self.buffer_size, (batch_size,), device="cuda")
        batch = self._buffer[indices]
        self._total_served += batch_size

        if self._total_served % self.buffer_size < batch_size:
            self._refill_thread = threading.Thread(target=self._refill_half, daemon=True)
            self._refill_thread.start()
        return batch
