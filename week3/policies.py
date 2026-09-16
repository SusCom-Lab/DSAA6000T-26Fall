"""Application data and scheduling policies, independent of CUDA API calls."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np


POLICIES = ("immediate", "later", "two-streams", "dependency")


@dataclass
class Slot:
    host_input: object
    host_output: object
    x: object
    y: object
    done: object
    batch: int | None = None


class Experiment:
    def __init__(self, runtime, elements):
        self.runtime = runtime
        self.elements = elements
        self.streams = [runtime.stream(), runtime.stream()]
        # Every policy uses the same two slots, allocated outside measurement.
        self.slots = [Slot(
            runtime.host_buffer(elements), runtime.host_buffer(elements),
            runtime.device_buffer(elements), runtime.device_buffer(elements),
            runtime.event(),
        ) for _ in range(2)]
        self.z = runtime.device_buffer(elements)
        self.ready = runtime.event()
        self.base = np.linspace(-1, 1, elements, dtype=np.float32)
        self.reference = np.empty_like(self.base)

    def prepare(self, slot, batch):
        # Real CPU work, identical for every policy; no artificial sleeps.
        np.add(self.base, np.float32(batch % 17) / np.float32(16),
               out=slot.host_input)

    def consume(self, slot, validate, squared=False):
        if slot.batch is None:
            return
        self.runtime.synchronize(slot.done)
        if validate:
            # Check all elements before the slot may be overwritten. Recreate
            # the expected input independently so unsafe input reuse is caught.
            np.add(self.base, np.float32(slot.batch % 17) / np.float32(16),
                   out=self.reference)
            np.multiply(self.reference, np.float32(2), out=self.reference)
            np.add(self.reference, np.float32(1), out=self.reference)
            if squared:
                np.square(self.reference, out=self.reference)
            np.testing.assert_allclose(slot.host_output, self.reference,
                                       rtol=1e-6, atol=1e-6)
        slot.batch = None

    def submit(self, slot, batch, stream):
        r = self.runtime
        r.copy_to_device(slot.host_input, slot.x, stream)
        r.launch("affine", slot.x, slot.y, stream)
        r.copy_to_host(slot.y, slot.host_output, stream)
        r.record_event(slot.done, stream)
        slot.batch = batch

    def run(self, policy, batches, validate=False):
        if policy not in POLICIES or batches < 1:
            raise ValueError("Use a known policy and at least one batch")
        r = self.runtime
        r.finish()
        start = perf_counter()
        try:
            if policy == "dependency":
                self._dependency(batches, validate)
            else:
                self._independent(policy, batches, validate)
            # Each final D2H has completed via consume(), including partial rings.
            return (perf_counter() - start) * 1000
        finally:
            # Also drain on exceptions before any pinned/device buffer is freed.
            r.finish()

    def _independent(self, policy, batches, validate):
        for batch in range(batches):
            slot = self.slots[batch % 2]
            self.consume(slot, validate)  # Safe reuse of input AND output.
            self.prepare(slot, batch)
            if policy == "later" and batch:
                # CPU prepares the next separate buffer while the prior batch
                # runs, then waits before submitting another GPU batch.
                self.consume(self.slots[(batch - 1) % 2], validate)
            stream = self.streams[batch % 2 if policy == "two-streams" else 0]
            self.submit(slot, batch, stream)
            if policy == "immediate":
                self.consume(slot, validate)
        for slot in self.slots:
            self.consume(slot, validate)

    def _dependency(self, batches, validate):
        r = self.runtime
        a, b = self.streams
        slot = self.slots[0]
        for batch in range(batches):
            self.prepare(slot, batch)
            r.copy_to_device(slot.host_input, slot.x, a)
            r.launch("affine", slot.x, slot.y, a)
            r.record_event(self.ready, a)
            r.wait_event(self.ready, b)
            r.launch("square", slot.y, self.z, b)
            r.copy_to_host(self.z, slot.host_output, b)
            r.record_event(slot.done, b)
            slot.batch = batch
            self.consume(slot, validate, squared=True)
