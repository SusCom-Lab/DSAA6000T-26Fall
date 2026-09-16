"""Deferred execution tests; opt into real CUDA with WEEK3_TEST_CUDA=1."""

from collections import deque
import os
from types import SimpleNamespace
import unittest

import numpy as np

from policies import Experiment, POLICIES


class DeferredRuntime:
    """Execute queued work only when waited on, including cross-stream waits.

    This checks ownership and ordering, and does not simulate GPU performance.
    """

    def __init__(self):
        self.streams = []
        self.log = []

    def host_buffer(self, elements):
        return np.full(elements, np.nan, dtype=np.float32)

    device_buffer = host_buffer

    def stream(self):
        stream = SimpleNamespace(queue=deque(), submitted=0, completed=0)
        self.streams.append(stream)
        return stream

    def event(self):
        return SimpleNamespace(marker=None)

    def enqueue(self, stream, action):
        stream.queue.append(action)
        stream.submitted += 1

    def copy_to_device(self, host, device, stream):
        # Read the source at execution time, not submission time, so overwriting
        # an in-flight pinned input produces incorrect results in this test.
        self.enqueue(stream, lambda: np.copyto(device, host))

    def copy_to_host(self, device, host, stream):
        self.enqueue(stream, lambda: np.copyto(host, device))

    def launch(self, name, source, target, stream):
        def compute():
            if name == "affine":
                np.multiply(source, np.float32(2), out=target)
                np.add(target, np.float32(1), out=target)
            else:
                np.square(source, out=target)
        self.enqueue(stream, compute)

    def record_event(self, event, stream):
        event.marker = (stream, stream.submitted)

    def drain(self, marker):
        stream, target = marker
        while stream.completed < target:
            stream.queue.popleft()()
            stream.completed += 1

    def wait_event(self, event, stream):
        marker = event.marker
        self.log.append("device wait")
        self.enqueue(stream, lambda: self.drain(marker))

    def synchronize(self, event):
        self.drain(event.marker)

    def finish(self):
        # Consumer first: an absent dependency must not pass accidentally.
        for stream in reversed(self.streams):
            self.drain((stream, stream.submitted))

class TestDeferredPolicies(unittest.TestCase):
    def test_all_batches_and_partial_buffer_cycles(self):
        for elements in (1, 257, 1025):
            runtime = DeferredRuntime()
            experiment = Experiment(runtime, elements)
            for batches in (1, 2, 3, 7):
                for policy in POLICIES:
                    with self.subTest(elements=elements, batches=batches, policy=policy):
                        experiment.run(policy, batches, validate=True)
                        self.assertTrue(all(not s.queue for s in runtime.streams))

    def test_later_prepares_before_waiting(self):
        class ObservedExperiment(Experiment):
            def prepare(self, slot, batch):
                self.runtime.log.append(f"prepare batch {batch}")
                super().prepare(slot, batch)

            def consume(self, slot, validate, squared=False):
                if slot.batch is not None:
                    self.runtime.log.append(f"CPU wait batch {slot.batch}")
                super().consume(slot, validate, squared)

        for policy, prepare_before_wait in (("immediate", False), ("later", True)):
            runtime = DeferredRuntime()
            ObservedExperiment(runtime, 257).run(policy, 3, validate=True)
            prepare = runtime.log.index("prepare batch 1")
            wait = runtime.log.index("CPU wait batch 0")
            self.assertEqual(prepare < wait, prepare_before_wait)

    def test_dependency_needs_device_wait(self):
        runtime = DeferredRuntime()
        experiment = Experiment(runtime, 257)
        runtime.wait_event = lambda event, stream: None
        with self.assertRaises(AssertionError):
            experiment.run("dependency", 1, validate=True)

    def test_validation_detects_corrupt_results(self):
        runtime = DeferredRuntime()
        runtime.launch = lambda name, source, target, stream: runtime.enqueue(
            stream, lambda: target.fill(123))
        with self.assertRaises(AssertionError):
            Experiment(runtime, 257).run("two-streams", 5, validate=True)
        self.assertTrue(all(not s.queue for s in runtime.streams))


@unittest.skipUnless(os.environ.get("WEEK3_TEST_CUDA") == "1",
                     "set WEEK3_TEST_CUDA=1 on a CUDA host")
class TestCudaPolicies(unittest.TestCase):
    def test_real_cuda(self):
        from runtime import CudaRuntime

        runtime = CudaRuntime()
        for elements in (1, 257, 1048579):
            experiment = Experiment(runtime, elements)
            for batches in (1, 3, 6):
                for policy in POLICIES:
                    with self.subTest(elements=elements, batches=batches, policy=policy):
                        experiment.run(policy, batches, validate=True)


if __name__ == "__main__":
    unittest.main()
