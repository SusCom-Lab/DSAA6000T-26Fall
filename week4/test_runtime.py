"""Planner failure paths, capability masks, and CPU execution contracts."""

import unittest
from unittest.mock import Mock, patch

from backends import CpuBackend, CudaBackend, make_inputs, validate
from runtime import Mask, Policy, Probe, Runtime, UnsupportedRequest


def fake_backend(cls):
    backend = Mock()
    backend.name = cls.name
    backend.capabilities = cls.capabilities
    backend.available.return_value = Probe(True, "test device available")
    backend.probe.return_value = Probe(True, "test probe passed")
    return backend


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.cuda = fake_backend(CudaBackend)
        self.cpu = fake_backend(CpuBackend)
        self.runtime = Runtime([self.cuda, self.cpu], emit=lambda _: None)

    def test_irregular_shape_rejects_combined_before_probe(self):
        plan = self.runtime.plan((64, 130, 128))
        self.assertEqual((plan.backend, plan.decision), ("cuda", "decompose"))
        self.assertEqual(plan.capability.path, "separate")
        self.cuda.probe.assert_called_once_with(plan.capability)
        self.cpu.available.assert_not_called()

    def test_failed_combined_probe_uses_separate_on_same_gpu(self):
        self.cuda.probe.side_effect = [Probe(False, "unsupported"), Probe(True, "ok")]
        plan = self.runtime.plan((64, 128, 128))
        self.assertEqual((plan.backend, plan.decision), ("cuda", "decompose"))

    def test_fp16_mask_does_not_poison_next_request(self):
        promoted = self.runtime.plan((64, 128, 128), mask=Mask(disable_fp16=True))
        self.assertEqual((promoted.backend, promoted.capability.dtype), ("cuda", "fp32"))
        self.cuda.probe.assert_called_once_with(promoted.capability)
        native = self.runtime.plan((64, 128, 128))
        self.assertEqual((native.decision, native.capability.dtype), ("direct", "fp16"))

    def test_offline_mask_never_touches_cuda(self):
        plan = self.runtime.plan((64, 128, 128), mask=Mask(disable_cuda=True))
        self.assertEqual((plan.backend, plan.decision), ("cpu", "offload"))
        self.cuda.available.assert_not_called()
        self.cuda.probe.assert_not_called()

    def test_real_unavailability_falls_back(self):
        self.cuda.available.return_value = Probe(False, "no GPU")
        plan = self.runtime.plan((64, 128, 128))
        self.assertEqual(plan.backend, "cpu")
        self.cuda.probe.assert_not_called()

    def test_policy_can_forbid_precision_change(self):
        with self.assertRaises(UnsupportedRequest):
            self.runtime.plan((64, 128, 128), Policy(allow_promotion=False),
                              Mask(disable_fp16=True))
        self.cuda.probe.assert_not_called()
        self.cpu.probe.assert_not_called()

    def test_policy_can_forbid_device_change(self):
        with self.assertRaises(UnsupportedRequest):
            self.runtime.plan((64, 128, 128), Policy(allow_offload=False),
                              Mask(disable_cuda=True))
        self.cpu.available.assert_not_called()

    def test_invalid_dimensions_rejected_before_discovery(self):
        with self.assertRaises(ValueError):
            self.runtime.plan((64, 0, 128))
        self.cuda.available.assert_not_called()


class CpuExecutionTests(unittest.TestCase):
    def test_probe_is_cached(self):
        backend = CpuBackend()
        cap = backend.capabilities[0]
        with patch.object(backend, "execute", wraps=backend.execute) as execute:
            self.assertTrue(backend.probe(cap).ok)
            self.assertTrue(backend.probe(cap).ok)
            execute.assert_called_once()

    def test_irregular_cpu_output_and_validation_detects_corruption(self):
        runtime = Runtime([CudaBackend(), CpuBackend()], emit=lambda _: None)
        inputs = make_inputs(5, 13, 7)
        output, plan = runtime.linear(*inputs, mask=Mask(disable_cuda=True))
        self.assertEqual(plan.backend, "cpu")
        self.assertEqual(tuple(output.shape), (5, 7))
        validate(output, *inputs)
        output = output.clone()
        output[0, 0] += 1
        with self.assertRaises(AssertionError):
            validate(output, *inputs)

    def test_incompatible_shapes_rejected(self):
        runtime = Runtime([CpuBackend()], emit=lambda _: None)
        x, w, b = make_inputs(5, 13, 7)
        with self.assertRaises(ValueError):
            runtime.linear(x, w[:-1], b, policy=Policy(preferred_backend="cpu"))


if __name__ == "__main__":
    unittest.main()
