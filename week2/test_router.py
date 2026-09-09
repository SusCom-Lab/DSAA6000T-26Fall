"""GPU selection regressions; all hardware/toolkit calls are mocked."""
from pathlib import Path
import unittest
from unittest.mock import patch

from router import discover_gpu_device


class GPUSelectionTests(unittest.TestCase):
    def test_selected_index_uses_live_uuid(self):
        with (
            patch("router.output", side_effect=[
                "GPU-abcd-1234",
                "nvidia.com/gpu=GPU-dead-0000\nnvidia.com/gpu=GPU-abcd-1234",
            ]) as run,
            patch("router.Path.glob", return_value=[]),
        ):
            self.assertEqual(discover_gpu_device("2"), "nvidia.com/gpu=GPU-abcd-1234")
            self.assertIn("--id=2", run.call_args_list[0].args[0])

    def test_missing_uuid_never_falls_back_to_all_or_index(self):
        with (
            patch("router.output", side_effect=[
                "GPU-abcd-1234", "nvidia.com/gpu=all\nnvidia.com/gpu=0",
            ]),
            patch("router.Path.glob", return_value=[]),
        ):
            with self.assertRaisesRegex(RuntimeError, "No UUID-named CDI"):
                discover_gpu_device("0")

    def test_json_specs_work_without_toolkit_cli(self):
        spec = '{"kind":"example.com/gpu","devices":[{"name":"GPU-abcd-1234"}]}'
        with (
            patch("router.output", side_effect=["GPU-abcd-1234", FileNotFoundError()]),
            patch("router.Path.glob", return_value=[Path("gpu.json")]),
            patch("router.Path.read_text", return_value=spec),
        ):
            self.assertEqual(discover_gpu_device("0"), "example.com/gpu=GPU-abcd-1234")

    def test_invalid_index_rejected_before_hardware_query(self):
        with patch("router.output") as run:
            for index in ("-1", "all", "0,1", ""):
                with self.assertRaises(ValueError):
                    discover_gpu_device(index)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
