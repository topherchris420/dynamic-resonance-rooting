import subprocess
import sys
import textwrap
import unittest


class MissingScipyRuntimeTest(unittest.TestCase):
    def test_core_resonance_workflow_runs_without_scipy(self):
        script = textwrap.dedent("""
            import importlib.abc
            import sys

            import numpy as np

            class BlockScipy(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "scipy" or fullname.startswith("scipy."):
                        raise ModuleNotFoundError("No module named 'scipy'")
                    return None

            sys.meta_path.insert(0, BlockScipy())
            sys.path.insert(0, "src")

            from drr_framework.modules import DepthCalculator, ResonanceDetector

            sampling_rate = 100.0
            t = np.arange(0, 2, 1 / sampling_rate)
            data = np.sin(2 * np.pi * 5 * t)

            detector_result = ResonanceDetector().detect(
                data,
                method="welch",
                sampling_rate=sampling_rate,
                peak_height_ratio=0.2,
            )
            depth_result = DepthCalculator().calculate(
                data,
                window_size=128,
                sampling_rate=sampling_rate,
                resonance_frequencies=detector_result["dominant_freq"],
            )

            assert detector_result["dominant_freq"].size > 0
            assert abs(float(detector_result["dominant_freq"][0]) - 5.0) <= 1.0
            assert 0.0 <= depth_result["resonance_depth"] <= 1.0
            """)
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=".",
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
