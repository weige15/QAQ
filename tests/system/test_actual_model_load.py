import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = Path(
    os.environ.get(
        "QAQ_MODEL_SNAPSHOT",
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
        "snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
    )
).expanduser()


@unittest.skipUnless(
    os.environ.get("QAQ_RUN_RESOURCE_HEAVY") == "1",
    "resource-heavy actual-model test; set QAQ_RUN_RESOURCE_HEAVY=1",
)
class ActualModelLoadTest(unittest.TestCase):
    def test_pinned_model_loads_and_inspects_offline(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "actual.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/inspect_actual_model.py"),
                    "--mode",
                    "actual",
                    "--model-path",
                    str(SNAPSHOT),
                    "--device",
                    os.environ.get("QAQ_MODEL_DEVICE", "cuda:3"),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
            )
            import json

            evidence = json.loads(output.read_text())
            self.assertEqual(evidence["s00_mapping_comparison"]["result"], "MATCH")
            self.assertEqual(evidence["total_target_count"], 252)
            self.assertTrue(evidence["full_precision_smoke_forward"]["finite_values"])
            self.assertFalse(evidence["quantization_performed"])


if __name__ == "__main__":
    unittest.main()
