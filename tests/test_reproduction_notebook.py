"""Validate the notebook's own code and interactive episode execution offline."""
import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest

import torch


def notebook_namespace():
    root = Path(__file__).resolve().parent.parent
    notebook = json.loads((root / "DSPy_GRPO.ipynb").read_text(encoding="utf-8"))
    module = ModuleType("_autodspy_notebook_validation")
    sys.modules[module.__name__] = module
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        compile(source, f"DSPy_GRPO.ipynb:{cell['id']}", "exec")
        if set(cell["metadata"].get("tags", [])) & {
            "bootstrap", "definition", "configuration", "offline-tests"
        }:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source, f"DSPy_GRPO.ipynb:{cell['id']}", "exec"), module.__dict__)
    return notebook, module.__dict__


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook, cls.ns = notebook_namespace()

    def test_self_contained_cells_and_offline_suite(self):
        # Validate functions embedded in the notebook, not module proxies.
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                for node in ast.walk(ast.parse("".join(cell["source"]))):
                    if isinstance(node, ast.ImportFrom):
                        self.assertFalse((node.module or "").startswith("autodspy_reproduction"))
                    elif isinstance(node, ast.Import):
                        self.assertFalse(any(name.name.startswith("autodspy_reproduction") for name in node.names))
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(self.ns[name])
                                  for name in ("NumericalTests", "LossTests", "PolicyTests", "RunTests"))
        result = unittest.TestResult()
        suite.run(result)
        self.assertTrue(result.wasSuccessful(), repr(result.errors + result.failures))
        self.assertEqual(result.testsRun, 18)

    def test_manual_episode_iteration_matches_validated_module(self):
        from autodspy_reproduction.training import train as module_train
        ns = self.ns
        examples = [ns["Example"](i, f"fixture {i}", "#### 42", "42", "train") for i in range(3)]
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            config = ns["Config"](device="cpu", episodes=2, samples_per_episode=2,
                                   checkpoint_every=1, output_root=directory)
            provenance = {"data": {"kind": "fixture"}, "policy": {"kind": "fixture"},
                          "execution": {"kind": "fixture"}}
            ns["seed_everything"](42)
            expected_policy = ns["fixture_policy"]()
            reference = module_train(config, expected_policy, examples, ns["FixtureExecutor"](), provenance)
            ns["seed_everything"](42)
            policy = ns["fixture_policy"]()
            iterator = ns["train_episodes"](config, policy, examples, ns["FixtureExecutor"](), provenance)
            first = next(iterator)
            self.assertEqual(first["metrics"]["episode"], 1)
            self.assertTrue((first["run"] / "episode_0001.pt").exists())
            # Pausing for inspection must not change policy or sampling RNG.
            second = next(iterator)
            self.assertEqual(second["metrics"]["episode"], 2)
            with self.assertRaises(StopIteration) as completion:
                next(iterator)
            self.assertEqual(completion.exception.value, first["run"])
            self.assertTrue(torch.equal(expected_policy.model.logits, policy.model.logits))
            expected_metrics = [json.loads(line) for line in (reference / "metrics.jsonl").read_text().splitlines()]
            for actual, expected in zip((first["metrics"], second["metrics"]), expected_metrics):
                for key in ("loss", "gradient_norm", "mean_reward", "train_indices"):
                    self.assertEqual(actual[key], expected[key])
            status = json.loads((first["run"] / "status.json").read_text())
            self.assertEqual(status["status"], "training_complete")

    def test_provenance_uses_notebook_code_without_output_or_flag_changes(self):
        manifest = self.ns["environment_manifest"]()
        sources = manifest["source_sha256"]
        self.assertEqual(manifest["entrypoint"], "DSPy_GRPO.ipynb")
        self.assertIn("notebook-cell:policy-definition", sources)
        self.assertIn("notebook-cell:training-definition", sources)
        self.assertNotIn("notebook-cell:experiment-configuration", sources)
        self.assertGreaterEqual(len(sources), 10)


if __name__ == "__main__":
    unittest.main()
