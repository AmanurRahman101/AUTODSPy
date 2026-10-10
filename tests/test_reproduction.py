"""Offline engineering tests; fixtures never represent Llama benchmark results."""
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
import contextlib
import io
import json
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch
from autodspy_reproduction.config import Config, load_config
from autodspy_reproduction.data import Example, check_no_leakage
from autodspy_reproduction.execution import DSPyExecutor, Execution, pipeline_pair
from autodspy_reproduction.loss import action_loss, group_advantages, group_loss
from autodspy_reproduction.policy import MODULES, SIGNATURES, Policy, action_distribution
from autodspy_reproduction.reward import canonical_answer, compute_reward, exact_match, normalize_number, parse_judge_score
from autodspy_reproduction.runtime import load_checkpoint, save_checkpoint, seed_everything
from autodspy_reproduction.training import train, update_episode
from autodspy_reproduction.evaluation import evaluate


class Inputs(dict):
    def to(self, device):
        return Inputs({key: value.to(device) for key, value in self.items()})


class FixtureTokenizer:
    def encode(self, text, add_special_tokens=False):
        if text in MODULES:
            return [MODULES.index(text)]
        if text == "stop":
            return [9]
        return [2 + SIGNATURES.index(text) % 7, 11]

    def __call__(self, text, **kwargs):
        return Inputs(input_ids=torch.tensor([[0, 1]]))


class FixtureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.linspace(-0.2, 0.2, 12))

    def forward(self, **kwargs):
        return SimpleNamespace(logits=self.logits[None, None, :].expand(1, 2, -1))


class FixtureExecutor:
    def execute(self, prompt, pipeline):
        return Execution("[42]" if pipeline[0] == "CoT" else "[0]", 0.001)

    def judge(self, prompt):
        return "[0]"


def fixture_policy():
    return Policy(FixtureModel(), FixtureTokenizer())


class NumericalTests(unittest.TestCase):
    def test_signed_commas_decimals(self):
        cases = {"[-1,234.50]": Decimal("-1234.5"), "[+00012.00]": Decimal("12"),
                 "Result: -.5": Decimal("-.5"), "3 then 4.0": Decimal("4"),
                 "reasoning 10\n#### -2": Decimal("-2"), "[0]": Decimal("0")}
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(normalize_number(text), expected)
        self.assertTrue(exact_match("[1,000.00]", "1000"))
        self.assertFalse(exact_match("[12]", "2"))
        self.assertIsNone(normalize_number("nothing"))
        self.assertIsNone(normalize_number("[3 or 4]"))

    def test_canonical_preserves_value(self):
        self.assertEqual(canonical_answer("2+2=4\n#### -1,000.50"), "-1,000.50")
        with self.assertRaises(ValueError):
            canonical_answer("42")

    def test_binary_judge_rejects_substring_false_positive(self):
        self.assertEqual(parse_judge_score("[1.0]", binary=True), 1)
        self.assertEqual(parse_judge_score("[0.75]"), .75)
        for text in ("[0.1]", "[10]", "nan", "score 1 or 0", "[inf]"):
            with self.assertRaises(ValueError):
                parse_judge_score(text, binary=True)

    def test_reward_exact_and_fallback(self):
        judge = lambda _: "[0.75]"
        self.assertEqual(compute_reward("q", "[42]", "42", judge).value, 1)
        result = compute_reward("q", "[0]", "42", judge)
        self.assertEqual(result.value, .75)
        self.assertTrue(result.fallback_used)
        self.assertEqual(compute_reward("q", None, "42", judge).value, 0)
        self.assertIsNotNone(compute_reward("q", "[0]", "42", judge, binary=True).error)

    def test_infrastructure_errors_propagate(self):
        def broken(_):
            raise ConnectionError("offline")
        with self.assertRaises(ConnectionError):
            compute_reward("q", "[0]", "42", broken)


class LossTests(unittest.TestCase):
    def test_centered_and_population_normalization(self):
        self.assertTrue(torch.allclose(group_advantages([0, 1, 1]), torch.tensor([-2/3, 1/3, 1/3])))
        standardized = group_advantages([0, 1, 1], "standardized")
        self.assertAlmostEqual(float(standardized.std(correction=0)), 1, places=6)
        for value in (0., 1., .1234567):
            self.assertTrue(torch.equal(group_advantages([value]*5), torch.zeros(5)))
            self.assertTrue(torch.equal(group_advantages([value]*5, "standardized"), torch.zeros(5)))
        with self.assertRaises(ValueError):
            group_advantages([0, float("nan")])

    def test_notebook_loss_matches_released_formula(self):
        current = torch.tensor([[-.2, -.7, 0], [-.9, -.3, 0]], requires_grad=True)
        old = current.detach().clone().requires_grad_()
        loss = group_loss(current, old, [1., 0.], torch.zeros_like(current))
        expected = -(current[0].sum()*.5 + current[1].sum()*-.5)
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertIsNone(old.grad)
        self.assertTrue(torch.isfinite(current.grad).all())
        self.assertGreater(float(current.grad.abs().sum()), 0)

    def test_clipping_both_advantage_signs(self):
        logs = torch.log(torch.tensor([1.5, .5], requires_grad=True))
        loss = action_loss(logs, torch.zeros(2), torch.tensor([1., -1.]), torch.zeros(2), "clipped")
        self.assertTrue(torch.allclose(loss, torch.tensor([-1.2, .8])))

    def test_clipped_entropy_and_behavior_detachment(self):
        current = torch.tensor(-.5, requires_grad=True)
        old = torch.tensor(-.5, requires_grad=True)
        entropy = torch.tensor(.7, requires_grad=True)
        loss = action_loss(current, old, 1., entropy, "clipped", .2, .01)
        self.assertAlmostEqual(float(loss.detach()), -1.007, places=6)
        loss.backward()
        self.assertIsNone(old.grad)
        self.assertAlmostEqual(float(current.grad), -1, places=6)
        self.assertAlmostEqual(float(entropy.grad), -.01, places=6)

    def test_equal_rewards_zero_gradient(self):
        logits = torch.tensor([[-.5, -.3], [-.7, -.2]], requires_grad=True)
        loss = group_loss(logits, logits.detach(), [1, 1], torch.zeros_like(logits))
        loss.backward()
        self.assertEqual(float(loss), 0)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))


class PolicyTests(unittest.TestCase):
    def test_duplicate_tokens_are_separate_action_slots(self):
        tokenizer = FixtureTokenizer()
        logits = torch.arange(12, dtype=torch.float32, requires_grad=True)
        actions = (SIGNATURES[0], SIGNATURES[7], SIGNATURES[1])
        dist = action_distribution(logits, tokenizer, actions)
        expected = torch.softmax(logits[[2, 2, 3]], -1)
        self.assertTrue(torch.allclose(dist.probs, expected))
        self.assertEqual(float(dist.probs[0]), float(dist.probs[1]))
        dist.log_prob(torch.tensor(0)).backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_generation_stop_and_replay_probability(self):
        policy = fixture_policy()
        trajectory = policy.generate("q")
        self.assertEqual(len(trajectory.pipeline), 3)
        self.assertEqual(trajectory.pipeline[-1], "stop")
        self.assertEqual(trajectory.decisions[-1].old_log_prob, 0)
        for decision in trajectory.decisions:
            new, _ = policy.score(decision)
            self.assertAlmostEqual(float(new.detach()), decision.old_log_prob, places=6)
        self.assertIsNotNone(pipeline_pair(trajectory.pipeline))
        self.assertIsNotNone(pipeline_pair(trajectory.pipeline[:-1]))
        self.assertIsNone(pipeline_pair(("ReAct", "question -> answer")))
        self.assertIsNone(pipeline_pair(("stop",)))

    def test_replay_updates_parameters(self):
        seed_everything(42)
        policy = fixture_policy()
        groups = [([policy.generate("q") for _ in range(5)], [0, 1, 0, 1, 0])]
        before = policy.model.logits.detach().clone()
        metrics = update_episode(policy, torch.optim.Adam(policy.model.parameters(), lr=1e-4), groups, Config())
        self.assertGreater(metrics["gradient_norm"], 0)
        self.assertFalse(torch.equal(before, policy.model.logits))

    def test_accumulated_replay_matches_full_graph_gradient(self):
        seed_everything(42)
        policy = fixture_policy()
        groups = [([policy.generate("q") for _ in range(5)], [0, 1, 0, 1, 0]),
                  ([policy.generate("q2") for _ in range(5)], [1, 0, 1, 0, 1])]
        expected_policy = fixture_policy()
        loss = 0
        for trajectories, rewards in groups:
            advantages = group_advantages(rewards)
            for trajectory, advantage in zip(trajectories, advantages):
                for decision in trajectory.decisions[:-1]:
                    logprob, _ = expected_policy.score(decision)
                    loss = loss-logprob*advantage/len(groups)
        loss.backward()
        expected_gradient = expected_policy.model.logits.grad.clone()
        # SGD with lr=1 makes its parameter delta exactly the accumulated gradient.
        before = policy.model.logits.detach().clone()
        update_episode(policy, torch.optim.SGD(policy.model.parameters(), lr=1), groups, Config())
        self.assertTrue(torch.allclose(before-policy.model.logits, expected_gradient, atol=1e-6))


class RunTests(unittest.TestCase):
    def test_dspy_execution_fields_and_adapter_failure(self):
        import dspy
        from dspy.utils.exceptions import AdapterParseError
        executor = DSPyExecutor(Config(device="cpu"))
        self.assertEqual(executor.lm.kwargs["num_gpu"], 0)
        self.assertEqual(executor.lm.kwargs["num_ctx"], 8192)
        self.assertFalse(executor.lm.cache)
        program = Mock(return_value=dspy.Prediction(summary="[42]"))
        with patch.object(dspy, "Predict", return_value=program):
            result = executor.execute("q", ("Predict", "text -> summary", "stop"))
        self.assertEqual(result.response, "[42]")
        self.assertIn("text", program.call_args.kwargs)
        self.assertIn("square brackets", program.call_args.kwargs["text"])
        failed = Mock(side_effect=AdapterParseError("test", dspy.Signature("text -> summary"), "malformed"))
        with patch.object(dspy, "Predict", return_value=failed):
            result = executor.execute("q", ("Predict", "text -> summary"))
        self.assertIsNone(result.response)
        self.assertIn("adapter_parse_error", result.error)

    def test_configs_and_dataset_split_guard(self):
        full = load_config("configs/grpo_gsm8k.yaml")
        smoke = load_config("configs/grpo_smoke_test.yaml")
        self.assertEqual((full.episodes, full.samples_per_episode, full.group_size), (200, 200, 5))
        self.assertEqual((smoke.episodes, smoke.group_size), (1, 5))
        with self.assertRaises(ValueError):
            replace(full, execution_model="llama3.2:3b").validate()
        check_no_leakage([Example(0, "a", "#### 1", "1", "train")],
                         [Example(0, "b", "#### 2", "2", "test")])
        with self.assertRaises(ValueError):
            check_no_leakage([Example(0, "a", "#### 1", "1", "train")],
                             [Example(0, "a", "#### 1", "1", "test")])

    def test_checkpoint_refuses_overwrite_and_restores_rng(self):
        seed_everything(42)
        policy = fixture_policy()
        sampler = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory)/"checkpoint.pt"
            save_checkpoint(file, policy, None, Config(), 0, sampler, {})
            expected = torch.rand(3)
            load_checkpoint(file, policy, sampler=sampler, restore_rng=True)
            self.assertTrue(torch.equal(torch.rand(3), expected))
            with self.assertRaises(FileExistsError):
                save_checkpoint(file, policy, None, Config(), 0, sampler, {})

    def test_training_checkpoint_resume_and_four_way_evaluation(self):
        examples = [Example(i, f"fixture {i}", "#### 42", "42", "train") for i in range(3)]
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            config = Config(device="cpu", episodes=2, samples_per_episode=2, checkpoint_every=1,
                            output_root=directory)
            provenance = {"data": {"kind": "fixture"}, "policy": {"kind": "fixture"},
                          "execution": {"kind": "fixture"}}
            seed_everything(42)
            policy = fixture_policy()
            run = train(config, policy, examples, FixtureExecutor(), provenance)
            final = policy.model.logits.detach().clone()
            checkpoint = run/"episode_0002.pt"
            self.assertTrue(checkpoint.exists())
            metrics = [json.loads(line) for line in (run/"metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(metrics), 2)
            self.assertEqual(metrics[0]["pipeline_evaluations"], 10)
            resumed_policy = fixture_policy()
            seed_everything(0)  # Must be replaced by checkpoint RNG.
            resumed = train(config, resumed_policy, examples, FixtureExecutor(), provenance,
                            resume=run/"episode_0001.pt")
            self.assertTrue(torch.equal(final, resumed_policy.model.logits))
            self.assertEqual(json.loads((resumed/"status.json").read_text())["completed_episodes"], 2)
            held_out = [Example(0, "held out fixture", "#### 42", "42", "test")]
            eval_run = evaluate(config, policy, held_out, FixtureExecutor(), checkpoint, provenance)
            results = json.loads((eval_run/"results.json").read_text())
            self.assertEqual(set(results), {"static_predict", "static_cot", "untrained_policy", "trained_policy"})
            self.assertTrue(all(row["count"] == 1 for row in results.values()))


if __name__ == "__main__":
    unittest.main()
