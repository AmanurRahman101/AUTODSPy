"""Offline verification suite for all three APEX-GRPO v3 Colab-ready notebooks."""

from __future__ import annotations

import ast
import copy
import json
import os
import random
import re
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

import litellm
import matplotlib
import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel


matplotlib.use("Agg", force=True)

AUTODSPY_DIR = Path(__file__).resolve().parents[1]
COLAB_DIR = AUTODSPY_DIR / "Colab Notebooks"

NOTEBOOK_SPECS = [
    ("gsm8k", COLAB_DIR / "APEX_GRPO_v3_GSM8K_Colab_Ready.ipynb"),
    ("hotpotqa", COLAB_DIR / "APEX_GRPO_v3_HotpotQA_Colab_Ready.ipynb"),
    ("mixed_routing", COLAB_DIR / "APEX_GRPO_v3_Mixed_Routing_Colab_Ready.ipynb"),
]


class DeterministicMockTokenizer:
    """Offline tokenizer mapping CoT/Predict/stop and A-Z labels to unique single tokens."""

    def __init__(self) -> None:
        self.eos_token = "<|endoftext|>"
        self.eos_token_id = 50256
        self.pad_token = None
        self.pad_token_id = 50256
        self.truncation_side = "left"
        self._fixed = {"CoT": 200, "Predict": 201, "stop": 202}
        for i in range(26):
            ch = chr(ord("A") + i)
            self._fixed[ch] = 100 + i
            self._fixed[" " + ch] = 100 + i

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text in self._fixed:
            return [self._fixed[text]]
        tokens = text.strip().split()
        if not tokens:
            return [self.eos_token_id]
        ids = []
        for tok in tokens:
            if tok in self._fixed:
                ids.append(self._fixed[tok])
            else:
                ids.append(250 + (zlib.crc32(tok.encode("utf-8")) % 49000))
        return ids

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        rev = {v: k for k, v in self._fixed.items()}
        return " ".join(rev.get(int(t), f"tok{int(t)}") for t in token_ids)

    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 128,
    ) -> dict[str, torch.Tensor]:
        ids = self.encode(text, add_special_tokens=False)
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[-max_length:] if self.truncation_side == "left" else ids[:max_length]
        t_ids = torch.tensor([ids], dtype=torch.long)
        mask = torch.ones_like(t_ids)
        return {"input_ids": t_ids, "attention_mask": mask}


def make_tiny_gpt2_factory():
    """Return a deterministic GPT2LMHeadModel factory preserving caller RNG."""
    tiny_cfg = GPT2Config(vocab_size=50257, n_positions=128, n_embd=32, n_layer=1, n_head=2)

    def _from_pretrained(*args, **kwargs):
        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        try:
            torch.manual_seed(13579)
            model = GPT2LMHeadModel(copy.deepcopy(tiny_cfg))
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
        return model

    return _from_pretrained


def cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    return src if isinstance(src, str) else "".join(src)


def strip_ipython_magics(code: str) -> str:
    kept = []
    for line in code.splitlines():
        if line.lstrip().startswith(("!", "%")):
            kept.append(f"# magic stripped: {line}")
        else:
            kept.append(line)
    return "\n".join(kept)


def load_isolated_namespace(nb_path: Path) -> dict:
    """Execute definition and disabled-op cells (9..50, 52..59) in a fresh namespace."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    ns: dict = {"__name__": f"nb_{nb_path.stem}"}
    indices = list(range(9, 51)) + list(range(52, 60))
    for idx in indices:
        cell = nb["cells"][idx]
        if cell.get("cell_type") != "code":
            continue
        code = strip_ipython_magics(cell_source(cell))
        exec(compile(code, f"{nb_path.name}#cell_{idx}", "exec"), ns)
        if idx == 9 and "plt" in ns:
            ns["plt"].show = lambda *a, **k: None
        if idx == 13:
            ns["IN_COLAB"] = False
            ns["BACKUP_TO_DRIVE"] = False
    return ns


def make_synthetic_datasets(dataset_mode: str) -> dict:
    """Create disjoint synthetic GSM8K and HotpotQA splits with easy/hard questions."""
    gsm_train = []
    for i in range(16):
        tag = "EASY_Q" if i % 2 == 0 else "HARD_Q"
        gsm_train.append({
            "question": f"{tag} GSM8K Train #{i}: Compute {10 + i} + {20 + i}. [ID_GTR_{i}]",
            "answer": f"Step by step reasoning. #### {100 + i}",
        })
    gsm_test = [
        {
            "question": f"TEST_G GSM8K Test #{i}: Compute {50 + i} + 1. [ID_GTE_{i}]",
            "answer": f"Reasoning #### {200 + i}",
        }
        for i in range(10)
    ]

    hp_train = []
    for i in range(16):
        tag = "EASY_Q" if i % 2 == 0 else "HARD_Q"
        hp_train.append({
            "id": f"hp_tr_{i}",
            "question": f"{tag} Hotpot Train #{i} entity? [ID_HTR_{i}]",
            "answer": f"Answer{i}",
            "context": {"title": [f"Title{i}"], "sentences": [[f"Fact about Answer{i}."]]},
            "type": "bridge",
            "level": "medium",
            "supporting_facts": {"title": [f"Title{i}"], "sent_id": [0]},
        })
    hp_val = [
        {
            "id": f"hp_va_{i}",
            "question": f"TEST_H Hotpot Val #{i} entity? [ID_HVA_{i}]",
            "answer": f"ValAnswer{i}",
            "context": {"title": [f"ValTitle{i}"], "sentences": [[f"Fact about ValAnswer{i}."]]},
            "type": "bridge",
            "level": "medium",
            "supporting_facts": {"title": [f"ValTitle{i}"], "sent_id": [0]},
        }
        for i in range(10)
    ]

    gsm_ds = {"train": gsm_train, "test": gsm_test}
    hp_ds = {"train": hp_train, "validation": hp_val}
    if dataset_mode == "gsm8k":
        return gsm_ds
    if dataset_mode == "hotpotqa":
        return hp_ds
    return {"gsm8k": gsm_ds, "hotpotqa": hp_ds}


def install_offline_mocks(ns: dict, domain_mode: str) -> None:
    """Patch HF models, datasets, Ollama HTTP endpoints, and DSPy pipeline execution."""
    tiny_factory = make_tiny_gpt2_factory()
    tok_factory = lambda *a, **k: DeterministicMockTokenizer()

    ns["GPT2LMHeadModel"].from_pretrained = staticmethod(tiny_factory)
    ns["AutoModelForCausalLM"].from_pretrained = staticmethod(tiny_factory)
    ns["GPT2Tokenizer"].from_pretrained = staticmethod(tok_factory)
    ns["AutoTokenizer"].from_pretrained = staticmethod(tok_factory)

    synth = make_synthetic_datasets(domain_mode)

    class FakeDatasetInfo:
        def __init__(self, sha: str) -> None:
            self.sha = sha

    class FakeHfApi:
        def dataset_info(self, name: str):
            return FakeDatasetInfo(f"sha-{name.replace('/', '-')}")

    def fake_load_dataset(name: str, config: str | None = None, revision: str | None = None):
        if domain_mode == "mixed_routing":
            return synth["gsm8k"] if "gsm8k" in name else synth["hotpotqa"]
        return synth

    ns["HfApi"] = FakeHfApi
    ns["load_dataset"] = fake_load_dataset

    def fake_json_request(host: str, route: str, payload=None, timeout: int = 15):
        if route == "/api/version":
            return {"version": "0.6.0-mock"}
        if route == "/api/tags":
            return {
                "models": [
                    {"name": "llama3.1:8b", "model": "llama3.1:8b", "digest": "a" * 64},
                    {"name": "llama3.2:3b", "model": "llama3.2:3b", "digest": "b" * 64},
                ]
            }
        if route == "/api/show":
            return {"template": "{{ .Prompt }}", "system": "", "parameters": "temperature 0.0"}
        if route == "/api/ps":
            return {"models": []}
        raise RuntimeError(f"Unexpected mock Ollama route: {route}")

    ns["json_request"] = fake_json_request

    sigs = ns["SIGNATURES"]
    winning_sigs = set(sigs[:4])

    def extract_index_target(prompt: str) -> tuple[bool, str]:
        m_gtr = re.search(r"\[ID_GTR_(\d+)\]", prompt)
        if m_gtr:
            return True, str(100 + int(m_gtr.group(1)))
        m_gte = re.search(r"\[ID_GTE_(\d+)\]", prompt)
        if m_gte:
            return True, str(200 + int(m_gte.group(1)))
        m_htr = re.search(r"\[ID_HTR_(\d+)\]", prompt)
        if m_htr:
            return False, f"Answer{int(m_htr.group(1))}"
        m_hva = re.search(r"\[ID_HVA_(\d+)\]", prompt)
        if m_hva:
            return False, f"ValAnswer{int(m_hva.group(1))}"
        return True, "0"

    def fake_execute_pipeline(trajectory: dict, prompt: str, lm) -> dict:
        # Record a physical LM call on the MeasuredLM ledger without hitting network.
        lm("execute mock pipeline")
        is_gsm, target_val = extract_index_target(prompt)
        is_easy = "EASY_Q" in prompt
        is_test = "TEST_G" in prompt or "TEST_H" in prompt
        good_pipe = (trajectory["module"] == "CoT" and trajectory["signature"] in winning_sigs)

        if is_easy or is_test or good_pipe:
            out = f"#### {target_val}" if is_gsm else f"Reasoning leads to [{target_val}]"
        else:
            out = "#### 99999" if is_gsm else "Unformatted response [WrongAnswer]"
        return dict(output=out, success=True, error=None, seconds=0.002)

    ns["execute_pipeline"] = fake_execute_pipeline


def mock_dspy_lm_forward(self, *args, **kwargs):
    return litellm.ModelResponse(
        choices=[{"message": {"role": "assistant", "content": "0.0"}}],
        usage={"prompt_tokens": 10, "completion_tokens": 2},
        model="ollama_chat/llama3.1:8b",
    )


class TestApexV3Notebooks(unittest.TestCase):
    """Verify structure, algorithms, profiles, and offline end-to-end execution for v3."""

    def test_structure_cleared_outputs_sections_and_ast(self):
        expected_sections = [f"# {n}." for n in range(2, 22)]
        for domain_mode, nb_path in NOTEBOOK_SPECS:
            with self.subTest(domain=domain_mode):
                self.assertTrue(nb_path.is_file(), f"Missing notebook: {nb_path}")
                nb = json.loads(nb_path.read_text(encoding="utf-8"))
                cells = nb["cells"]
                self.assertEqual(len(cells), 60, f"{nb_path.name} must have 60 cells")

                markdown_blob = "\n".join(cell_source(c) for c in cells if c.get("cell_type") == "markdown")
                self.assertIn("# APEX-GRPO v3:", markdown_blob)
                for sec in expected_sections:
                    self.assertIn(sec, markdown_blob, f"Missing section {sec} in {nb_path.name}")

                for idx, cell in enumerate(cells):
                    if cell.get("cell_type") == "code":
                        self.assertEqual(cell.get("outputs"), [], f"Cell {idx} outputs not cleared")
                        self.assertIsNone(cell.get("execution_count"), f"Cell {idx} execution_count not None")
                        ast.parse(strip_ipython_magics(cell_source(cell)), filename=f"{nb_path.name}#cell_{idx}")

    def test_profiles_backup_paths_and_self_checks(self):
        expected_profiles = {
            "apex_v3",
            "static_k5_original",
            "static_k5_stabilized",
            "static_k5_debiased",
            "static_k5_matched",
            "static_k12_matched",
            "ablation_no_prealloc",
            "ablation_with_replacement",
            "ablation_kmin5",
        }
        for domain_mode, nb_path in NOTEBOOK_SPECS:
            with self.subTest(domain=domain_mode):
                ns = load_isolated_namespace(nb_path)
                self.assertEqual(set(ns["RUN_PROFILES"].keys()), expected_profiles)
                ns["adaptive_self_checks"]()

                dirs = set()
                for pname in expected_profiles:
                    cfg = ns["build_config"](run_profile=pname, seed=43, preset="smoke", policy_device="cpu")
                    d = str(ns["resolve_drive_backup_dir"](cfg))
                    self.assertIn(pname, d)
                    self.assertIn("seed43", d)
                    self.assertNotIn(d, dirs)
                    dirs.add(d)

                cfg_qwen = ns["build_config"](
                    run_profile="apex_v3",
                    policy_model="Qwen/Qwen2.5-1.5B",
                    use_lora=True,
                    seed=42,
                    policy_device="cpu",
                )
                self.assertIsNone(cfg_qwen["policy_revision"])
                self.assertIn("Qwen2.5-1.5B_llama3.1-8b", str(ns["resolve_drive_backup_dir"](cfg_qwen)))

    def test_debiasing_without_replacement_and_contrast_updates(self):
        _, nb_path = NOTEBOOK_SPECS[0]
        ns = load_isolated_namespace(nb_path)
        install_offline_mocks(ns, "gsm8k")

        cfg = ns["build_config"](run_profile="apex_v3", seed=42, preset="smoke", policy_device="cpu")
        model, tok, opt, first_tokens, _, init_bias = ns["initialize_policy"](cfg)
        self.assertIsNotNone(init_bias)

        # Neutral prefix must yield exact uniform 1/30 distribution when debiased_logits=True.
        pi, prob_array = ns["compute_pipeline_distribution"](
            model, tok, first_tokens, "Solve the task.", cfg, init_logit_bias=init_bias
        )
        self.assertEqual(len(pi), 30)
        self.assertLess(float(np.max(np.abs(prob_array - (1.0 / 30.0)))), 1e-3)

        # Without-replacement sampling draws strictly distinct untried pairs and replays accurately.
        tried = set()
        for step in range(8):
            traj = ns["sample_untried_pipeline"](
                model, tok, first_tokens, "HARD_Q test prompt", tried, cfg, init_logit_bias=init_bias
            )
            key = (traj["module"], traj["signature"])
            self.assertNotIn(key, tried)
            tried.add(key)
            self.assertGreater(traj["importance_weight"], 0.0)
            self.assertLessEqual(traj["importance_weight"], 1.0 + 1e-9)
            replayed = ns["replay_log_probs"](
                model, tok, first_tokens, "HARD_Q test prompt", traj, cfg, init_logit_bias=init_bias
            )
            self.assertTrue(
                np.allclose([float(x.detach()) for x in replayed], traj["log_probs"], rtol=2e-5, atol=2e-5)
            )

        # Contrast gating + gradient clipping checks inside update_policy.
        session = dict(
            config=cfg,
            policy=model,
            tokenizer=tok,
            optimizer=opt,
            first_tokens=first_tokens,
            init_logit_bias=init_bias,
            evaluation_only=False,
        )
        t_list = [
            ns["generate_pipeline"](model, tok, first_tokens, "HARD_Q gate", cfg, init_logit_bias=init_bias)
            for _ in range(2)
        ]

        res_zv = ns["update_policy"](session, "HARD_Q gate", t_list, [1.0, 1.0])
        self.assertFalse(res_zv["optimizer_step"])
        self.assertEqual(res_zv["skip_reason"], "zero_variance_group")

        res_bc = ns["update_policy"](session, "HARD_Q gate", t_list, [0.8, 0.9])
        self.assertFalse(res_bc["optimizer_step"])
        self.assertEqual(res_bc["skip_reason"], "below_contrast_threshold")

        res_ok = ns["update_policy"](session, "HARD_Q gate", t_list, [1.0, 0.0])
        self.assertTrue(res_ok["optimizer_step"])
        self.assertIsNone(res_ok["skip_reason"])
        self.assertLessEqual(res_ok["gradient_norm"], 1.0 + 1e-5)

    def test_end_to_end_all_three_notebooks(self):
        with patch.object(litellm, "ModelResponse", litellm.ModelResponse):
            for domain_mode, nb_path in NOTEBOOK_SPECS:
                with self.subTest(domain=domain_mode), tempfile.TemporaryDirectory() as tmpdir:
                    ns = load_isolated_namespace(nb_path)
                    install_offline_mocks(ns, domain_mode)

                    common = dict(
                        seed=42,
                        preset="smoke",
                        num_episodes=4,
                        train_pool_size=6,
                        validation_size=4,
                        test_size=4,
                        schedule_episodes=4,
                        validation_interval=2,
                        checkpoint_interval=2,
                        policy_device="cpu",
                        results_root=tmpdir,
                        eval_decoding="greedy",
                        eval_cache_enabled=True,
                        reward_mode="original_semantic",
                        legacy_reward_protocol=True,
                    )

                    with patch.object(ns["dspy"].LM, "forward", new=mock_dspy_lm_forward):
                        # 1. Zero-RL baselines
                        cfg_z = ns["build_config"](run_profile="apex_v3", **common)
                        z_res = ns["run_zero_rl_baselines"](cfg_z)
                        self.assertEqual(set(z_res["baselines"].keys()), {"dspy_predict", "dspy_cot", "random_pipeline"})

                        # 2. Run static_k5_original, static_k5_matched, and apex_v3 end-to-end
                        run_dirs = []
                        sessions = {}
                        for prof in ("static_k5_original", "static_k5_matched", "apex_v3"):
                            cfg = ns["build_config"](run_profile=prof, **common)
                            sess = ns["initialize_experiment"](cfg, create_run=True)
                            ns["train_grpo"](sess)
                            best_ckpt = json.loads((sess["run_dir"] / "best.json").read_text(encoding="utf-8"))["checkpoint"]
                            eval_sess = ns["load_selected_experiment"](best_ckpt, cfg, for_resume=False)
                            test_res = ns["test_model"](eval_sess)

                            # Repeat eval on the same session to verify eval_cache hit behavior
                            if cfg["eval_cache_enabled"]:
                                second_eval = ns["evaluate_module"](eval_sess, "test")
                                self.assertEqual(second_eval["eval_cache_hits"], cfg["test_size"])
                                self.assertEqual(second_eval["eval_physical_llm_calls"], 0)

                            if domain_mode == "hotpotqa":
                                self.assertIn("target_in_prediction_rate", test_res)
                            elif domain_mode == "mixed_routing":
                                self.assertIn("routing_tv_distance", test_res)
                                self.assertIn("gsm8k_em", test_res)
                                self.assertIn("hotpotqa_em", test_res)

                            summary = json.loads((sess["run_dir"] / "summary.json").read_text(encoding="utf-8"))
                            self.assertTrue(summary["completed"])
                            ns["verify_adaptive_artifacts"](sess["run_dir"], summary)
                            run_dirs.append(sess["run_dir"])
                            sessions[prof] = (sess, cfg)

                        # 3. Verify APEX-v3 exercised both K=2 early exit and K>2 without-replacement rescue
                        apex_rows = sessions["apex_v3"][0]["state"]["train"]
                        self.assertTrue(any(r["k"] == 2 and r["stop_reason"] == "all_correct_exit" for r in apex_rows))
                        self.assertTrue(any(r["k"] > 2 and r["rescue_samples_count"] > 0 for r in apex_rows))

                        # 4. Cross-run verification across static_k5_original, static_k5_matched, apex_v3
                        compared = ns["compare_runs"](run_dirs, comparison_axis="episodes")
                        self.assertEqual(len(compared), 3)

                        # 5. Plotting checks (plot_adaptive_results + Cell 59 plot_checkpoint_figures)
                        fig = ns["plot_adaptive_results"](sessions["apex_v3"][0])
                        ns["plt"].close(fig)
                        fig_dir = ns["plot_checkpoint_figures"](sessions["apex_v3"][0]["run_dir"])
                        self.assertTrue((fig_dir / "best_checkpoint.png").is_file())
                        self.assertTrue((fig_dir / "best_checkpoint.pdf").is_file())

                        # 6. Mid-run checkpoint resume check (interrupt after Ep 1, then resume Ep 2..4)
                        cfg_res_i1 = ns["build_config"](run_profile="apex_v3", **{**common, "checkpoint_interval": 1, "validation_interval": 2})
                        res_sess = ns["initialize_experiment"](cfg_res_i1, create_run=True)
                        orig_save = ns["save_checkpoint"]
                        save_calls = {"n": 0}

                        def interrupt_after_ep1(session, validation=None):
                            save_calls["n"] += 1
                            path = orig_save(session, validation=validation)
                            if save_calls["n"] >= 2:
                                raise RuntimeError("Simulated Colab disconnect after Episode 1 checkpoint")
                            return path

                        ns["save_checkpoint"] = interrupt_after_ep1
                        with self.assertRaises(RuntimeError):
                            ns["train_grpo"](res_sess)
                        ns["save_checkpoint"] = orig_save

                        latest_ptr = json.loads((res_sess["run_dir"] / "latest.json").read_text(encoding="utf-8"))
                        self.assertEqual(latest_ptr["episode"], 1)
                        resumed_cfg = json.loads((res_sess["run_dir"] / "config.json").read_text(encoding="utf-8"))["config"]
                        resumed_sess = ns["load_selected_experiment"](latest_ptr["checkpoint"], resumed_cfg, for_resume=True)
                        ns["train_grpo"](resumed_sess)
                        resumed_summary = json.loads((resumed_sess["run_dir"] / "summary.json").read_text(encoding="utf-8"))
                        self.assertTrue(resumed_summary["completed"])
                        ns["verify_adaptive_artifacts"](resumed_sess["run_dir"], resumed_summary)

                        for s, _ in sessions.values():
                            ns["release_experiment"](s)
                        ns["release_experiment"](resumed_sess)


if __name__ == "__main__":
    unittest.main()
