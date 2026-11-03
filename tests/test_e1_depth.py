import os
import sys
import math
import unittest
import tempfile
import shutil
import subprocess
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.config import load_config
from src.experiments.runner import ExperimentRunner
from src.model.gpt import GPT

class TestE1DepthInfrastructure(unittest.TestCase):
    """
    Test suite for E1 Transformer Depth Experiment Infrastructure.
    Verifies configuration loading, architecture parameter scaling, forward passes,
    checkpoint/results isolation, accidental training prevention, independent RNGs,
    and checkpoint loading/evaluation safeguards.
    """

    def setUp(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.configs_dir = os.path.join(self.project_root, 'experiments', 'gpt4', 'e1_depth', 'configs')
        self.config_files = {
            'baseline': os.path.join(self.configs_dir, 'baseline.yaml'),
            'e1_a': os.path.join(self.configs_dir, 'e1_a.yaml'),
            'e1_b': os.path.join(self.configs_dir, 'e1_b.yaml'),
            'e1_c': os.path.join(self.configs_dir, 'e1_c.yaml')
        }
        self.expected_layers = {
            'baseline': 2,
            'e1_a': 3,
            'e1_b': 4,
            'e1_c': 6
        }
        self.expected_params = {
            'baseline': 60800,
            'e1_a': 73504,
            'e1_b': 86208,
            'e1_c': 111616
        }

    def test_all_configurations_exist_and_load(self):
        """Verify all four experiment configurations exist and parse correctly."""
        for name, path in self.config_files.items():
            self.assertTrue(os.path.isfile(path), f"Configuration file missing: {path}")
            cfg = load_config(path)
            self.assertIn('model', cfg, f"Missing 'model' section in {name}")
            self.assertIn('training', cfg, f"Missing 'training' section in {name}")
            self.assertIn('data', cfg, f"Missing 'data' section in {name}")
            self.assertIn('paths', cfg, f"Missing 'paths' section in {name}")

    def test_only_layers_change(self):
        """Verify that ONLY the number of layers changes across E1 configurations."""
        baseline_cfg = load_config(self.config_files['baseline'])
        b_model = baseline_cfg['model']
        b_train = baseline_cfg['training']
        b_data = baseline_cfg['data']

        for name in ['e1_a', 'e1_b', 'e1_c']:
            exp_cfg = load_config(self.config_files[name])
            m = exp_cfg['model']
            t = exp_cfg['training']
            d = exp_cfg['data']

            # Controlled architecture parameters must be identical
            self.assertEqual(m['vocab_size'], b_model['vocab_size'])
            self.assertEqual(m['embedding_dim'], b_model['embedding_dim'])
            self.assertEqual(m['max_context_length'], b_model['max_context_length'])
            self.assertEqual(m['num_heads'], b_model['num_heads'])
            self.assertEqual(m['feed_forward_dim'], b_model['feed_forward_dim'])

            # Controlled training parameters must be identical
            self.assertEqual(t['batch_size'], b_train['batch_size'])
            self.assertEqual(float(t['learning_rate']), float(b_train['learning_rate']))
            self.assertEqual(t['train_steps'], b_train['train_steps'])
            self.assertEqual(t['val_interval'], b_train['val_interval'])
            self.assertEqual(t['seed'], b_train['seed'])

            # Controlled data parameters must be identical
            self.assertEqual(d['tokenizer_type'], b_data['tokenizer_type'])
            self.assertEqual(d['tokenizer_path'], b_data['tokenizer_path'])
            self.assertEqual(d['data_dir'], b_data['data_dir'])

            # Independent variable: num_layers must match expectation and differ from baseline
            self.assertEqual(m['num_layers'], self.expected_layers[name])
            self.assertNotEqual(m['num_layers'], b_model['num_layers'])

    def test_parameter_counts_exact_scaling(self):
        """Verify exact parameter counts for all four depths (12,704 params per layer)."""
        layer_delta = 12704
        for name, path in self.config_files.items():
            runner = ExperimentRunner(config_path=path, device="cpu")
            param_info = runner.count_parameters()
            expected = self.expected_params[name]
            self.assertEqual(
                param_info['trainable'], expected,
                f"Config {name} expected {expected} trainable parameters, got {param_info['trainable']}"
            )
            self.assertEqual(param_info['non_trainable'], 0)
            self.assertEqual(param_info['total'], expected)

        # Confirm step deltas between consecutive depth increments
        p_base = self.expected_params['baseline']
        p_a = self.expected_params['e1_a']
        p_b = self.expected_params['e1_b']
        p_c = self.expected_params['e1_c']

        self.assertEqual(p_a - p_base, layer_delta * 1)
        self.assertEqual(p_b - p_base, layer_delta * 2)
        self.assertEqual(p_c - p_base, layer_delta * 4)

    def test_forward_pass_finite_and_shapes(self):
        """Verify forward pass on synthetic input produces correct shapes without NaN/Inf."""
        for name, path in self.config_files.items():
            runner = ExperimentRunner(config_path=path, device="cpu")
            model = runner.build_model()
            model.eval()

            batch_size = 2
            seq_len = 64
            dummy_input = torch.randint(0, 512, (batch_size, seq_len))

            with torch.no_grad():
                logits = model(dummy_input)

            self.assertEqual(logits.shape, (batch_size, seq_len, 512))
            self.assertTrue(torch.all(torch.isfinite(logits)), f"Non-finite logits detected in {name}")

    def test_isolation_disjoint_checkpoint_and_result_paths(self):
        """Verify each experiment outputs to distinct directories and never collides with canonical GPT-3."""
        checkpoint_paths = set()
        checkpoint_dirs = set()
        canonical_ckpt = os.path.abspath("checkpoints/gpt3/gpt3_baseline.pt")

        for name, path in self.config_files.items():
            runner = ExperimentRunner(config_path=path, device="cpu")
            insp = runner.inspect()
            ckpt_path = os.path.abspath(insp['paths']['checkpoint_path'])
            ckpt_dir = os.path.abspath(insp['paths']['checkpoint_dir'])

            self.assertNotIn(ckpt_path, checkpoint_paths, f"Duplicate checkpoint path for {name}")
            self.assertNotIn(ckpt_dir, checkpoint_dirs, f"Duplicate checkpoint directory for {name}")
            self.assertNotEqual(ckpt_path, canonical_ckpt, f"Collision with canonical GPT-3 checkpoint for {name}")

            checkpoint_paths.add(ckpt_path)
            checkpoint_dirs.add(ckpt_dir)

    def test_runner_does_not_train_accidentally(self):
        """Verify that inspect mode and default CLI execution never trigger training."""
        for name, path in self.config_files.items():
            runner = ExperimentRunner(config_path=path, device="cpu")
            insp = runner.inspect()
            # Assert no checkpoint was created by inspection
            self.assertFalse(os.path.exists(insp['paths']['checkpoint_path']))

        # Test CLI invocation default behavior
        cmd = [sys.executable, "scripts/run_experiment.py", "--config", self.config_files['e1_a']]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        self.assertEqual(res.returncode, 0)
        self.assertIn("NO TRAINING PERFORMED", res.stdout)
        self.assertFalse(os.path.exists("checkpoints/gpt4/e1_depth/e1_a/e1_a_baseline.pt"))

    def test_reproducibility_with_fixed_seed(self):
        """Verify model initialization and sampling are reproducible under identical seed."""
        cfg_path = self.config_files['e1_a']
        runner1 = ExperimentRunner(config_path=cfg_path, device="cpu", seed=42)
        m1 = runner1.build_model()

        runner2 = ExperimentRunner(config_path=cfg_path, device="cpu", seed=42)
        m2 = runner2.build_model()

        for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
            self.assertEqual(n1, n2)
            self.assertTrue(torch.equal(p1, p2), f"Weight divergence in parameter {n1}")

        # Forward pass on same input gives identical output
        dummy = torch.randint(0, 512, (2, 64))
        with torch.no_grad():
            out1 = m1(dummy)
            out2 = m2(dummy)
        self.assertTrue(torch.equal(out1, out2))

    def test_model_initialization_reproducible_after_unrelated_random_ops(self):
        """Verify build_model() yields identical weights even if unrelated RNG calls occur before it."""
        cfg_path = self.config_files['e1_a']

        # Runner 1: clean run
        runner1 = ExperimentRunner(config_path=cfg_path, device="cpu", seed=42)
        m1 = runner1.build_model()

        # Scramble global RNG with intensive unrelated random draws
        _ = torch.rand(10000)
        _ = torch.randn(500, 500)
        _ = torch.randint(0, 100, (1000,))

        # Runner 2: created after random operations, build_model called after more random ops
        runner2 = ExperimentRunner(config_path=cfg_path, device="cpu", seed=42)
        _ = torch.rand(5000)
        m2 = runner2.build_model()

        for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
            self.assertEqual(n1, n2)
            self.assertTrue(torch.equal(p1, p2), f"Weight divergence after random ops in {n1}")

    def test_saved_checkpoint_loaded_and_evaluated_correctly(self):
        """Verify that an explicit checkpoint can be saved, strictly loaded, and evaluated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = self.config_files['e1_a']
            runner = ExperimentRunner(config_path=cfg_path, device="cpu", seed=42)
            model = runner.build_model()

            ckpt_path = os.path.join(tmpdir, "test_e1_a.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": {},
                "step": 100,
                "config": runner.model_cfg
            }, ckpt_path)

            # Load checkpoint into new runner
            runner2 = ExperimentRunner(config_path=cfg_path, device="cpu")
            ckpt_info = runner2.load_checkpoint(ckpt_path)
            self.assertTrue(runner2.checkpoint_loaded)
            self.assertEqual(ckpt_info["step"], 100)

            # Evaluate with loaded checkpoint
            eval_res = runner2.evaluate(batch_count=2, split="val", checkpoint_path=ckpt_path)
            self.assertEqual(eval_res["checkpoint_evaluated"], ckpt_path)
            self.assertTrue(math.isfinite(eval_res["loss"]))
            self.assertTrue(math.isfinite(eval_res["perplexity"]))

    def test_missing_or_incompatible_checkpoint_fails_clearly(self):
        """Verify that missing or architecturally mismatched checkpoints raise clear exceptions."""
        cfg_path = self.config_files['e1_a']
        runner = ExperimentRunner(config_path=cfg_path, device="cpu")

        # Missing checkpoint
        with self.assertRaises(FileNotFoundError):
            runner.load_checkpoint("non_existent_checkpoint_file.pt")

        with self.assertRaises(FileNotFoundError):
            runner.evaluate(require_checkpoint=True)

        # Incompatible checkpoint (2 layers loaded into 3 layers config)
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_ckpt_path = os.path.join(tmpdir, "incompatible.pt")
            m_bad = GPT(vocab_size=512, embedding_dim=32, max_context_length=64, num_heads=4, feed_forward_dim=128, num_layers=2)
            torch.save({
                "model_state_dict": m_bad.state_dict(),
                "config": {"num_layers": 2, "vocab_size": 512, "embedding_dim": 32, "max_context_length": 64, "num_heads": 4, "feed_forward_dim": 128}
            }, bad_ckpt_path)

            with self.assertRaises(ValueError):
                runner.load_checkpoint(bad_ckpt_path)

    def test_training_batch_sequences_match_across_all_four_configurations(self):
        """Verify independent train_generator produces identical training sequence IDs across all depths."""
        batch_sequences = {}
        for name, path in self.config_files.items():
            runner = ExperimentRunner(config_path=path, device="cpu", seed=42)
            _, dataset = runner.load_tokenizer_and_dataset()
            generator = torch.Generator().manual_seed(runner.seed)

            batches = []
            for _ in range(10):
                x, y = dataset.get_batch(split="train", batch_size=16, generator=generator)
                batches.append((x.clone(), y.clone()))
            batch_sequences[name] = batches

        # Compare baseline against e1_a, e1_b, e1_c
        b_batches = batch_sequences['baseline']
        for name in ['e1_a', 'e1_b', 'e1_c']:
            for i in range(10):
                self.assertTrue(
                    torch.equal(b_batches[i][0], batch_sequences[name][i][0]),
                    f"Train input batch {i} mismatch between baseline and {name}"
                )
                self.assertTrue(
                    torch.equal(b_batches[i][1], batch_sequences[name][i][1]),
                    f"Train target batch {i} mismatch between baseline and {name}"
                )

    def test_validation_batch_sequences_match_across_all_four_configurations(self):
        """Verify independent val_generator produces identical validation sequence IDs across all depths."""
        val_sequences = {}
        for name, path in self.config_files.items():
            runner = ExperimentRunner(config_path=path, device="cpu", seed=42)
            _, dataset = runner.load_tokenizer_and_dataset()
            val_gen = torch.Generator().manual_seed(runner.val_seed)

            batches = []
            for _ in range(10):
                x, y = dataset.get_batch(split="val", batch_size=16, generator=val_gen)
                batches.append((x.clone(), y.clone()))
            val_sequences[name] = batches

        b_val = val_sequences['baseline']
        for name in ['e1_a', 'e1_b', 'e1_c']:
            for i in range(10):
                self.assertTrue(
                    torch.equal(b_val[i][0], val_sequences[name][i][0]),
                    f"Val input batch {i} mismatch between baseline and {name}"
                )
                self.assertTrue(
                    torch.equal(b_val[i][1], val_sequences[name][i][1]),
                    f"Val target batch {i} mismatch between baseline and {name}"
                )

    def test_evaluation_does_not_change_training_batch_sequence(self):
        """Verify that running evaluation does not advance or modify the training batch generator."""
        cfg_path = self.config_files['e1_a']

        # Runner A: draws 5 training batches, runs evaluation, draws 5 more training batches
        runner_a = ExperimentRunner(config_path=cfg_path, device="cpu", seed=42)
        _, ds_a = runner_a.load_tokenizer_and_dataset()
        gen_a = torch.Generator().manual_seed(runner_a.seed)

        first_5_a = [ds_a.get_batch(split="train", batch_size=16, generator=gen_a)[0] for _ in range(5)]
        # Run evaluation (should use independent eval generator)
        _ = runner_a.evaluate(batch_count=5, split="val")
        next_5_a = [ds_a.get_batch(split="train", batch_size=16, generator=gen_a)[0] for _ in range(5)]

        # Runner B: draws 10 training batches uninterrupted
        runner_b = ExperimentRunner(config_path=cfg_path, device="cpu", seed=42)
        _, ds_b = runner_b.load_tokenizer_and_dataset()
        gen_b = torch.Generator().manual_seed(runner_b.seed)

        all_10_b = [ds_b.get_batch(split="train", batch_size=16, generator=gen_b)[0] for _ in range(10)]

        for i in range(5):
            self.assertTrue(torch.equal(first_5_a[i], all_10_b[i]), f"Early train batch {i} mismatch")
            self.assertTrue(torch.equal(next_5_a[i], all_10_b[5 + i]), f"Post-eval train batch {5 + i} mismatch")

    def test_memory_estimation_structure(self):
        """Verify memory estimation returns expected schema and scales with depth."""
        mem_base = ExperimentRunner(self.config_files['baseline'], device="cpu").estimate_memory()
        mem_c = ExperimentRunner(self.config_files['e1_c'], device="cpu").estimate_memory()

        for k in ['estimated_param_memory_mb', 'estimated_grad_memory_mb', 'estimated_optimizer_memory_mb', 'estimated_activation_memory_mb', 'estimated_total_training_mb']:
            self.assertIn(k, mem_base)
            self.assertGreater(mem_base[k], 0)
            self.assertGreater(mem_c[k], mem_base[k], f"Memory for {k} did not scale from 2 to 6 layers")

if __name__ == '__main__':
    unittest.main()
