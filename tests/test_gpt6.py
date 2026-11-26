import os
import sys
import tempfile
import unittest
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.experiments.runner import ExperimentRunner
from src.generation.generate import generate
from src.model.gpt import GPT
from src.tokenizer.subword import SubwordTokenizer


class TestGPT6ModernStack(unittest.TestCase):
    def test_modern_gpt_forward_tied_weights_and_backward(self):
        model = GPT(
            vocab_size=512,
            embedding_dim=64,
            max_context_length=64,
            num_heads=4,
            feed_forward_dim=176,
            num_layers=2,
            norm_type="rmsnorm",
            ffn_type="swiglu",
            pos_encoding="rope",
            tie_weights=True,
        )
        self.assertIsNone(model.embedding.positional_embedding)
        self.assertIs(model.lm_head.linear.weight, model.embedding.token_embedding.embedding.weight)
        self.assertIsNone(model.lm_head.linear.bias)

        x = torch.randint(0, 512, (2, 16))
        logits = model(x)
        self.assertEqual(logits.shape, (2, 16, 512))
        logits.sum().backward()
        for p in model.parameters():
            self.assertTrue(torch.all(torch.isfinite(p)).item())

    def test_classic_gpt_backward_compatibility(self):
        model = GPT(
            vocab_size=512,
            embedding_dim=32,
            max_context_length=64,
            num_heads=4,
            feed_forward_dim=128,
            num_layers=6,
        )
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 111616)

    def test_byte_level_bpe_lossless_roundtrip(self):
        tok = SubwordTokenizer(byte_level=True)
        texts = [
            "Once upon a time, a man lived in a house!",
            "Python: def add(a, b): return a + b\n",
            "Hello world! Testing spaces   and symbols: 100%.",
        ]
        tok.build_vocab(texts, vocab_size=320)
        sample = "once a man lived naked! 100%"
        ids = tok.encode(sample)
        self.assertNotIn(tok.tokenizer.token_to_id("[UNK]"), ids)
        self.assertEqual(tok.decode(ids), sample)

    def test_generate_topk_topp_repetition_penalty(self):
        tok = SubwordTokenizer(byte_level=True)
        tok.build_vocab(["hello world test sentence for tiny gpt"], vocab_size=300)
        model = GPT(
            vocab_size=tok.vocab_size,
            embedding_dim=32,
            max_context_length=32,
            num_heads=4,
            feed_forward_dim=64,
            num_layers=2,
            norm_type="rmsnorm",
            ffn_type="swiglu",
            pos_encoding="rope",
            tie_weights=True,
        )
        gen = torch.Generator().manual_seed(42)
        out = generate(
            model,
            tok,
            "hello",
            max_new_tokens=10,
            temperature=0.8,
            generator=gen,
            top_k=20,
            top_p=0.9,
            repetition_penalty=1.2,
        )
        self.assertTrue(out.startswith("hello"))

    def test_gpt6_configs_inspect(self):
        for cfg in ["ablation_classic.yaml", "ablation_modern.yaml", "gpt6_full.yaml"]:
            r = ExperimentRunner(f"experiments/gpt6/configs/{cfg}", device="cpu")
            insp = r.inspect()
            self.assertGreater(insp["parameters"]["trainable"], 1_000_000)

    def test_runner_periodic_checkpoint_and_auto_resume(self):
        tmp = tempfile.mkdtemp()
        (np.arange(0, 1000, dtype=np.uint16) % 300).tofile(os.path.join(tmp, "train.bin"))
        (np.arange(0, 300, dtype=np.uint16) % 300).tofile(os.path.join(tmp, "val.bin"))
        cfg = {
            "experiment": {"name": "resume_test", "family": "test"},
            "model": {
                "vocab_size": 300,
                "embedding_dim": 32,
                "max_context_length": 16,
                "num_heads": 4,
                "feed_forward_dim": 64,
                "num_layers": 2,
                "norm_type": "rmsnorm",
                "ffn_type": "swiglu",
                "pos_encoding": "rope",
                "tie_weights": True,
            },
            "training": {
                "batch_size": 4,
                "learning_rate": 3e-4,
                "lr_schedule": "cosine",
                "warmup_steps": 1,
                "grad_clip": 1.0,
                "checkpoint_interval": 2,
                "auto_resume": True,
                "train_steps": 4,
                "val_interval": 2,
                "seed": 42,
            },
            "data": {"tokenizer_path": "data/gpt5/gpt5_tokenizer.json", "tokenizer_type": "subword", "data_dir": tmp},
            "paths": {"checkpoint_dir": os.path.join(tmp, "ckpt"), "results_dir": os.path.join(tmp, "res")},
        }
        cfg_path = os.path.join(tmp, "cfg.yaml")
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)

        r1 = ExperimentRunner(cfg_path, device="cpu")
        r1.train(steps=2)
        ckpt1 = torch.load(r1.checkpoint_path, map_location="cpu", weights_only=False)
        self.assertEqual(ckpt1["step"], 2)

        r2 = ExperimentRunner(cfg_path, device="cpu")
        r2.train(steps=4)
        ckpt2 = torch.load(r2.checkpoint_path, map_location="cpu", weights_only=False)
        self.assertEqual(ckpt2["step"], 4)


if __name__ == "__main__":
    unittest.main()

