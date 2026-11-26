import hashlib
import json
import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.experiments.runner import ExperimentRunner
from src.model.gpt import GPT
from src.tokenizer.subword import SubwordTokenizer


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class TestGPT5(unittest.TestCase):
    CONFIG_PATH = "experiments/gpt5/configs/gpt5.yaml"
    CKPT_PATH = "checkpoints/gpt5/gpt5_baseline.pt"
    MANIFEST_PATH = "data/gpt5/dataset_manifest.json"
    TOKENIZER_PATH = "data/processed/gpt3_tokenizer.json"

    def test_config_and_frozen_architecture(self):
        runner = ExperimentRunner(self.CONFIG_PATH, device="cpu")
        insp = runner.inspect()
        dims = insp["dimensions"]
        self.assertEqual(dims["vocab_size"], 512)
        self.assertEqual(dims["embedding_dim"], 32)
        self.assertEqual(dims["max_context_length"], 64)
        self.assertEqual(dims["num_heads"], 4)
        self.assertEqual(dims["feed_forward_dim"], 128)
        self.assertEqual(dims["num_layers"], 6)
        self.assertEqual(insp["parameters"]["trainable"], 111616)
        self.assertEqual(insp["parameters"]["total"], 111616)

    def test_tokenizer_unchanged(self):
        with open("configs/artifact_manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)
        expected_sha = manifest["gpt3"]["artifacts"]["tokenizer"]["sha256"]
        self.assertEqual(sha256_file(self.TOKENIZER_PATH), expected_sha)

        tok = SubwordTokenizer()
        tok.load(self.TOKENIZER_PATH)
        self.assertEqual(tok.vocab_size, 512)

    def test_dataset_manifest_and_zero_overlap(self):
        self.assertTrue(os.path.isfile(self.MANIFEST_PATH))
        with open(self.MANIFEST_PATH, "r", encoding="utf-8") as f:
            m = json.load(f)
        self.assertEqual(m["train"]["document_count"], 100000)
        self.assertEqual(m["val"]["document_count"], 2000)
        self.assertGreater(m["train"]["token_count"], 1000000)

        with open("data/processed/val.jsonl", "r", encoding="utf-8") as f:
            old_val = {json.loads(line)["text"] for line in f if line.strip()}
        with open("data/gpt5/train.jsonl", "r", encoding="utf-8") as f:
            gpt5_train = {json.loads(line)["text"] for line in f if line.strip()}
        with open("data/gpt5/val.jsonl", "r", encoding="utf-8") as f:
            gpt5_val = {json.loads(line)["text"] for line in f if line.strip()}

        self.assertEqual(len(gpt5_train & old_val), 0)
        self.assertEqual(len(gpt5_train & gpt5_val), 0)
        self.assertEqual(len(gpt5_val & old_val), 0)

    def test_checkpoint_strict_reload_and_finite(self):
        self.assertTrue(os.path.isfile(self.CKPT_PATH))
        ckpt = torch.load(self.CKPT_PATH, map_location="cpu", weights_only=False)
        self.assertEqual(ckpt["step"], 10000)
        cfg = ckpt["config"]
        self.assertEqual(cfg["num_layers"], 6)
        self.assertEqual(cfg["vocab_size"], 512)
        self.assertEqual(cfg["tokenizer_type"], "subword")

        model_cfg = {k: v for k, v in cfg.items() if k != "tokenizer_type"}
        model = GPT(**model_cfg)
        res = model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.assertEqual(len(res.missing_keys), 0)
        self.assertEqual(len(res.unexpected_keys), 0)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 111616)
        for p in model.parameters():
            self.assertTrue(torch.all(torch.isfinite(p)).item())

    def test_canonical_gpt1_to_gpt3_artifacts_untouched(self):
        with open("configs/artifact_manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)
        for name in ("gpt1", "gpt2", "gpt3"):
            info = manifest[name]["artifacts"]["checkpoint"]
            self.assertEqual(sha256_file(info["local_path"]), info["sha256"])

    def test_gpt5_full_checkpoint_and_config(self):
        full_cfg = "experiments/gpt5/configs/gpt5_full_run.yaml"
        full_ckpt = "checkpoints/gpt5_full/gpt5_full_baseline.pt"
        full_manifest = "data/gpt5_full/dataset_manifest.json"
        self.assertTrue(os.path.isfile(full_cfg))
        self.assertTrue(os.path.isfile(full_ckpt))
        self.assertTrue(os.path.isfile(full_manifest))

        ckpt = torch.load(full_ckpt, map_location="cpu", weights_only=False)
        self.assertEqual(ckpt["step"], 30000)
        cfg = ckpt["config"]
        self.assertEqual(cfg["num_layers"], 6)
        self.assertEqual(cfg["embedding_dim"], 192)
        self.assertEqual(cfg["max_context_length"], 128)
        self.assertEqual(cfg["vocab_size"], 2048)

        model_cfg = {k: v for k, v in cfg.items() if k != "tokenizer_type"}
        model = GPT(**model_cfg)
        res = model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.assertEqual(len(res.missing_keys), 0)
        self.assertEqual(len(res.unexpected_keys), 0)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 3482624)
        for p in model.parameters():
            self.assertTrue(torch.all(torch.isfinite(p)).item())


if __name__ == "__main__":
    unittest.main()
