import os
import sys
import math
import time
import json
import yaml
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

from src.model.gpt import GPT
from src.utils.device import resolve_device
from src.utils.config import load_config
from src.tokenizer.factory import get_tokenizer
from src.data.language_dataset import LanguageDataset
from src.training.optimizer import create_optimizer
from src.training.step import train_step, validation_step

class ExperimentRunner:
    """
    Modular, reusable experiment runner for architecture explorations (e.g., E1 depth scaling).
    Provides strict separation between configuration inspection, evaluation, and training.
    Features:
      - Fully reproducible model initialization via dedicated seeding in build_model().
      - Independent seeded torch.Generators for training and validation batch sampling.
      - Safe checkpoint loading and evaluation with strict architecture validation.
      - Training is NEVER executed unless explicitly called via train().
    """

    def __init__(self, config_path: str, device: str = "auto", seed: Optional[int] = None):
        self.config_path = os.path.abspath(config_path)
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        self.raw_config = load_config(self.config_path)
        self._parse_config()

        # Seed and device resolution
        if seed is not None:
            self.seed = seed
        else:
            self.seed = self.training_cfg.get("seed", 42)
        self.val_seed = self.training_cfg.get("val_seed", self.seed + 10000)

        # Initialize global RNG for general safety
        torch.manual_seed(self.seed)

        self.device = resolve_device(device)
        self.model: Optional[GPT] = None
        self.tokenizer = None
        self.dataset: Optional[LanguageDataset] = None

        # Checkpoint tracking
        self.checkpoint_loaded = False
        self.loaded_checkpoint_path: Optional[str] = None

        # Independent Generators
        self.train_generator: Optional[torch.Generator] = None
        self.val_generator: Optional[torch.Generator] = None

    def _parse_config(self):
        """Parse and normalize configuration schema."""
        cfg = self.raw_config
        self.exp_meta = cfg.get("experiment", {})
        self.name = self.exp_meta.get("name", os.path.splitext(os.path.basename(self.config_path))[0])
        self.family = self.exp_meta.get("family", "architecture_experiments")

        self.model_cfg = cfg.get("model", {})
        if not self.model_cfg:
            raise ValueError(f"Missing 'model' configuration block in {self.config_path}")

        self.training_cfg = cfg.get("training", {})
        self.data_cfg = cfg.get("data", {})
        self.paths_cfg = cfg.get("paths", {})

        # Default paths if not explicitly configured
        self.checkpoint_dir = self.paths_cfg.get(
            "checkpoint_dir",
            os.path.join("checkpoints", "gpt4", self.family, self.name)
        )
        self.results_dir = self.paths_cfg.get(
            "results_dir",
            os.path.join("experiments", "gpt4", self.family, "results")
        )
        self.checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.name}_baseline.pt")

    def build_model(self, seed: Optional[int] = None) -> GPT:
        """
        Instantiate the GPT model from configuration onto the target device.
        Guarantees reproducible initialization by explicitly seeding PyTorch RNG
        immediately before model instantiation, isolating it from any prior RNG calls.
        """
        init_seed = seed if seed is not None else self.seed
        torch.manual_seed(init_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(init_seed)

        self.model = GPT(**self.model_cfg).to(self.device)
        self.checkpoint_loaded = False
        self.loaded_checkpoint_path = None
        return self.model

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable, non-trainable, and total parameters."""
        if self.model is None:
            self.build_model()
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        return {
            "trainable": trainable,
            "non_trainable": non_trainable,
            "total": trainable + non_trainable
        }

    def get_model_dimensions(self) -> Dict[str, Any]:
        """Return model dimensional hyperparameters."""
        return {
            "vocab_size": self.model_cfg.get("vocab_size"),
            "embedding_dim": self.model_cfg.get("embedding_dim"),
            "max_context_length": self.model_cfg.get("max_context_length"),
            "num_heads": self.model_cfg.get("num_heads"),
            "feed_forward_dim": self.model_cfg.get("feed_forward_dim"),
            "num_layers": self.model_cfg.get("num_layers"),
        }

    def estimate_memory(self, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Estimate memory requirements (parameters, gradients, optimizer states, activations).
        Also reports actual hardware allocated memory where supported.
        """
        params = self.count_parameters()["trainable"]
        bs = batch_size if batch_size is not None else self.training_cfg.get("batch_size", 16)
        ctx = self.model_cfg.get("max_context_length", 64)
        emb = self.model_cfg.get("embedding_dim", 32)
        layers = self.model_cfg.get("num_layers", 2)

        # 4 bytes per float32
        param_bytes = params * 4
        grad_bytes = params * 4
        # AdamW maintains two fp32 states (momentum and variance) per parameter
        optimizer_bytes = 2 * params * 4
        # Rough activation estimate per layer: batch * context * embedding * intermediate states
        activation_bytes = bs * ctx * emb * layers * 4 * 4

        total_static_bytes = param_bytes + grad_bytes + optimizer_bytes
        total_training_bytes = total_static_bytes + activation_bytes

        hardware_allocated = 0
        if torch.cuda.is_available() and self.device.type == "cuda":
            hardware_allocated = torch.cuda.memory_allocated(self.device)

        param_mb = round(param_bytes / (1024 * 1024), 4)
        grad_mb = round(grad_bytes / (1024 * 1024), 4)
        opt_mb = round(optimizer_bytes / (1024 * 1024), 4)
        act_mb = round(activation_bytes / (1024 * 1024), 4)
        tot_mb = round(total_training_bytes / (1024 * 1024), 4)
        hw_mb = round(hardware_allocated / (1024 * 1024), 4)

        return {
            "estimated_param_memory_mb": param_mb,
            "estimated_grad_memory_mb": grad_mb,
            "estimated_optimizer_memory_mb": opt_mb,
            "estimated_activation_memory_mb": act_mb,
            "estimated_total_training_mb": tot_mb,
            "measured_hardware_allocated_mb": hw_mb,
            # Backward-compatible aliases
            "param_memory_mb": param_mb,
            "grad_memory_mb": grad_mb,
            "optimizer_memory_mb": opt_mb,
            "activation_memory_mb": act_mb,
            "hardware_allocated_mb": hw_mb
        }

    def load_tokenizer_and_dataset(self) -> Tuple[Any, LanguageDataset]:
        """Load tokenizer and dataset according to experiment configuration."""
        tok_type = self.data_cfg.get("tokenizer_type", "subword")
        tok_path = self.data_cfg.get("tokenizer_path", "data/processed/gpt3_tokenizer.json")
        data_dir = self.data_cfg.get("data_dir", "data/processed")

        self.tokenizer = get_tokenizer(tok_type)
        self.tokenizer.load(tok_path)

        self.dataset = LanguageDataset(
            self.tokenizer,
            data_dir=data_dir,
            context_length=self.model_cfg.get("max_context_length", 64)
        )
        return self.tokenizer, self.dataset

    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Explicitly load a trained checkpoint into self.model.
        Validates checkpoint file existence, structure, and architecture compatibility.
        """
        path = checkpoint_path if checkpoint_path is not None else self.checkpoint_path
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint file not found: '{path}'. Cannot evaluate missing checkpoint.")

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            raise ValueError(f"Invalid checkpoint format at '{path}': missing 'model_state_dict'")

        # Validate architectural compatibility if config metadata present
        if "config" in ckpt:
            ckpt_cfg = ckpt["config"]
            for key in ["vocab_size", "embedding_dim", "max_context_length", "num_heads", "feed_forward_dim", "num_layers"]:
                if key in ckpt_cfg and key in self.model_cfg:
                    if ckpt_cfg[key] != self.model_cfg[key]:
                        raise ValueError(
                            f"Checkpoint architecture mismatch for '{key}': "
                            f"checkpoint has {ckpt_cfg[key]}, but current model config specifies {self.model_cfg[key]}"
                        )

        if self.model is None:
            self.build_model()

        # Strict weight loading
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.eval()
        self.checkpoint_loaded = True
        self.loaded_checkpoint_path = path
        return ckpt

    def inspect(self) -> Dict[str, Any]:
        """
        Read-only preflight inspection of the experiment configuration and model.
        Does NOT execute any training.
        """
        if self.model is None:
            self.build_model()

        param_info = self.count_parameters()
        dims = self.get_model_dimensions()
        mem = self.estimate_memory()

        checkpoint_exists = os.path.isfile(self.checkpoint_path)
        checkpoint_info = None
        if checkpoint_exists:
            try:
                ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
                checkpoint_info = {
                    "step": ckpt.get("step"),
                    "keys_count": len(ckpt.get("model_state_dict", {})),
                    "file_size_bytes": os.path.getsize(self.checkpoint_path)
                }
            except Exception as e:
                checkpoint_info = {"error": str(e)}

        return {
            "name": self.name,
            "family": self.family,
            "config_path": self.config_path,
            "device": str(self.device),
            "seed": self.seed,
            "val_seed": self.val_seed,
            "parameters": param_info,
            "dimensions": dims,
            "memory": mem,
            "paths": {
                "checkpoint_path": self.checkpoint_path,
                "checkpoint_dir": self.checkpoint_dir,
                "results_dir": self.results_dir
            },
            "checkpoint_exists": checkpoint_exists,
            "checkpoint_info": checkpoint_info,
            "checkpoint_loaded": self.checkpoint_loaded,
            "loaded_checkpoint_path": self.loaded_checkpoint_path
        }

    def evaluate(
        self,
        batch_count: int = 50,
        split: str = "val",
        seed: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        require_checkpoint: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate model loss and perplexity on dataset split without modifying weights.
        Uses an independent seeded torch.Generator to ensure validation sampling is
        fully isolated and does not alter any training generator state.

        Args:
            batch_count: Number of batches to evaluate.
            split: Dataset split ('val' or 'train').
            seed: Seed override for evaluation batch sampling generator.
            checkpoint_path: Optional explicit path to checkpoint file.
            require_checkpoint: If True, requires that a checkpoint is loaded or present.
        """
        # Handle checkpoint loading
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        elif require_checkpoint:
            if not self.checkpoint_loaded:
                if not os.path.isfile(self.checkpoint_path):
                    raise FileNotFoundError(
                        f"Checkpoint required for evaluation but not found at '{self.checkpoint_path}'. "
                        "To explicitly evaluate an untrained model, pass require_checkpoint=False."
                    )
                self.load_checkpoint(self.checkpoint_path)

        if self.model is None:
            self.build_model()
        if self.dataset is None:
            self.load_tokenizer_and_dataset()

        # Dedicated isolated evaluation generator
        eval_seed = seed if seed is not None else self.val_seed
        eval_generator = torch.Generator().manual_seed(eval_seed)

        self.model.eval()
        batch_size = self.training_cfg.get("batch_size", 16)
        ctx_len = self.model_cfg.get("max_context_length", 64)

        total_loss = 0.0
        losses = []

        with torch.no_grad():
            for _ in range(batch_count):
                x, y = self.dataset.get_batch(split=split, batch_size=batch_size, generator=eval_generator)
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                val_loss = loss.item()
                losses.append(val_loss)
                total_loss += val_loss

        avg_loss = total_loss / batch_count if batch_count > 0 else float("inf")
        perplexity = math.exp(avg_loss) if math.isfinite(avg_loss) else float("inf")

        return {
            "experiment": self.name,
            "split": split,
            "batch_count": batch_count,
            "batch_size": batch_size,
            "tokens_evaluated": batch_count * batch_size * ctx_len,
            "loss": round(avg_loss, 4),
            "perplexity": round(perplexity, 4),
            "eval_seed": eval_seed,
            "checkpoint_evaluated": self.loaded_checkpoint_path if self.checkpoint_loaded else "untrained_weights"
        }

    def train(self, steps: Optional[int] = None, overwrite: bool = False, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute training loop only when explicitly invoked.
        Uses separate independent seeded generators for training and validation sampling:
          - self.train_generator: only advanced by training batch draws.
          - self.val_generator: only advanced by validation interval batch draws.
        """
        if self.model is None:
            self.build_model()
        if self.dataset is None:
            self.load_tokenizer_and_dataset()

        if os.path.exists(self.checkpoint_path) and not overwrite and not resume_from:
            raise FileExistsError(
                f"Checkpoint {self.checkpoint_path} already exists. Set overwrite=True or resume_from to proceed."
            )

        # Independent Generators for training and validation
        self.train_generator = torch.Generator().manual_seed(self.seed)
        self.val_generator = torch.Generator().manual_seed(self.val_seed)

        learning_rate = float(self.training_cfg.get("learning_rate", 3e-4))
        train_steps = steps if steps is not None else self.training_cfg.get("train_steps", 5000)
        val_interval = self.training_cfg.get("val_interval", 100)
        batch_size = self.training_cfg.get("batch_size", 16)

        optimizer = create_optimizer(self.model, learning_rate=learning_rate)

        start_step = 0
        if resume_from and os.path.exists(resume_from):
            ckpt = torch.load(resume_from, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_step = ckpt.get("step", 0)

        start_time = time.time()
        min_val_loss = float("inf")
        history = []

        for step in range(start_step + 1, train_steps + 1):
            x, y = self.dataset.get_batch(split="train", batch_size=batch_size, generator=self.train_generator)
            x, y = x.to(self.device), y.to(self.device)
            loss = train_step(self.model, optimizer, x, y)

            if step % val_interval == 0 or step == train_steps:
                val_x, val_y = self.dataset.get_batch(split="val", batch_size=batch_size, generator=self.val_generator)
                val_x, val_y = val_x.to(self.device), val_y.to(self.device)
                val_loss = validation_step(self.model, val_x, val_y)
                val_val = val_loss.item()
                if val_val < min_val_loss:
                    min_val_loss = val_val

                history.append({
                    "step": step,
                    "train_loss": round(loss.item(), 4),
                    "val_loss": round(val_val, 4)
                })

        elapsed = time.time() - start_time

        # Save checkpoint
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": train_steps,
            "config": {
                "tokenizer_type": self.data_cfg.get("tokenizer_type", "subword"),
                **self.model_cfg
            }
        }, self.checkpoint_path)

        self.checkpoint_loaded = True
        self.loaded_checkpoint_path = self.checkpoint_path

        results = {
            "name": self.name,
            "train_steps": train_steps,
            "elapsed_seconds": round(elapsed, 2),
            "final_train_loss": history[-1]["train_loss"] if history else None,
            "final_val_loss": history[-1]["val_loss"] if history else None,
            "min_val_loss": round(min_val_loss, 4),
            "checkpoint_path": self.checkpoint_path
        }
        self.save_results(results)
        return results

    def save_results(self, results_data: Dict[str, Any], filename: Optional[str] = None):
        """Save results in structured JSON format in the experiment results directory."""
        os.makedirs(self.results_dir, exist_ok=True)
        fn = filename if filename is not None else f"{self.name}_results.json"
        out_path = os.path.join(self.results_dir, fn)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)
        return out_path
