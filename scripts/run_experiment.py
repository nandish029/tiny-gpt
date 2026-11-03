#!/usr/bin/env python3
import os
import sys
import argparse
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.experiments.runner import ExperimentRunner

def main():
    parser = argparse.ArgumentParser(
        description="Tiny GPT Reusable Experiment Runner (E1 Depth and Architecture Experiments)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration YAML (e.g., experiments/gpt4/e1_depth/configs/e1_a.yaml)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run on (default: auto)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override"
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        default=True,
        help="Inspect configuration, parameter counts, model dimensions, and memory without training (default mode)"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation on validation dataset split using a checkpoint or explicitly allowed untrained model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to specific checkpoint file to load for evaluation (default: experiment's canonical checkpoint_path)"
    )
    parser.add_argument(
        "--allow-untrained",
        action="store_true",
        help="Explicitly permit evaluating untrained randomly-initialized weights rather than requiring a checkpoint"
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=50,
        help="Number of batches to evaluate (default: 50)"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Explicit flag required to initiate model training"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override training steps"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing checkpoint when training"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON summary"
    )

    args = parser.parse_args()

    runner = ExperimentRunner(
        config_path=args.config,
        device=args.device,
        seed=args.seed
    )

    # 1. Training mode - only if explicitly invoked
    if args.train:
        print(f"=== TRAINING EXPERIMENT: {runner.name} ===")
        print(f"Configuration: {args.config}")
        print(f"Device: {runner.device} | Seed: {runner.seed}")
        results = runner.train(steps=args.steps, overwrite=args.overwrite)
        print(json.dumps(results, indent=2))
        return

    # 2. Evaluation mode
    if args.evaluate:
        print(f"=== EVALUATING EXPERIMENT: {runner.name} ===")
        print(f"Configuration: {args.config}")
        require_ckpt = not args.allow_untrained
        eval_results = runner.evaluate(
            batch_count=args.eval_batches,
            split="val",
            checkpoint_path=args.checkpoint,
            require_checkpoint=require_ckpt
        )
        print(json.dumps(eval_results, indent=2))
        return

    # 3. Default mode: Preflight Inspection (read-only, no training)
    inspection = runner.inspect()
    if args.json:
        print(json.dumps(inspection, indent=2))
    else:
        print(f"============================================================")
        print(f" EXPERIMENT PREFLIGHT INSPECTION: {inspection['name'].upper()}")
        print(f"============================================================")
        print(f"Family:           {inspection['family']}")
        print(f"Config Path:      {inspection['config_path']}")
        print(f"Device:           {inspection['device']}")
        print(f"Random Seed:      {inspection['seed']}")
        print(f"Validation Seed:  {inspection['val_seed']}")
        print(f"------------------------------------------------------------")
        print(f"MODEL DIMENSIONS:")
        for k, v in inspection['dimensions'].items():
            print(f"  {k:<20}: {v}")
        print(f"------------------------------------------------------------")
        print(f"PARAMETER COUNTS:")
        print(f"  Trainable Parameters: {inspection['parameters']['trainable']:,}")
        print(f"  Total Parameters:     {inspection['parameters']['total']:,}")
        print(f"------------------------------------------------------------")
        print(f"ESTIMATED MEMORY:")
        print(f"  Param Memory:         {inspection['memory']['estimated_param_memory_mb']} MB")
        print(f"  Grad Memory:          {inspection['memory']['estimated_grad_memory_mb']} MB")
        print(f"  Optimizer Memory:     {inspection['memory']['estimated_optimizer_memory_mb']} MB")
        print(f"  Activation Memory:    {inspection['memory']['estimated_activation_memory_mb']} MB")
        print(f"  Total Training Est:   {inspection['memory']['estimated_total_training_mb']} MB")
        print(f"------------------------------------------------------------")
        print(f"PATHS:")
        print(f"  Checkpoint File:      {inspection['paths']['checkpoint_path']}")
        print(f"  Results Directory:    {inspection['paths']['results_dir']}")
        print(f"  Checkpoint Exists:    {inspection['checkpoint_exists']}")
        print(f"============================================================")
        print(f"[STATUS] Model and configuration verified. (NO TRAINING PERFORMED)")

if __name__ == "__main__":
    main()
