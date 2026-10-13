# Tiny GPT

This project builds a small decoder-only GPT-like language model from scratch and progressively develops five model versions.

## Project Structure & Philosophy

This project strictly separates source code, environments, and generated artifacts:
- **GitHub:** Contains source code, configurations, tests, and documentation.
- **Hugging Face Hub:** Will contain canonical trained model artifacts (checkpoints and tokenizers).
- **Generated Datasets:** Built reproducibly from scripts; not committed to the repository.
- **Colab:** An optional training environment, not required for inference.
- **Docker:** A reproducible execution environment, decoupled from the source code.

---

## 🟢 Status Accuracy

**IMPLEMENTED NOW:**
- repository
- training infrastructure
- verification
- tests
- Docker
- CPU/CUDA support
- artifact cleanup
- checkpoint overwrite protection

**PLANNED:**
- canonical CPU retraining
- publishing canonical checkpoints
- actual Hugging Face download workflow
- fresh-machine artifact verification using canonical models

---

## 🛠️ Executable Workflow (Currently Implemented)

### A. Clone Repository
```bash
git clone https://github.com/nandish029/tiny-gpt.git
cd tiny-gpt
```

### B. Build Docker Environment
```bash
docker build -f docker/Dockerfile -t tiny-gpt:latest .
```

### C. Prepare Dataset
```bash
python src/data/prepare.py --num_samples 100000 --seed 42 --data_dir data/processed
```

### D. Build Tokenizer (GPT-3 Subword Example)
```bash
python scripts/train_subword_tokenizer.py --data data/processed/train.jsonl --vocab_size 512 --out data/processed/gpt3_tokenizer.json
```

---

## 🚧 Canonical Workflow (PLANNED)

> [!WARNING]
> **Notice:** The project is currently in a pre-training hardening phase. The commands below represent the **PLANNED** canonical workflow for downloading canonical artifacts and training GPT-1/2/3. The actual canonical CPU artifacts have not yet been published to Hugging Face.

### E. Train GPT-1 (CPU) [PLANNED]
```bash
python scripts/train_gpt1.py --config configs/gpt1.yaml --tokenizer data/processed/tokenizer.json --device cpu
```

### F. Train GPT-2 (CPU) [PLANNED]
```bash
python scripts/train_gpt1.py --config configs/gpt2.yaml --tokenizer data/processed/tokenizer.json --device cpu
```

### G. Train GPT-3 (CPU) [PLANNED]
```bash
python scripts/train_gpt1.py --config configs/gpt3.yaml --tokenizer data/processed/gpt3_tokenizer.json --tokenizer_type subword --device cpu
```

### H. Resume Training [PLANNED]
To resume training, provide the `--resume` flag and the path to the checkpoint:
```bash
python scripts/train_gpt1.py --config configs/gpt1.yaml --tokenizer data/processed/tokenizer.json --resume checkpoints/gpt1/gpt1_baseline.pt --device cpu
```

### I. Evaluate Model [PLANNED]
```bash
python scripts/evaluate_gpt1.py --checkpoint checkpoints/gpt1/gpt1_baseline.pt --data_dir data/processed
```

### J. Verify Model Integrity [PLANNED]
```bash
python scripts/verify_model.py --model gpt1 --device cpu
```

### K. Download Model from Hugging Face [PLANNED]
*Artifact download interface prepared; canonical artifacts are not yet published.*
```bash
python scripts/download_model.py --model gpt1
```

### L. Run Native CPU Inference [PLANNED]
```bash
python scripts/chat.py --checkpoint checkpoints/gpt1/gpt1_baseline.pt --tokenizer data/processed/tokenizer.json --config configs/gpt1.yaml --device cpu
```

### M. Run Native CUDA Inference [PLANNED]
```bash
python scripts/chat.py --checkpoint checkpoints/gpt1/gpt1_baseline.pt --tokenizer data/processed/tokenizer.json --config configs/gpt1.yaml --device cuda
```

### N. Run Docker CPU Inference [PLANNED]
```bash
docker run -it -v "$(pwd):/app" --rm tiny-gpt:latest python scripts/chat.py --checkpoint checkpoints/gpt1/gpt1_baseline.pt --tokenizer data/processed/tokenizer.json --config configs/gpt1.yaml --device cpu
```

### O. Run Docker CUDA Inference [PLANNED]
```bash
docker run --gpus all -it -v "$(pwd):/app" --rm tiny-gpt:latest python scripts/chat.py --checkpoint checkpoints/gpt1/gpt1_baseline.pt --tokenizer data/processed/tokenizer.json --config configs/gpt1.yaml --device cuda
```

---

## Testing & Validation

The codebase includes an extensive test suite verifying mathematical correctness, architectural constraints, checkpoint integrity, and execution safety. Checkpoints are strictly protected from accidental overwrite unless the `--overwrite` flag is passed.

To run the test suite:
```bash
python -m unittest discover tests
```
