# SWE-Bench Few-Shot Learning Study

This repository contains code for conducting few-shot learning experiments on the SWE-Bench dataset using CodeLlama models. The study investigates how different numbers of examples (0, 1, or 2) affect the model's performance in bug fixing tasks.

## Overview

The project consists of three main inference scripts:
- `inference_n_0.py`: Zero-shot learning (no examples)
- `inference_n_1.py`: One-shot learning (1 example)
- `inference_n_2.py`: Two-shot learning (2 examples)

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- CUDA-compatible GPU (recommended)

## Model Access

This project uses the CodeLlama-7b-Python model from Meta. To access the model:

1. Visit [CodeLlama-7b-Python-hf](https://huggingface.co/meta-llama/CodeLlama-7b-Python-hf)
2. Request access by accepting the Meta license agreement
3. Log in to Hugging Face with your account
4. Once approved, you can use the model with your Hugging Face token

Note: The model requires accepting Meta's license agreement and may take some time to get approved. Make sure to read and comply with the [Meta Llama 2 Community License Agreement](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd swe_bench_few_shots_study
```

2. Install dependencies using uv (recommended) or pip:

Using uv (faster):
```bash
# Install uv if you haven't already
pip install uv

# Create and activate a virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
uv pip install -r requirements.txt

# Optional: Install all dependencies including development tools
uv pip install -r requirements.txt --all-extras
```

Using pip:
```bash
# Install core dependencies
pip install -r requirements.txt

# Optional: Install all dependencies including development tools
pip install -r requirements.txt --all-extras
```

3. Set up your Hugging Face token:
```bash
huggingface-cli login
```

## Optional Dependencies

The project includes several optional dependencies that can be installed for different purposes:

### Performance Improvements
- `flash-attn`: Faster attention computation (Linux/Mac only)
- `bitsandbytes`: 8-bit quantization for reduced memory usage
- `xformers`: Memory efficient attention mechanisms
- `einops`: Enhanced tensor operations

### Development Tools
- `black`: Code formatting
- `isort`: Import sorting
- `flake8`: Code linting
- `pytest`: Testing framework
- `pytest-cov`: Test coverage reporting
- `jupyter` & `ipykernel`: Notebook development support

### Monitoring and Logging
- `wandb`: Experiment tracking and visualization
- `tensorboard`: Training visualization

## Usage

Each script can be run with various command-line arguments:

```bash
python inference_n_[0|1|2].py [options]
```

### Command Line Arguments

- `--batch_size`: Batch size for inference (default: 1)
- `--log_level`: Logging level (DEBUG, INFO, WARNING, ERROR) (default: INFO)
- `--dry_run`: Process only 4 examples for testing
- `--model`: Model ID from Hugging Face or local path (default: "meta-llama/CodeLlama-7b-Python-hf")
- `--use_torch_weights`: Use PyTorch weights instead of safetensors
- `--low_cpu_mem_usage`: Use low CPU memory usage when loading the model
- `--trust_remote_code`: Allow models that require custom code to be loaded
- `--golden_example_id`: Instance ID of the golden example to use (for n_1 and n_2)

### Example Commands

1. Run zero-shot inference:
```bash
python inference_n_0.py --batch_size 1 --model meta-llama/CodeLlama-7b-Python-hf
```

2. Run one-shot inference with a specific example:
```bash
python inference_n_1.py --golden_example_id "example_123" --batch_size 1
```

3. Run two-shot inference with default examples:
```bash
python inference_n_2.py --batch_size 1 --dry_run
```

## Output

Results are saved in the `swe_bench_results` directory with the following naming convention:
- `swe_bench_results_N_0.json`: Zero-shot results
- `swe_bench_results_N_1.json`: One-shot results
- `swe_bench_results_N_2.json`: Two-shot results

If running in dry-run mode, files will have a `_dry_run` suffix.

## Logging

Logs are saved in files named `inference_YYYYMMDD_HHMMSS.log` in the current directory, containing detailed information about the inference process.

## Notes

- The scripts use Flash Attention 2 if available, falling back to eager attention if not
  - Flash Attention is automatically installed on Linux/Mac systems for better performance
  - On Windows, the code will fall back to standard attention
- Models are loaded in bfloat16 precision by default to save memory
- Intermediate results are saved periodically to prevent data loss
- The code includes robust error handling for model loading with multiple fallback options
- Make sure you have accepted the Meta license agreement before using the model