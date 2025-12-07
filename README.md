# Notes2Tone

OMR benchmarking framework for evaluating music score transcription models on the PRAIG/SMB dataset.

## Features

- Benchmark OMR models with OMR-NED metrics
- Custom MusicXML → **kern converter
- GPU acceleration (CUDA 11.x support)
- Interactive dataset browser

## Requirements

- Python 3.9+
- NVIDIA GPU with CUDA 11.8 (optional, for GPU acceleration)
- HuggingFace account for dataset access

## Quick Start

### 1. Installation

```bash
# Install uv if needed
pip install uv

# For GPU (CUDA 11.x - GTX 1060/1070/1080, RTX 2000/3000)
uv sync --extra gpu --extra models

# For CUDA 12.x (RTX 4000+)
uv sync --extra models
uv pip install onnxruntime-gpu>=1.19.0

# CPU only
uv sync --extra models
```

### 2. Setup HuggingFace

Create `.env` file:
```
HF_TOKEN=your_token_here
```

Get token from https://huggingface.co/settings/tokens  
Request access: https://huggingface.co/datasets/PRAIG/SMB

### 3. Run Benchmark

```bash
python -m benchmarks.benchmark --models oemer --limit 10
```

### 4. Browse Dataset

```bash
python dataset_viewer.py
```

## Project Structure

```
benchmarks/
├── datasets/      # Dataset loaders (SMB)
├── models/        # OMR model wrappers (OeMeR)
├── converters/    # MusicXML → **kern
├── eval/          # OMR-NED metrics
└── benchmark.py   # Main CLI
```

## Adding Models

Create class in `benchmarks/models/`:

```python
from benchmarks.models.base_model import BaseOMRModel

class MyModel(BaseOMRModel):
    def _predict_impl(self, image_path: str, debug_dir: Path) -> str:
        # Return **kern notation
        return kern_output
```

Register in `benchmarks/benchmark.py`.

## Troubleshooting

**GPU not working:**
```bash
nvidia-smi             # Verify GPU
```

**HF authentication error:** Check `.env` file and dataset access request

**First run slow:** Model compilation takes 2-5 min initially

## Metrics

**OMR-NED:** Normalized Edit Distance (lower = better, 0.0 = perfect)

## Acknowledgments

- PRAIG/SMB dataset: https://huggingface.co/datasets/PRAIG/SMB
- OeMeR: https://github.com/BreezeWhite/oemer
- music21: http://web.mit.edu/music21/