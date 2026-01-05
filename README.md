# Notes2Tone

OMR benchmarking framework for evaluating music score transcription models on the PRAIG/SMB dataset.


## Requirements

- Python 3.10+
- HuggingFace account for dataset access

## Quick Start

### 1. Installation

```bash
# Create Python 3.10 environment
python3.10 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
# for cuda12
pip install -e ".[gpu-cuda12]"

# for cuda11
pip install -e ".[gpu-cuda11]"

#for cpu
pip install -e ".[cpu]"
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
# Benchmark all models
python -m benchmarks.benchmark --models all --limit 10

# Benchmark specific models
python -m benchmarks.benchmark --models oemer homr --limit 10

# Benchmark with Audiveris
python -m benchmarks.benchmark --models audiveris --limit 5
```

### 4. Model Installation

**OeMeR:**
```bash
pip install oemer
```

**homr:**
```bash
# Clone and install with Poetry
git clone https://github.com/liebharc/homr
cd homr
poetry install --only main,gpu  # for GPU
poetry install --only main      # for CPU only
```

**Audiveris:**
- Windows: Download .msi from [releases](https://github.com/Audiveris/audiveris/releases)
- Linux: Download .deb or use Flatpak from [Flathub](https://flathub.org/apps/org.audiveris.audiveris)
- macOS: Download .dmg from releases
- Requires Java 11+ ([download here](https://adoptium.net/))

### 5. Browse Dataset

```bash
python dataset_viewer.py
```

## Project Structure

```
benchmarks/
├── datasets/      # Dataset loaders (SMB)
├── models/        # OMR model wrappers (OeMeR, homr, Audiveris)
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
nvidia-smi
nvtop
```

**HF authentication error:** Check `.env` file and dataset access request

**First run slow:** Model compilation takes 2-5 min initially

## Metrics

**OMR-NED:** Normalized Edit Distance (lower = better, 0.0 = perfect)

## Acknowledgments

- PRAIG/SMB dataset: https://huggingface.co/datasets/PRAIG/SMB
- OeMeR: https://github.com/BreezeWhite/oemer
- music21: http://web.mit.edu/music21/