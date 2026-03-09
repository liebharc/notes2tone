# Notes2Tone

OMR benchmarking framework for evaluating music score transcription models on the PRAIG/SMB dataset.


## Requirements

- Python 3.10+
- HuggingFace account for dataset access

## Installation

### 1. Clone notes2tone Repository

```bash
git clone https://github.com/yourusername/notes2tone.git
cd notes2tone
```

### 2. Create Conda Environment

```bash
conda create -n notes2tone python=3.12 -y
conda activate notes2tone
```

### 3. Install notes2tone

```bash
cd /notes2tone
pip install -e .
```

### 4. Install ONNX Runtime (resolves the cuda and cudnn installtion)

```bash
conda install onnxruntime=1.19.2
```

**Verify GPU:**
```python
import onnxruntime as ort
print(ort.get_available_providers())
# Should show: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

### 5. Install OEMER

**Option A: Simple**
```bash
pip install oemer   # already installed with notes2tone dependencys
```

**Option B:**
```bash
# Clone OEMER
cd ~/work  # or your preferred location
git clone https://github.com/BreezeWhite/oemer.git
cd oemer

# Install
pip install -e .
```

If using Option B, specify the path in the notebook:
```python
oemer = OemerModel(oemer_module_path="/path/to/oemer")
```

### 6. Install HOMR

```bash
# Clone repository
cd ~/work  # or your preferred location
git clone https://github.com/liebharc/homr.git
cd homr

# Install Poetry
pip install poetry
poetry config virtualenvs.create false

# Install dependencies
poetry install --only main,gpu  # For GPU support
# or: poetry install --only main  # For CPU only
# or: poetry install  # For development

# Test installation
poetry run homr <image_path>
```

### 7. Setup HuggingFace Authentication

Create `.env` file in notes2tone project root:
```
HF_TOKEN=your_token_here
```

Get token: https://huggingface.co/settings/tokens  
Request access: https://huggingface.co/datasets/PRAIG/SMB

### 7. Setup Jupyter Kernel

```bash
pip install ipykernel
python -m ipykernel install --user --name notes2tone --display-name "notes2tone"
```

### 8. Specify the path of oemer and homr in notebook
```bash
# check in terminal with activated environment

which oemer
which homr
```

## Quick Start

### Run Benchmark

```bash
# Benchmark all models
python -m benchmarks.benchmark --models all --limit 10

# Benchmark specific models
python -m benchmarks.benchmark --models oemer homr --limit 10
```

### Browse Dataset

```bash
python dataset_viewer.py
```

### Run Audiveris Pipeline

```bash
python -m benchmarks.audiveris_pipeline \
  --audiveris-path /path/to/Audiveris \
  --audiveris-upscale-factor 2.0 \
  --audiveris-upscale-max-side 3500
```

If Audiveris skips many pages because staff lines are too close, increase `--audiveris-upscale-factor` (e.g. `2.5`).

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