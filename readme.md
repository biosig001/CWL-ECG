# CWL-ECG

PyTorch implementation of **CWL-ECG: Cycle-Informed Waveform Learning for ECG Delineation with Improved P-Wave Detection**.

## Installation

Use Python 3.10 and an NVIDIA GPU on Linux or WSL. Install CUDA-enabled PyTorch 2.10 before installing the requirements.

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install setuptools wheel packaging
pip install --no-build-isolation -r requirements.txt
```

## Data

Download [LUDB v1.0.1](https://physionet.org/content/ludb/1.0.1/) and keep `RECORDS` and the `data/` folder together under the LUDB root.

## Run

```bash
python main.py --data-dir /path/to/ludb
```

The script runs record-level five-fold cross-validation and reports sample classification and boundary delineation metrics. Results are saved to `results/metrics.json`. Set `--output-dir` to change the output location.
