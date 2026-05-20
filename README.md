# Cross-Task Federated Backbone Aggregation with Selective State Space Models for Building Energy Analytics

Code and experiment artefacts for the ACM BALANCES 2026 submission.

A single **MambaMixer** (selective SSM) is trained to serve two building-analytics
tasks at once — **hourly load forecasting** (ASHRAE) and **energy anomaly
detection** (LEAD 1.0) — across many buildings without moving raw meter data
off-site. The model splits into a shared **multi-scale BiMamba backbone** and
two small task-specific heads. Each federated round, the backbone is averaged
across *all* clients regardless of task; each head is averaged only within
its task group.

---

## 1. Repository layout

```
cross-task-fl/
├── main.py                       # top-level entry (FL + centralized + baselines + plots)
├── preprocess.py                 # build 50/50 building splits with 70/20/10 temporal slices
├── configs/
│   └── config.py                 # ExperimentConfig dataclass — every knob lives here
├── models/
│   ├── mamba_mixer.py            # MambaMixer: multi-scale patching, BiMamba SSM, FiLM gate, heads
│   └── baselines/                # LSTM, LSTM-AE, ANN-AE, Informer, MSD-Mixer
├── trainers/
│   ├── multitask_fed_trainer.py  # cross-task FL (FedAvg / FedProx) + dual/single_task/local_only
│   └── centralized_trainer.py    # pooled-data upper bound + per-baseline training
├── data_provider/
│   ├── ashrae_dataset.py         # get_ashrae_fl_data, get_ashrae_centralized_loaders
│   └── lead_dataset.py           # get_lead_fl_data, get_lead_centralized_loaders
├── experiments/
│   └── run_baselines.py          # centralized baselines on identical splits
├── utils/                        # metrics + logging helpers
├── visualization/                # research-paper plots from saved JSON results
├── data/                         # processed CSVs + split_metadata.json  ← download or generate
│   ├── ashrae/processed/         #   ashrae_clean.csv, split_metadata.json
│   └── lead/processed/           #   lead_clean.csv,   split_metadata.json
├── checkpoints/                  # trained .pt model weights              ← download (optional)
├── raw/                          # raw ASHRAE / LEAD CSVs                  ← you provide (if preprocessing)
├── results/                      # (created at run time) JSON metrics, CSVs, figures
└── logs/                         # (created at run time) text + CSV training logs
```

The empty `data/` and `checkpoints/` directories in the repo are placeholders
for downloaded artefacts (see §3).

---

## 2. Environment

- Python **3.10+**
- CUDA-capable GPU recommended (any single GPU is sufficient; multi-GPU is
  optional, see `max_federated_gpus`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm
# Optional: install the official Mamba kernels for the selective scan (speed-up only)
# pip install mamba-ssm causal-conv1d
```

The model falls back to a pure-PyTorch BiMamba implementation when
`mamba-ssm` is not available — kernel choice affects speed, not numerical
results.

---

## 3. Getting the data and checkpoints

You have **three options**, depending on what you want to do:

### Option A — Download our processed data + checkpoints (fastest)

Use this if you want to reproduce numbers or evaluate from saved weights
without rebuilding the splits from raw CSVs.

> **Google Drive folders** (mirror the directory names in this repo —
> download each folder and drop its contents into the matching folder
> at the repo root):
>
> - **`data/`** — processed CSVs + split metadata: <https://drive.google.com/drive/folders/1LF8J_PGBY_kDS_8bhGCvhYv3dfNm7CAs?usp=sharing>
> - **`checkpoints/`** — all trained `.pt` weights: <https://drive.google.com/drive/folders/1-n8dFPlXdeFq0R_m31qjI_qtafUhxAHy?usp=sharing>
> - **`data_provider/`** — dataset wrappers (already present in the repo; mirror link in case the local copy is missing): <https://drive.google.com/drive/folders/1evbm-pJVtrYKOth3CZXRskOnwrHzCbx7?usp=sharing>

After downloading, the layout should be:

```
cross-task-fl/
├── data/
│   ├── ashrae/processed/
│   │   ├── ashrae_clean.csv
│   │   └── split_metadata.json
│   └── lead/processed/
│       ├── lead_clean.csv
│       └── split_metadata.json
└── checkpoints/
    ├── centralized_forecasting_model.pt
    ├── centralized_anomaly_model.pt
    ├── fed_dual_fedavg_forecasting_model.pt
    ├── fed_dual_fedavg_anomaly_model.pt
    ├── fed_dual_fedprox_forecasting_model.pt
    ├── fed_dual_fedprox_anomaly_model.pt
    ├── fed_single_task_fedavg_forecasting_model.pt
    ├── fed_single_task_fedavg_anomaly_model.pt
    ├── fed_local_only_fedavg_forecasting_model.pt
    └── fed_local_only_fedavg_anomaly_model.pt
```

With these in place you can **skip `python preprocess.py`** and go straight
to §4.

### Option B — Use only the processed data (re-train models yourself)

Download just the **`data/`** folder linked above, drop it in place as
shown, and run any training command from §4. Training will overwrite/extend
`checkpoints/`.

### Option C — Rebuild everything from raw CSVs

Download the raw datasets yourself:

| Dataset | Task | Source | Notes |
|---|---|---|---|
| **ASHRAE Great Energy Predictor III** | hourly load forecasting | Kaggle: `ashrae-energy-prediction` (`train.csv`) | electricity meters only (`meter == 0`) |
| **LEAD 1.0** | energy anomaly detection | Kaggle: `lead1-0-an-large-scale-energy-anomaly-detection` (`train.csv`) | expert-labelled anomalies on ASHRAE electricity series |

Point the preprocessor at them with environment variables, then run it:

```bash
export ASHRAE_RAW_CSV=/path/to/ashrae_train.csv
export LEAD_RAW_CSV=/path/to/lead_train.csv
python preprocess.py
```

Or place the files at the default locations `raw/train_ashrae.csv` and
`raw/train_lead.csv`. The preprocessor will write the same `data/.../processed/`
files described in Option A.

Expected raw schema (enforced by [preprocess.py](preprocess.py)):
- ASHRAE: `building_id, timestamp, meter, meter_reading`
- LEAD:   `building_id, timestamp, meter_reading, anomaly`

The whole pipeline is deterministic — building selection, train/test split,
and torch RNG are all seeded by `seed = 42` in
[configs/config.py](configs/config.py), so Option C reproduces Option A
byte-for-byte on the same Python/NumPy versions.

---

## 4. Reproducing the headline results

Assuming `data/` is populated (Option A or B finished, or `python preprocess.py`
has run).

```bash
# Proposed: cross-task FL with FedAvg (dual aggregation across all 70 clients)
#   Also runs the centralised upper bound + visualisation in one go.
python main.py

# Cross-task FL with FedProx (mu = 0.01, set in config)
python main.py --federated --strategy fedprox

# Ablations
python main.py --federated --mode single_task   # average within each task only
python main.py --federated --mode local_only    # no parameter exchange

# Centralised upper bound only (pooled data)
python main.py --centralized

# SOTA baselines on identical splits: LSTM, Informer, MSD-Mixer, LSTM-AE, ANN-AE
python main.py --baselines
#   or directly, with finer control:
python -m experiments.run_baselines
python -m experiments.run_baselines --task forecasting --model informer

# Regenerate figures from whatever JSON files are already in results/
python main.py --visualize
```

### One-shot full reproduction

```bash
# Skip the first line if you downloaded data/ from the Drive bundle.
python preprocess.py

python main.py                                    # proposed + centralized + plots
python main.py --federated --strategy fedprox     # FedProx variant
python main.py --federated --mode single_task     # ablation
python main.py --federated --mode local_only      # ablation
python main.py --baselines                        # SOTA baselines
python main.py --visualize                        # consolidate all plots
```

End-to-end on a single modern GPU (RTX 3090 / A100) is on the order of
hours, dominated by the centralised baselines.

### Evaluating from the downloaded checkpoints (no training)

`main.py` does not yet expose a dedicated `--evaluate-only` flag, but the
saved `.pt` files in `checkpoints/` are plain `state_dict`s for the
MambaMixer architecture built by `build_forecasting_model` / `build_anomaly_model`
in [main.py](main.py). A minimal evaluation snippet looks like:

```python
import torch
from configs.config import ExperimentConfig
from main import build_forecasting_model, build_anomaly_model
from data_provider.ashrae_dataset import get_ashrae_centralized_loaders
from data_provider.lead_dataset  import get_lead_centralized_loaders
from utils.metrics import compute_forecasting_metrics, compute_anomaly_metrics, find_threshold_on_validation

cfg = ExperimentConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fc = build_forecasting_model(cfg).to(device).eval()
fc.load_state_dict(torch.load("checkpoints/fed_dual_fedavg_forecasting_model.pt", map_location=device))
# … then iterate the test loader from get_ashrae_centralized_loaders() and call compute_forecasting_metrics
```

The anomaly model uses `build_anomaly_model` and
`get_lead_centralized_loaders`; pick the threshold on the validation loader
with `find_threshold_on_validation` before computing test metrics, exactly
as [main.py:200-209](main.py#L200-L209) does.

---

## 5. Key configuration knobs

All knobs live in [configs/config.py](configs/config.py); the CLI flags
in `main.py` only override `aggregation_mode` and `fl_strategy`. To change
any other value, edit the dataclass.

| Knob | Default | Meaning |
|---|---|---|
| `seed` | 42 | controls building selection, splits, and torch RNG |
| `seq_len` / `pred_len` | 128 / 24 | input window and forecasting horizon (hours) |
| `patch_sizes` | (24,12,6,2,1) | multi-scale patch sizes consumed by the BiMamba backbone |
| `hid_chn`, `hid_pch`, `hid_pred`, `d_ssm`, `state_size`, `expand`, `conv_kernel` | see file | MambaMixer dims |
| `num_rounds` / `local_epochs` | 10 / 5 | FL schedule |
| `client_lr`, `weight_decay`, `grad_clip` | 1e-3, 0.01, 1.0 | client AdamW + clipping |
| `participation_rate` | 1.0 | 100% participation (all 70 clients per round) |
| `aggregation_mode` | `dual` | `dual` (proposed), `single_task`, `local_only` |
| `fl_strategy` | `fedavg` | `fedavg`, `fedprox` (`fedprox_mu = 0.01`) |
| `mask_rate` | 0.25 | reconstruction mask rate for the anomaly head |
| `clean_only` | True | train autoencoders on clean windows only |
| `lambda_mse`, `lambda_acf`, `acf_cutoff` | 0.1, 0.3, 2 | residual auxiliary loss weights |
| `centralized_max_epochs` / `centralized_early_stop` | 100 / 15 | centralised training schedule |
| `batch_size` / `num_workers` | 32 / 0 | DataLoader |
| `max_federated_gpus` | 1 | bump to 2 to train the two task groups in parallel |

---

## 6. Outputs

After a full run:

```
results/
├── federated_dual_fedavg_results.json        # proposed
├── federated_dual_fedprox_results.json       # FedProx
├── federated_single_task_fedavg_results.json # ablation
├── federated_local_only_fedavg_results.json  # ablation
├── centralized_results.json                  # pooled-data upper bound
├── baseline_results.json                     # LSTM / Informer / MSD-Mixer / LSTM-AE / ANN-AE
├── comparison.json                           # fed vs centralised side-by-side
└── figures/                                  # paper plots
checkpoints/   # .pt state_dicts for every trained model
logs/          # text logs + per-round/per-epoch CSV
```

Each `*_results.json` contains the full config snapshot, model parameter
counts (total / shared backbone / per-head), the round-by-round history,
and the final metrics:

- **Forecasting**: `mse`, `rmse`, `mae`, `mape`, `r2`
- **Anomaly**: `f1`, `auc_roc`, `auc_pr`, `precision`, `recall`, plus the
  validation-selected threshold

All numbers reported in the paper come from these JSONs.

---

## 7. Troubleshooting

- **`FileNotFoundError: .../split_metadata.json`** — `data/` is empty.
  Either drop the downloaded `data/` bundle in place (Option A/B) or run
  `python preprocess.py` after setting the raw-CSV env vars (Option C).
- **`ModuleNotFoundError: mamba_ssm`** — safe to ignore; the pure-PyTorch
  BiMamba path is used automatically.
- **CUDA OOM** — lower `batch_size` in [configs/config.py](configs/config.py),
  or set `max_federated_gpus = 1` to disable threaded multi-GPU training.
- **Checkpoint shape mismatch when loading** — make sure the
  `ExperimentConfig` values for `seq_len`, `pred_len`, `patch_sizes`, and
  the MambaMixer hidden-dim fields are unchanged from defaults; the
  released checkpoints were trained with the defaults in
  [configs/config.py](configs/config.py).

---

## 8. License & citation

Released under the repository's [LICENSE](LICENSE). If you use this code,
please cite the BALANCES 2026 submission.
