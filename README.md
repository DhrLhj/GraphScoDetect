# GraphScoDetect

## Overview
GraphScoDetect is a two-stage deep learning framework for scoliosis screening from multi-channel wearable motion signals. The model first performs encoder pretraining to learn asymmetry-aware representations, and then performs joint training of the encoder and classifier. The implementation used here follows a graph-based formulation in which channel-wise signal segments are embedded as graph nodes, processed by graph message passing layers, temporally modeled by a bidirectional LSTM, and finally classified by an MLP head.

---

## Repository contents

The current release contains the following core files:

```text
GraphScoDetect/
├── README.md
├── requirements.txt
├── graph_scodetect_revised.py
└── data2/
    ├── train01zhee/
    └── test01zhee/
```

If your repository uses a different entry script name, replace `graph_scodetect_revised.py` with the actual file name.

---

## 1. System requirements

### Operating system
- Linux is recommended.
- The code is written in Python and can also run on other operating systems supported by PyTorch.

### Python
- Python 3.10 or later is recommended.

### Main software dependencies
- numpy
- pandas
- scipy
- scikit-learn
- torch

### Suggested `requirements.txt`

```text
numpy>=1.23
pandas>=1.5
scipy>=1.10
scikit-learn>=1.2
torch>=2.0
openpyxl>=3.1
```

`openpyxl` is recommended because the input data are Excel files (`.xls` / `.xlsx`).

### Non-standard hardware
- No non-standard hardware is required to run the code.
- A CUDA-enabled GPU is recommended for faster training, but the code automatically falls back to CPU if no GPU is available.

### Versions the software has been tested on
Please replace the following placeholders with your verified environment before submission:
- OS tested on: `[Ubuntu 22.04]`
- Python tested on: `[3.10.13]`
- PyTorch tested on: `[2.2.0+cu121]`

---

## 2. Installation guide

### Step 1. Create a Python environment

```bash
python -m venv graphscodetect_env
source graphscodetect_env/bin/activate
```

On Windows:

```bash
graphscodetect_env\Scripts\activate
```

### Step 2. Install dependencies

```bash
pip install -r requirements.txt
```

### Typical installation time
- On a normal desktop or laptop computer with a stable internet connection, installation typically takes **5-15 minutes**, depending mainly on the PyTorch installation source and network speed.

---

## 3. Input data format

The current implementation expects the following directory structure:

```text
data2/
├── train01zhee/
│   ├── sample_subjectA_0.xlsx
│   ├── sample_subjectB_1.xlsx
│   └── ...
└── test01zhee/
    ├── sample_subjectC_0.xlsx
    ├── sample_subjectD_1.xlsx
    └── ...
```

### File format
Each Excel file should contain one time column and six signal channels:

- `time`
- `channel1`
- `channel2`
- `channel3`
- `channel4`
- `channel5`
- `channel6`

The code reads the file as a table, drops the `time` column, standardizes each channel independently, resamples the signal to length 500, and splits it into 20 segments of length 25.

### Label format
The class label is parsed from the trailing numeric token in the file name. For example:
- `subject01_0.xlsx` -> label `0`
- `subject02_1.xlsx` -> label `1`

### Subject ID format
The subject ID is inferred from the file name by removing the trailing numeric label token. This subject ID is used for the inter-subject contrastive loss.

If your file naming convention differs, modify the functions:
- `parse_label_from_filename()`
- `parse_subject_id_from_filename()`

in the main script.

The simulated demo files included in this repository follow the same naming convention, so the code can run without modification.


---

## 4. Demo instructions

### Demo dataset
This repository includes a small **simulated demo dataset** under the default folder structure:

```text
data2/
├── train01zhee/
└── test01zhee/

### Running the demo
If your demo data are stored directly in the default folder structure (`data2/train01zhee` and `data2/test01zhee`), run:

```bash
python graph_scodetect_revised.py
```

### Expected console output
A successful run prints messages similar to:

```text
Loaded N samples from data2/train01zhee
Signal shape: (N, 20, 6, 25)
Classes: {...}
=== Stage 1: encoder pretraining ===
[pretrain] epoch=001 loss=...
...
=== Stage 2: joint training ===
[train] epoch=001 total=... intra=... inter=... class=... test_acc=...%
...
Final test accuracy: ...%
```

### Typical demo runtime
- CPU: a small demo typically runs in a few minutes.
- GPU: the same demo usually runs faster.

---

## 5. Instructions for use on your own data

### Step 1. Prepare your files
Organize your dataset into:

```text
data2/train01zhee/
data2/test01zhee/
```

Each sample must be an Excel file with one `time` column and six channel columns.

### Step 2. Check labels and subject IDs
Ensure that:
- the class label is encoded in the file name,
- the subject identifier can be recovered from the remaining file name stem.

### Step 3. Adjust configuration if needed
At the top of `graph_scodetect_revised.py`, you can modify:
- `PATH`
- `TRAIN_DIR`
- `TEST_DIR`
- `RESAMPLE_LEN`
- `SEGMENT_LEN`
- `BATCH_SIZE`
- `PRETRAIN_EPOCHS`
- `TRAIN_EPOCHS`
- `LR`
- `WEIGHT_DECAY`
- `LAMBDA_INTER`
- `GAMMA_CLASS`
- `TEMPERATURE`
- `INTRA_MARGIN`
- `NORMAL_LABEL`

### Step 4. Run training

```bash
python graph_scodetect_revised.py
```

### Step 5. Save results
The current script prints results to the console. If you want saved checkpoints, logs, or prediction files, add a save routine in `main()` or redirect terminal output to a log file:

```bash
python graph_scodetect_revised.py > results/run.log 2>&1
```

---

## 6. Expected outputs

The script performs:
1. dataset loading and preprocessing,
2. encoder pretraining,
3. joint training,
4. test-set evaluation.

Expected outputs include:
- printed class mapping,
- per-epoch pretraining losses,
- per-epoch joint training losses,
- per-epoch test accuracy,
- final test accuracy.

The current implementation returns:
- `model`
- `history`
- `y_true`
- `y_pred`

inside the Python runtime at the end of `main()`.

---

## 7. Method summary

The implementation follows a two-stage GraphScoDetect pipeline:

1. **Data preprocessing**
   - read Excel files,
   - remove the `time` column,
   - skip invalid files containing `--`,
   - fill missing values,
   - perform per-channel standardization,
   - resample each signal to length 500,
   - split each signal into 20 segments of length 25.

2. **Graph-based encoder**
   - each channel segment is embedded by a small MLP,
   - embedded channel features are treated as graph nodes,
   - graph message passing layers learn inter-channel relationships,
   - segment-level and sample-level representations are obtained by averaging.

3. **Temporal modeling and classification**
   - segment representations are processed by a bidirectional LSTM,
   - the pooled temporal representation is passed to an MLP classifier.

4. **Training objective**
   - stage 1: encoder pretraining with intra-subject asymmetry loss and inter-subject contrastive loss,
   - stage 2: joint training with total loss
     `L = L_intra + lambda * L_inter + gamma * L_class`.

---

