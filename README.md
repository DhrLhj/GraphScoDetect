# GraphScoDetect v5 Reproducibility Package

## Overview

This repository is the consolidated code package for the current GraphScoDetect scoliosis-identification experiments. It follows the compact organization style of the provided code-package example, while retaining **all training settings used by the current v5 study** rather than only one train/test split.

The package contains:

1. data validation and task construction;
2. subject-independent split generation;
3. main model training for Tasks 2/4/6 under GKF3/GKF5/GKF7/GKF10/LOSO/LOCO;
4. subject-level multi-seed result aggregation, including AUROC;
5. LOSO channel/sensor-number/sensor-modality experiments for Tasks 2 and 4;
6. Macro-AUPRC, misclassification, curve-location, and curve-group analyses;
7. Fig. 4f single-model feature extraction and PCA visualization.

The package includes a small **non-clinical demo dataset** under `data/train01zhee` and `data/test01zhee` for interface checks. The full clinical dataset and identifiable label workbook used for the paper are not included; pass them separately to the full protocol runner.

---

## Repository contents

```text
GraphScoDetect_v5_reproducibility/
├── README.md
├── Requirements.txt
├── EXPERIMENTS.md
├── FILE_MAP.md
├── models.py
│
├── 00_validate_inputs.py
├── 01_prepare_tasks.py
├── 02_make_all_subject_splits.py
├── 03_train_all_protocols.py
├── 04_summarize_results.py
├── 05_run_all_protocols.sh
├── 06_run_all_training.sh
├── 07_run_all_postprocessing.sh
│
├── sensor_experiments/
│   ├── 00_check_baseline_loso_config.py
│   ├── 01_train_sensor_experiments_loso.py
│   ├── 02_summarize_sensor_experiments_loso.py
│   ├── 03_run_sensor_experiments_loso.sh
│   └── 04_wait_then_run_sensor.sh
│
├── postprocessing/
│   ├── 01_calculate_task4_macro_auprc.py
│   ├── 02_analyze_task4_misclassifications.py
│   ├── 03_analyze_sensor_by_curve_location.py
│   └── 04_analyze_curve_group_statistics.py
│
├── visualization/
│   ├── 01_export_single_model_subject_features.py
│   ├── 02_plot_pca_single_model.py
│   └── 03_run_fig4f_single_model_pca.sh
│
└── data/
    └── README.md
```

See `EXPERIMENTS.md` for the full experiment matrix.

---

## 1. System requirements

### Recommended platform

- Linux
- Python 3.10+
- CUDA-capable GPU recommended for training

### Install dependencies

```bash
python -m venv graphscodetect_env
source graphscodetect_env/bin/activate
pip install -r Requirements.txt
```

Main dependencies:

```text
numpy
pandas
scipy
scikit-learn
torch
openpyxl
matplotlib
```

---

## 2. Input data

The code does not modify the original input data.

The current v5 pipeline expects an existing QC/preprocessed source root with at least:

```text
SOURCE_DATA_ROOT/
└── 4/
    ├── data_4class.npy
    ├── names_4class.npy
    └── dataset_4class.json
```

The subject-level label workbook is passed separately. The expected schema is documented in:

```text
data/README.md
```

Current tasks:

```text
Task 2: binary scoliosis screening
        Normal / Scoliosis

Task 4: four-class severity classification
        Normal / Mild / Moderate / Severe

Task 6: primary curve-location classification
        Thoracic / Thoracolumbar / Lumbar
```

Rows explicitly marked with `不要数据` are excluded by the current v5 preparation logic.

---

## 3. Main model

`models.py` has been replaced by the user-supplied revised **GraphScoDetect** implementation. Its main representation flow is:

```text
raw 6-channel sequence
→ per-channel standardization + resampling to 500 points
→ 20 temporal segments × 6 channels × 25 samples
→ SegmentEmbedding (MLP)
→ learnable symmetric GraphMessagePassing × 2
→ channel pooling to segment representations
→ bidirectional LSTM
→ temporal mean pooling (256-D temporal_repr)
→ Linear(256,128) + ReLU + Dropout(0.2)
→ Linear(128,num_classes)
```

The revised training objective follows the supplied code:

```text
Stage 1 (20 epochs): L_intra + 0.5 * L_inter
Stage 2 (80 epochs): L_intra + 0.5 * L_inter + 1.0 * L_class
Optimizer: Adam, LR=1e-4, weight_decay=1e-4
```

For Fig. 4f, the exported pre-classifier feature is `temporal_repr`, i.e. the **256-D pooled BiLSTM representation before the classifier**.

---

## 4. Main experiment protocols

All of the following are implemented and included in the full runner:

```text
gkf3
gkf5
gkf7
gkf10
loso
loco
```

The manuscript training configuration used by `06_run_all_training.sh` is:

```text
Total epochs = 100 (20 encoder pretraining + 80 joint training)
Batch size   = 8
Seeds        = 42 43 44
Learning rate= 1e-4
Weight decay = 1e-4
Classifier dropout = 0.2
AMP          = enabled when requested
Class-weighted CE = disabled by default to match the supplied revised objective
Save model checkpoints = enabled
```

For LOCO:

```text
青海 = train-only center
```

It can participate in training but is not used as a held-out LOCO test center.

---

## 5. Run all main protocols

Run the complete Task 2/4/6 protocol matrix:

```bash
CUDA_VISIBLE_DEVICES=0 \
EPOCHS=100 \
PRETRAIN_EPOCHS=20 \
JOINT_EPOCHS=80 \
BATCH_SIZE=8 \
SEEDS="42 43 44" \
NUM_WORKERS=4 \
AMP=1 \
SAVE_MODEL=1 \
PROTOCOLS="gkf3 gkf5 gkf7 gkf10 loso loco" \
bash 05_run_all_protocols.sh \
  /path/to/SOURCE_DATA_ROOT \
  /path/to/label_v5.xlsx \
  /path/to/GraphScoDetect_v5_reproducibility \
  /path/to/output/main_protocols
```

The script sequentially performs:

```text
input validation
→ Task 2/4/6 construction
→ split generation
→ gkf3 training
→ gkf5 training
→ gkf7 training
→ gkf10 training
→ loso training
→ loco training
→ subject-level summary
```

Training is resumable. A completed fold is skipped only when the saved run configuration matches the requested configuration.

---

## 6. One-command full training package

To reproduce all current training experiments, including the LOSO sensor experiments:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash 06_run_all_training.sh \
  /path/to/SOURCE_DATA_ROOT \
  /path/to/label_v5.xlsx \
  /path/to/output_root
```

This executes:

```text
A. Main Tasks 2/4/6:
   GKF3 + GKF5 + GKF7 + GKF10 + LOSO + LOCO

B. Sensor experiments for Tasks 2/4:
   LOSO channel ablation
   LOSO sensor-number configurations
   LOSO sensor-modality configurations
```

Default outputs:

```text
output_root/
├── main_protocols/
└── sensor_loso/
```

Set:

```bash
RUN_SENSOR=0
```

if only the main protocol training is required.

---

## 7. LOSO sensor experiments

Sensor/channel map:

```text
S1 -> ch0
S2 -> ch1
S3 -> ch2,ch3
S4 -> ch4,ch5
```

### Channel ablation

```text
All sensors
Remove S1
Remove S2
Remove S3
Remove S4
```

### Different sensor counts

Task 2:

```text
4: S1+S2+S3+S4
3: S1+S3+S4
2: S3+S4
1: S4
```

Task 4:

```text
4: S1+S2+S3+S4
3: S1+S3+S4
2: S1+S3
1: S3
```

### Different sensor modalities

```text
Uniaxial-only = S1+S2+S3(ch1)+S4(ch1)
Vector-only   = S3+S4
Combined      = S1+S2+S3+S4
```

Run sensor experiments independently after the baseline LOSO is complete:

```bash
CUDA_VISIBLE_DEVICES=0 \
SEEDS="42 43 44" \
NUM_WORKERS=4 \
AMP=1 \
UNIAXIAL_DEFINITION=image \
bash sensor_experiments/03_run_sensor_experiments_loso.sh \
  /path/to/output/main_protocols \
  /path/to/GraphScoDetect_v5_reproducibility \
  /path/to/GraphScoDetect_v5_reproducibility \
  /path/to/output/sensor_loso
```

The sensor scripts read the baseline LOSO run configurations so that epochs, batch size, learning rate, weight decay, dropout, class weighting, dataset fingerprint, and split fingerprint remain aligned with the main experiment.

---

## 8. Saved outputs

For every main training fold, the code can save:

```text
run_config.json
metrics.json
predictions.json
training_history.csv
train.npy
test.npy
model.pt              # when SAVE_MODEL=1
```

The main subject-level summary contains files such as:

```text
01_main_subject_ensemble_metrics.csv
02_metrics_by_seed.csv
03_seed_mean_std.csv
04_per_class_subject_ensemble.csv
05_confusion_matrix_counts.csv
06_confusion_matrix_row_normalized.csv
07_subject_ensemble_predictions.csv
08_subject_predictions_by_seed.csv
...
14_loco_per_center_ensemble_metrics.csv
...
```

Main metrics include:

```text
Accuracy
Sensitivity
Specificity
Balanced Accuracy
Micro-F1
Macro-F1
Weighted-F1
AUROC
```

---

## 9. Macro-AUPRC

Task-4 Macro-AUPRC is calculated from the existing subject-level ensemble probabilities; no retraining is required:

```bash
python postprocessing/01_calculate_task4_macro_auprc.py \
  --summary_dir /path/to/output/main_protocols/summary \
  --out_dir /path/to/output/analysis/task4_auprc
```

The script reports:

```text
Macro-AUPRC
Micro-AUPRC
Weighted-AUPRC
Normal OvR AUPRC
Mild OvR AUPRC
Moderate OvR AUPRC
Severe OvR AUPRC
```

for GKF3/GKF5/GKF7/GKF10/LOSO/LOCO.

---

## 10. Four-class misclassification analysis

```bash
python postprocessing/02_analyze_task4_misclassifications.py \
  --summary_dir /path/to/output/main_protocols/summary \
  --out_dir /path/to/output/analysis/task4_misclassification
```

This produces one-row-per-error-event records containing the experiment protocol, subject, true class, predicted class, and error transition, as well as a subject-level cross-protocol summary.

---

## 11. Sensor contribution by primary curve location

```bash
python postprocessing/03_analyze_sensor_by_curve_location.py \
  --base_out_root /path/to/output/main_protocols \
  --sensor_out_root /path/to/output/sensor_loso \
  --out_dir /path/to/output/analysis/sensor_by_curve_location \
  --seeds 42 43 44
```

The output includes both absolute subgroup accuracy and:

```text
Delta Acc = Acc(Remove Si) - Acc(All)
```

with paired subject bootstrap confidence intervals.

---

## 12. Curve-group descriptive statistics

```bash
python postprocessing/04_analyze_curve_group_statistics.py \
  --subject_csv /path/to/output/main_protocols/summary/07_subject_ensemble_predictions.csv \
  --protocol loso \
  --out_dir /path/to/output/analysis/curve_group_statistics
```

---

## 13. Fig. 4f PCA visualization

The visualization uses one fixed representative Task-4 LOSO checkpoint (default `seed42/fold0`) for all subjects so that all embeddings share one latent coordinate system.

```bash
CUDA_VISIBLE_DEVICES=0 \
SEED=42 \
FOLD=0 \
BATCH_SIZE=256 \
NUM_WORKERS=4 \
STANDARDIZE=0 \
bash visualization/03_run_fig4f_single_model_pca.sh \
  /path/to/output/main_protocols \
  /path/to/GraphScoDetect_v5_reproducibility \
  /path/to/output/analysis/fig4f_single_model
```

Outputs include:

```text
task4_single_model_seed42_fold0_subject_features.csv
Fig4f_PCA_coordinates.csv
Fig4f_PCA_variance.csv
Fig4f_PCA_single_model.pdf
Fig4f_PCA_single_model.png
Fig4f_PCA_single_model.svg
```

The feature CSV contains the 128-D pre-classifier representation and does not export prediction probabilities.

---

## 14. Run all postprocessing

After all training finishes:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash 07_run_all_postprocessing.sh /path/to/output_root
```

This runs the current AUPRC, misclassification, sensor-location, curve-group, and PCA analyses.

---

## 15. Reproducibility notes

- Main evaluation is subject-level.
- Repeated signal segments from the same subject are first aggregated at the subject level.
- Subject probabilities are then averaged across seeds 42/43/44 for the main ensemble result.
- Grouped cross-validation and LOSO splits are subject-independent.
- LOCO is center-independent with `青海` train-only by default.
- Do not compare seed-mean Accuracy with the final seed-probability-ensemble Accuracy as though they were the same statistic.
- For patient-only curve-location subgroups in Task 2, subgroup Accuracy is numerically equivalent to patient sensitivity because the subgroup contains no negative controls.
