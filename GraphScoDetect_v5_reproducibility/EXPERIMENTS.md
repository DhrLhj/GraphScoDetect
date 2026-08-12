# Experiment index

This package contains the training code for all experiment settings used in the current v5 study.

## A. Main subject-independent experiments

Tasks:

| Task | Problem | Classes |
|---|---|---|
| 2 | Binary scoliosis screening | Normal / Scoliosis |
| 4 | Severity classification | Normal / Mild / Moderate / Severe |
| 6 | Primary curve-location classification | Thoracic / Thoracolumbar / Lumbar |

Protocols:

```text
gkf3
gkf5
gkf7
gkf10
loso
loco
```

Default manuscript training setup used by `06_run_all_training.sh`:

```text
Epochs       20 encoder pretraining + 80 joint training
Batch size   8
Seeds        42, 43, 44
Optimizer    Adam
LR           1e-4
Weight decay 1e-4
Classifier dropout 0.2
AMP          optional/enabled by runner
Class-weighted CE disabled by default (revised objective)
LOCO train-only center: 青海
Model checkpoints: saved
```

Core scripts:

```text
00_validate_inputs.py
01_prepare_tasks.py
02_make_all_subject_splits.py
03_train_all_protocols.py
04_summarize_results.py
05_run_all_protocols.sh
```

## B. LOSO sensor experiments

Tasks:

```text
Task 2 binary
Task 4 four-class
```

Protocol:

```text
LOSO
```

The non-full configurations reuse the exact baseline training implementation and hyperparameters from the completed LOSO baseline `run_config.json` files.

### Channel ablation

```text
All sensors : S1+S2+S3+S4
Remove S1  : S2+S3+S4
Remove S2  : S1+S3+S4
Remove S3  : S1+S2+S4
Remove S4  : S1+S2+S3
```

### Sensor-number configurations

Task 2:

```text
4 sensors: S1+S2+S3+S4
3 sensors: S1+S3+S4
2 sensors: S3+S4
1 sensor : S4
```

Task 4:

```text
4 sensors: S1+S2+S3+S4
3 sensors: S1+S3+S4
2 sensors: S1+S3
1 sensor : S3
```

### Sensor-modality configurations

```text
Uniaxial-only: S1 + S2 + S3(ch1) + S4(ch1)
Vector-only  : S3 + S4
Combined     : S1 + S2 + S3 + S4
```

Physical channel map:

```text
S1 -> ch0
S2 -> ch1
S3 -> ch2, ch3
S4 -> ch4, ch5
```

Scripts:

```text
sensor_experiments/00_check_baseline_loso_config.py
sensor_experiments/01_train_sensor_experiments_loso.py
sensor_experiments/02_summarize_sensor_experiments_loso.py
sensor_experiments/03_run_sensor_experiments_loso.sh
```

## C. Derived metrics / statistics

No retraining is required for these scripts.

```text
postprocessing/01_calculate_task4_macro_auprc.py
postprocessing/02_analyze_task4_misclassifications.py
postprocessing/03_analyze_sensor_by_curve_location.py
postprocessing/04_analyze_curve_group_statistics.py
```

## D. Fig. 4f PCA visualization

A single representative Task-4 LOSO checkpoint (default seed 42 / fold 0) is used to extract the revised model's pre-classifier `temporal_repr` (256-D pooled BiLSTM representation), followed by PCA to 2-D.

```text
visualization/01_export_single_model_subject_features.py
visualization/02_plot_pca_single_model.py
visualization/03_run_fig4f_single_model_pca.sh
```

This is a qualitative representation visualization. Quantitative generalization metrics remain based on strict out-of-fold LOSO predictions.
