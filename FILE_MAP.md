# File naming map

The repository uses clean, function-oriented names rather than historical `v3/v4/fixed/final` names.

| Consolidated name | Historical/current source role |
|---|---|
| `models.py` | user-supplied `graph_scodetect_revised(20260812-082938).py` (revised GraphScoDetect) |
| `00_validate_inputs.py` | v5 input validation |
| `01_prepare_tasks.py` | `01_prepare_tasks_from_label_v3.py` used by the v5 label pipeline |
| `02_make_all_subject_splits.py` | subject-independent GKF/LOSO/LOCO split generator |
| `03_train_all_protocols.py` | `03_train_stgcn_all_protocols.py` |
| `04_summarize_results.py` | `04_summarize_full_protocol_results.py` |
| `05_run_all_protocols.sh` | `05_run_selected_protocols_sequential.sh` |
| `sensor_experiments/01_train_sensor_experiments_loso.py` | `01_train_sensor_loso_exact.py` |
| `sensor_experiments/02_summarize_sensor_experiments_loso.py` | `02_summarize_sensor_loso_exact.py` |
| `sensor_experiments/03_run_sensor_experiments_loso.sh` | `03_run_sensor_loso_exact.sh` |
| `postprocessing/01_calculate_task4_macro_auprc.py` | current Macro-AUPRC analysis |
| `postprocessing/02_analyze_task4_misclassifications.py` | all-protocol Task-4 error analysis |
| `postprocessing/03_analyze_sensor_by_curve_location.py` | LOSO sensor removal × curve-location analysis |
| `postprocessing/04_analyze_curve_group_statistics.py` | curve-group descriptive/statistical analysis |
| `visualization/01_export_single_model_subject_features.py` | single-model Fig.4f feature export |
| `visualization/02_plot_pca_single_model.py` | PCA plot generation |
| `visualization/03_run_fig4f_single_model_pca.sh` | one-command Fig.4f runner |

The clinical `label_v5.xlsx` file is intentionally not copied into this archive because it contains subject-level clinical information. Supply it externally when running the code.
