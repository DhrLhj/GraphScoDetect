# Data interface and packaged demo audit

## Packaged demo data (authoritative for this archive)

The `data/` directory currently contains a small, non-clinical interface demo used to verify the revised model's Excel input format:

```text
data/
├── train01zhee/   4 subjects
└── test01zhee/    2 subjects
```

Total demo subjects: **6**.

Labels are parsed from the trailing integer in each filename, exactly as in the revised `models.py`:

```text
0 = healthy / normal
1 = patient / scoliosis
```

Demo label counts: **label 0 = 3**, **label 1 = 3**.

Each demo workbook contains seven columns:

```text
time, channel1, channel2, channel3, channel4, channel5, channel6
```

The six signal channels are standardized per channel, resampled to 500 points, then divided into 20 segments of length 25 by the revised model pipeline.

## Full paper experiments

The demo set above is **not** the full clinical dataset and does not contain the four-class severity or three-class curve-location labels required to reproduce Tasks 4 and 6. The full experiment runner therefore still accepts the external QC source root and subject-level clinical label workbook.

Full task definitions remain:

- Task 2: Normal/Control vs Scoliosis/Patient.
- Task 4: Normal, Mild, Moderate, Severe.
- Task 6: primary curve location: Thoracic, Thoracolumbar, Lumbar.

Rows whose note contains `不要数据` are excluded from all full-data experiments. For LOCO, `青海` is train-only by default.

No fixed subject/sample count is hard-coded in the revised package; counts are inferred from the supplied full-data inputs.
