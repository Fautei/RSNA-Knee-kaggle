# RSNA Knee Abnormality Detection

Kaggle competition solution for detecting abnormalities in knee MRI scans.

Multilabel classification over 12 findings: ACL, MCL, medial/lateral meniscus, medial/lateral OA, PF OA, effusion, synovitis, Baker's cyst, contusion, fracture.

## How to run

Train:

```bash
python main.py --config_path configs/exp1.yaml --run_name exp1
```

Export to TorchScript:

```bash
python export.py --config_path configs/exp1.yaml --ckpt_path checkpoints/exp1/<checkpoint_name>
```

OOF predictions (K-fold):

```bash
python oof_preds.py --config_path configs/exp1.yaml --run_name oof --output_csv_path ./oof
```

## Results

Final ranking ~196th with 5 different models ensemble.

## Structure

- `main.py` / `export.py` / `oof_preds.py` — training, TorchScript export, K-fold OOF
- `datasets/` — DICOM series loading with MONAI transforms
- `models/` — timm encoder + query-based cross-attention pooling head
- `notebooks/` — EDA, baselines, validation, label refinement
- `configs/` — YAML configs (`default.yaml`, `exp1.yaml`, `extra_data.yaml`)
- `labels/` — label CSVs and prediction files
- `utils/` — transforms, config loading, logging
- `scripts/` — viewer, label API helper