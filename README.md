# FedALign-MedFedAlignPro

This repository extends the original FedALign codebase with **MedFedAlignPro**, a Kaggle-ready federated domain generalization pipeline for **binary referable diabetic retinopathy classification across multiple fundus datasets**.

Base method:
- FedAlign: Federated Domain Generalization with Cross-Client Feature Alignment
- CVPR 2025 Workshop paper: [FedAlign paper](https://openaccess.thecvf.com/content/CVPR2025W/FedVision/html/Gupta_FedAlign_Federated_Domain_Generalization_with_Cross-Client_Feature_Alignment_CVPRW_2025_paper.html)

---

## Overview

The repo now contains two tracks:

- Original visual-domain FedDG baselines such as `FedAlign`, `FedAvg`, `FedProx`, `FedSR`, `FedIIR`, `FedADG`, and `CCST`
- `MedFedAlignPro`, a new medical extension that replaces cross-client style-pool sharing with **reliability-weighted class prototype alignment**

### MedFedAlignPro Highlights

- Binary diabetic retinopathy classification: `0 = non_referable_dr`, `1 = referable_dr`
- Multi-domain DG with default domains: `aptos`, `idrid`, `messidor`, `messidor2`
- Kaggle/Hugging Face data loading
- GPU auto-selection when available
- Metrics: accuracy, macro-F1, referable-DR F1
- Plots: round curves, confusion matrices, prototype heatmap, optional t-SNE

---

## Installation

```bash
pip install -r requirements.txt
```

---

## How to Run

### Original FedAlign

```bash
python main.py FedAlign -d minidomainnet
```

### MedFedAlignPro

Quick 2-domain smoke run:

```bash
python main.py MedFedAlignPro -d medical_dr --domains aptos,idrid --round 1 --num_epochs 1 --batch_size 8
```

Full DR run with all available fundus domains:

```bash
python main.py MedFedAlignPro -d medical_dr --domains aptos,idrid,messidor,messidor2 --round 5 --num_epochs 1 --batch_size 16
```

Run a single held-out domain:

```bash
python main.py MedFedAlignPro -d medical_dr --domains aptos,idrid,messidor,messidor2 --heldout_domain messidor2
```

### Kaggle

If you want Kaggle to clone this repository directly, use:

```bash
git clone https://github.com/bakisama/FedALign-MedFedAlignPro.git
cd FedALign-MedFedAlignPro
pip install -r requirements.txt
python main.py MedFedAlignPro -d medical_dr --domains aptos,idrid --round 1 --num_epochs 1 --batch_size 8
```

There is also a helper script:

```bash
bash scripts/run_kaggle_medfedalignpro.sh
```

You can override defaults, for example:

```bash
REPO_DIR=/kaggle/working/FedALign-MedFedAlignPro DOMAINS=aptos,idrid,messidor,messidor2 ROUND=3 BATCH_SIZE=16 bash scripts/run_kaggle_medfedalignpro.sh
```

## Outputs

MedFedAlignPro writes results to:

```bash
out/MedFedAlignPro/medical_dr/<run_timestamp>/
```

Per held-out domain:
- `round_metrics.csv`
- `round_curves.png`
- `validation_confusion_matrix.png`
- `test_confusion_matrix.png`
- `prototype_heatmap.png`
- `summary.csv`
- `checkpoint.pth`

Cross-domain summary:
- `overall_summary.csv`

## Notes

- Domains are discovered from Kaggle-style mounted folders or explicit environment variables such as `APTOS_ROOT` and `IDRID_ROOT`
- Labels are harmonized into binary `referable_dr` vs `non_referable_dr`
- Metadata is inferred from CSV/XLS/XLSX files with common image-id and grade column names

## Citation

If you use the original FedAlign method, please cite:

```bash
@inproceedings{gupta2025fedalign,
  title={FedAlign: Federated Domain Generalization with Cross-Client Feature Alignment},
  author={Gupta, Sunny and Sutar, Vinay and Singh, Varunav and Sethi, Amit},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={1801--1810},
  year={2025}
}
```
