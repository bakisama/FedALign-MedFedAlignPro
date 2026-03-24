# FedALign-MedFedAlignPro

This repository extends the original FedALign codebase with **MedFedAlignPro**, a Kaggle-ready federated domain generalization pipeline for **multi-site chest X-ray pneumonia classification under scanner/protocol shift**.

Base method:
- FedAlign: Federated Domain Generalization with Cross-Client Feature Alignment
- CVPR 2025 Workshop paper: [FedAlign paper](https://openaccess.thecvf.com/content/CVPR2025W/FedVision/html/Gupta_FedAlign_Federated_Domain_Generalization_with_Cross-Client_Feature_Alignment_CVPRW_2025_paper.html)

---

## Overview

The repo now contains two tracks:

- Original visual-domain FedDG baselines such as `FedAlign`, `FedAvg`, `FedProx`, `FedSR`, `FedIIR`, `FedADG`, and `CCST`
- `MedFedAlignPro`, a new medical extension that replaces cross-client style-pool sharing with **reliability-weighted class prototype alignment**

### MedFedAlignPro Highlights

- Chest X-ray binary classification: `0 = non-pneumonia`, `1 = pneumonia`
- Multi-domain DG with default domains: `nih`, `guangzhou`, `rsna`
- Kaggle/Hugging Face data loading
- GPU auto-selection when available
- Metrics: accuracy, macro-F1, pneumonia-F1
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
python main.py MedFedAlignPro -d medical_cxr --domains nih,guangzhou --round 1 --num_epochs 1 --batch_size 8
```

Full medical run with RSNA when attached on Kaggle:

```bash
python main.py MedFedAlignPro -d medical_cxr --domains nih,guangzhou,rsna --round 5 --num_epochs 1 --batch_size 16
```

Run a single held-out domain:

```bash
python main.py MedFedAlignPro -d medical_cxr --domains nih,guangzhou,rsna --heldout_domain rsna
```

### Kaggle

If you want Kaggle to clone this repository directly, use:

```bash
git clone https://github.com/bakisama/FedALign-MedFedAlignPro.git
cd FedALign-MedFedAlignPro
pip install -r requirements.txt
python main.py MedFedAlignPro -d medical_cxr --domains nih,guangzhou --round 1 --num_epochs 1 --batch_size 8
```

There is also a helper script:

```bash
bash scripts/run_kaggle_medfedalignpro.sh
```

You can override defaults, for example:

```bash
REPO_DIR=/kaggle/working/FedALign-MedFedAlignPro DOMAINS=nih,guangzhou,rsna ROUND=3 BATCH_SIZE=16 bash scripts/run_kaggle_medfedalignpro.sh
```

## Outputs

MedFedAlignPro writes results to:

```bash
out/MedFedAlignPro/medical_cxr/<run_timestamp>/
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

- NIH and Guangzhou are loaded via Hugging Face `datasets`
- RSNA is auto-detected from common Kaggle input paths
- RSNA DICOM reading uses `pydicom`
- If Kaggle Internet is disabled, Hugging Face loading will fail unless cached already

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
