# FedAlign: Federated Domain Generalization with Cross-Client Feature Alignment

**Authors:** Sunny Gupta, Vinay Sutar, Varunav Singh, Amit Sethi  
**Conference:** CVPR 2025  
**Paper:** (https://openaccess.thecvf.com/content/CVPR2025W/FedVision/html/Gupta_FedAlign_Federated_Domain_Generalization_with_Cross-Client_Feature_Alignment_CVPRW_2025_paper.html)

---

## Overview

**FedAlign** is a lightweight, privacy-preserving framework for **Federated Domain Generalization (FedDG)** that enhances generalization to unseen domains without compromising data privacy. It addresses key challenges such as:

- Strict privacy constraints in Federated Learning (FL)
- Non-i.i.d. data distributions across clients
- Limited domain diversity

### Key Contributions

- **Cross-Client Feature Extension Module**  
  Enhances local domain representation using domain-invariant feature perturbation and selective cross-client feature transfer.

- **Dual-Stage Alignment Module**  
  Aligns both feature embeddings and predictions across clients to extract robust, domain-invariant representations.

- **Lightweight & Scalable**  
  Requires minimal computation and communication overhead, making it suitable for real-world federated setups.

---

Supported datasets include:
- MiniDomainNet
- PACS
- OfficeHome

---

## How to Run

You can run the training script using the following command:

```bash
python main.py [Algorithm] -d [Dataset Name] [other arguments]

python main.py FedAlign -d minidomainnet
```
If you find this work useful, please cite as

```bash
@inproceedings{gupta2025fedalign,
  title={FedAlign: Federated Domain Generalization with Cross-Client Feature Alignment},
  author={Gupta, Sunny and Sutar, Vinay and Singh, Varunav and Sethi, Amit},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={1801--1810},
  year={2025}
}
```
