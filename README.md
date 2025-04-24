# EASE2025
#Replication Package

This repository reproduces every table, figure, and statistic of  
**"Co-Change Graph Entropy: A New Process Metric for Defect Prediction"** (accepted to EASE 2025).  
All code is pure **Python 3.11**—no R, Jupyter, or external GUIs.

---

## Directory Overview

| Phase | Script(s) | Purpose |
|-------|-----------|---------|
| **Pre‑processing** | `preprocess_0.py` | Tag every commit with its **release** (SmartSHARK) |
| | `preprocess_1.py` | Collect files **+ defect labels** for each release |
| | `preprocess_2.py` | Compute per‑file **process metrics** |
| | `preprocess_3.py` | Compute **change** & **co‑change entropy** |
| **Correlation** | `correlation_0.py` | Correlate both entropy measures with *defect counts* |
| **Prediction** | `prediction_0.py` | Train & evaluate defect classifiers |
| | `prediction_1.py` | Statistical analysis of the classifier metrics |
| **Representation** | `representation_0.py` | Generate pie‑chart comparisons used in the paper |
| **Dataset Table** | `dataset_table.py` | Build the project / release overview table |


---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| **SmartSHARK** | ≥ 0.2 | Follow official install guide |
| **MongoDB** | ≥ 6.0 | SmartSHARK backend |

---
