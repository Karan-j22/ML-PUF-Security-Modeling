# PUF Security Vulnerability Modeling (CS771A Project)

This project explores the vulnerability of **Physical Unclonable Functions (PUFs)** to machine learning attacks.

## 🚀 Overview
- Implemented feature maps to model **Multi-Level PUFs (ML-PUFs)** using linear classifiers.
- Demonstrated that simple linear models can accurately predict responses from ML-PUFs.
- Recovered intrinsic **delay parameters** of Arbiter PUFs from learned linear models.
- Benchmarked accuracy, training time, and robustness under different hyperparameters.

## 📂 Files
- `submit.py` — main implementation (`my_map`, `my_fit`, `my_decode`)
- `report.pdf` — detailed report (math, derivations, experiments)
- `data/` — training, test, and model files

## 🛠️ Tech Stack
- Python, NumPy, Scikit-learn  
- Only linear models (`LinearSVC`, `LogisticRegression`) were used.

## ▶️ Usage
```bash
# Clone the repo
git clone <your-repo-url>
cd <repo-folder>

# Run evaluation (example)
python submit.py
