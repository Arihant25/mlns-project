# Machine Learning for Natural Sciences Project

This is the step-by-step command list to run the full experiment pipeline.

## Project file structure

```text
mlns-project/
├─ requirements.txt
├─ README.md
├─ plan.md / proposal.md / golden_findings.md
├─ caco/
│  ├─ data/                  # shared Caco-2 dataset + cached tensors/smiles
│  ├─ mlp/                   # MLP baselines, evidential MLP, retraining, weighting
│  ├─ gsl/                   # GSL and evidential GSL training + curation scripts
│  ├─ xgboost/               # XGBoost on MolFormer embeddings
│  └─ results/               # all generated reports/metrics/model artifacts
└─ lipo/
	├─ data/                  # shared Lipophilicity dataset + cached tensors/smiles
	├─ mlp/                   # MLP baselines, evidential MLP, retraining, weighting
	├─ gsl/                   # GSL and evidential GSL training + curation scripts
	├─ xgboost/               # XGBoost on MolFormer embeddings
	└─ results/               # all generated reports/metrics/model artifacts
```

### What each subfolder does

- `data/`: Input datasets and generated split artifacts (`train/val/test embeddings`, `targets`, and `smiles`).
- `mlp/`: Non-graph experiments (baseline MLP, evidential MLP, retraining, soft-weighting, golden MLP comparison).
- `gsl/`: Graph Structure Learning experiments (baseline GSL, evidential GSL, uncertainty-based curation, golden dataset creation).
- `xgboost/`: Tree-based baseline using MolFormer embeddings.
- `results/`: Output reports and saved model weights from each phase.

## 1) Environment setup (once)

Run from the repo root:

```powershell
pip install -r requirements.txt
```

## 2) Run full pipeline for Caco-2

### 2.1 Phase 1 + 2 (embeddings, MLP, GSL, evidential models)

```powershell
cd caco/mlp
python generate_embeddings.py
python train_mlp.py
python train_evidential_mlp.py

cd ../gsl
python train_gsl.py
python train_evidential_gsl.py
```

### 2.2 Phase 3A/3B/3C (curation, retrain, soft-weighting)

```powershell
cd caco/gsl
python curate_dataset.py

cd ../mlp
python train_retrain_mlp.py
python train_weighted_mlp.py
```

### 2.3 Golden dataset + comparisons

```powershell
cd caco/gsl
python create_golden_dataset.py
```

```powershell
cd caco/mlp
python golden_mlp.py

cd ../xgboost
python train_xgboost_molformer.py
```

## 3) Run full pipeline for Lipophilicity

### 3.1 Phase 1 + 2 (embeddings, MLP, GSL, evidential models)

```powershell
cd lipo/mlp
python generate_embeddings.py
python train_mlp.py
python train_evidential_mlp.py

cd ../gsl
python train_gsl.py
python train_evidential_gsl.py
```

### 3.2 Phase 3A/3B/3C (curation, retrain, soft-weighting)

```powershell
cd lipo/gsl
python curate_dataset.py

cd ../mlp
python train_retrain_mlp.py
python train_weighted_mlp.py
```

### 3.3 Golden dataset + comparisons

```powershell
cd lipo/gsl
python create_golden_dataset.py

cd ../mlp
python golden_mlp.py
```

```powershell
cd lipo/xgboost
python train_xgboost_molformer.py
```

## 4) Where outputs are saved

- Caco outputs: `caco/results/`
- Lipo outputs: `lipo/results/`

Main files include:

- `phase1a_metrics.txt`
- `phase1b_metrics.txt`
- `phase2a_metrics.txt`
- `phase2b_metrics.txt`
- `phase3a_curation_report.txt`
- `phase3b_retrain_report.txt`
- `phase3c_soft_weighting_report.txt`
- `golden_dataset_report.txt`
- `golden_mlp_report.txt`
- `xgboost_molformer_report.txt`
