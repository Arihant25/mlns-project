# Phase A — data prep
cd aqsol/mlp && python generate_embeddings.py
cd ../minimol && python generate_embeddings_minimol.py
cd ../chemprop && python export_chemprop_data.py

# Phase B — baselines
cd ../mlp && python train_mlp.py
cd ../gsl && python train_gsl.py
cd ../minimol && python train_minimol_mlp.py         # MiniMol as-is
cd ../chemprop && python train_chemprop_baseline.py  # Chemprop as-is

# Phase C — evidential
cd ../gsl && python train_evidential_gsl.py          # → evidential_gsl.pt
cd ../minimol && python train_minimol_evidential_gsl.py

# Phase D — soft weighting
cd ../gsl && python train_weighted_gsl.py
cd ../mlp && python train_weighted_mlp.py
cd ../minimol && python train_minimol_weighted_gsl.py

# Phase E — Chemprop soft weighting
cd ../chemprop && python export_sample_weights.py
cd . && python train_chemprop_weighted.py