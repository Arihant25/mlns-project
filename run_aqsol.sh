# Phase A — data prep
cd aqsol/minimol && python generate_embeddings_minimol.py
cd ../chemprop && python export_chemprop_data.py

# Phase B — as-is baselines (TDC leaderboard top-2, 5 seeds each)
cd ../minimol && python train_minimol_mlp.py         # MiniMol as-is  (#1, MAE 0.741)
cd ../chemprop && python train_chemprop_baseline.py  # Chemprop-RDKit as-is (#2, MAE 0.761)

# Phase C — evidential uncertainty (internal step, feeds soft weighting)
cd ../minimol && python train_minimol_evidential_gsl.py  # -> results/evidential_gsl_minimol.pt

# Phase D — our contribution: soft weighting on top of both baselines
cd ../minimol && python train_minimol_weighted_mlp.py    # MiniMol + soft-weighted MLP
cd ../chemprop && python export_sample_weights.py        # export uncertainty weights
cd . && python train_chemprop_weighted.py                # Chemprop-RDKit + soft weighting
