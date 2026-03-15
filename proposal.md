# Aleatoric Graph Structure Learning for Automated Dataset Curation in ADMET Prediction

**Inesh Shukla\*** 1  
**Arihant Tripathy\*** 1  

## Abstract

Public ADMET datasets are routinely treated as ground truth for molecular property prediction, yet inter-laboratory assay variability introduces severe, unrecorded label noise. Standard graph neural networks (GNNs) operating over fixed molecular structures propagate this noise during neighborhood aggregation, particularly at activity cliffs where structurally similar molecules exhibit conflicting measurements. Furthermore, standard evaluation metrics systematically penalize models that learn underlying chemistry over noisy assay artifacts.

To address this, we propose an uncertainty-aware Graph Structure Learning (GSL) framework that serves as both a predictive model and an automated dataset curation engine. Operating on frozen chemical language representations, our architecture learns a continuous molecule-molecule adjacency graph and dynamically gates message-passing contributions using aleatoric uncertainty derived from an evidential regression head.

By isolating irreducible data noise via an error-scaled loss function, our model identifies anomalous labels without requiring manual cross-database chemical audits. We outline a filter-and-retrain protocol demonstrating that removing data dynamically flagged by our evidential gate improves the generalizability of standard baseline models, establishing our framework as a scalable solution for robust ADMET prediction. :contentReference[oaicite:0]{index=0}

---

# 1. Introduction

Accurate prediction of scalar molecular properties, such as **Caco-2 intestinal permeability**, is a canonical task in computational drug discovery. Both graph neural networks (GNNs) (Yang et al., 2019) and chemical language models (Ross et al., 2022) have reported state-of-the-art results on the **Therapeutics Data Commons (TDC)** benchmarks (Huang et al., 2021). However, implicit in these comparisons is the flawed assumption that benchmark labels are universally reliable. :contentReference[oaicite:1]{index=1}

Biological assays like **Caco-2** are highly sensitive to experimental covariates—such as **pH, incubation time, and cell passage number**—that are rarely recorded in dataset metadata (Hubatsch et al., 2007). Consequently, when different laboratories measure the same compound under subtly different protocols, the discrepancy manifests as label variance that is indistinguishable from the target signal. Standard training objectives minimize expected loss under this corrupted distribution, effectively creating a systematic bias against robust predictors. :contentReference[oaicite:2]{index=2}

This problem is compounded in graph-based molecular property prediction. GNNs operate by iteratively aggregating feature information from local neighborhoods via message passing. If structurally similar molecules carry inconsistent labels due to inter-laboratory variability, neighborhood aggregation propagates the noise. This failure mode is particularly severe at **activity cliffs** (van Tilborg et al., 2022): pairs of structurally similar molecules with large differences in measured activity. :contentReference[oaicite:3]{index=3}

To address noise propagation, we introduce an **uncertainty-aware Graph Structure Learning (GSL)** architecture. Unlike existing methods that optimize purely for prediction, our model utilizes a **sender-indexed aleatoric uncertainty gate**, trained via an error-scaled evidential deep learning objective (Amini et al., 2020), to suppress message-passing contributions from high-noise neighbors.

Crucially, we utilize this evidential gate not just as an architectural enhancement, but as a **dynamic dataset curation tool**, allowing us to automatically filter noisy labels and improve the performance of baseline models without relying on highly manual, cross-database chemical audits (Fourches et al., 2010). :contentReference[oaicite:4]{index=4}

---

# 2. Background and Related Work

## 2.1 Molecular Property Prediction and GSL

Standard GNNs operate on a fixed molecular graph defined by **covalent bonding**. **Graph Structure Learning (GSL)** methods relax this constraint by jointly learning the graph topology and node representations from data (Chen et al., 2020).

Recent adaptations, such as **GSL-MPP** (Zhao et al., 2024), have applied this paradigm to learn **molecule–molecule similarity graphs over an entire dataset**. However, current GSL implementations assume reliable training labels and remain vulnerable to noise propagation. :contentReference[oaicite:5]{index=5}

## 2.2 Evidential Deep Learning for Regression

Uncertainty in neural network predictions can be decomposed into:

- **Epistemic uncertainty** (model ignorance)
- **Aleatoric uncertainty** (irreducible data noise)

**Evidential Deep Learning (EDL)** operationalizes this in a regression setting by placing a **Normal–Inverse–Gamma (NIG)** prior over the predictive distribution (Amini et al., 2020). Soleimany et al. demonstrated that this framework applies directly to molecular property prediction (Soleimany et al., 2021).

Our work departs from prior applications by utilizing **aleatoric uncertainty dynamically within the GSL message-passing mechanism itself**. :contentReference[oaicite:6]{index=6}

---

# 3. Proposed Framework

## 3.1 Evidential GSL Architecture

Our architecture replaces fixed bond graphs with a **learned molecule–molecule adjacency matrix**.

1. Molecules are encoded using **frozen MolFormer-XL embeddings** (Ross et al., 2022).
2. A **graph structure learner** dynamically generates a symmetric adjacency matrix \( A_{ij} \) over the batch.
3. Predictions are routed through an **evidential regression head** that outputs the parameters  
   \( (\mu, \lambda, \alpha, \beta) \) of a **Normal–Inverse–Gamma (NIG)** distribution.

The **aleatoric uncertainty** \( u \), isolating inherent label noise, is analytically computed as:

\[
u = \frac{\beta}{\alpha - 1}
\] 
:contentReference[oaicite:7]{index=7}

---

## 3.2 Resolving Canonicalized Datasets via Activity Cliffs

Many ADMET datasets, including **TDC**, are canonicalized to provide **one aggregated label per molecule**. In this regime, aleatoric uncertainty must rely on **structural similarity rather than exact duplicates**.

Our GSL module acts as a **forcing function**, learning edges between structurally highly similar molecules in latent space.

At an **activity cliff**, structurally similar molecules are forced to share representations despite conflicting labels. The network cannot smoothly map this localized graph neighborhood to distinct values without incurring high residual error, forcing the evidential head to output **elevated aleatoric uncertainty** to minimize its penalty. :contentReference[oaicite:8]{index=8}

---

## 3.3 Sender-Indexed Uncertainty Gate

To prevent noise propagation during message passing, we introduce a **sender-indexed gate** based on aleatoric uncertainty.

In subsequent rounds of message passing, contributions from **high-noise neighbors are suppressed**.

To ensure a noisy node does not suppress its own representation, we include a **skip connection** isolating the self-update:

\[
h_i^{(l+1)} = h_i^{(l)} + \sum_{j \in N(i)} A_{ij} \cdot \sigma(-\gamma u_j) \cdot W h_j^{(l)}
\]

where:

- \( u_j \) = aleatoric uncertainty of neighbor \( j \)
- \( W \) = learnable weight matrix
- \( \gamma \) = scaling hyperparameter
- \( \sigma \) = monotonic bounding function (e.g., sigmoid)

initialized near-open. :contentReference[oaicite:9]{index=9}

---

## 3.4 Error-Scaled Evidential Loss Function

To incentivize correct identification of noisy data, we employ an **error-scaled KL divergence regularizer**.

Standard evidential regression minimizes:

- Negative Log-Likelihood (NLL)
- KL regularization

We scale the KL penalty by the residual error:

\[
L = L_{NLL} + \lambda_{KL} |y_i - \mu_i| \cdot KL(NIG \parallel Prior)
\]

This imposes a **severe penalty if the model makes a large prediction error while expressing low uncertainty**, encouraging the model to assign high aleatoric uncertainty to contradictory labels. :contentReference[oaicite:10]{index=10}

---

## 3.5 The Filter-and-Retrain Protocol

To validate the architecture as an **automated dataset curation engine**, we propose the following protocol:

1. Train the **evidential GSL model** on the noisy ADMET benchmark.
2. Flag molecules whose aleatoric uncertainty exceeds a threshold \( u > \tau \).
3. Remove flagged molecules from the dataset.
4. Retrain **standard baseline models** on the curated dataset.

This demonstrates **improved generalization**, validating the framework as a scalable automated curation mechanism. :contentReference[oaicite:11]{index=11}

---

# References

Amini, A., Schwarting, W., Soleimany, A., and Rus, D.  
*Deep evidential regression.* Advances in Neural Information Processing Systems, 33, 2020.

Chen, Y., Wu, L., and Zaki, M. J.  
*Iterative deep graph learning for graph neural networks: Better and robust node embeddings.* NeurIPS, 2020.

Fourches, D., Muratov, E., and Tropsha, A.  
*Trust, but verify: On the importance of chemical structure curation in chemoinformatics and QSAR modeling research.* Journal of Chemical Information and Modeling, 50(7):1189–1204, 2010.

Huang, K., Fu, T., Gao, W., Zhao, Y., Roohani, Y., Leskovec, J., Coley, C. W., Xiao, C., Sun, J., and Zitnik, M.  
*Therapeutics Data Commons: Machine learning datasets and tasks for drug discovery and development.* NeurIPS Datasets and Benchmarks, 2021.

Hubatsch, I., Ragnarsson, E. G. E., and Artursson, P.  
*Determination of drug permeability and prediction of drug absorption in Caco-2 monolayers.* Nature Protocols, 2(9):2111–2119, 2007.

Ross, J., Belgodere, B., Chenthamarakshan, V., Padhi, I., Mroueh, Y., and Das, P.  
*Large-scale chemical language representations capture molecular structure and properties.* Nature Machine Intelligence, 4(12):1256–1264, 2022.

Soleimany, A. P., Amini, A., Goldman, S., Rus, D., Bhatia, S. N., and Coley, C. W.  
*Evidential deep learning for guided molecular property prediction and discovery.* ACS Central Science, 7(8):1356–1367, 2021.

van Tilborg, D., Alenicheva, A., and Grisoni, F.  
*Exposing the limitations of molecular machine learning with activity cliffs.* Journal of Chemical Information and Modeling, 62(23):5938–5951, 2022.

Yang, K., Swanson, K., Jin, W., Coley, C., Eiden, P., Gao, H., Guzman-Perez, A., Hopper, T., Kelley, B., Mathea, M., Palmer, A., Settels, V., Jaakkola, T., Jensen, K., and Barzilay, R.  
*Analyzing learned molecular representations for property prediction.* Journal of Chemical Information and Modeling, 59(8):3370–3388, 2019.

Zhao, B., Xu, W., Guan, J., and Zhou, S.  
*Molecular property prediction based on graph structure learning.* Bioinformatics, 40(5):btae304, 2024.