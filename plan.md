### Phase 1: Baseline Implementations (Deterministic)

#### Subphase 1A: MLP on MolFormer
**Objective:**  
Establish a non-graph baseline using frozen chemical language representations.

**Experiments:**  
Train a standard Multi-Layer Perceptron (MLP) using frozen MolFormer-XL embeddings to predict scalar molecular properties such as Caco-2 permeability.

**Evaluation Metrics:**  
- Mean Absolute Error (MAE)  
- Root Mean Square Error (RMSE)

**Expected Results:**  
A functional deterministic baseline that establishes a performance floor but remains vulnerable to unrecorded label noise.


#### Subphase 1B: Simple Graph Structure Learning (GSL)

**Objective:**  
Implement a deterministic graph structure learning mechanism without uncertainty gating.

**Experiments:**  
Develop the GSL module to dynamically generate a symmetric molecule–molecule adjacency matrix over each batch using the MolFormer embeddings. The aggregated representations are then routed through a standard (non-evidential) regression head.

**Evaluation Metrics:**  
- Mean Absolute Error (MAE)  
- Root Mean Square Error (RMSE)  
- Visual or statistical inspection of the learned adjacency matrix to verify whether structurally similar molecules are connected.

**Expected Results:**  
A working GSL model that aggregates local neighborhood information. However, it will likely propagate noise when structurally similar molecules exhibit conflicting measurements at activity cliffs.


---

### Phase 2: Integration of Evidential Deep Learning (EDL)

**Objective:**  
Introduce aleatoric uncertainty estimation to both baseline architectures to isolate irreducible data noise.

**Experiments:**

**Evidential MLP**  
Replace the standard regression head in the Phase 1A MLP with an evidential regression head that outputs Normal-Inverse-Gamma (NIG) parameters  
$(\mu, \lambda, \alpha, \beta)$.

Aleatoric uncertainty is calculated analytically as:

\[
u = \frac{\beta}{\alpha - 1}
\]

**Evidential GSL**  
Add the same evidential head to the Phase 1B GSL model. Implement the sender-indexed aleatoric uncertainty gate in the message-passing mechanism to suppress contributions from high-noise neighbors. Include an explicit skip connection to isolate the self-update.

**Training**  
Train both models using the error-scaled KL divergence regularizer to penalize confident errors.

**Evaluation Metrics:**  
- Prediction error (MAE, RMSE)  
- Correlation between estimated aleatoric uncertainty ($u$) and prediction residuals.

**Expected Results:**  
The Evidential GSL model should output elevated aleatoric uncertainty at activity cliffs by forcing structurally similar molecules with conflicting labels to share representations. This should demonstrate superiority over the Evidential MLP, which lacks the structural bottleneck needed to flag this specific type of noise.


---

### Phase 3: Automated Dataset Curation (Filter-and-Retrain)

**Objective:**  
Validate the Evidential GSL framework as an automated dataset curation engine.

**Experiments:**

1. Train the fully assembled Evidential GSL model on a noisy ADMET benchmark dataset.
2. Identify molecules whose aleatoric uncertainty exceeds a defined threshold ($u > \tau$).
3. Remove these molecules to form a curated dataset.
4. Retrain standard baseline models (e.g., standard GNNs) entirely on this curated dataset.

**Evaluation Metrics:**  
Compare the generalization performance (MAE, RMSE) of models retrained on the curated dataset with those trained on the original uncurated dataset.

**Expected Results:**  
Removing dynamically flagged noisy data should improve the generalization ability and predictive accuracy of standard baseline models, demonstrating the framework's viability as a scalable dataset curation solution.