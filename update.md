### Summary of Research Results and Outcomes

The following table summarizes the performance of the various architectures and experimental protocols tested on the **Caco-2** dataset using frozen **MolFormer-XL** embeddings.

| Phase | Architecture / Strategy | MAE (Mean) | Outcome & Key Insight |
| --- | --- | --- | --- |
| **1A** | **Baseline MLP** | **0.3869** | Established the deterministic performance floor for non-graph models.

 |
| **1B** | **Simple GSL** | **0.5045** | <br>**Failure**: Confirmed that standard GSL propagates unrecorded assay noise through neighborhood aggregation.

 |
| **2A** | **Evidential MLP** | **0.4376** | <br>**Ablation**: Proved that isolated evidential heads are "confidently wrong" ($\rho = -0.14$) without structural context.

 |
| **2B** | **Evidential GSL** | **0.3648** | <br>**Success**: The uncertainty gate successfully suppressed noise, achieving the best neural performance and ranking **#16** on the TDC leaderboard.

 |
| **3B** | **Hard Filtering MLP** | **0.4694*** | <br>**Failure**: Removing high-uncertainty samples caused "data starvation" and deleted critical chemical activity cliffs.

 |
| **3C** | **Soft-Weighting MLP** | **0.3973** | <br>**Qualified Success**: ($\gamma=0.5$) Prevented the performance collapse of Phase 3B by down-weighting rather than deleting noisy samples.

 |

**Results shown for 10% removal; performance degraded further at 15%.*

---

### Critical Analysis of Key Outcomes

* 
**The "Smear Effect" Validated**: Phase 1B empirically proved that standard GNN aggregation is detrimental when labels are inconsistent across structurally similar molecules, a common issue in inter-laboratory biological assays.


* 
**Internal Gating vs. External Curation**: The most significant finding is that **aleatoric uncertainty gating** is a powerful *architectural* tool for robust prediction, but a dangerous *curation* tool for physical data removal.


* 
**Activity Cliff Dilemma**: The failure of hard filtering suggests that the model's aleatoric head successfully identifies "contradictions" , but these contradictions include both **assay noise** (to be ignored) and **genuine chemical complexity** (to be learned).


* 
**Soft-Weighting Stability**: Phase 3C demonstrated that an exponential decay penalty ($w_i = \exp(-\gamma \cdot u_i)$) is the most stable way to utilize uncertainty for downstream models, as it preserves the full chemical feature space while reducing the impact of outliers.



---

### Potential Next Steps

Since we have identified that the MLP might be too "simple" to fully exploit the soft-weighting, and that Caco-2's small size limits curation benefits, we could:

1. **Test on Larger Datasets**: Replicate the Evidential GSL on the **TDC Lipophilicity** dataset ($N \approx 4,200$) to see if the increased data density allows for more effective curation without starvation.
2. **Weighted XGBoost Baseline**: Apply the $u$ values from Phase 2B as sample weights for a **Gradient Boosted Tree** model, which is often more sensitive to the "noise" identified by the evidential head than an MLP.
3. **Cross-Dataset Validation**: Use the model trained on Caco-2 to flag noise in a different but related permeability dataset (e.g., H-PAMPA) to test the transferability of the evidential gate.
