***

### Hand-off: Scaling the Two-Stage Evidential Meta-Learner
Refer meta/run_hopfield_poc.py for current setup

**The TL;DR:** The core architecture (`maml_hopfield_two_stage`) is mathematically locked and proven on a small scale. It successfully combines an Entropy Gate (to block out-of-distribution context retrieval) with an Evidential Gate (to stabilize within-episode message passing). Your job is to scale the data infrastructure, run the rigorous evaluation protocol, and generate the final publication-ready results table.

#### 1. The Code: What Stays and What Changes
* **DO NOT TOUCH:** The math and inner/outer loop gradient flow in `maml_hopfield_two_stage`. It is working perfectly. Write your code in a new file.
* **CHANGE:** The data pipeline and the evaluation wrapper.

#### 2. Scale the Context Set (The ChEMBL Upgrade)
We must replace our current limited TDC context set with an independent, massive memory bank to prove we aren't leaking data.
* **Action:** Sample 50,000 to 100,000 diverse molecules from ChEMBL. Use RDKit to run a Murcko Scaffold split to ensure structural diversity.
* **Pre-compute:** Run these molecules through MolFormer *once*. Save the resulting `[N, 768]` tensor to disk.
* **Integration:** Swap this pre-computed tensor in as our new `ctx_v` (Context Values) for the Hopfield retrieval. 

#### 3. Implement Leave-One-Task-Out (LOTO) Cross-Validation
We are ditching the fixed test-set approach for the gold-standard meta-learning evaluation.
* **The Pool:** Select 8-10 standard TDC Regression tasks.
* **The Loop:** Write a wrapper that iterates through the tasks. For each Fold:
    * Meta-train the outer loop on $N-1$ tasks.
    * Meta-test on the 1 held-out task at $K \in \{10, 20, 50\}$ shots.
* Record the RMSE and standard deviation for each fold.

#### 4. Baseline Generation (The MHNfs Run)
We need a strict apples-to-apples comparison against our primary target, MHNfs.
* **Action:** Clone the official MHNfs repository.
* **The Run:** Feed MHNfs the *exact same* K-shot support sets and MolFormer embeddings that our model uses. The only difference in the pipeline should be their blind Hopfield retrieval vs. our Two-Stage Gated retrieval.

#### 5. Assemble the Final Results Table
Format the final output table to explicitly separate the paradigms so reviewers don't accuse us of an unfair comparison.
* **Category 1: Strict Few-Shot (No Context Set)**
    * Pull numbers directly from the ADKF-IFT and Pin-Tuning papers (if our LOTO splits align with standard FS-Mol tasks). 
    * Include our own `maml_static_gnn` baseline.
* **Category 2: Context-Augmented (The Main Event)**
    * MHNfs (Your fresh run).
    * `maml_hopfield_no_gating`
    * `maml_hopfield_stage_one_gating_only` 
    * `maml_hopfield_stage_two_gating_only` 
    * `maml_hopfield_two_stage` (Our final model).