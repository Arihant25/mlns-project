# Reviewer 2 (Empirical / Systems-Focused)

## Summary

The paper reports leave-one-task-out (LOTO) evaluation on 7 TDC ADMET regression tasks at K=10, 20, 50 with 5 seeds per condition, comparing a proposed two-stage gated Hopfield+evidential meta-learner against five baselines. Tables give per-task RMSE with paired-t significance; figures show a shot-curve and a per-task relative-RMSE bar chart.

## Strengths

1. **LOTO evaluation with statistical significance is done right.** Paired-t tests over matched seeds, Wilcoxon as a robustness check, and a win-count summary table are all better than the usual "mean over 3 seeds" that is common in molecular ML. The reporting of `p < 0.10` as "approaching significance" under n=5 is honest.

2. **Per-seed standard deviations are surfaced in the main tables, not just means.** The observation that Hop-NoGate has 3x the variance of TwoStage on Caco2-K10 is exactly the kind of numerical signal that readers care about, and it is clearly displayed.

3. **The failure case is acknowledged and analysed, not hidden.** Clearance-Microsome is pointed at directly, the mechanism (task-level noise floor) is named, and the paper does not pretend the method is universal. This is unusually good scientific hygiene.

4. **Reproducibility signals are strong for the experiments that were run.** Fixed seeds (0-4), fixed orthogonal projection for Hopfield keys (seed 42), explicit hyperparameters and FOMAML partitioning, caching paths for context sets and embeddings. A motivated reader could re-run this.

## Weaknesses

1. **7 tasks is modest.** FS-Mol uses 157 test tasks precisely because single-task variance on ADMET-style endpoints is too high to reliably compare methods. 7 tasks × 5 seeds = 35 data points per method per K. The paired-t wins on Caco2 and PPBR are real, but the "significantly worse on Clearance-Microsome" is equally statistically supported by the same test. The paper should acknowledge that with n_tasks=7 the multiple-comparisons risk is real.

2. **No comparison to MHNfs itself.** The paper frames Hop-NoGate as an MHNfs analogue, but the original MHNfs paper's architecture includes context modules, class-conditional attention, and training-time context augmentation that this ablation does not implement. Running the actual MHNfs codebase (which is public) on the same 7 tasks would either strengthen the paper dramatically or expose a weakness. Currently the reader has to trust that "the MHNfs analogue inside our FOMAML loop" is a fair proxy, and I do not.

3. **No FS-Mol evaluation.** Given that the recent strong baselines (ADKF-IFT, Pin-Tuning, UniMatch) report on FS-Mol, not reporting there leaves the method's position in the overall leaderboard unclear. The Section 6.4 defence (FS-Mol is classification, NIG is regression) is fair but incomplete: the architecture has a DirHead form, and even a small FS-Mol subset at K=16 would calibrate the reader's expectations.

4. **Classification results are not reported in the main paper.** The authors say in Section 6.5 that preliminary classification runs showed MLP-indistinguishable AUROC. That is an important result and belongs in the paper even as a negative. Hiding it in a limitations paragraph creates an impression of selection bias.

5. **Compute/wall-time numbers are missing.** What's the meta-training cost per condition? A reader cannot tell whether TwoStage is 2x or 10x MLP. For a submission that emphasises practical ADMET, deployment cost matters.

6. **The VDss K=50 rows have unusually large standard deviations (4.35-5.06).** Looking at per-seed means, seed 2 for multiple conditions diverges to RMSE ~14 while others sit at ~24-28. This is called out neither in the paper nor the supplementary. The paired-t test is robust to this, but the means themselves are being pulled by seed 2 — the paper should either exclude seed 2 from VDss or note this explicitly.

7. **No ablation on context-set size.** Handoff notes suggest CTX_SIZE ranges from 1k (QUICK) to 50k (GPU). Running at 5k vs 50k would test whether retrieval quality scales, and is a cheap experiment.

8. **"Gradient-isolated" is claimed but not verified.** Did you check that gradients don't flow to `W_q` from the detached branch? A simple assertion in a training script or a gradient-norm ablation (zero gradient of `W_q` from support loss) would confirm the design.

## Questions for Authors

- How many GPU-hours does a full LOTO run take per condition?
- What is the wall-time cost of a meta-test episode at K=50 (one inner adapt + forward)? This determines deployment viability.
- Can you report classification results (HIA, CYP2C9-Substrate) with error bars in the main paper, even if they are negative?
- What is the effect of context-set size? A 1k vs 50k comparison at K=20 would sharpen the claim that retrieval quality correlates with bank size.
- Is seed 2 on VDss a known-pathological case? If so, report it; if not, investigate it.
- Why not include a Murcko-scaffold meta-test split (train on fold A's molecules, test on fold B's) as an additional robustness check beyond LOTO?
- Have you measured the memory footprint of the ChEMBL 50k context at inference time?

## Overall Assessment

The experiments are run with discipline, the statistical testing is honest, and the negative finding on Clearance is bravely reported. However, the scope (7 tasks, no FS-Mol, no real MHNfs, no classification in main body) is the dominant weakness. Every individual shortcut is defensible; the aggregate is that a reader cannot confidently place this method in the broader literature.

The main revision asks:
- Add an actual MHNfs reference run on the 7 tasks.
- Report classification results in the main paper, not a limitations paragraph.
- Add a context-size ablation.
- Add compute/wall-time numbers.
- Note the VDss seed-2 anomaly explicitly.

With those additions the paper becomes genuinely informative for few-shot ADMET practitioners. Without them it sits in a middle tier.

**Score: 5/10**
**Confidence: 5/5**
