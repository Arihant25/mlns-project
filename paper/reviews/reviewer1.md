# Reviewer 1 (Theory-Focused / Skeptical)

## Summary

The paper proposes a meta-learning architecture for few-shot ADMET regression that combines Modern Hopfield context retrieval with an evidential uncertainty gate. Two gates are applied multiplicatively at the edges of a learned episode-level adjacency: (i) a zero-parameter entropy gate derived from the Shannon entropy of Hopfield attention, and (ii) a Normal-Inverse-Gamma aleatoric-uncertainty gate with the critical detail that the NIG head takes detached features, isolating it from Hopfield retrieval gradients. Evaluation is done by leave-one-task-out on 7 TDC ADMET regression tasks with 5 meta-training seeds.

## Strengths

1. **The stop-gradient design is non-obvious and well-motivated.** Decoupling retrieval learning from within-episode label-noise modelling via `stop()` on the NIG head's input is a clean trick and the paper flags it explicitly as a design discovered through instability of the naive variant. This is a contribution worth highlighting.

2. **The entropy gate is parameter-free and plausibly OOD.** Using `1 - H(A_hop_i)/log(M)` as a gating signal is elegant: the normaliser is exact (log of the uniform distribution over M atoms), so the gate is bounded in [0,1] without tuning. The idea that retrieval diffusion is a proxy for OOD is defensible and fits with the Modern Hopfield literature's storage-capacity framing.

3. **The ablation ladder is clean.** MLP → StaticGNN → DenseGSL → Hop-NoGate → Hop-Entropy → TwoStage is well-structured and isolates the contribution of each mechanism.

## Weaknesses

1. **Conflation of aleatoric and epistemic uncertainty.** The paper uses `u = β/(α-1)`, which is the NIG *aleatoric* uncertainty. It then argues this suppresses molecules "whose own labels the model cannot explain." But aleatoric uncertainty is *irreducible data noise*, not model ignorance — for cross-task molecules or scaffold-novel molecules, the epistemic uncertainty `β/(ν(α-1))` is the correct quantity. The paper should clarify whether this design choice is intentional (and if so, why aleatoric is preferred over epistemic in this context) or acknowledge that using aleatoric in this role is an approximation.

2. **Entropy gate as "OOD detector" is empirically asserted but not empirically verified.** The paper claims `G_ctx` "serves as an OOD detector" but provides no direct evidence: no plot of gate values against some OOD-ness metric (e.g., Euclidean distance to nearest context molecule), no correlation between gate values and prediction error, no reliability diagram. This claim should either be demonstrated or softened.

3. **"Blind Hopfield retrieval never wins" overstates the finding.** The Hop-NoGate condition is not MHNfs. It is MHNfs-adjacent — it uses the same retrieval mechanism but trains under the same FOMAML loop as the proposed method, with the same MolAttention layer. A real MHNfs run would use the original paper's full cross-attention stack, context modules, and training procedure. The paper should not claim comparison to MHNfs itself without that apples-to-apples run. Call it the "unconditioned Hopfield" ablation and be explicit.

4. **No theoretical justification for multiplicative gate combination.** Why `G = G_ctx * G_ep` and not `min(G_ctx, G_ep)`, `G_ctx + G_ep - G_ctx·G_ep`, or a learned combination? A sentence or short argument (e.g., both gates must pass for the edge to survive, consistent with the AND-semantics of "reliable retrieval AND reliable support") would strengthen the design.

5. **The claim of "first architecture combining retrieval and evidential GSL" is not defended against UnGSL.** The paper cites UnGSL (Han et al. 2025) as "combining evidential uncertainty with graph structure learning," but doesn't distinguish the current work from UnGSL beyond the meta-learning framing. A sentence on architectural differences would clarify novelty.

6. **FOMAML parameter partition is asymmetric without clear justification.** The Hopfield query/blend parameters go in `θ_out`, while the evidential gate's `γ` goes in `θ_in`. Why should the evidential scaling adapt to the support set but the retrieval not? If retrieval is task-agnostic, fine; if adapting it per task would help, the partition is a weakness. A sensitivity ablation here would be useful.

## Questions for Authors

- Did you try using `β/(ν(α-1))` (epistemic) rather than `β/(α-1)` (aleatoric) as the gating signal? If so, what happened?
- What is the distribution of `G_ctx` values across the held-out task versus the meta-training tasks? Do they differ systematically as the "OOD detector" framing would predict?
- In Section 4.3 you define Hop-NoGate as "the MHNfs analogue inside our FOMAML loop." Can you quantify how much of the Hop-NoGate vs MHNfs gap is due to architectural differences rather than the training loop?
- Is the observed Clearance-Microsome failure mode reproducible if you swap the backbone (e.g., ECFP counts + MLP rather than MolFormer)?
- The entropy gate is applied edge-wise (`G_i * G_j`). Did you consider sender-only gating (`G_i` alone)? Node-level gating would halve the contribution to the dynamic range.

## Overall Assessment

The architectural idea is genuine, the stop-gradient trick is non-obvious and well-argued, and the ablation table is honest. The main theoretical concerns are (1) the aleatoric/epistemic conflation in the evidential gate, and (2) the lack of direct empirical validation of the OOD-detection claim. Both are addressable by rewrites plus one additional experiment each.

The scope (7 regression tasks, no FS-Mol, no real MHNfs) is a genuine limitation that the paper acknowledges but cannot remove at submission time.

**Score: 6/10**
**Confidence: 4/5**
