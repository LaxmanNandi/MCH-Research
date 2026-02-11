# Paper 4: Claims and Evidence Table

| Claim | Evidence | Strength | Limits / Confounds | Next Validation |
|------|----------|----------|--------------------|-----------------|
| DRCI tracks an MI proxy (entanglement signal) | Pooled correlation r = 0.74, p = 3.0e-42 over N = 240 model-position points (8 model-domain runs × 30 positions) | Strong | Proxy relies on variance ratio; may miss non-linear MI effects | Recompute with alternative MI estimators or embedding models |
| Entanglement is bidirectional (convergent vs divergent) | Observed Var_Ratio < 1 with positive DRCI (convergent) and Var_Ratio > 1 with negative DRCI (divergent) | Strong | Requires explicit sign convention; thresholds may be dataset-specific | Test across additional domains and models |
| SOVEREIGN corresponds to divergent entanglement | Negative DRCI co-occurs with Var_Ratio > 1 in multiple model-domain runs | Moderate | Needs prevalence statistics across all 22 model-domain runs | Expand to full model set; report prevalence by domain |
| Llama safety anomaly at medical P30 | Var_Ratio = 2.64 (Maverick) and 7.46 (Scout) with negative DRCI at summarization | Strong (within current dataset) | Based on a single task position (P30) and single domain | Evaluate P5-P30 Type 2 prompts and other medical cases |
| Medical domain increases variance on average | Mean Var_Ratio ~ 1.30 vs philosophy ~ 1.01 | Moderate | Domain choices are narrow (STEMI vs consciousness) | Add new medical cases and non-medical domains |
| Variance ratio is sufficient for entanglement screening | DRCI correlates strongly with MI_Proxy defined by Var_Ratio | Moderate | Sufficiency depends on embedding and trial variance | Compare to k-NN entropy and mutual information baselines |
| Two model classes emerge (convergent vs divergent at Type 2) | Convergent models show Var_Ratio < 1 at P30; divergent models show Var_Ratio > 1 | Moderate | May be prompt-specific or architecture-specific | Test with alternative Type 2 prompts and positions |
| Predictability reframe (DRCI as predictability change) | DRCI monotonic with variance reduction/increase | Strong (conceptual + empirical) | Requires careful framing in safety vs creativity contexts | Validate with human eval and task success metrics |
| ESI is a useful stability metric | ESI cleanly separates extreme divergence (ESI < 1) from stable models | Exploratory | Thresholds are provisional | Calibrate thresholds with human safety outcomes |
