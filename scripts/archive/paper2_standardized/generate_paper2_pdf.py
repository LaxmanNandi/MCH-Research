"""
Generate Paper 2 PDF using fpdf2 (pure Python, no external dependencies).
Academic formatting with Times New Roman, figures, and appendix table.

DEPRECATED (February 2026): This script contains hardcoded v1 values that are
NO LONGER CORRECT. Gemini Flash Medical ΔRCI was corrected from -0.133 to +0.427
in v2. The authoritative Paper 2 document is now Paper2_Manuscript.tex (v2),
compiled via LaTeX. Do not use this script for generating submission PDFs.
"""
from pathlib import Path
from fpdf import FPDF

PAPER_DIR = Path(__file__).resolve().parent.parent / "papers" / "paper2_standardized"
FIG_DIR = PAPER_DIR / "figures"
OUTPUT_PDF = PAPER_DIR / "Paper2_Manuscript.pdf"


class AcademicPDF(FPDF):
    def __init__(self):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.set_auto_page_break(auto=True, margin=25)

    def header(self):
        if self.page_no() > 1:
            self.set_font('Times', 'I', 9)
            self.set_text_color(128)
            self.cell(0, 8, 'Laxman (2026) - Standardized Context Sensitivity Benchmark', align='C')
            self.ln(4)
            self.set_text_color(0)

    def footer(self):
        self.set_y(-15)
        self.set_font('Times', '', 10)
        self.set_text_color(128)
        self.cell(0, 10, str(self.page_no()), align='C')
        self.set_text_color(0)

    def section_heading(self, num, title):
        self.ln(6)
        self.set_font('Times', 'B', 14)
        self.cell(0, 8, f'{num}. {title}', new_x='LMARGIN', new_y='NEXT')
        self.set_draw_color(180)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def subsection_heading(self, num, title):
        self.ln(4)
        self.set_font('Times', 'BI', 12)
        self.cell(0, 7, f'{num} {title}', new_x='LMARGIN', new_y='NEXT')
        self.ln(2)

    def body_text(self, text):
        self.set_font('Times', '', 11)
        self.multi_cell(0, 5.5, text, new_x='LMARGIN', new_y='NEXT')
        self.ln(1)

    def bold_text(self, text):
        self.set_font('Times', 'B', 11)
        self.multi_cell(0, 5.5, text, new_x='LMARGIN', new_y='NEXT')
        self.ln(1)

    def bullet(self, text):
        self.set_font('Times', '', 11)
        x = self.get_x()
        self.cell(8, 5.5, '-')
        self.multi_cell(0, 5.5, text, new_x='LMARGIN', new_y='NEXT')

    def numbered_item(self, num, text):
        self.set_font('Times', '', 11)
        self.cell(8, 5.5, f'{num}.')
        self.multi_cell(0, 5.5, text, new_x='LMARGIN', new_y='NEXT')

    def add_figure(self, filename, caption, fig_num):
        fig_path = str(FIG_DIR / filename)
        # Check remaining space
        if self.get_y() > 160:
            self.add_page()
        self.ln(4)
        img_w = self.w - self.l_margin - self.r_margin
        self.image(fig_path, x=self.l_margin, w=img_w)
        self.ln(2)
        self.set_font('Times', 'I', 9)
        self.multi_cell(0, 4.5, f'Figure {fig_num}. {caption}', new_x='LMARGIN', new_y='NEXT')
        self.ln(4)


def build_pdf():
    pdf = AcademicPDF()
    pdf.add_page()

    # ---- TITLE ----
    pdf.set_font('Times', 'B', 18)
    pdf.multi_cell(0, 9, 'Standardized Context Sensitivity Benchmark\nAcross 25 LLM-Domain Configurations', align='C')
    pdf.ln(6)
    pdf.set_font('Times', '', 12)
    pdf.cell(0, 6, 'Dr. Laxman M M, MBBS', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Times', 'I', 11)
    pdf.cell(0, 6, 'Primary Health Centre Manchi, Karnataka, India', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Times', '', 11)
    pdf.cell(0, 6, 'February 2026', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(6)

    # ---- ABSTRACT ----
    pdf.set_draw_color(150)
    x0 = pdf.l_margin
    y0 = pdf.get_y()
    pdf.set_font('Times', 'B', 12)
    pdf.cell(0, 7, 'Abstract', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(2)
    pdf.set_font('Times', '', 10)
    abstract = (
        'We present a standardized cross-domain framework for measuring context sensitivity in large language models '
        '(LLMs) using the Delta Relational Coherence Index (dRCI). Across 25 model-domain runs (14 unique models, '
        '50 trials each, 112,500 total responses), we compare medical (closed-goal) and philosophical (open-goal) '
        'reasoning domains using a three-condition protocol (TRUE/COLD/SCRAMBLED). We find that: (1) both domains '
        'elicit robust positive context sensitivity (mean dRCI: philosophy=0.317, medical=0.308), with no significant '
        'domain-level difference (U=51, p=0.149); (2) medical domain exhibits substantially higher inter-model '
        'variance (SD=0.131 vs 0.045), driven by a Gemini Flash safety-filter anomaly (dRCI=-0.133); (3) vendor '
        'signatures show marginal differentiation (F(7,17)=2.31, p=0.075), with Moonshot (Kimi K2) showing highest '
        'context sensitivity and Google lowest; (4) the expected information hierarchy (dRCI_COLD > dRCI_SCRAMBLED) '
        'holds in 24/25 model-domain runs, validating that even scrambled context retains partial information; and '
        '(5) position-level analysis reveals domain-specific temporal signatures consistent with theoretical '
        'predictions. This dataset provides the first standardized benchmark for cross-domain context sensitivity '
        'measurement in state-of-the-art LLMs.'
    )
    pdf.multi_cell(0, 4.5, abstract, new_x='LMARGIN', new_y='NEXT')
    pdf.ln(2)
    pdf.set_font('Times', 'B', 10)
    pdf.cell(18, 4.5, 'Keywords: ')
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 4.5, 'Context sensitivity, dRCI, cross-domain AI evaluation, medical reasoning, philosophical reasoning, LLM benchmarking', new_x='LMARGIN', new_y='NEXT')
    y1 = pdf.get_y() + 2
    pdf.rect(x0, y0 - 2, pdf.w - pdf.l_margin - pdf.r_margin, y1 - y0 + 4)
    pdf.ln(4)

    # ---- 1. INTRODUCTION ----
    pdf.section_heading('1', 'Introduction')

    pdf.subsection_heading('1.1', 'Background')
    pdf.body_text(
        'Large language models increasingly serve as reasoning tools across diverse domains, from medical '
        'diagnostics to philosophical inquiry. In-context learning -- the ability to adapt behavior based on '
        'conversational history -- is fundamental to modern LLMs [1], yet how domain structure shapes this context '
        'sensitivity remains poorly understood. Current benchmarks focus primarily on accuracy and task completion [2], '
        'with context evaluation itself underdeveloped [3]. Following the operant tradition [4], we treat model outputs '
        'as behavioral data rather than cognitive states, measuring what models do with context rather than inferring '
        'internal representations.'
    )
    pdf.body_text(
        'Prior work [5] introduced the Delta Relational Coherence Index (dRCI) and demonstrated dramatic behavioral '
        'mode-switching between domains using 7 closed models. However, that study used aggregate metrics, mixed trial '
        'definitions, and lacked open-weight model comparisons.'
    )

    pdf.subsection_heading('1.2', 'Research Gap')
    pdf.body_text(
        'Current LLM benchmarks are increasingly saturated and redundant [2], measuring task accuracy rather than '
        'behavioral dynamics. No existing benchmark provides:'
    )
    pdf.bullet('Standardized cross-domain context sensitivity measurement')
    pdf.bullet('Unified methodology across open and closed architectures')
    pdf.bullet('Position-level temporal analysis across task types')
    pdf.bullet('Systematic vendor-level behavioral characterization')

    pdf.subsection_heading('1.3', 'Research Questions')
    pdf.numbered_item(1, 'RQ1: How does domain structure (closed-goal vs open-goal) affect aggregate context sensitivity?')
    pdf.numbered_item(2, 'RQ2: Do temporal dynamics differ systematically between domains at the position level?')
    pdf.numbered_item(3, 'RQ3: Are architectural differences (open vs closed models) domain-specific?')
    pdf.numbered_item(4, 'RQ4: Do vendor-level behavioral signatures persist across domains?')

    pdf.subsection_heading('1.4', 'Contributions')
    pdf.numbered_item(1, 'Standardized framework: Unified 50-trial methodology with corrected trial definition across 14 models and 2 domains')
    pdf.numbered_item(2, 'Cross-domain validation: First systematic comparison of dRCI in medical vs philosophical reasoning')
    pdf.numbered_item(3, 'Architectural diversity: Balanced open (7) and closed (5-6) model inclusion in both domains')
    pdf.numbered_item(4, 'Baseline dataset: 25 model-domain runs providing reproducible benchmarks for 14 state-of-the-art LLMs')
    pdf.numbered_item(5, 'Anomaly detection: Identification of safety-filter-induced context sensitivity inversion (Gemini Flash medical)')

    # ---- 2. RELATED WORK ----
    pdf.section_heading('2', 'Related Work')

    pdf.subsection_heading('2.1', 'Context Sensitivity in LLMs')
    pdf.body_text(
        'Transformer architectures process context through self-attention mechanisms [6], enabling in-context learning [1] '
        'that underpins modern LLM capabilities. However, measuring how models use conversational context -- beyond whether '
        'they produce correct answers -- remains underdeveloped [3]. Recent work on decoupling safety behaviors into '
        'orthogonal subspaces [7] provides independent evidence that model behaviors can be decomposed along interpretable '
        'dimensions, supporting our approach of isolating context sensitivity as a measurable behavioral axis.'
    )

    pdf.subsection_heading('2.2', 'Cross-Domain AI Evaluation')
    pdf.body_text(
        'Domain-specific evaluation has advanced significantly, with medical AI benchmarks demonstrating that LLMs can '
        'encode clinical knowledge [9] and safety alignment methods shaping model behavior through constitutional '
        'principles [10]. Yet cross-domain behavioral comparison remains rare: existing benchmarks (MMLU, HELM) measure '
        'accuracy within domains but do not track how the same model\'s behavioral dynamics shift across task structures. '
        'Our dRCI framework addresses this gap by providing a domain-agnostic metric that captures context sensitivity '
        'independent of correctness.'
    )

    pdf.subsection_heading('2.3', 'Paper 1 Foundation')
    pdf.body_text(
        'The Mirror-Coherence Hypothesis [5] introduced the dRCI metric and three-condition protocol '
        '(TRUE/COLD/SCRAMBLED), demonstrating domain-dependent behavioral mode-switching (Cohen\'s d > 2.7) across '
        '7 closed models. That study established the "presence > absence" principle -- that even scrambled context '
        'retains partial information -- but was limited to aggregate-only analysis, mixed trial methodology, and '
        'closed-weight models exclusively.'
    )

    # ---- 3. METHODOLOGY ----
    pdf.section_heading('3', 'Methodology')

    pdf.subsection_heading('3.1', 'Experimental Design')
    pdf.bold_text('Three-condition protocol applied to each trial:')
    pdf.bullet('TRUE: Model receives coherent 29-message conversational history before prompt')
    pdf.bullet('COLD: Model receives prompt with no prior context')
    pdf.bullet('SCRAMBLED: Model receives same 29 messages in randomized order before prompt')
    pdf.ln(2)
    pdf.set_font('Times', 'B', 11)
    pdf.cell(0, 6, 'dRCI = mean(RCI_TRUE) - mean(RCI_COLD)', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(2)
    pdf.body_text(
        'Where RCI is computed via cosine similarity of response embeddings using Sentence-BERT [8] '
        '(all-MiniLM-L6-v2, 384D). This embedding-based approach captures semantic similarity without requiring '
        'domain-specific annotation, enabling cross-domain comparison.'
    )

    pdf.subsection_heading('3.2', 'Domains')
    pdf.body_text('Medical (closed-goal): 52-year-old STEMI case with diagnostic/therapeutic targets. '
                  'Philosophy (open-goal): Consciousness inquiry with no single correct answer. '
                  'Both use 30 prompts per trial. Expected patterns: U-shaped + P30 spike (medical) vs Inverted-U (philosophy) [5].')

    pdf.subsection_heading('3.3', 'Models')
    pdf.body_text(
        '14 unique models across 25 model-domain runs from 8 vendors: OpenAI (GPT-4o, GPT-4o-mini, GPT-5.2), '
        'Anthropic (Claude Haiku, Claude Opus), Google (Gemini Flash), DeepSeek (V3.1), Moonshot (Kimi K2), '
        'Meta (Llama 4 Maverick, Llama 4 Scout), Mistral (Mistral Small 24B, Ministral 14B), Alibaba (Qwen3 235B). '
        'Medical: 13 models (6 closed + 7 open). Philosophy: 12 models (5 closed + 7 open). '
        '12 models appear in both domains (paired comparison).'
    )

    pdf.subsection_heading('3.4', 'Parameters')
    pdf.bullet('Trials per model: 50 (standardized), meeting empirically derived evaluation requirements [11]')
    pdf.bullet('Temperature: 0.7')
    pdf.bullet('Embedding model: sentence-transformers/all-MiniLM-L6-v2 (384D) [8]')
    pdf.bullet('API providers: Direct API (closed), Together AI (open)')
    pdf.bullet('Information-theoretic grounding: Position-level MI estimation between context and response [12]')

    pdf.subsection_heading('3.5', 'Data Scale')
    pdf.body_text('Unique models: 14. Model-domain runs: 25. Trials per run: 50. Prompts per trial: 30. '
                  'Conditions per trial: 3 (TRUE, COLD, SCRAMBLED). Total trials: 1,250. Total responses: 112,500.')

    # ---- 4. RESULTS ----
    pdf.section_heading('4', 'Results')

    pdf.subsection_heading('4.1', 'Dataset Overview')
    pdf.add_figure('fig1_dataset_overview.png',
                   'Mean dRCI by model and domain across 25 model-domain runs (14 unique models, 50 trials each).', 1)
    pdf.body_text(
        '23/25 model-domain runs show positive dRCI (context enhances coherence). Kimi K2 shows highest sensitivity '
        'in both domains (philosophy: 0.428, medical: 0.417). Gemini Flash medical is the sole negative outlier '
        '(dRCI = -0.133), attributed to safety-filter interference. Claude Opus appears only in medical domain.'
    )

    pdf.subsection_heading('4.2', 'Domain Comparison')
    pdf.add_figure('fig2_domain_comparison.png',
                   'Left: Violin plots comparing philosophy (n=12) and medical (n=13) dRCI distributions. '
                   'Right: Paired bar chart for 12 models tested in both domains.', 2)
    pdf.body_text(
        'Aggregate comparison: No significant difference between domains (Mann-Whitney U=51, p=0.149). '
        'Philosophy: mean dRCI = 0.317 +/- 0.045 (n=12). Medical: mean dRCI = 0.308 +/- 0.131 (n=13). '
        'Notable exceptions: Gemini Flash (divergence of 0.471), GPT-5.2 (higher in medical), '
        'Kimi K2 (consistently highest in both).'
    )

    pdf.subsection_heading('4.3', 'Vendor Signatures')
    pdf.add_figure('fig3_vendor_signatures.png',
                   'Mean dRCI by vendor, sorted by descending mean. Error bars show SEM. '
                   'ANOVA: F(7,17)=2.31, p=0.075.', 3)
    pdf.body_text(
        'One-way ANOVA across 8 vendors: F(7,17) = 2.31, p = 0.075 (marginal significance). '
        'Ranking: (1) Moonshot 0.423, (2) Mistral 0.352, (3) Anthropic 0.336, (4) Alibaba 0.325, '
        '(5) DeepSeek 0.312, (6) OpenAI 0.310, (7) Meta 0.301, (8) Google 0.103. '
        'Google\'s low ranking is entirely driven by the Gemini Flash medical anomaly.'
    )

    pdf.subsection_heading('4.4', 'Position-Level Patterns')
    pdf.add_figure('fig4_position_patterns.png',
                   'Position-level dRCI trajectories across 30 prompt positions. Left: Philosophy. '
                   'Right: Medical. Bold lines show domain mean; thin lines show individual models.', 4)
    pdf.body_text(
        'Philosophy domain (12 models): Noisy but elevated sensitivity, slight upward trend, no dramatic P30 effect. '
        'Medical domain (12 models with position data): Higher amplitude oscillations, several models show elevated P30, '
        'greater inter-model variability. Patterns consistent with theoretical predictions (inverted-U philosophy, '
        'U-shaped medical).'
    )

    pdf.subsection_heading('4.5', 'Information Hierarchy')
    pdf.add_figure('fig5_information_hierarchy.png',
                   'dRCI computed with SCRAMBLED vs COLD baselines. Expected hierarchy: dRCI_COLD > dRCI_SCRAMBLED. '
                   'Hierarchy holds in 24/25 testable runs (96%).', 5)
    pdf.body_text(
        'The theoretical prediction from [5] -- that scrambled context should retain partial information compared to '
        'no context -- was tested across 25 model-domain runs. Logic: If scrambled retains partial info, SCRAMBLED '
        'responses should be closer to TRUE than COLD responses are, yielding dRCI_COLD > dRCI_SCRAMBLED. '
        'Observed: Hierarchy holds in 24/25 runs (96%). This strongly validates the "presence > absence" claim. '
        'Sole exception: Gemini Flash medical, where safety filters distort the COLD baseline.'
    )

    pdf.subsection_heading('4.6', 'Model Rankings')
    pdf.add_figure('fig6_model_rankings.png',
                   'Model rankings by mean dRCI with 95% confidence intervals. Left: Philosophy (12 models). '
                   'Right: Medical (13 models). (C)=Closed, (O)=Open. Dashed red line shows domain mean.', 6)
    pdf.body_text(
        'Philosophy top 3: (1) Kimi K2 (O): 0.428, (2) Ministral 14B (O): 0.373, (3) Gemini Flash (C): 0.338. '
        'Medical top 3: (1) Kimi K2 (O): 0.417, (2) Ministral 14B (O): 0.391, (3) GPT-5.2 (C): 0.379. '
        'Cross-domain consistency: Kimi K2 and Ministral 14B rank #1 and #2 in both domains.'
    )

    # ---- 5. DISCUSSION ----
    pdf.section_heading('5', 'Discussion')

    pdf.subsection_heading('5.1', 'Domain Invariance of Aggregate dRCI')
    pdf.body_text(
        'The lack of significant domain-level difference (p=0.149) suggests that aggregate context sensitivity is '
        'relatively domain-invariant. This supports dRCI as a generalizable metric rather than a domain-specific '
        'artifact. However, the medical domain\'s much higher variance (SD=0.131 vs 0.045) indicates that closed-goal '
        'tasks create more extreme behavioral differentiation between models.'
    )

    pdf.subsection_heading('5.2', 'The Gemini Flash Medical Anomaly')
    pdf.body_text(
        'Gemini Flash shows the most dramatic domain effect: positive in philosophy (0.338) but negative in medical '
        '(-0.133). This is attributed to safety filters -- shaped by constitutional AI principles [10] and RLHF '
        'training [14] -- that activate on medical content, disrupting coherent context utilization. This finding '
        'aligns with recent evidence that quality benchmarks do not predict safety behavior [13], and has important '
        'implications for medical AI deployment [9]: safety mechanisms can paradoxically reduce response quality by '
        'interfering with context integration.'
    )

    pdf.subsection_heading('5.3', 'Open vs Closed Architecture')
    pdf.body_text(
        'Open models show competitive or superior context sensitivity in both domains: Medical open mean: 0.348 vs '
        'closed mean: 0.257 (excluding Gemini Flash: 0.335). Philosophy open mean: 0.325 vs closed mean: 0.306. '
        'This suggests that open-weight models, despite generally smaller parameter counts, can achieve comparable '
        'context sensitivity.'
    )

    pdf.subsection_heading('5.4', 'Vendor Clustering')
    pdf.body_text(
        'The marginal vendor effect (p=0.075) suggests that organizational-level design decisions -- training data, '
        'RLHF procedures [14], safety tuning [10] -- create subtle but potentially meaningful behavioral signatures. '
        'Moonshot\'s consistent dominance and Google\'s safety-filter-driven anomaly represent the extremes.'
    )

    pdf.subsection_heading('5.5', 'Information Hierarchy Validation')
    pdf.body_text(
        'The near-universal confirmation of the expected hierarchy (dRCI_COLD > dRCI_SCRAMBLED in 24/25 runs) is a '
        'significant methodological validation. It confirms that scrambled context retains partial information -- even '
        'disrupted conversational structure provides extractable signal. This validates the three-condition protocol as '
        'a well-ordered measurement framework and confirms the "presence > absence" principle [5] at scale.'
    )

    pdf.subsection_heading('5.6', 'Limitations')
    pdf.numbered_item(1, 'Single scenario per domain: One medical case (STEMI) and one philosophical topic (consciousness)')
    pdf.numbered_item(2, 'Embedding model ceiling: all-MiniLM-L6-v2 [8] may not capture all semantic distinctions')
    pdf.numbered_item(3, 'Temperature fixed at 0.7: Other settings may yield different patterns')
    pdf.numbered_item(4, 'Claude Opus: Medical only (absent from philosophy); recovered data lacks response text')
    pdf.numbered_item(5, 'Position-level noise: 50 trials provide limited statistical power for 30-position analysis')

    # ---- 6. CONCLUSION ----
    pdf.section_heading('6', 'Conclusion')
    pdf.body_text(
        'This study establishes a standardized cross-domain framework for measuring context sensitivity in LLMs. '
        'Across 14 models and 112,500 responses, we find that:'
    )
    pdf.numbered_item(1, 'Context sensitivity is robust and positive for nearly all models in both domains (23/25 runs)')
    pdf.numbered_item(2, 'Domain structure shapes variance, not mean: Medical and philosophical domains yield similar average dRCI but dramatically different inter-model spread')
    pdf.numbered_item(3, 'Safety mechanisms can invert context sensitivity: Gemini Flash medical anomaly demonstrates deployment-critical risk')
    pdf.numbered_item(4, 'Open models compete with closed: No systematic architectural disadvantage for open-weight models')
    pdf.numbered_item(5, 'Vendor signatures are detectable: Organizational design choices create marginal but consistent behavioral patterns')
    pdf.ln(2)
    pdf.body_text(
        'This dataset and methodology -- building on the dRCI framework [5] and addressing gaps in current LLM '
        'evaluation [2, 3] -- provide the foundation for deeper analyses of temporal dynamics (Paper 3) and '
        'information-theoretic mechanisms (Paper 4).'
    )

    # ---- DATA AVAILABILITY ----
    pdf.section_heading('', 'Data Availability')
    pdf.body_text('All experimental data and analysis code are available at: https://github.com/LaxmanNandi/MCH-Experiments')

    # ---- REFERENCES ----
    pdf.section_heading('', 'References')
    pdf.set_font('Times', '', 9)
    refs = [
        '[1] Brown, T., Mann, B., Ryder, N., et al. (2020). Language Models are Few-Shot Learners. Advances in Neural Information Processing Systems, 33. arXiv:2005.14165.',
        '[2] Subramani, N., Srinivasan, R., & Hovy, E. (2025). SimBA: Simplifying Benchmark Analysis. Findings of EMNLP 2025. DOI: 10.18653/v1/2025.findings-emnlp.711.',
        '[3] Xu, Y., et al. (2025). Does Context Matter? ContextualJudgeBench for Evaluating LLM-based Judges. Proceedings of ACL 2025. DOI: 10.18653/v1/2025.acl-long.470.',
        '[4] Skinner, B. F. (1957). Verbal Behavior. Copley Publishing Group.',
        '[5] Laxman, M M. (2026). Context Curves Behavior: Measuring AI Relational Dynamics with dRCI. Preprints.org. DOI: 10.20944/preprints202601.1881.v2.',
        '[6] Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems, 30. arXiv:1706.03762.',
        '[7] Mou, X., et al. (2025). Decoupling Safety into Orthogonal Subspace. arXiv:2510.09004.',
        '[8] Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. Proceedings of EMNLP 2019. arXiv:1908.10084.',
        '[9] Singhal, K., Azizi, S., Tu, T., et al. (2023). Large Language Models Encode Clinical Knowledge. Nature, 620, 172-180.',
        '[10] Bai, Y., Jones, A., Ndousse, K., et al. (2022). Constitutional AI: Harmlessness from AI Feedback. arXiv:2212.08073.',
        '[11] NIH PMC. (2025). Empirically derived evaluation requirements for responsible deployments of AI in safety-critical settings. npj Digital Medicine. DOI: 10.1038/s41746-025-01784-y.',
        '[12] Nguyen, T., et al. (2025). A Framework for Neural Topic Modeling with Mutual Information. Neurocomputing. DOI: 10.1016/j.neucom.2025.130420.',
        '[13] Datasaur. (2025). LLM Scorecard 2025. https://datasaur.ai/blog-posts/llm-scorecard-22-8-2025.',
        '[14] Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35. arXiv:2203.02155.',
    ]
    for ref in refs:
        pdf.multi_cell(0, 4, ref, new_x='LMARGIN', new_y='NEXT')
        pdf.ln(1)

    # ---- APPENDIX ----
    pdf.add_page()
    pdf.set_font('Times', 'B', 14)
    pdf.cell(0, 8, 'Appendix A: Complete Per-Model Statistics (50 trials each)', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(4)

    pdf.set_font('Times', 'B', 8)
    col_widths = [38, 22, 14, 10, 24, 14, 16]
    headers = ['Model', 'Domain', 'Type', 'n', 'Mean dRCI', 'SD', '95% CI']
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 5, h, border=1, align='C')
    pdf.ln()

    data = [
        ('GPT-4o', 'Philosophy', 'Closed', '50', '0.283', '0.011', '+/-0.003'),
        ('GPT-4o-mini', 'Philosophy', 'Closed', '50', '0.269', '0.009', '+/-0.002'),
        ('GPT-5.2', 'Philosophy', 'Closed', '50', '0.308', '0.015', '+/-0.004'),
        ('Claude Haiku', 'Philosophy', 'Closed', '50', '0.331', '0.012', '+/-0.003'),
        ('Gemini Flash', 'Philosophy', 'Closed', '50', '0.338', '0.022', '+/-0.006'),
        ('DeepSeek V3.1', 'Philosophy', 'Open', '50', '0.304', '0.014', '+/-0.004'),
        ('Kimi K2', 'Philosophy', 'Open', '50', '0.428', '0.022', '+/-0.006'),
        ('Llama 4 Maverick', 'Philosophy', 'Open', '50', '0.269', '0.012', '+/-0.003'),
        ('Llama 4 Scout', 'Philosophy', 'Open', '50', '0.298', '0.011', '+/-0.003'),
        ('Ministral 14B', 'Philosophy', 'Open', '50', '0.373', '0.015', '+/-0.004'),
        ('Mistral Small 24B', 'Philosophy', 'Open', '50', '0.281', '0.009', '+/-0.003'),
        ('Qwen3 235B', 'Philosophy', 'Open', '50', '0.322', '0.009', '+/-0.003'),
        ('', '', '', '', '', '', ''),
        ('GPT-4o', 'Medical', 'Closed', '50', '0.299', '0.010', '+/-0.003'),
        ('GPT-4o-mini', 'Medical', 'Closed', '50', '0.319', '0.010', '+/-0.003'),
        ('GPT-5.2', 'Medical', 'Closed', '50', '0.379', '0.021', '+/-0.006'),
        ('Claude Haiku', 'Medical', 'Closed', '50', '0.340', '0.010', '+/-0.003'),
        ('Claude Opus', 'Medical', 'Closed', '50', '0.339', '0.017', '+/-0.005'),
        ('Gemini Flash', 'Medical', 'Closed', '50', '-0.133', '0.026', '+/-0.007'),
        ('DeepSeek V3.1', 'Medical', 'Open', '50', '0.320', '0.010', '+/-0.003'),
        ('Kimi K2', 'Medical', 'Open', '50', '0.417', '0.016', '+/-0.004'),
        ('Llama 4 Maverick', 'Medical', 'Open', '50', '0.316', '0.012', '+/-0.003'),
        ('Llama 4 Scout', 'Medical', 'Open', '50', '0.323', '0.011', '+/-0.003'),
        ('Ministral 14B', 'Medical', 'Open', '50', '0.391', '0.014', '+/-0.004'),
        ('Mistral Small 24B', 'Medical', 'Open', '50', '0.365', '0.015', '+/-0.004'),
        ('Qwen3 235B', 'Medical', 'Open', '50', '0.328', '0.010', '+/-0.003'),
    ]

    pdf.set_font('Times', '', 8)
    for row in data:
        if row[0] == '':
            pdf.ln(2)
            continue
        for i, val in enumerate(row):
            pdf.cell(col_widths[i], 4.5, val, border=1, align='C' if i >= 3 else 'L')
        pdf.ln()

    # Save
    pdf.output(str(OUTPUT_PDF))
    size_kb = OUTPUT_PDF.stat().st_size / 1024
    print(f"PDF generated: {OUTPUT_PDF} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    build_pdf()
