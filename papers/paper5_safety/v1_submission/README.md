# Paper 5 v1 Submission - LaTeX Manuscript

## File Information
- **Filename**: Paper5_v1.tex
- **Created**: February 21, 2026
- **Lines**: 474
- **Size**: 29 KB

## Content Structure

### Sections
1. Abstract
2. Introduction
3. Methods (4 subsections)
4. Results (5 subsections)
5. Discussion (6 subsections including Future Directions)
6. Conclusion
7. Acknowledgments
8. Data Availability
9. References

### Figures (6 total)
1. fig1_safety_matrix.png - The 2×2 predictability matrix
2. fig2_llama_variability.png - Trial-level variability in Llama models
3. fig3_archetypes_embedding.png - Response distribution archetypes
4. fig4_one_dimension_failure.png - Why neither dimension alone suffices
5. fig5_position_var_ratio.png - Position-level Var_Ratio curves
6. fig6_deployment_flowchart.png - Deployment decision framework

### Tables (6 total)
1. Model set (8 medical models)
2. 16-element clinical rubric
3. Cross-model accuracy and variance with class assignments
4. Correlation analysis (Var_Ratio vs Accuracy)
5. Clinical element hit rates (DIVERGENT vs IDEAL)
6. Empirically motivated Var_Ratio thresholds

### Key Content
- **Four-class taxonomy**: IDEAL, EMPTY, DIVERGENT, RICH
- **Stochastic incompleteness** failure mode
- **Independence finding**: r=-0.24, p=0.56 (Var_Ratio vs Accuracy)
- **Zero hallucinations** in Llama trials (100 trials analyzed)
- **Llama anomaly**: Scout Var_Ratio=7.46, Maverick Var_Ratio=2.64
- **Dataset**: N=360 from Paper 2 (8 medical models at P30)

### Bibliography
14 references including:
- Papers 1-4, 6 from MCH Research Program
- Laban et al. 2025 (Lost in Conversation)
- Asgari et al. 2025 (clinical safety framework)
- Reimers & Gurevych 2019 (Sentence-BERT)
- Clinical AI references (Rajkomar, Singhal, Thirunavukarasu)

### Template Source
Based on Paper 4 template structure:
- Same package configuration
- Same formatting style
- Same hyperref settings
- Program box with context
- Professional layout

## Future Directions Section
Includes mention of CUD experiments "in preparation" with preliminary findings from 4 pilot models (DeepSeek V3.1, Gemini Flash, Llama 4 Maverick, Qwen3 235B).

## Compilation Notes
- Requires figures in ../figures/ directory (all 6 figures present)
- Uses natbib citation style
- Float package for [H] placement
- Ready for pdflatex compilation
