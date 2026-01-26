# ΔRCI Visualization: From Words to Vectors

Interactive visualization explaining how text embeddings and cosine similarity power the ΔRCI metric.

## View Live
[https://laxmannandi.github.io/MCH-Research/paper1/visualizations/embedding_cosine_drci_explainer.html](https://laxmannandi.github.io/MCH-Research/paper1/visualizations/embedding_cosine_drci_explainer.html)

## Stages Explained

| Stage | Concept |
|-------|---------|
| 01 - Tokenization | Text → Token units |
| 02 - Embedding Lookup | Tokens → Pre-trained vectors (384-dim) |
| 03 - Vector Space | Similar meanings → Similar directions |
| 04 - Cosine Similarity | Angle between vectors = semantic similarity |

## Connection to ΔRCI
```
ΔRCI = RCI_contextual − RCI_baseline
```

- **CONVERGENT (+ve):** Model output aligns with conversation history
- **SOVEREIGN (−ve):** Model ignores conversation history

## Citation

If using this visualization, please cite:
> Dr. Laxman M M. "Context Curves Behavior: Measuring AI Relational Dynamics with ΔRCI"
> Preprints 2026, 195793. DOI: 10.20944/preprints202501.1881.v1

## Tech

Pure HTML/CSS - no dependencies. Open in any browser.
