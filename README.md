# Narrative-Aware Propaganda Detection

Investigating whether giving a large language model a news article's dominant **narrative** as additional context improves its ability to detect fine-grained **propaganda techniques**.

Term paper for *Advanced Topics in Computational Text and Media Sciences* (MSc NLP). Builds on:

- Sinelnik & Hovy (2024) — [*Narratives at Conflict: Computational Analysis of News Framing in Multilingual Disinformation Campaigns*](https://aclanthology.org/2024.acl-srw.21/)
- Gaeta et al. (2025) — [*Towards a LLM-based intelligent system for detecting propaganda within textual content*](https://doi.org/10.1016/j.compeleceng.2025.110765)

## Research question

> Does knowledge of an article's broader narrative/frame help an LLM identify the propaganda techniques used within it?

**RQ1** — Are particular propaganda techniques systematically associated with particular narratives?  
**RQ2** — Does providing narrative context improve LLM propaganda-technique classification vs. article-only input?  
**RQ3** — Which techniques benefit most (or least, or negatively) from narrative context?  

## Approach

```
SemEval-2020 Task 11 article subset
        │
        ├── gold propaganda-technique labels (existing)
        └── narrative labels (this project)
                │
                ▼
        LLM propaganda classifier
                │
     ┌──────────┴──────────┐
     ▼                     ▼
Article only         Article + Narrative
     │                     │
     └──────────┬──────────┘
                ▼
     Macro-F1 / per-technique F1
                ▼
         Error analysis
```

Narrative labels are added on top of an existing propaganda-annotated corpus (rather than trying to merge two separately-annotated datasets), which avoids both the dataset-alignment problem and the circularity risk of using one LLM to generate labels that the same LLM is then evaluated on.

## Repository structure

```
.
├── data/
│   ├── raw/                # SemEval-2020 Task 11 files (see Data & License below)
│   └── processed/          # topic-subset CSVs with narrative_label column
├── setup_semeval2020_subset.py   # pulls the dataset and builds the topic subset
├── narrative_labeling/     # scripts/notebooks for narrative annotation
├── experiments/
│   ├── exp1_association/   # RQ1: narrative x technique association analysis
│   └── exp2_classification/# RQ2/RQ3: LLM prompting, Colab notebooks
├── results/                # figures, tables, metrics
└── paper/                  # term paper draft and final PDF
```

## Data & license

This project uses **SemEval-2020 Task 11: Detection of Propaganda Techniques in News Articles** (Da San Martino, Barrón-Cedeño, Wachsmuth, Petrov & Nakov, 2020), obtained from the official Zenodo record.

- Source: [https://zenodo.org/records/3952415](https://zenodo.org/records/3952415)
- DOI: [10.5281/zenodo.3952415](https://doi.org/10.5281/zenodo.3952415)
- License: **CC BY 4.0** — redistribution is permitted with attribution, which is why the raw dataset files are included in `data/raw/` in this repository.

```bibtex
@InProceedings{SemEval20-11-DaSanMartino,
author = "Da San Martino, Giovanni and
  Barr\'{o}n-Cede\~no, Alberto and
  Wachsmuth, Henning and
  Petrov, Rostislav and
  Nakov, Preslav",
 title = "{SemEval}-2020 Task 11: {D}etection of Propaganda Techniques in News 
Articles",
 pages = "",
 abstract = "We describe the outcome of the SemEval 2020 Task 11 on the 
	detection of propaganda in news articles. We present two tasks. In the 
	first task, systems are asked to identify specific text spans in a free 
	text where propaganda is being applied. In the second task, systems are 
	asked to identify the propaganda technique being applied in a text span. We 
	describe the construction of the evaluation framework (dataset and 
	evaluation metrics) as well as the approaches explored by the different 
	participants. ",
crossref = "SemEval20"
}
```

Narrative labels added in `data/processed/` are original annotations produced for this project (see `narrative_labeling/` for methodology).

## Setup

```bash
git clone https://github.com/mohammedOwaiskh/narrative-aware-propaganda-detection.git
cd narrative-aware-propaganda-detection
pip install -r requirements.txt
python semeval_data_extractor.py
```

This downloads the dataset and writes a topic-filtered CSV (article id, text, gold propaganda techniques, empty `narrative_label` column) to `data/processed/`.

## Models

- Narrative-aware LLM classifier: small quantized open models (Qwen2.5-3B / Gemma-3-4B), run on free-tier Google Colab
- Optional baseline: SBERT embeddings + logistic regression, or RoBERTa fine-tuning

## Status

Work in progress — see `paper/` for the current draft and `results/` for the latest experiment outputs.

## Author

Mohammed Owais Khan, MSc Natural Language Processing, Universität Trier

## License

Code in this repository is released under the MIT License, except where noted above (dataset files retain their original CC BY 4.0 license from the SemEval-2020 Task 11 organizers).