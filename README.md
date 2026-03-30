# Evaluating Large Language Models for Strategic Knowledge Extraction in Capability-Based Planning

This repository contains the source code and experimental pipeline for the paper:

> **Evaluating Large Language Models for Strategic Knowledge Extraction in Capability-Based Planning**
> Hein Kolk, Julia Garcia-Fernandez, Julia Bronkhorst, Roos Bakker
> *TNO, The Hague, The Netherlands*
> Proceedings of LREC-COLING 2026

## Overview

Large national organizations like the Dutch National Police use Capability-Based Planning (CBP) to guide strategic decision-making. CBP requires a comprehensive understanding of the strategic landscape, yet relevant knowledge is scattered across hundreds of unstructured internal documents. This repository provides:

- A method for automatically extracting strategic entities (Strategies, Goals, Trends) from policy documents and structuring them into a CBP-oriented knowledge graph.
- **StratID**, a newly created corpus of annotated strategic police documents (8 sections from the public *Begroting en beheerplan politie 2025-2029* document), annotated by four domain experts and three LLMs.
- The full evaluation pipeline, including Inter-Annotator Agreement (IAA) analysis and the Alternative Annotator Test (Alt-Test).

The study evaluates GPT-4.1, GPT-4.1-mini, GPT-5-mini, Mistral-Small-3.1-24B, Gemma3-27B, and Qwen3-4B on the extraction and capability-linking tasks, and applies the Alt-Test to assess whether an LLM can serve as a reliable substitute for a human domain expert.

## Repository Structure

```
capo/
|-- data/                           # Input data (StratID corpus, capabilities list)
|   `-- capabilities_final_list.xlsx
|-- docs/                           # Annotation guidelines (human & LLM)
|   `-- annotation_instructions.docx
|-- model/                          # LLMExtractor class (Azure OpenAI + Ollama)
|   `-- extractor.py
|-- notebooks/                      # Jupyter notebook for LLM extraction
|   `-- extract_and_evaluate.ipynb
|-- schemas/                        # Pydantic output schemas (Strategy, Goal, Trend, etc.)
|-- scripts/
|   |-- final_pipeline/             # ALT-Test preparation and execution (Table 1)
|   |   |-- analysis_logic.py
|   |   |-- analysis_logic_links.py
|   |   `-- alt_test.py
|   |-- iaa_sensitivity_analysis/   # Threshold sensitivity analysis (Figure 2)
|   |   |-- sensitivity_analysis.py
|   |   `-- plot_results.py
|   `-- supporting_utilities/       # Data formatting, IAA calculation, descriptive stats
|       |-- data_formatting.py
|       |-- data_cleaning.py
|       |-- iaa_calculator.py
|       |-- analyze_human_stats_411.py
|       `-- count_entities_by_threshold.py
|-- utils/                          # I/O helpers and annotation bundle formatter
|-- annotation_instructions.py      # Full annotation guidelines as a system prompt
|-- prompts.py                      # LLM prompt templates for each extraction step
|-- requirements.txt
`-- LICENSE
```

> **Note on data:** Annotation files (JSON) are excluded from version control (see `.gitignore`). To reproduce the results, place the human annotation export from Label Studio in `data/human_annotations/original_annotations.json` and the LLM annotation outputs in `data/llm_annotations/`. The StratID corpus and annotation data are available upon request.

## Setup

**1. Clone the repository:**
```bash
git clone https://github.com/capopaper/capo.git
cd capo
```

**2. Create and activate a virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
python -m spacy download nl_core_news_sm
```

**4. Configure API credentials** (required for Step 1 only):

Create a `.env` file in the project root:
```
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
AZURE_API_VERSION=...
# For local Ollama models:
OLLAMA_ENDPOINT=...
```

## Reproducing the Paper's Results

All commands should be run from the root `capo/` directory.

### Step 1 -- LLM-Based Knowledge Extraction

Open and run `notebooks/extract_and_evaluate.ipynb`.

This notebook instantiates `LLMExtractor` and runs the full annotation pipeline (strategy, goal, and trend extraction; grouping; and capability linking) for each document segment. Raw LLM annotation bundles are written to `output/`.

### Step 2 -- Combine Annotations

```bash
python scripts/supporting_utilities/data_formatting.py
```

Reads the Label Studio human annotation export and the LLM annotation output and merges them into a single unified file.

- **Input:** `data/human_annotations/original_annotations.json`, `data/llm_annotations/azure_gpt-4.1.json`
- **Output:** `output/annotations_cleaned.json`

### Step 3 -- Sensitivity Analysis (Figure 2)

```bash
python scripts/iaa_sensitivity_analysis/sensitivity_analysis.py
python scripts/iaa_sensitivity_analysis/plot_results.py
```

Sweeps the semantic similarity threshold from 0.45 to 0.95, computes Krippendorff's alpha and pairwise Jaccard index at each level, and generates Figure 2 of the paper.

- **Output:** `output/iaa_thresh_single_column.png`

### Step 4 -- ALT-Test Evaluation (Table 1)

```bash
# Prepare concept-level annotation matrices for entities (Strategies, Goals, Trends)
python scripts/final_pipeline/analysis_logic.py

# Prepare link-level annotation matrices (Strategy -> Capability/Goal/Trend)
python scripts/final_pipeline/analysis_logic_links.py

# Run the Alternative Annotator Test
python scripts/final_pipeline/alt_test.py
```

Produces the winning rate (omega) and advantage probability (rho) values reported in Table 1 across epsilon in {0.1, 0.2, 0.3, 0.4}. Results are printed to console.

### Step 5 -- Descriptive Statistics (Optional)

```bash
python scripts/supporting_utilities/analyze_human_stats_411.py
python scripts/supporting_utilities/count_entities_by_threshold.py
```

Prints the annotation summary statistics reported in the appendix.

## Key Design Decisions

| Component | Choice | Reason |
|---|---|---|
| Semantic matching | `paraphrase-multilingual-MiniLM-L12-v2` + cosine similarity | Handles multilingual (Dutch) free-text spans across annotators |
| Threshold tau | 0.8 | Compromise between entity-level alpha and capability-link Jaccard (see Figure 2) |
| LLM output format | Pydantic-constrained structured outputs | Ensures schema compliance; directly mirrors the ontology |
| Annotation approach | Task decomposition + output chaining | Reduces cognitive load; enables per-step ablation |
| Evaluation | Alt-Test (Calderon et al., 2025) | Designed for subjective tasks without a gold standard |

## Citation

If you use this code or the StratID corpus, please cite:

```bibtex
@inproceedings{kolk2026capo,
  title     = {Evaluating Large Language Models for Strategic Knowledge Extraction in Capability-Based Planning},
  author    = {Kolk, Hein and Garc{\'i}a-Fern{\'a}ndez, Julia and Bronkhorst, Julia and Bakker, Roos},
  booktitle = {Proceedings of LREC-COLING 2026},
  year      = {2026},
}
```

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
