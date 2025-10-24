# Evaluating LLMs for Strategic Knowledge Extraction in Capability-Based Planning

This repository contains the source code, data, and experimental pipeline for the paper: "Evaluating Large Language Models for Strategic Knowledge Extraction in Capability-Based Planning".

The full paper is available for review in the repository.

## Repository Structure

```
capo/
├── data/ # Input data (human & LLM annotations, capabilities)
├── model/ # Contains the core LLMExtractor class
├── notebooks/ # Jupyter notebook for the extraction process
├── output/ # All generated files (e.g., plots, test data)
├── schemas/ # Pydantic schemas for data validation
├── scripts/ # All Python analysis scripts
├── utils/ # Helper functions for formatting and I/O
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/capo.git
    cd capo
    ```

2.  **Create a Python Virtual Environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    python -m spacy download nl_core_news_sm
    ```

## How to Reproduce the Paper's Results

The analysis is a multi-step process. Please run the notebook and scripts in the following order.

**Important:** All commands should be run from the root `capo/` directory.

### Step 1: LLM-Based Knowledge Extraction

*   **Action:** Open and run the Jupyter Notebook `notebooks/extract_and_evaluate.ipynb`.
*   **Purpose:** This notebook uses LLMs to extract strategies, goals, and trends from text documents.
*   **Input:** Text files located in `data/`.
*   **Output:** Generates raw LLM annotation JSON files into the `output/` directory.

### Step 2: Format All Annotations for Analysis

*   **Action:** Run the data formatting script.
    ```bash
    python scripts/supporting_utilities/data_formating.py
    ```
*   **Purpose:** This script combines the raw human annotations from `data/human_annotations/` and the generated LLM annotations into a single, unified file.
*   **Output:** Creates `output/annotations_cleaned.json`, which is the input for all subsequent analysis steps.

### Step 3: Generate Sensitivity Analysis Plot (Figure 2)

*   **Action:** Run the analysis and plotting scripts sequentially.
    ```bash
    python scripts/iaa_sensitivity_analysis/sensitivity_analysis.py
    python scripts/iaa_sensitivity_analysis/plot_results.py
    ```
*   **Purpose:** The first script calculates IAA scores across different similarity thresholds. The second script generates the plot shown in Figure 2 of the paper.
*   **Output:** Saves plot to `output/iaa_thresh_single_column.png`.

### Step 4: Run the ALT-Test Evaluation (Table 1)

*   **Action:** First, prepare the formatted data for the ALT-Test, then run the test.
    ```bash
    # 1. Prepare data for entities (Strategies, Goals, Trends)
    python scripts/final_pipeline/analysis_logic.py

    # 2. Prepare data for links (Strategy -> Capability, etc.)
    python scripts/final_pipeline/analysis_logic_links.py

    # 3. Run the ALT-Test
    python scripts/final_pipeline/alt_test.py
    ```
*   **Purpose:** These scripts create the specific JSON files required for the ALT-Test and then execute the statistical evaluation.
*   **Output:** The results from Table 1 will be printed to your console.

### Step 5: (Optional) Generate Descriptive Statistics

*   **Action:** Run the supporting utility scripts to see the annotation stats from the paper's appendix.
    ```bash
    python scripts/supporting_utilities/analyze_human_stats_411.py
    python scripts/supporting_utilities/count_entities_by_threshold.py
    ```
*   **Output:** The summary tables will be printed to your console.

## Citation

If you use this work, please cite our paper:
```
@article{YourLastName_2024_CaPo,
  author    = {Author Names},
  title     = {Evaluating Large Language Models for Strategic Knowledge Extraction in Capability-Based Planning},
  journal   = {Conference/Journal Name},
  year      = {2024},
}
```