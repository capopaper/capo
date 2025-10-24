import json
import numpy as np
import pandas as pd
from collections import defaultdict
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.final_pipeline.analysis_logic import create_data_set

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.append(str(SCRIPTS_DIR))

def analyze_entity_counts_by_threshold():
    """
    Analyzes the number of unique concepts found by humans and the LLM
    across a range of similarity thresholds.
    """
    input_path = PROJECT_ROOT / "output" / "annotations_cleaned.json"

    print(f"Loading cleaned annotation data from '{input_path}'...")
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{input_path}' not found. Please run 'scripts/supporting_utilities/data_formating.py' first.")
        return

    # --- Configuration ---
    # Extract document IDs and define concepts to analyze
    document_ids = sorted([doc.get('document_id') for doc in data[0].get("documents_annotated")])
    concepts_to_analyze = ["strategies", "goals", "trends"]
    
    # Define the range of thresholds to test, matching your sensitivity analyses
    thresholds = np.arange(0.95, 0.44, -0.05)
    
    # --- Analysis Loop ---
    results = []
    print(f"Analyzing {len(concepts_to_analyze)} concepts across {len(thresholds)} thresholds...")

    for threshold in thresholds:
        print(f"  Processing threshold: {threshold:.2f}")
        
        current_threshold_counts = {"threshold": f"{threshold:.2f}"}

        for concept_name in concepts_to_analyze:
            # Reuse your existing robust logic to group concepts for this threshold
            grouped_concepts_by_doc = create_data_set(data, concept_name, document_ids, threshold)

            # Initialize counters for this specific concept and threshold
            total_human_concepts = 0
            total_llm_concepts = 0

            # Iterate through all documents and their grouped concepts
            for doc_id, groups in grouped_concepts_by_doc.items():
                for group in groups:
                    # Check if the LLM contributed to this concept group
                    llm_contributed = len(group['contributions'].get('LLM', [])) > 0
                    if llm_contributed:
                        total_llm_concepts += 1
                    
                    # Check if any human contributed to this concept group
                    human_contributed = any(
                        len(contributions) > 0
                        for annotator, contributions in group['contributions'].items()
                        if annotator != 'LLM'
                    )
                    if human_contributed:
                        total_human_concepts += 1
            
            # Store the final counts for this concept
            current_threshold_counts[f"human_{concept_name}"] = total_human_concepts
            current_threshold_counts[f"llm_{concept_name}"] = total_llm_concepts
            
        results.append(current_threshold_counts)

    # --- Display Results ---
    if not results:
        print("\nNo results were generated. Please check your data and configuration.")
        return
        
    print("\n--- Analysis Complete: Entity Counts by Threshold ---")
    
    # Use pandas for a clean, well-formatted table
    df = pd.DataFrame(results)
    
    # Set the threshold as the index for better readability
    df.set_index('threshold', inplace=True)
    
    # Define a logical column order
    column_order = [
        'human_strategies', 'llm_strategies',
        'human_goals', 'llm_goals',
        'human_trends', 'llm_trends'
    ]
    df = df[column_order]

    # Print the full table without truncation
    print(df.to_string())
    
    print("\nTable shows the total number of unique concept groups identified across all documents.")


if __name__ == "__main__":
    analyze_entity_counts_by_threshold()