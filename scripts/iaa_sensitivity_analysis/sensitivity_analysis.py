import json
import numpy as np
import sys
import os
from collections import defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.append(str(SCRIPTS_DIR))

from final_pipeline.analysis_logic import (
    create_data_set, 
    create_final_annotation_files, 
    similarity, 
    basic_normalize
)
from supporting_utilities.iaa_calculator import calculate_iaa

def calculate_capability_overlap(data, strategy_threshold):
    """
    Calculates the pairwise Jaccard index for capability links.
    Returns the overall average Jaccard score across all pairs.
    """
    annotator_data = {blob['annotator_id']: blob for blob in data if blob['annotator_id'] != 'LLM'}
    annotator_ids = sorted(annotator_data.keys())
    
    doc_map = defaultdict(lambda: defaultdict(list))
    for ann_id, blob in annotator_data.items():
        for doc in blob['documents_annotated']:
            doc_id = doc['document_id']
            for rel in doc.get('strategy_relations', []):
                strategy = rel.get('strategy', '').strip().lower()
                if strategy:
                    doc_map[doc_id][ann_id].append(rel)

    overall_scores = []
    
    for ann1_id, ann2_id in combinations(annotator_ids, 2):
        pair_scores = []
        num_shared_strategies = 0
        
        for doc_id in doc_map.keys():
            strats1 = doc_map[doc_id].get(ann1_id, [])
            strats2 = doc_map[doc_id].get(ann2_id, [])

            if not strats1 or not strats2:
                continue

            for rel1 in strats1:
                for rel2 in strats2:
                    strat1_text = rel1.get('strategy', '').strip().lower()
                    strat2_text = rel2.get('strategy', '').strip().lower()
                    
                    if similarity(strat1_text, strat2_text) >= strategy_threshold:
                        num_shared_strategies += 1
                        caps1 = set(basic_normalize(rel1.get('requires', [])))
                        caps2 = set(basic_normalize(rel2.get('requires', [])))
                        
                        if not caps1 and not caps2: jaccard = 1.0
                        elif not caps1 or not caps2: jaccard = 0.0
                        else: jaccard = len(caps1.intersection(caps2)) / len(caps1.union(caps2))
                        
                        pair_scores.append(jaccard)

        if pair_scores:
            overall_scores.extend(pair_scores)

    return np.mean(overall_scores) if overall_scores else 0.0


def main():
    print("Loading data...")
    input_path = PROJECT_ROOT / "output" / "annotations_cleaned.json"
    output_path = PROJECT_ROOT / "output" / "sensitivity_analysis_results.json"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    document_ids = sorted([doc.get('document_id') for doc in data[0].get("documents_annotated")])
    all_annotators = sorted([str(blob['annotator_id']) for blob in data])

    # Define the range of thresholds to test
    thresholds = np.arange(0.95, 0.44, -0.1)
    
    results = []

    for threshold in thresholds:
        print(f"\n--- Processing for threshold: {threshold:.2f} ---")
        
        # --- 1. Process standard concepts ---
        print("Grouping standard concepts (Strategies, Goals, Trends)...")
        strategies_by_doc = create_data_set(data, "strategies", document_ids, threshold)
        goals_by_doc = create_data_set(data, "goals", document_ids, threshold)
        trends_by_doc = create_data_set(data, "trends", document_ids, threshold)

        # --- 2. Generate annotation files to calculate Kappa/Alpha ---
        h_strat, _ = create_final_annotation_files(strategies_by_doc, all_annotators)
        h_goal, _ = create_final_annotation_files(goals_by_doc, all_annotators)
        h_trend, _ = create_final_annotation_files(trends_by_doc, all_annotators)

        # --- 3. Calculate Kappa/Alpha scores (by capturing print output) ---
        # This is a bit of a hack, but it reuses your existing IAA script's logic
        import io
        from contextlib import redirect_stdout
        
        def get_krippendorff(data_dict):
            f = io.StringIO()
            with redirect_stdout(f):
                calculate_iaa(data_dict)
            output = f.getvalue()
            try:
                # Find the line with Krippendorff's Alpha and extract the value
                alpha_line = [line for line in output.split('\n') if "Krippendorff's Alpha" in line][0]
                alpha_score = float(alpha_line.split(':')[-1].strip())
                return alpha_score
            except (IndexError, ValueError):
                return np.nan

        print("Calculating Krippendorff's Alpha...")
        strat_alpha = get_krippendorff(h_strat)
        goal_alpha = get_krippendorff(h_goal)
        trend_alpha = get_krippendorff(h_trend)

        # --- 4. Calculate Jaccard score for capabilities ---
        # We use the separate, correct logic for this
        print("Calculating Jaccard Index for Capability Links...")
        human_full_data = [blob for blob in data if blob['annotator_id'] != 'LLM']
        jaccard_score = calculate_capability_overlap(human_full_data, strategy_threshold=threshold)
        
        # --- 5. Store results ---
        results.append({
            "threshold": threshold,
            "strategy_alpha": strat_alpha,
            "goal_alpha": goal_alpha,
            "trend_alpha": trend_alpha,
            "capability_jaccard": jaccard_score
        })
        print(f"Scores for {threshold:.2f}: Trends={trend_alpha:.3f}, Goals={goal_alpha:.3f}, Strategies={strat_alpha:.3f}, Links={jaccard_score:.3f}")

    # --- 6. Save results to a file for plotting ---
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nAnalysis complete. Results saved to {output_path}")
    print("Now run 'python scripts/iaa_sensitivity_analysis/plot_results.py' to generate the chart.")

if __name__ == "__main__":
    from collections import defaultdict
    main()