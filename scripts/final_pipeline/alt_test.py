# alt_test.py (formerly alt_test_v2.py)

import json
import os
import numpy as np
from scipy.stats import ttest_rel
from typing import Any, List, Dict
import argparse # Import the argparse library

def by_procedure(p_values: List[float], q: float = 0.05) -> List[int]:
    """
    Benjamini-Yekutieli procedure to control the False Discovery Rate.
    """
    p_values = np.array(p_values, dtype=float)
    m = len(p_values)
    if m == 0:
        return []

    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]
    
    c_m = np.sum(1.0 / np.arange(1, m + 1))
    by_thresholds = (np.arange(1, m + 1) / m) * (q / c_m)
    
    rejected_mask = sorted_pvals <= by_thresholds
    
    if not np.any(rejected_mask):
        return []

    max_i = np.max(np.where(rejected_mask))
    rejected_sorted_indices = sorted_indices[:max_i + 1]
    
    return list(rejected_sorted_indices)

def calculate_accuracy(prediction: Any, references: List[Any]) -> float:
    """Calculates the alignment score (accuracy) against a list of references."""
    if not references:
        return 0.0
    return np.mean([prediction == ref for ref in references])

def run_alt_test(
    llm_annotations: Dict[str, Any],
    human_annotations: Dict[str, Dict[str, Any]],
    epsilon: float,
    min_human_annotations: int = 3,
    min_instances_per_human: int = 10,
    verbose_annotator: str = None
):
    """
    Runs the Alternative Annotator Test (ALT-Test).
    (This function remains unchanged)
    """
    if len(human_annotations) < min_human_annotations:
        print(f"Warning: Not enough human annotators ({len(human_annotations)}). Need at least {min_human_annotations}.")
        return 0.0, 0.0

    p_values = []; all_p_f = []; humans_in_test = []
    for excluded_h_id in sorted(human_annotations.keys()):
        common_instances = [
            i for i in human_annotations[excluded_h_id]
            if i in llm_annotations and
               all(i in human_annotations[other_h] for other_h in human_annotations if other_h != excluded_h_id)
        ]
        if len(common_instances) < min_instances_per_human: continue
        humans_in_test.append(excluded_h_id)
        llm_instance_scores = []; human_instance_scores = []
        if verbose_annotator == excluded_h_id:
            print("\n" + "="*80); print(f"DETAILED BREAKDOWN FOR EXCLUDED ANNOTATOR: {excluded_h_id}"); print("="*80)
            print(f"{'Instance ID':<25} | {'Human Ann':<10} | {'LLM Ann':<10} | {'Ground Truth Anns':<25} | {'Human Score':<12} | {'LLM Score':<12}"); print("-"*80)
        for i in sorted(common_instances):
            ground_truth_anns = [human_annotations[other_h][i] for other_h in human_annotations if other_h != excluded_h_id]
            llm_ann = llm_annotations[i]; human_ann = human_annotations[excluded_h_id][i]
            llm_score = calculate_accuracy(llm_ann, ground_truth_anns); human_score = calculate_accuracy(human_ann, ground_truth_anns)
            llm_instance_scores.append(llm_score); human_instance_scores.append(human_score)
            if verbose_annotator == excluded_h_id:
                gt_str = ", ".join([str(ann[0]) for ann in ground_truth_anns])
                print(f"{i.split('_')[-1]:<25} | {str(human_ann[0]):<10} | {str(llm_ann[0]):<10} | {gt_str:<25} | {llm_score:<12.2f} | {human_score:<12.2f}")
        p_f = np.mean([1 if llm_s >= h_s else 0 for llm_s, h_s in zip(llm_instance_scores, human_instance_scores)]); all_p_f.append(p_f)
        if verbose_annotator == excluded_h_id:
            print("-"*80); print(f"p^f (LLM's Advantage Prob.): {p_f:.3f}"); print(f"Mean LLM Score: {np.mean(llm_instance_scores):.3f}"); print(f"Mean Human Score: {np.mean(human_instance_scores):.3f}")
        adjusted_human_scores = [s - epsilon for s in human_instance_scores]
        _, p_value = ttest_rel(llm_instance_scores, adjusted_human_scores, alternative='greater')
        p_values.append(p_value)
        if verbose_annotator == excluded_h_id:
            print(f"Hypothesis test: H1: p^f > p^h - {epsilon}"); print(f"p-value from paired t-test: {p_value:.4f}"); print("="*80 + "\n")
    if not humans_in_test:
        print("Warning: No annotators met the minimum instance requirement."); return 0.0, 0.0
    rejected_indices = by_procedure(p_values, q=0.05)
    omega_winning_rate = len(rejected_indices) / len(humans_in_test)
    rho_advantage_prob = np.mean(all_p_f) if all_p_f else 0.0
    return omega_winning_rate, rho_advantage_prob

def main():
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    DATA_DIRS = {
        "Entities": PROJECT_ROOT / "output" / "alt_test_input" / "entities",
        "Links": PROJECT_ROOT / "output" / "alt_test_input" / "links"
    }
    
    EPSILON = 0.1
    MIN_INSTANCES = 5
    VERBOSE_ANNOTATOR_ID = None 

    print(f"--- Running ALT-Test with Epsilon = {EPSILON} ---")

    for dir_type, data_dir in DATA_DIRS.items():
        print(f"\n{'='*20} TESTING {dir_type.upper()} {'='*20}")
        if not data_dir.is_dir(): # Use .is_dir() for pathlib objects
            print(f"Directory not found: '{data_dir}'. Skipping.")
            continue

        concepts = sorted(list(set(
            f.replace('humans_annotations_', '').replace('.json', '')
            for f in os.listdir(data_dir) if f.startswith('humans_annotations_')
        )))
        
        if not concepts:
            print(f"No concept files found in '{data_dir}'.")
            continue
            
        print(f"Found concepts: {', '.join(concepts)}")
        
        all_humans_data = {}
        all_llm_data = {}

        for concept in concepts:
            print("\n" + "-"*50)
            print(f"Testing concept: '{concept}'")
            print("-"*50)
            
            humans_file = data_dir / f"humans_annotations_{concept}.json" 
            llm_file = data_dir / f"llm_annotations_{concept}.json"    

            with open(humans_file, 'r') as f: humans_data = json.load(f)
            with open(llm_file, 'r') as f: llm_data = json.load(f)

            omega, rho = run_alt_test(
                llm_annotations=llm_data,
                human_annotations=humans_data,
                epsilon=EPSILON,
                min_instances_per_human=MIN_INSTANCES,
                verbose_annotator=VERBOSE_ANNOTATOR_ID 
            )
            
            print(f"Winning Rate (ω): {omega:.2f}")
            print(f"Advantage Probability (ρ): {rho:.2f}")
            conclusion = "PASSED" if omega >= 0.5 else "FAILED"
            print(f"Conclusion: {conclusion}")
            
            all_llm_data.update(llm_data)
            for annotator, annotations in humans_data.items():
                all_humans_data.setdefault(annotator, {}).update(annotations)
                
        if all_llm_data:
            print("\n" + "-"*50)
            print("Testing concept: 'Combined (All Concepts)'")
            print("-"*50)
            
            omega, rho = run_alt_test(
                llm_annotations=all_llm_data,
                human_annotations=all_humans_data,
                epsilon=EPSILON,
                min_instances_per_human=MIN_INSTANCES,
                verbose_annotator=VERBOSE_ANNOTATOR_ID
            )
            
            print(f"Winning Rate (ω): {omega:.2f}")
            print(f"Advantage Probability (ρ): {rho:.2f}")
            conclusion = "PASSED" if omega >= 0.5 else "FAILED"

if __name__ == "__main__":
    main()