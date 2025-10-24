import json
import os
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import krippendorff
from itertools import combinations
import numpy as np

def clean_cell(cell_value):
    """
    Safely extracts a boolean from a cell, converting to a float (1.0, 0.0, or nan).
    """
    if isinstance(cell_value, list) and len(cell_value) > 0 and isinstance(cell_value[0], bool):
        return 1.0 if cell_value[0] else 0.0
    return np.nan

def calculate_iaa(human_annotations: dict):
    """
    Calculates and prints Inter-Annotator Agreement metrics for a given dataset.
    """
    try:
        df = pd.DataFrame.from_dict(human_annotations)
        df = df.map(clean_cell)
    except Exception as e:
        print(f"Could not process data into DataFrame. Error: {e}")
        return

    annotators = df.columns.tolist()
    if len(annotators) < 2:
        print("  Not enough annotators (need at least 2) to calculate agreement.\n")
        return

    print("--- Pairwise Agreement and Cohen's Kappa ---")
    for ann1, ann2 in combinations(annotators, 2):
        pair_df = df[[ann1, ann2]].dropna()

        if len(pair_df) < 2:
            print(f"  {ann1} vs {ann2}: Not enough overlapping annotations.")
            continue
        
        if pair_df[ann1].nunique() < 2 or pair_df[ann2].nunique() < 2:
            agreement = (pair_df[ann1] == pair_df[ann2]).mean()
            print(f"  {ann1} vs {ann2}:")
            print(f"    - Overlapping Instances: {len(pair_df)}")
            print(f"    - Percent Agreement:     {agreement:.2%}")
            print(f"    - Cohen's Kappa:         N/A (no variance in ratings)")
            continue

        agreement = (pair_df[ann1] == pair_df[ann2]).mean()
        
        try:
            kappa = cohen_kappa_score(pair_df[ann1], pair_df[ann2], labels=[1, 0])
            print(f"  {ann1} vs {ann2}:")
            print(f"    - Overlapping Instances: {len(pair_df)}")
            print(f"    - Percent Agreement:     {agreement:.2%}")
            print(f"    - Cohen's Kappa:         {kappa:.3f}")

        except Exception as e:
            print(f"  Could not calculate Kappa for {ann1} vs {ann2}. Error: {e}")
            continue

    print("\n--- Overall Group Agreement ---")
    reliability_data = df.T.to_numpy()
    
    try:
        alpha = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement='nominal')
        print(f"  Krippendorff's Alpha (for {len(annotators)} annotators): {alpha:.3f}")
    except Exception as e:
        print(f"  FAILED to calculate Krippendorff's Alpha. Error: {e}")
        
    print("-" * 40 + "\n")

