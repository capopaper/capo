import json
import pandas as pd
import numpy as np

def analyze_human_annotation_stats():
    """
    Calculates the total and average number of entities and links created
    by human annotators across all documents.
    """
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    input_path = PROJECT_ROOT / "output" / "annotations_cleaned.json"

    print(f"Loading cleaned annotation data from '{input_path}'...")
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{input_path}' not found. Please run 'scripts/supporting_utilities/data_formating.py' first.")
        return

    # This dictionary will store the counts for each individual human annotator.
    human_stats = {}

    # --- Step 1: Iterate through data and count annotations for each human ---
    for annotator_blob in data:
        annotator_id = annotator_blob.get("annotator_id")

        # Skip the LLM, as we only want human stats
        if annotator_id == 'LLM':
            continue

        # Initialize counters for this specific human
        total_strategies = 0
        total_goals = 0
        total_trends = 0
        total_strat_goal_links = 0
        total_strat_trend_links = 0
        total_strat_cap_links = 0

        # Loop through all documents annotated by this person
        for doc in annotator_blob.get("documents_annotated", []):
            # Count the simple entities
            total_strategies += len(doc.get('strategies', []))
            total_goals += len(doc.get('goals', []))
            total_trends += len(doc.get('trends', []))

            # Count the links by iterating through the relations
            for relation in doc.get('strategy_relations', []):
                # The number of links is the number of items in each list
                total_strat_goal_links += len(relation.get('has_goal', []))
                total_strat_trend_links += len(relation.get('is_response_to', []))
                total_strat_cap_links += len(relation.get('requires', []))

        # Store the final counts for this human
        human_stats[f"Human_{annotator_id}"] = {
            'Strategies': total_strategies,
            'Goals': total_goals,
            'Trends': total_trends,
            'Strategy-Goal Links': total_strat_goal_links,
            'Strategy-Trend Links': total_strat_trend_links,
            'Strategy-Capability Links': total_strat_cap_links
        }

    if not human_stats:
        print("\nNo human annotator data found in the file.")
        return

    # --- Step 2: Use pandas to create and display clean tables ---
    
    # Create a DataFrame showing the detailed counts for each human
    df_detailed = pd.DataFrame.from_dict(human_stats, orient='index')
    
    print("\n--- Detailed Counts per Human Annotator ---")
    print(df_detailed.to_string())

    # Create a summary DataFrame with the average and standard deviation
    df_summary = pd.DataFrame({
        'Average': df_detailed.mean(),
        'Std. Dev.': df_detailed.std()
    })
    
    # Format the output for better readability
    df_summary['Average'] = df_summary['Average'].round(2)
    df_summary['Std. Dev.'] = df_summary['Std. Dev.'].round(2)
    df_summary.index.name = 'Annotation Type'

    print("\n\n--- Summary of Human Annotation Statistics ---")
    print(df_summary.to_string())
    print("\n'Average' is the mean count per human across all documents.")
    print("'Std. Dev.' shows the variation in counts among humans.")


if __name__ == "__main__":
    analyze_human_annotation_stats()