import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_sensitivity_results(results_file="sensitivity_analysis_results.json"):
    """
    Loads the analysis results and plots them on a polished, publication-ready line chart,
    optimized for a single-column paper format.
    """
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    
    results_file = PROJECT_ROOT / "output" / "sensitivity_analysis_results.json"
    output_image_file = PROJECT_ROOT / "output" / "iaa_thresh_single_column.png"

    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The results file '{results_file}' was not found.")
        print("Please run 'python scripts/iaa_sensitivity_analysis/sensitivity_analysis.py' first.")
        return

    df = pd.DataFrame(data)
    
    # --- MODIFIED: Simplified column names for a more compact legend ---
    df.rename(columns={
        "strategy_alpha": "Strategies (Alpha)",
        "goal_alpha": "Goals (Alpha)",
        "trend_alpha": "Trends (Alpha)",
        "capability_jaccard": "Links (Jaccard)"
    }, inplace=True)

    # Melt the DataFrame for Seaborn
    df_melted = df.melt(id_vars=['threshold'], 
                        value_vars=['Strategies (Alpha)', 'Goals (Alpha)', 'Trends (Alpha)', 'Links (Jaccard)'],
                        var_name='Concept Type', 
                        value_name='Agreement Score')

    # --- Plotting Setup ---
    # Using a global font size setting for consistency
    plt.rcParams.update({'font.size': 14}) # Base font size
    sns.set_theme(style="whitegrid", rc={"axes.facecolor": (0.95, 0.95, 0.95)})
    
    # --- MODIFIED: Changed figure size for a single column (width, height in inches) ---
    # A typical single-column width is ~3.5 inches. We'll make it slightly wider.
    # The height is increased to make it more portrait-oriented.
    fig, ax = plt.subplots(figsize=(6, 5))
    
    palette = {
        "Strategies (Alpha)": "#1f77b4", # Muted Blue
        "Goals (Alpha)": "#ff7f0e",      # Safety Orange
        "Trends (Alpha)": "#2ca02c",      # Cooked Asparagus Green
        "Links (Jaccard)": "#d62728" # Brick Red
    }

    # --- Create the Line Plot with adjusted element sizes ---
    sns.lineplot(
        data=df_melted, 
        x='threshold', 
        y='Agreement Score', 
        hue='Concept Type', 
        palette=palette,
        marker='o', 
        markersize=6,      # --- MODIFIED: Slightly smaller markers ---
        linewidth=2.0,     # --- MODIFIED: Slightly thinner lines ---
        ax=ax
    )

    # --- Formatting and Annotations ---
    # --- MODIFIED: Increased font sizes for all labels ---
    ax.set_xlabel('Similarity Threshold (Stricter → Lenient)', fontsize=12, labelpad=10)
    ax.set_ylabel('Agreement Score', fontsize=12, labelpad=10)
    
    ax.invert_xaxis()
    ax.tick_params(axis='both', which='major', labelsize=11) # Tick labels

    ax.axhline(0, color='gray', linestyle='--', linewidth=1.0, zorder=0)

    # --- MODIFIED: Adjusted legend for a smaller plot ---
    legend = ax.legend(title='Concept & Metric', fontsize=10, title_fontsize=11)
    
    # Final adjustments
    plt.grid(which='major', linestyle='--', linewidth='0.5', color='grey')
    plt.tight_layout(pad=0.5) # Reduced padding for compact layout
    
    # Save the plot
    plt.savefig(output_image_file, dpi=300, bbox_inches='tight')
    print(f"Single-column plot saved successfully as '{output_image_file}'")
    
    plt.show()


if __name__ == "__main__":
    plot_sensitivity_results()