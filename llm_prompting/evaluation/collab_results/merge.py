import pandas as pd
import os

INPUT_DIR = "." 

FINAL_DIR = "final_results"
os.makedirs(FINAL_DIR, exist_ok=True) 

file_pairs = [
    {
        "fin_file": "tier3_test_raw_predictions_with_fin.csv",
        "llama70b_file": "tier3_test_raw_predictions.csv",
        "output_file": "combined_tier3_test_raw_predictions.csv"
    },
    {
        "fin_file": "tier3_test_summary_with_fin.csv",
        "llama70b_file": "tier3_test_summary.csv",
        "output_file": "combined_tier3_test_summary.csv"
    },
    {
        "fin_file": "tier3_validation_prompt_selection_with_fin.csv",
        "llama70b_file": "tier3_validation_prompt_selection.csv",
        "output_file": "combined_tier3_validation_prompt_selection.csv"
    }
]

print(f"Merging evaluation files and saving to '{FINAL_DIR}' folder...")

for pair in file_pairs:
    path_fin = os.path.join(INPUT_DIR, pair["fin_file"])
    path_70b = os.path.join(INPUT_DIR, pair["llama70b_file"])
    path_out = os.path.join(FINAL_DIR, pair["output_file"])
    
    try:
        df_fin = pd.read_csv(path_fin)
        df_70b = pd.read_csv(path_70b)
        
        combined_df = pd.concat([df_fin, df_70b], ignore_index=True)
        
        combined_df.to_csv(path_out, index=False)
        print(f"Successfully created: {FINAL_DIR}/{pair['output_file']} (Total rows: {len(combined_df)})")
        
    except FileNotFoundError as e:
        print(f"Error finding a file: {e}. Please check your file names.")

print("\nMerge complete. Check the 'final_results' folder for your unified datasets!")