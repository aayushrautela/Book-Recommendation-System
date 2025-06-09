import pandas as pd
import os
import argparse
from surprise import Dataset, Reader, SVD as SVD_original # Import original SVD with a new name
from surprise.model_selection import GridSearchCV
import collaborative_recommender as cf
import matplotlib.pyplot as plt
import seaborn as sns

# --- Custom SVD Wrapper to Fix TypeError ---
# This wrapper ensures n_epochs and n_factors are always integers.
class SVD(SVD_original):
    def __init__(self, n_factors=100, n_epochs=20, **kwargs):
        # Explicitly cast the parameters to integers before passing them on.
        super().__init__(n_factors=int(n_factors), n_epochs=int(n_epochs), **kwargs)

# --- End of Custom Wrapper ---


# Define file paths for this final tuning run.
RESULTS_CSV_PATH = 'tuning_results_final_epochs_factors.csv'
HEATMAP_IMG_PATH = 'tuning_heatmap_final_epochs_factors.png'

def run_hyperparameter_tuning():
    """
    Performs a final, fine-tuned hyperparameter search for the SVD model.
    Saves numerical results to a CSV and generates a heatmap plot as a PNG.
    """
    if os.path.exists(RESULTS_CSV_PATH):
        print(f"Found existing results at '{RESULTS_CSV_PATH}'. Skipping tuning and generating plot from this file.")
        results_df = pd.read_csv(RESULTS_CSV_PATH)
    else:
        data, _ = cf.load_data_for_surprise()
        if data is None:
            print("Evaluation: Failed to load data. Aborting.")
            return

        # Final 2x2 param_grid, focusing on n_epochs and n_factors.
        param_grid = {
            'n_epochs': [35, 40],                   # New values to test for epochs
            'n_factors': [175, 200],                # New values to test for factors
            'lr_all': [0.01],                       # Fixed to best known value for this run
            'reg_all': [0.1]                        # Fixed to best known value for this run
        }

        # GridSearchCV will now use our custom SVD wrapper class.
        gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=1)

        num_combinations = len(param_grid['n_epochs']) * len(param_grid['n_factors'])
        print("Starting FINAL (2x2) hyperparameter search with GridSearchCV...")
        print(f"This will test {num_combinations} combinations for n_epochs and n_factors.")
        
        gs.fit(data)

        print("\n--- FINAL Hyperparameter Tuning Results ---")
        
        print(f"Best RMSE score: {gs.best_score['rmse']:.4f}")
        print("Best parameters for RMSE:")
        print(gs.best_params['rmse'])
        
        print(f"\nBest MAE score: {gs.best_score['mae']:.4f}")
        print("Best parameters for MAE:")
        print(gs.best_params['mae'])

        results_df = pd.DataFrame(gs.cv_results)
        results_df.to_csv(RESULTS_CSV_PATH, index=False)
        print(f"\nFinal tuning results saved to '{RESULTS_CSV_PATH}'")

    # Plotting Results for the final search
    print("\nGenerating heatmap of final tuning results...")
    
    results_subset_df = results_df[['param_n_factors', 'param_n_epochs', 'mean_test_rmse']]
    
    heatmap_data = results_subset_df.pivot_table(index='param_n_epochs', columns='param_n_factors', values='mean_test_rmse')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="viridis_r", cbar_kws={'label': 'RMSE Score (lower is better)'})
    
    plt.title('SVD Final Hyperparameter Results (RMSE)', fontsize=16)
    plt.xlabel('Number of Factors (n_factors)', fontsize=12)
    plt.ylabel('Number of Epochs (n_epochs)', fontsize=12)
    
    plt.savefig(HEATMAP_IMG_PATH)
    print(f"Heatmap saved to '{HEATMAP_IMG_PATH}'")

if __name__ == "__main__":
    run_hyperparameter_tuning()
