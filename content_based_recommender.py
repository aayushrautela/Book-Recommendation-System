import pandas as pd
import os
import argparse
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV
import collaborative_recommender as cf
import matplotlib.pyplot as plt
import seaborn as sns

# Define new file paths for this 3x3 run.
RESULTS_CSV_PATH = 'tuning_results_3x3_fine.csv'
HEATMAP_IMG_PATH = 'tuning_heatmap_3x3_fine.png'

def run_hyperparameter_tuning():
    """
    Performs a fine-tuned 3x3 hyperparameter search for the SVD model.
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

        # Fine-tuned 3x3 param_grid, focusing on lr_all and reg_all.
        param_grid = {
            'n_epochs': [40],                   # Fixed to best value from previous run
            'n_factors': [200],                 # Fixed to best value from previous run
            'lr_all': [0.008, 0.01, 0.012],      # Test values around the best lr_all=0.01
            'reg_all': [0.08, 0.1, 0.12]         # Test values around the best reg_all=0.1
        }

        gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=1)

        num_combinations = len(param_grid['lr_all']) * len(param_grid['reg_all'])
        print("Starting FINE-TUNED (3x3) hyperparameter search with GridSearchCV...")
        print(f"This will test {num_combinations} combinations for lr_all and reg_all.")
        
        gs.fit(data)

        print("\n--- FINE-TUNED (3x3) Hyperparameter Tuning Results ---")
        
        print(f"Best RMSE score: {gs.best_score['rmse']:.4f}")
        print("Best parameters for RMSE:")
        print(gs.best_params['rmse'])
        
        print(f"\nBest MAE score: {gs.best_score['mae']:.4f}")
        print("Best parameters for MAE:")
        print(gs.best_params['mae'])

        results_df = pd.DataFrame(gs.cv_results)
        results_df.to_csv(RESULTS_CSV_PATH, index=False)
        print(f"\nFine-tuned results saved to '{RESULTS_CSV_PATH}'")

    # Plotting Results for the fine-tuned search
    print("\nGenerating heatmap of fine-tuned results...")
    
    results_subset_df = results_df[['param_lr_all', 'param_reg_all', 'mean_test_rmse']]
    
    heatmap_data = results_subset_df.pivot_table(index='param_lr_all', columns='param_reg_all', values='mean_test_rmse')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="viridis_r", cbar_kws={'label': 'RMSE Score (lower is better)'})
    
    plt.title('SVD Fine-Tuned Hyperparameter Results (RMSE)', fontsize=16)
    plt.xlabel('Regularization Term (reg_all)', fontsize=12)
    plt.ylabel('Learning Rate (lr_all)', fontsize=12)
    
    plt.savefig(HEATMAP_IMG_PATH)
    print(f"Heatmap saved to '{HEATMAP_IMG_PATH}'")

if __name__ == "__main__":
    run_hyperparameter_tuning()
