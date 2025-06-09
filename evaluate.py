import pandas as pd
import os
import argparse
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV
import collaborative_recommender as cf
import matplotlib.pyplot as plt
import seaborn as sns

# Define new file paths for this final tuning run.
RESULTS_CSV_PATH = 'tuning_results_final.csv'
HEATMAP_IMG_PATH = 'tuning_heatmap_final.png'

def run_hyperparameter_tuning():
   
    if os.path.exists(RESULTS_CSV_PATH):
        print(f"Found existing results at '{RESULTS_CSV_PATH}'. Skipping tuning and generating plot from this file.")
        results_df = pd.read_csv(RESULTS_CSV_PATH)
    else:
        data, _ = cf.load_data_for_surprise()
        if data is None:
            print("Evaluation: Failed to load data. Aborting.")
            return

        param_grid = {
            'n_epochs': [30],                   
            'n_factors': [100],               
            'lr_all': [0.012, 0.015],            
            'reg_all': [0.06, 0.08]            
        }

        gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=1)

        num_combinations = len(param_grid['lr_all']) * len(param_grid['reg_all'])
        print("Starting hyperparameter search with GridSearchCV...")
        print(f"This will test {num_combinations} combinations for lr_all and reg_all.")
        
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
    
    results_subset_df = results_df[['param_lr_all', 'param_reg_all', 'mean_test_rmse']]
    
    heatmap_data = results_subset_df.pivot_table(index='param_lr_all', columns='param_reg_all', values='mean_test_rmse')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="viridis_r", cbar_kws={'label': 'RMSE Score (lower is better)'})
    
    plt.title('SVD Final Hyperparameter Results (RMSE)', fontsize=16)
    plt.xlabel('Regularization Term (reg_all)', fontsize=12)
    plt.ylabel('Learning Rate (lr_all)', fontsize=12)
    
    plt.savefig(HEATMAP_IMG_PATH)
    print(f"Heatmap saved to '{HEATMAP_IMG_PATH}'")

if __name__ == "__main__":
    run_hyperparameter_tuning()
