import pandas as pd
import os
import argparse
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV
import collaborative_recommender as cf
import matplotlib.pyplot as plt
import seaborn as sns

def run_hyperparameter_tuning():
    data, _ = cf.load_data_for_surprise()
    if data is None:
        print("Evaluation: Failed to load data. Aborting.")
        return

    # Expanded parameter grid for a nicer heatmap.
    param_grid = {
        'n_epochs': [20, 25, 30],           
        'n_factors': [50, 100, 150, 200],   
        'lr_all': [0.005],           
        'reg_all': [0.02]          
    }

    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=1)

    num_combinations = len(param_grid['n_epochs'])*len(param_grid['n_factors'])*len(param_grid['lr_all'])*len(param_grid['reg_all'])
    print("Starting hyperparameter tuning with GridSearchCV...")
    print(f"This will test {num_combinations} combinations sequentially and may take some time.")
    
    gs.fit(data)

    print("\n--- Hyperparameter Tuning Results ---")
    
    print(f"Best RMSE score: {gs.best_score['rmse']:.4f}")
    print("Best parameters for RMSE:")
    print(gs.best_params['rmse'])
    
    print(f"\nBest MAE score: {gs.best_score['mae']:.4f}")
    print("Best parameters for MAE:")
    print(gs.best_params['mae'])

    # --- Plotting Results ---
    print("\nGenerating heatmap of tuning results...")
    
    # Convert the grid search results to a pandas DataFrame
    results_df = pd.DataFrame(gs.cv_results)
    
    # We are interested in n_factors, n_epochs, and the mean_test_rmse
    results_subset_df = results_df[['param_n_factors', 'param_n_epochs', 'mean_test_rmse']]
    
    # Pivot the DataFrame to create a 2D grid for the heatmap
    heatmap_data = results_subset_df.pivot_table(index='param_n_epochs', columns='param_n_factors', values='mean_test_rmse')
    
    # Create the heatmap plot
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="viridis_r", cbar_kws={'label': 'RMSE Score (lower is better)'})
    
    plt.title('SVD Hyperparameter Tuning Results (RMSE)', fontsize=16)
    plt.xlabel('Number of Factors (n_factors)', fontsize=12)
    plt.ylabel('Number of Epochs (n_epochs)', fontsize=12)
    plt.show()


if __name__ == "__main__":
    run_hyperparameter_tuning()
