import pandas as pd
import os
import argparse
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV
import collaborative_recommender as cf

def run_hyperparameter_tuning():
    data, _ = cf.load_data_for_surprise()
    if data is None:
        print("Evaluation: Failed to load data. Aborting.")
        return

    param_grid = {
        'n_epochs': [20],
        'n_factors': [100, 150],
        'lr_all': [0.005],
        'reg_all': [0.02, 0.1]
    }
#n_jobs is the number of cpu cores to use. More cpu core, more ram pool needed.
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

if __name__ == "__main__":
    run_hyperparameter_tuning()
