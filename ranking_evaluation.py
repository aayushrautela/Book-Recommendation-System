import pandas as pd
import numpy as np
import os
import random
from tqdm import tqdm # A library to show progress bars for long loops
import matplotlib.pyplot as plt
import seaborn as sns

# Import our existing recommender modules
import content_based_recommender as cb
import collaborative_recommender as cf
import hybrid_recommender as hybrid

# --- Evaluation Configuration ---
N_TEST_USERS = 200        # How many users to test on (for speed).
# Define the different 'k' values we want to test for Precision@k.
K_VALUES = [3, 5, 10, 15] 
RATING_THRESHOLD = 4.0    # Consider books rated >= 4.0 as 'liked'.
PROFILE_SET_RATIO = 0.5   # Use 50% of a user's liked books as profile, 50% as hold-out.
LINE_PLOT_IMG_PATH = 'precision_at_k_comparison.png' # Filename for the output plot

def get_cf_recommendations_for_eval(model, user_id, profile_book_ids, all_books_df, n):
    """
    A special version of the CF recommendation function for evaluation.
    It predicts scores for all books EXCEPT those in the user's profile set.
    """
    all_book_ids = all_books_df['book_id'].unique()
    
    predictions = []
    for book_id in all_book_ids:
        if book_id not in profile_book_ids:
            predicted_rating = model.predict(uid=user_id, iid=book_id).est
            predictions.append({'book_id': book_id, 'score': predicted_rating})

    predictions.sort(key=lambda x: x['score'], reverse=True)
    
    top_recs = []
    for pred in predictions[:n]:
        book_id = pred['book_id']
        book_title_series = all_books_df.loc[all_books_df['book_id'] == book_id, 'title']
        title = book_title_series.iloc[0] if not book_title_series.empty else f"Book ID {book_id}"
        top_recs.append({'book_id': book_id, 'title': title, 'score': round(pred['score'], 4)})
        
    return top_recs

def calculate_precision_at_k(recommended_items, hold_out_items, k):
    """Calculates Precision@k for a single user."""
    top_k_recs = {rec['book_id'] for rec in recommended_items[:k]}
    hits = len(top_k_recs.intersection(hold_out_items))
    return hits / k

def run_ranking_evaluation():
    """
    Runs an offline evaluation to calculate Precision@k for all three models
    across multiple values of k.
    """
    print("--- Starting Ranking Evaluation (Precision@k) ---")

    print("Loading all models and data...")
    cf_model = cf.load_model()
    if cf_model is None: return
    all_ratings_data, all_books_info = cf.load_data_for_surprise()
    if all_ratings_data is None: return

    cb_prep_data = cb.load_and_prepare_content_data()
    cb_sim_matrix, cb_books_data, cb_book_map, _ = cb.build_content_model(cb_prep_data)
    if cb_sim_matrix is None: return
    print("All models and data ready.")

    all_user_ids = all_ratings_data.df['user_id'].unique()
    test_users = random.sample(list(all_user_ids), N_TEST_USERS)
    print(f"\nSelected {len(test_users)} random users for evaluation.")

    results = {
        'Content-Based': {k: [] for k in K_VALUES},
        'CF (SVD)': {k: [] for k in K_VALUES},
        'Hybrid': {k: [] for k in K_VALUES}
    }

    max_k = max(K_VALUES) 

    print(f"Calculating Precision@k for k in {K_VALUES}...")
    for user_id in tqdm(test_users, desc="Evaluating Users"):
        liked_books_df = all_ratings_data.df[
            (all_ratings_data.df['user_id'] == user_id) & 
            (all_ratings_data.df['rating'] >= RATING_THRESHOLD)
        ]
        
        if len(liked_books_df) < 2: continue

        profile_set_df = liked_books_df.sample(frac=PROFILE_SET_RATIO, random_state=42)
        hold_out_set_df = liked_books_df.drop(profile_set_df.index)
        
        profile_anchor_ids = profile_set_df['book_id'].tolist()
        profile_book_ids_set = set(profile_anchor_ids)
        hold_out_book_ids = set(hold_out_set_df['book_id'])
        
        # --- A) Generate recommendations for Content-Based Model ---
        cb_recs_aggregated = {}
        for anchor_id in profile_anchor_ids:
            if anchor_id not in cb_book_map: continue
            recs_for_anchor = cb.get_content_based_recommendations(anchor_id, max_k + 5, cb_sim_matrix, cb_books_data, cb_book_map)
            for rec in recs_for_anchor:
                if rec['book_id'] not in cb_recs_aggregated or rec['score'] > cb_recs_aggregated[rec['book_id']]['score']:
                    cb_recs_aggregated[rec['book_id']] = rec
        sorted_cb_recs = sorted(cb_recs_aggregated.values(), key=lambda x: x['score'], reverse=True)

        # --- B) Generate recommendations for Collaborative Filtering Model ---
        cf_recs = get_cf_recommendations_for_eval(cf_model, user_id, profile_book_ids_set, all_books_info, n=max_k)

        # --- C) Generate recommendations for Hybrid Model ---
        # Get a candidate pool from the special evaluation CF function
        hybrid_candidate_recs = get_cf_recommendations_for_eval(cf_model, user_id, profile_book_ids_set, all_books_info, n=50)
        
        # Manually perform the hybrid re-ranking on these valid candidates
        raw_cf_scores = {rec['book_id']: rec['score'] for rec in hybrid_candidate_recs}
        normalized_cf_scores = hybrid.normalize_scores(raw_cf_scores)
        candidate_ids_from_cf = list(normalized_cf_scores.keys())
        final_hybrid_scores = {}
        
        valid_hybrid_anchors = [aid for aid in profile_anchor_ids if aid in cb_book_map]

        for cand_id in candidate_ids_from_cf:
            norm_cf_score_val = normalized_cf_scores.get(cand_id, 0)
            avg_content_sim_score = 0.0

            if valid_hybrid_anchors: 
                content_sims = []
                for anchor_id in valid_hybrid_anchors:
                    if anchor_id in cb_book_map and cand_id in cb_book_map:
                        pair_similarity = cb.get_similarity_score_for_book_pair(anchor_id, cand_id, cb_sim_matrix, cb_book_map)
                        if isinstance(pair_similarity, (float, np.number)): 
                            content_sims.append(pair_similarity)
                if content_sims: 
                    avg_content_sim_score = sum(content_sims) / len(content_sims)
            
            norm_content_score_val = avg_content_sim_score
            hybrid_score = (0.5 * norm_content_score_val) + ((1 - 0.5) * norm_cf_score_val)
            final_hybrid_scores[cand_id] = hybrid_score

        sorted_hybrid_recs_items = sorted(final_hybrid_scores.items(), key=lambda item: item[1], reverse=True)
        
        # Format for precision calculation
        hybrid_recs_list = [{'book_id': book_id, 'score': h_score} for book_id, h_score in sorted_hybrid_recs_items]
        
        # --- Calculate precision for each k ---
        for k in K_VALUES:
            results['Content-Based'][k].append(calculate_precision_at_k(sorted_cb_recs, hold_out_book_ids, k=k))
            results['CF (SVD)'][k].append(calculate_precision_at_k(cf_recs, hold_out_book_ids, k=k))
            results['Hybrid'][k].append(calculate_precision_at_k(hybrid_recs_list, hold_out_book_ids, k=k))

    # --- Calculate and print average precision for each model and each k ---
    print("\n--- Ranking Evaluation Results ---")
    print(f"Number of users evaluated: {len(test_users)}")
    
    plot_data = []
    
    for model_name, k_results in results.items():
        print(f"\n--- {model_name} ---")
        for k, precisions in k_results.items():
            avg_precision = np.mean(precisions) if precisions else 0
            print(f"Average Precision@{k}: {avg_precision:.4f}")
            plot_data.append({'Model': model_name, 'k': k, 'Precision': avg_precision})
            
    # --- Plot the results ---
    print("\nGenerating line plot of evaluation results...")
    
    results_df = pd.DataFrame(plot_data)

    plt.figure(figsize=(10, 7))
    sns.lineplot(data=results_df, x='k', y='Precision', hue='Model', marker='o', style='Model', dashes=False)

    plt.title('Model Comparison: Precision@k', fontsize=16)
    plt.xlabel('Number of Recommendations (k)', fontsize=12)
    plt.ylabel('Average Precision', fontsize=12)
    plt.xticks(K_VALUES) 
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Recommender Model')
    
    plt.savefig(LINE_PLOT_IMG_PATH)
    print(f"Line plot saved to '{LINE_PLOT_IMG_PATH}'")


if __name__ == "__main__":
    run_ranking_evaluation()
