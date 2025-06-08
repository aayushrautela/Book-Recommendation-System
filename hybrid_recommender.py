import pandas as pd
import numpy as np
import os
import argparse 

import content_based_recommender as cb
import collaborative_recommender as cf # Refers to the SVD-based module

TOP_N_CF_CANDIDATES = 100 
TOP_N_ANCHOR_BOOKS = 5   
ALPHA = 0.4 # Weight for content score vs collaborative score; lesser the value more cf part
TOP_N_FINAL = 10

def get_top_n_anchor_books_for_user(user_id, ratings_data_to_use, books_info_df, top_n=TOP_N_ANCHOR_BOOKS):
    # Gets user's top N rated books as content anchors.
    user_ratings = ratings_data_to_use[ratings_data_to_use['user_id'] == user_id]
    if user_ratings.empty: return []

    sorted_user_ratings = user_ratings.sort_values(by='rating', ascending=False)
    top_rated_books_df = sorted_user_ratings.head(top_n)
    if top_rated_books_df.empty: return []
        
    anchor_book_ids = top_rated_books_df['book_id'].tolist()
    print(f"Hybrid: Identified {len(anchor_book_ids)} anchor(s) for User ID {user_id}.")
    return anchor_book_ids

def normalize_scores(scores_dict, new_min=0, new_max=1):
    # Scales a dictionary of scores to a 0-1 range.
    if not scores_dict: return {}
    values = list(scores_dict.values())
    if not values: return {}
    min_val, max_val = min(values), max(values)
    
    if max_val == min_val: 
        return {book_id: (new_min + new_max) / 2.0 for book_id in scores_dict}
    return {
        book_id: new_min + ((score - min_val) * (new_max - new_min) / (max_val - min_val))
        for book_id, score in scores_dict.items()
    }

def get_hybrid_recommendations(target_user_id, list_of_anchor_book_ids,
                               collab_model, all_books_info_df, # Now takes the trained CF model
                               content_cos_sim_matrix, content_books_data_for_cb, content_book_id_to_idx_map,
                               top_n_cf_candidates=TOP_N_CF_CANDIDATES, alpha=ALPHA, top_n_final=TOP_N_FINAL):
    # Generates hybrid recommendations using SVD model for CF part.
    if not list_of_anchor_book_ids and alpha > 0:
        print(f"Hybrid Warning: No anchor books for User ID {target_user_id} for content scoring.")

    # Get initial candidate books from the Collaborative Filtering SVD model.
    cf_recs_list = cf.get_top_n_cf_recommendations(collab_model, target_user_id, all_books_info_df, n=top_n_cf_candidates)
    
    if isinstance(cf_recs_list, str) or not cf_recs_list: 
        print(f"Hybrid: CF issue or no initial candidates: {cf_recs_list if isinstance(cf_recs_list, str) else 'empty list'}")
        return []

    raw_cf_scores = {rec['book_id']: rec['score'] for rec in cf_recs_list}
    normalized_cf_scores = normalize_scores(raw_cf_scores)
    
    candidate_ids_from_cf = list(normalized_cf_scores.keys())
    final_hybrid_scores = {}

    for cand_id in candidate_ids_from_cf:
        norm_cf_score_val = normalized_cf_scores.get(cand_id, 0)
        avg_content_sim_score = 0.0

        if list_of_anchor_book_ids and alpha > 0: 
            content_sims = []
            for anchor_id in list_of_anchor_book_ids:
                if anchor_id in content_book_id_to_idx_map and cand_id in content_book_id_to_idx_map:
                    pair_similarity = cb.get_similarity_score_for_book_pair(
                        anchor_id, cand_id, content_cos_sim_matrix, content_book_id_to_idx_map
                    )
                    if isinstance(pair_similarity, (float, np.number)): 
                        content_sims.append(pair_similarity)
            
            if content_sims: 
                avg_content_sim_score = sum(content_sims) / len(content_sims)
        
        norm_content_score_val = avg_content_sim_score 
        hybrid_score = (alpha * norm_content_score_val) + ((1 - alpha) * norm_cf_score_val)
        final_hybrid_scores[cand_id] = hybrid_score

    sorted_hybrid_recs = sorted(final_hybrid_scores.items(), key=lambda item: item[1], reverse=True)
    
    output_list = []
    for book_id, h_score in sorted_hybrid_recs[:top_n_final]:
        title_series = all_books_info_df.loc[all_books_info_df['book_id'] == book_id, 'title']
        title_val = title_series.iloc[0] if not title_series.empty else f"Book ID {book_id}"
        output_list.append({'book_id': book_id, 'title': title_val, 'hybrid_score': round(h_score, 4)})
    return output_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Book Recommender System.")
    parser.add_argument("user_id", type=int, help="User ID for whom to generate recommendations.")
    args = parser.parse_args()

    print(f"Initializing Hybrid Recommender System for User ID: {args.user_id}...")

    # 1. Setup Content-Based Model
    print("Setting up Content-Based Model...")
    cb_prep_data = cb.load_and_prepare_content_data()
    cb_sim_matrix, cb_books_data, cb_book_map, _ = (None, None, None, None)
    if cb_prep_data is not None and not cb_prep_data.empty:
        cb_sim_matrix, cb_books_data, cb_book_map, _ = cb.build_content_model(cb_prep_data)
    if cb_sim_matrix is None: exit("Hybrid Error: Failed Content-Based model initialization.")
    print("Content-Based Model setup complete.")

    # 2. Load the PRE-TRAINED Collaborative Filtering Model
    print("\nLoading Collaborative Filtering Model...")
    cf_model = cf.load_model()
    if cf_model is None:
        print("\nERROR: Collaborative Filtering model checkpoint not found.")
        print("Please run 'python collaborative_recommender.py train' first to train and save the model.")
        exit()
    print("Collaborative Filtering Model loaded successfully.")

    # We still need the original ratings data for anchor book selection
    # and the books_df for titles.
    all_ratings_data, all_books_info = cf.load_data_for_surprise()
    if all_ratings_data is None: exit("Hybrid Error: Could not load data for anchor book selection.")

    target_user = args.user_id
    print(f"\nGenerating recommendations for User ID: {target_user}...")
    
    # Get anchor books from all ratings for the target user
    # all_ratings_data from surprise is not a df, we need to access its .df attribute
    anchor_book_ids_list = get_top_n_anchor_books_for_user(
        target_user, all_ratings_data.df, all_books_info
    )
    valid_anchor_ids = [aid for aid in anchor_book_ids_list if aid in cb_book_map] if anchor_book_ids_list else []

    # Check if user exists in the CF model's training set
    try:
        cf_model.trainset.to_inner_uid(target_user)
    except ValueError:
        print(f"Warning: User ID {target_user} was not part of the training set for the CF model. Predictions may be based on global average.")

    hybrid_recs = get_hybrid_recommendations(
        target_user, valid_anchor_ids,
        cf_model, all_books_info, 
        cb_sim_matrix, cb_books_data, cb_book_map,
        alpha=ALPHA
    )

    if hybrid_recs:
        print(f"\nTop {TOP_N_FINAL} Hybrid Recommendations (alpha={ALPHA}):")
        for r_item in hybrid_recs:
            print(f"- \"{r_item['title']}\" (Book ID: {r_item['book_id']}, Hybrid Score: {r_item['hybrid_score']})")
    else:
        print(f"Hybrid: No recommendations were generated for User ID {target_user}.")
