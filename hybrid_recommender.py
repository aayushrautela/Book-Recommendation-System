import pandas as pd
import numpy as np
import os

import content_based_recommender as cb
import collaborative_recommender as cf

TOP_N_CF_CANDIDATES = 50
TOP_N_ANCHOR_BOOKS = 3
ALPHA = 0.5 # Weight for content score vs collaborative score
TOP_N_FINAL = 10

def get_top_n_anchor_books_for_user(user_id, all_ratings_df, books_info_df, top_n=TOP_N_ANCHOR_BOOKS):
    # Gets user's top N rated books as content anchors.
    user_ratings = all_ratings_df[all_ratings_df['user_id'] == user_id]
    if user_ratings.empty: return []

    sorted_user_ratings = user_ratings.sort_values(by='rating', ascending=False)
    top_rated_books_df = sorted_user_ratings.head(top_n)
    if top_rated_books_df.empty: return []
        
    anchor_book_ids = top_rated_books_df['book_id'].tolist()
    return anchor_book_ids

def normalize_scores(scores_dict, new_min=0, new_max=1):
    # Scales scores to a 0-1 range.
    if not scores_dict: return {}
    values = list(scores_dict.values())
    if not values: return {}
    min_val, max_val = min(values), max(values)
    
    if max_val == min_val: # All scores are identical
        return {book_id: (new_min + new_max) / 2.0 for book_id in scores_dict}
    return {
        book_id: new_min + ((score - min_val) * (new_max - new_min) / (max_val - min_val))
        for book_id, score in scores_dict.items()
    }

def get_hybrid_recommendations(target_user_id, list_of_anchor_book_ids,
                               collab_model_comps, books_data_cf_titles,
                               content_cos_sim_matrix, content_books_data_for_cb, content_book_id_to_idx_map,
                               top_n_cf_candidates=TOP_N_CF_CANDIDATES, alpha=ALPHA, top_n_final=TOP_N_FINAL):
    # Generates hybrid recommendations by combining CF and CB scores.

    if not list_of_anchor_book_ids:
        print(f"Hybrid Warning: No anchor books for User ID {target_user_id}. Content scores will be 0.")

    # Get initial candidates from Collaborative Filtering
    cf_recs_list = cf.get_user_based_collaborative_recommendations(
        target_user_id, collab_model_comps, books_data_cf_titles, top_n=top_n_cf_candidates
    )
    if isinstance(cf_recs_list, str) or not cf_recs_list: 
        print(f"Hybrid: CF issue or no initial recs: {cf_recs_list if isinstance(cf_recs_list, str) else 'empty list'}")
        return []

    raw_cf_scores = {rec['book_id']: rec['score'] for rec in cf_recs_list}
    normalized_cf_scores = normalize_scores(raw_cf_scores)
    
    candidate_ids = list(normalized_cf_scores.keys())
    final_hybrid_scores = {}

    for cand_id in candidate_ids:
        norm_cf = normalized_cf_scores.get(cand_id, 0)
        avg_content_sim = 0.0

        if list_of_anchor_book_ids: # Calculate content score if anchors exist
            content_sims = []
            for anchor_id in list_of_anchor_book_ids:
                if anchor_id in content_book_id_to_idx_map and cand_id in content_book_id_to_idx_map:
                    pair_sim = cb.get_similarity_score_for_book_pair(
                        anchor_id, cand_id, content_cos_sim_matrix, content_book_id_to_idx_map
                    )
                    if isinstance(pair_sim, (float, np.float32, np.float64)):
                        content_sims.append(pair_sim)
            if content_sims: avg_content_sim = sum(content_sims) / len(content_sims)
        
        norm_content = avg_content_sim # Already 0-1

        hybrid_score = (alpha * norm_content) + ((1 - alpha) * norm_cf)
        final_hybrid_scores[cand_id] = hybrid_score

    sorted_recs = sorted(final_hybrid_scores.items(), key=lambda item: item[1], reverse=True)
    
    output_list = []
    for book_id, h_score in sorted_recs[:top_n_final]:
        title_s = books_data_cf_titles.loc[books_data_cf_titles['book_id'] == book_id, 'title']
        title = title_s.iloc[0] if not title_s.empty else f"Book ID {book_id}"
        output_list.append({'book_id': book_id, 'title': title, 'hybrid_score': round(h_score, 4)})
    return output_list

if __name__ == "__main__":
    print("Initializing Hybrid Recommender System...")

    cb_data = cb.load_and_prepare_content_data()
    cb_cos_sim, cb_books_df, cb_id_to_idx, _ = (None, None, None, None)
    if cb_data is not None and not cb_data.empty:
        cb_cos_sim, cb_books_df, cb_id_to_idx, _ = cb.build_content_model(cb_data)
    if cb_cos_sim is None: exit("Hybrid Error: Failed Content-Based model init.")
    print("Content-Based Model setup complete.")

    cf_ratings, cf_book_titles = cf.load_and_filter_collaborative_data(
        cf.MIN_RATINGS_PER_USER, cf.MIN_RATINGS_PER_BOOK
    )
    cf_model_comps = None
    if cf_ratings is not None and cf_book_titles is not None:
        cf_model_comps = cf.build_collaborative_model_components(cf_ratings)
    if cf_model_comps is None: exit("Hybrid Error: Failed Collaborative Filtering model init.")
    print("Collaborative Filtering Model setup complete.")

    example_user = 75 if 75 in cf_model_comps["unique_user_ids_filtered"] else (
        cf_model_comps["unique_user_ids_filtered"][0] if cf_model_comps["unique_user_ids_filtered"].size > 0 else None
    )
    if example_user is None: exit("\nHybrid Error: No example user ID found.")

    print(f"\nGenerating Hybrid Recommendations for User ID: {example_user}...")
    
    anchor_ids = get_top_n_anchor_books_for_user(
        example_user, cf_model_comps["filtered_ratings_df"], cf_book_titles
    )

    valid_anchors = [aid for aid in anchor_ids if aid in cb_id_to_idx] if anchor_ids else []
    if anchor_ids and not valid_anchors:
        print(f"Hybrid Warning: User {example_user}'s anchor books not in content model. Content score will be 0.")
    elif anchor_ids and len(valid_anchors) < len(anchor_ids):
        print(f"Hybrid Warning: Some anchor books for user {example_user} not in content model.")

    hybrid_recs = get_hybrid_recommendations(
        example_user, valid_anchors,
        cf_model_comps, cf_book_titles,
        cb_cos_sim, cb_books_df, cb_id_to_idx,
        alpha=ALPHA 
    )

    if hybrid_recs:
        print(f"\nTop {TOP_N_FINAL} Hybrid Recommendations (alpha={ALPHA}):")
        for r in hybrid_recs:
            print(f"- \"{r['title']}\" (Book ID: {r['book_id']}, Hybrid Score: {r['hybrid_score']})")
    else:
        print("Hybrid: No recommendations generated.")