import pandas as pd
import numpy as np
import os
import argparse 

import content_based_recommender as cb
import collaborative_recommender as cf

TOP_N_CF_CANDIDATES = 50
TOP_N_ANCHOR_BOOKS = 5
ALPHA = 0.5 # Weight: content score vs collaborative score
TOP_N_FINAL = 10

def get_top_n_anchor_books_for_user(user_id, ratings_data_to_use, books_info_df, top_n=TOP_N_ANCHOR_BOOKS):
    # Gets user's top N rated books as content anchors from the provided ratings data.
    user_ratings = ratings_data_to_use[ratings_data_to_use['user_id'] == user_id]
    if user_ratings.empty: return []

    sorted_user_ratings = user_ratings.sort_values(by='rating', ascending=False)
    top_rated_books_df = sorted_user_ratings.head(top_n)
    if top_rated_books_df.empty: return []
        
    anchor_book_ids = top_rated_books_df['book_id'].tolist()
    print(f"Hybrid: Identified {len(anchor_book_ids)} anchor(s) for User ID {user_id}.") # Feedback
    return anchor_book_ids

def normalize_scores(scores_dict, new_min=0, new_max=1):
    # Scales a dictionary of scores to a 0-1 range.
    if not scores_dict: return {}
    values = list(scores_dict.values())
    if not values: return {}
    min_val, max_val = min(values), max(values)
    
    if max_val == min_val: # Avoid division by zero if all scores are identical
        return {book_id: (new_min + new_max) / 2.0 for book_id in scores_dict}
    return {
        book_id: new_min + ((score - min_val) * (new_max - new_min) / (max_val - min_val))
        for book_id, score in scores_dict.items()
    }

def get_hybrid_recommendations(target_user_id, list_of_anchor_book_ids,
                               collab_model_comps, books_data_cf_titles,
                               content_cos_sim_matrix, content_books_data_for_cb, content_book_id_to_idx_map,
                               top_n_cf_candidates=TOP_N_CF_CANDIDATES, alpha=ALPHA, top_n_final=TOP_N_FINAL):
    # Generates hybrid recommendations.
    if not list_of_anchor_book_ids and alpha > 0: # Content part has weight but no anchors
        print(f"Hybrid Warning: No anchor books for User ID {target_user_id} to use for content scoring.")

    # Get CF candidates
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
        avg_content_sim = 0.0 # Default content score

        if list_of_anchor_book_ids and alpha > 0: # Only compute if anchors exist and content matters
            content_sims = []
            for anchor_id in list_of_anchor_book_ids:
                # Ensure both anchor and candidate are in the content model's map
                if anchor_id in content_book_id_to_idx_map and cand_id in content_book_id_to_idx_map:
                    pair_sim = cb.get_similarity_score_for_book_pair(
                        anchor_id, cand_id, content_cos_sim_matrix, content_book_id_to_idx_map
                    )
                    if isinstance(pair_sim, (float, np.float32, np.float64)): # Check for valid score
                        content_sims.append(pair_sim)
            if content_sims: avg_content_sim = sum(content_sims) / len(content_sims)
        
        norm_content = avg_content_sim # Assumed to be 0-1 from cosine similarity

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
    parser = argparse.ArgumentParser(description="Hybrid Book Recommender System.")
    parser.add_argument("user_id", type=int, help="User ID for whom to generate recommendations.")
    args = parser.parse_args()

    print(f"Initializing Hybrid Recommender System for User ID: {args.user_id}...")

    # Setup Content-Based Model
    cb_data = cb.load_and_prepare_content_data()
    cb_cos_sim, cb_books_df_for_cb, cb_id_to_idx, _ = (None, None, None, None)
    if cb_data is not None and not cb_data.empty:
        cb_cos_sim, cb_books_df_for_cb, cb_id_to_idx, _ = cb.build_content_model(cb_data)
    if cb_cos_sim is None: exit("Hybrid Error: Failed Content-Based model init.")
    print("Content-Based Model setup complete.")

    # Setup Collaborative Filtering Model (and get original ratings)
    cf_ratings_filtered, cf_book_titles, cf_ratings_original = cf.load_and_filter_collaborative_data(
        cf.MIN_RATINGS_PER_USER, cf.MIN_RATINGS_PER_BOOK
    )
    cf_model_comps = None
    if cf_ratings_filtered is not None and not cf_ratings_filtered.empty and cf_book_titles is not None:
        cf_model_comps = cf.build_collaborative_model_components(cf_ratings_filtered)
    
    if cf_ratings_original is None or cf_book_titles is None : 
        exit("Hybrid Error: Failed to load basic ratings/book title data.")
    
    if cf_model_comps is None : 
        print("Collaborative Filtering Model setup failed or produced no model. Fallback may be used.")
    else: 
        print("Collaborative Filtering Model setup complete.")

    target_user_for_hybrid = args.user_id
    print(f"\nGenerating recommendations for User ID: {target_user_for_hybrid}...")
    
    user_in_cf_model = cf_model_comps and target_user_for_hybrid in cf_model_comps["unique_user_ids_filtered"]

    if user_in_cf_model:
        print(f"User ID {target_user_for_hybrid} found in CF model. Proceeding with Hybrid Recommendations.")
        # For users in CF model, anchor books from their (filtered) CF interactions might be more relevant for hybrid
        anchor_ids = get_top_n_anchor_books_for_user(
            target_user_for_hybrid, cf_model_comps["filtered_ratings_df"], cf_book_titles
        )
        valid_anchors = [aid for aid in anchor_ids if aid in cb_id_to_idx] if anchor_ids else []
        
        hybrid_recs = get_hybrid_recommendations(
            target_user_for_hybrid, valid_anchors,
            cf_model_comps, cf_book_titles,
            cb_cos_sim, cb_books_df_for_cb, cb_id_to_idx,
            alpha=ALPHA 
        )
        if hybrid_recs:
            print(f"\nTop {TOP_N_FINAL} Hybrid Recommendations (alpha={ALPHA}):")
            for r in hybrid_recs:
                print(f"- \"{r['title']}\" (Book ID: {r['book_id']}, Hybrid Score: {r['hybrid_score']})")
        else:
            print(f"Hybrid: No hybrid recommendations generated for User ID {target_user_for_hybrid}.")

    else: # Fallback to Content-Based
        print(f"User ID {target_user_for_hybrid} not in CF model or CF model failed. Attempting Content-Based Fallback.")
        anchor_ids = get_top_n_anchor_books_for_user(
            target_user_for_hybrid, cf_ratings_original, cf_book_titles # Use original ratings for fallback anchors
        )
        if not anchor_ids:
            print(f"Fallback: No anchor books found for User ID {target_user_for_hybrid}. Cannot provide content-based recommendations.")
        else:
            valid_anchors_for_cb = [aid for aid in anchor_ids if aid in cb_id_to_idx]
            if not valid_anchors_for_cb:
                print(f"Fallback: Anchor books for User {target_user_for_hybrid} found, but none exist in content model.")
            else:
                print(f"Fallback: Using {len(valid_anchors_for_cb)} anchor book(s) for pure content-based recommendations.")
                # Simple fallback: use the first valid anchor book
                first_valid_anchor = valid_anchors_for_cb[0]
                cb_fallback_recs = cb.get_content_based_recommendations(
                    first_valid_anchor, top_n=TOP_N_FINAL,
                    cos_sim_matrix=cb_cos_sim, books_data_df=cb_books_df_for_cb,
                    book_id_to_idx_map=cb_id_to_idx
                )
                if isinstance(cb_fallback_recs, str): 
                    print(f"Fallback Error: {cb_fallback_recs}")
                elif cb_fallback_recs:
                    print(f"\nTop {TOP_N_FINAL} Content-Based Fallback Recommendations (based on anchor ID {first_valid_anchor}):")
                    for r in cb_fallback_recs:
                        print(f"- \"{r['title']}\" (Book ID: {r['book_id']}, Similarity Score: {r['score']})")
                else:
                    print(f"Fallback: No content-based recommendations for anchor ID {first_valid_anchor}.")