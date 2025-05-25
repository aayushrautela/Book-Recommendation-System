import pandas as pd
import numpy as np
import os
import argparse 

import content_based_recommender as cb
import collaborative_recommender as cf 

TOP_N_CF_CANDIDATES = 50 # Initial candidates from CF for hybrid processing.
TOP_N_ANCHOR_BOOKS = 3   # Number of user's top books for content matching.
ALPHA = 0.5              # Weight for combining content vs. collaborative scores.
TOP_N_FINAL = 10         # Final number of hybrid recommendations.

def get_top_n_anchor_books_for_user(user_id, ratings_data_to_use, books_info_df, top_n=TOP_N_ANCHOR_BOOKS):
    # Gets user's top N rated books to use as content anchors.
    user_ratings = ratings_data_to_use[ratings_data_to_use['user_id'] == user_id]
    if user_ratings.empty: return []

    sorted_user_ratings = user_ratings.sort_values(by='rating', ascending=False)
    top_rated_books_df = sorted_user_ratings.head(top_n)
    if top_rated_books_df.empty: return []
        
    anchor_book_ids = top_rated_books_df['book_id'].tolist()
    print(f"Hybrid: Identified {len(anchor_book_ids)} anchor(s) for User ID {user_id}.") # User feedback.
    return anchor_book_ids

def normalize_scores(scores_dict, new_min=0, new_max=1):
    # Scales a dictionary of scores to a 0-1 range.
    if not scores_dict: return {}
    values = list(scores_dict.values())
    if not values: return {}
    min_val, max_val = min(values), max(values)
    
    if max_val == min_val: # All scores are identical.
        return {book_id: (new_min + new_max) / 2.0 for book_id in scores_dict}
    return {
        book_id: new_min + ((score - min_val) * (new_max - new_min) / (max_val - min_val))
        for book_id, score in scores_dict.items()
    }

def get_hybrid_recommendations(target_user_id, list_of_anchor_book_ids,
                               collab_model_comps, books_info_for_titles, 
                               content_cos_sim_matrix, content_books_data_for_cb, content_book_id_to_idx_map,
                               top_n_cf_candidates=TOP_N_CF_CANDIDATES, alpha=ALPHA, top_n_final=TOP_N_FINAL):
    # Generates hybrid recommendations by combining CF and CB scores.
    if not list_of_anchor_book_ids and alpha > 0: # Content has weight but no anchors.
        print(f"Hybrid Warning: No anchor books for User ID {target_user_id} for content scoring.")

    # Get CF candidates; CF function uses its internal default for num_similar_users.
    cf_recs_list = cf.get_user_based_collaborative_recommendations_on_fly(
        target_user_id, collab_model_comps, books_info_for_titles, 
        top_n_recs=top_n_cf_candidates 
    )
    if isinstance(cf_recs_list, str) or not cf_recs_list: # Handle error or empty list from CF.
        print(f"Hybrid: CF issue or no initial candidates: {cf_recs_list if isinstance(cf_recs_list, str) else 'empty list'}")
        return []

    raw_cf_scores = {rec['book_id']: rec['score'] for rec in cf_recs_list}
    normalized_cf_scores = normalize_scores(raw_cf_scores) # Normalize CF scores to 0-1.
    
    candidate_ids_from_cf = list(normalized_cf_scores.keys())
    final_hybrid_scores = {} # Stores {book_id: final_hybrid_score}.

    for cand_id in candidate_ids_from_cf:
        norm_cf_score_val = normalized_cf_scores.get(cand_id, 0)
        avg_content_sim_score = 0.0 # Default content score.

        # Calculate average content similarity if anchors exist and content component has weight.
        if list_of_anchor_book_ids and alpha > 0: 
            content_similarities_for_candidate = []
            for anchor_id in list_of_anchor_book_ids:
                # Both anchor and candidate must be known to the content model.
                if anchor_id in content_book_id_to_idx_map and cand_id in content_book_id_to_idx_map:
                    pair_similarity = cb.get_similarity_score_for_book_pair(
                        anchor_id, cand_id, content_cos_sim_matrix, content_book_id_to_idx_map
                    )
                    if isinstance(pair_similarity, (float, np.number)): # Check for valid numeric score.
                        content_similarities_for_candidate.append(pair_similarity)
            
            if content_similarities_for_candidate: # If any valid scores found.
                avg_content_sim_score = sum(content_similarities_for_candidate) / len(content_similarities_for_candidate)
        
        norm_content_score_val = avg_content_sim_score # Cosine similarity is already 0-1.

        hybrid_score = (alpha * norm_content_score_val) + ((1 - alpha) * norm_cf_score_val) # Weighted sum.
        final_hybrid_scores[cand_id] = hybrid_score

    sorted_hybrid_recs = sorted(final_hybrid_scores.items(), key=lambda item: item[1], reverse=True)
    
    output_list = []
    for book_id, h_score in sorted_hybrid_recs[:top_n_final]: # Format top N final results.
        title_series = books_info_for_titles.loc[books_info_for_titles['book_id'] == book_id, 'title']
        title_val = title_series.iloc[0] if not title_series.empty else f"Book ID {book_id}"
        output_list.append({'book_id': book_id, 'title': title_val, 'hybrid_score': round(h_score, 4)})
    return output_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Book Recommender System.")
    parser.add_argument("user_id", type=int, help="User ID for recommendations.")
    args = parser.parse_args()

    print(f"Initializing Hybrid Recommender System for User ID: {args.user_id}...")

    # Setup Content-Based model.
    cb_prep_data = cb.load_and_prepare_content_data()
    cb_sim_matrix, cb_books_data, cb_book_map, _ = (None, None, None, None)
    if cb_prep_data is not None and not cb_prep_data.empty:
        cb_sim_matrix, cb_books_data, cb_book_map, _ = cb.build_content_model(cb_prep_data)
    if cb_sim_matrix is None: exit("Hybrid Error: Failed Content-Based model initialization.")
    print("Content-Based Model setup complete.")

    # Setup Collaborative Filtering model.
    print("\nSetting up Collaborative Filtering Model...")
    cf_all_ratings, cf_all_books, _ = cf.load_all_collaborative_data() 
    
    cf_model_components_dict = None 
    if cf_all_ratings is None or cf_all_books is None : 
        exit("Hybrid Error: Failed to load data for CF model.")
    
    cf_model_components_dict = cf.build_collaborative_model_components_on_all_data(cf_all_ratings) 
    
    if cf_model_components_dict is None : 
        print("CF Model components could not be built. Fallback may be attempted.")
    else: 
        print("Collaborative Filtering Model setup complete.")

    target_user = args.user_id
    print(f"\nGenerating recommendations for User ID: {target_user}...")
    
    user_has_ratings_flag = not cf_all_ratings[cf_all_ratings['user_id'] == target_user].empty

    if not user_has_ratings_flag:
        print(f"User ID {target_user} has no ratings. Cannot generate recommendations.")
    else:
        # Get anchor books from all ratings for the target user.
        anchor_book_ids_list = get_top_n_anchor_books_for_user(
            target_user, cf_all_ratings, cf_all_books 
        )
        # Filter anchor books to those present in the content model.
        valid_anchor_ids = [aid for aid in anchor_book_ids_list if aid in cb_book_map] if anchor_book_ids_list else []

        # Check if user can be processed by the CF component.
        user_in_cf_scope_flag = cf_model_components_dict and target_user in cf_model_components_dict["user_to_idx"]

        if user_in_cf_scope_flag: 
            print(f"User ID {target_user} processable by CF. Attempting Hybrid Recommendations.")
            hybrid_recommendations = get_hybrid_recommendations(
                target_user, valid_anchor_ids,
                cf_model_components_dict, cf_all_books, 
                cb_sim_matrix, cb_books_data, cb_book_map,
                alpha=ALPHA 
            )
            if hybrid_recommendations:
                print(f"\nTop {TOP_N_FINAL} Hybrid Recommendations (alpha={ALPHA}):")
                for r_item in hybrid_recommendations:
                    print(f"- \"{r_item['title']}\" (Book ID: {r_item['book_id']}, Hybrid Score: {r_item['hybrid_score']})")
            else: 
                print(f"Hybrid: No hybrid recommendations generated for User ID {target_user}.")
                # Deeper fallback if hybrid yields nothing but anchors existed.
                if valid_anchor_ids: 
                     print("Attempting pure Content-Based due to empty hybrid result with valid anchors.")
                     # (Fallback logic as previously implemented)
                     aggregated_fallback_recs = {} 
                     for anchor_id_val in valid_anchor_ids:
                         cb_recs_one_anchor = cb.get_content_based_recommendations(
                             anchor_id_val, top_n=(TOP_N_FINAL + 5), 
                             cos_sim_matrix=cb_sim_matrix, books_data_df=cb_books_data, 
                             book_id_to_idx_map=cb_book_map
                         )
                         if isinstance(cb_recs_one_anchor, list):
                             for rec_item in cb_recs_one_anchor:
                                 rec_book_id_val = rec_item['book_id']
                                 if rec_book_id_val not in valid_anchor_ids: 
                                     if rec_book_id_val not in aggregated_fallback_recs or \
                                        rec_item['score'] > aggregated_fallback_recs[rec_book_id_val]['highest_score']:
                                         aggregated_fallback_recs[rec_book_id_val] = {
                                             'title': rec_item['title'], 'highest_score': rec_item['score']
                                         }
                     if aggregated_fallback_recs:
                         sorted_aggregated_list = sorted(
                             aggregated_fallback_recs.items(), key=lambda item: item[1]['highest_score'], reverse=True
                         )
                         print(f"\nTop {TOP_N_FINAL} Content-Based Fallback Recommendations (aggregated):")
                         for i, (b_id, r_data) in enumerate(sorted_aggregated_list):
                             if i >= TOP_N_FINAL: break
                             print(f"- \"{r_data['title']}\" (Book ID: {b_id}, Highest Similarity Score: {r_data['highest_score']})")
                     else: print(f"Fallback: No content-based recommendations could be aggregated.")
        
        elif valid_anchor_ids: # User not in CF scope, but has valid anchors for CB fallback.
            print(f"User ID {target_user} not in CF scope or CF model error. Attempting Content-Based Fallback.")
            
            aggregated_fallback_recs = {} 
            for anchor_id_val in valid_anchor_ids:
                cb_recs_one_anchor = cb.get_content_based_recommendations(
                    anchor_id_val, top_n=(TOP_N_FINAL + 5), 
                    cos_sim_matrix=cb_sim_matrix, books_data_df=cb_books_data, 
                    book_id_to_idx_map=cb_book_map
                )
                if isinstance(cb_recs_one_anchor, list):
                    for rec_item in cb_recs_one_anchor:
                        rec_book_id_val = rec_item['book_id']
                        if rec_book_id_val not in valid_anchor_ids: 
                            if rec_book_id_val not in aggregated_fallback_recs or \
                               rec_item['score'] > aggregated_fallback_recs[rec_book_id_val]['highest_score']:
                                aggregated_fallback_recs[rec_book_id_val] = {
                                    'title': rec_item['title'], 'highest_score': rec_item['score']
                                }
            
            if aggregated_fallback_recs:
                sorted_aggregated_list = sorted(
                    aggregated_fallback_recs.items(), key=lambda item: item[1]['highest_score'], reverse=True
                )
                print(f"\nTop {TOP_N_FINAL} Content-Based Fallback Recommendations (aggregated):")
                for i, (b_id, r_data) in enumerate(sorted_aggregated_list):
                    if i >= TOP_N_FINAL: break
                    print(f"- \"{r_data['title']}\" (Book ID: {b_id}, Highest Similarity Score: {r_data['highest_score']})")
            else:
                print(f"Fallback: No content-based recommendations could be aggregated.")
        else: # No ratings or no valid anchors.
            print(f"User ID {target_user} has no valid anchor books in content model. Cannot provide recommendations.")