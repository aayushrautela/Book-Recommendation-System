import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.sparse import csr_matrix
import argparse
import heapq 

DATA_DIR = 'data'
MIN_RATINGS_FOR_NEIGHBOR_CONSIDERATION = 175 # Min ratings for a user to be a potential neighbor. Lower if its takking too lomg.

DEFAULT_K_SIMILAR = 10 # Default number of similar users to find.
DEFAULT_N_RECS = 5     # Default number of recommendations to return.

def load_all_collaborative_data():
    # Loads all ratings and book titles.
    try:
        ratings_all_df = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'))
        books_info_df = pd.read_csv(os.path.join(DATA_DIR, 'books.csv'))[['book_id', 'title']]
    except FileNotFoundError as e:
        print(f"Collaborative: Error loading data: {e}")
        return None, None, None 
    return ratings_all_df, books_info_df, ratings_all_df 

def build_collaborative_model_components_on_all_data(all_ratings_df_input):
    # Prepares data for CF: sparse matrix (mean-centered), ID maps, user averages.
    if all_ratings_df_input is None or all_ratings_df_input.empty: 
        print("Collaborative: Ratings data empty for model build.")
        return None

    all_ratings_df = all_ratings_df_input.copy()

    user_averages = all_ratings_df.groupby('user_id')['rating'].mean()
    all_ratings_df['rating_centered'] = all_ratings_df.apply(
        lambda row: row['rating'] - user_averages.get(row['user_id'], row['rating']), axis=1
    )

    u_ids = sorted(all_ratings_df['user_id'].unique())
    user_map = {uid: i for i, uid in enumerate(u_ids)}
    idx_to_user = {i: uid for uid, i in user_map.items()}
    
    b_ids = sorted(all_ratings_df['book_id'].unique())
    book_map = {bid: i for i, bid in enumerate(b_ids)}
    idx_to_book = {i: bid for bid, i in book_map.items()}

    all_ratings_df['user_idx'] = all_ratings_df['user_id'].map(user_map)
    all_ratings_df['book_idx'] = all_ratings_df['book_id'].map(book_map)
    
    existing_ratings_set = set(zip(all_ratings_df['user_idx'], all_ratings_df['book_idx']))

    try:
        # Sparse matrix uses mean-centered ratings.
        sp_matrix = csr_matrix(
            (all_ratings_df['rating_centered'].values, 
             (all_ratings_df['user_idx'].values, all_ratings_df['book_idx'].values)),
            shape=(len(u_ids), len(b_ids))
        )
    except Exception as e:
        print(f"Collaborative: Sparse matrix error: {e}")
        return None
    
    return {
        "sparse_user_item_matrix": sp_matrix, 
        "user_to_idx": user_map, "idx_to_user": idx_to_user,
        "book_to_idx": book_map, "idx_to_book": idx_to_book,
        "all_user_ids": u_ids, "all_book_ids": b_ids,
        "user_averages": user_averages, 
        "existing_ratings_set": existing_ratings_set
    }

def find_top_n_similar_users_on_fly(target_user_m_idx, sparse_matrix, n=DEFAULT_K_SIMILAR, min_ratings_neighbor=MIN_RATINGS_FOR_NEIGHBOR_CONSIDERATION):
    # Finds top N similar users by on-the-fly similarity calculation.
    if target_user_m_idx is None or target_user_m_idx >= sparse_matrix.shape[0]: return []

    target_vector = sparse_matrix[target_user_m_idx] 
    user_rating_counts = np.array(sparse_matrix.getnnz(axis=1)) 

    similar_users_heap = [] # Min-heap for top N
    for other_user_m_idx in range(sparse_matrix.shape[0]):
        if other_user_m_idx == target_user_m_idx: continue 
        if user_rating_counts[other_user_m_idx] < min_ratings_neighbor: continue

        other_vector = sparse_matrix[other_user_m_idx]
        similarity = cosine_similarity(target_vector, other_vector)[0, 0]

        if similarity > 0: 
            if len(similar_users_heap) < n:
                heapq.heappush(similar_users_heap, (similarity, other_user_m_idx))
            else: 
                heapq.heappushpop(similar_users_heap, (similarity, other_user_m_idx))
    
    return sorted(similar_users_heap, key=lambda x: x[0], reverse=True) # Sort by similarity

def get_collaborative_score_for_user_book_pair_on_fly(user_id, book_id, model_comps, top_k_similar=DEFAULT_K_SIMILAR):
    # Predicts score for a single user-book pair.
    if model_comps is None: return "CF Model components missing."
    
    sp_matrix = model_comps["sparse_user_item_matrix"] 
    user_map = model_comps["user_to_idx"]
    book_map = model_comps["book_to_idx"]
    user_averages = model_comps["user_averages"]
    existing_ratings_set = model_comps["existing_ratings_set"]

    if user_id not in user_map: return f"User {user_id} not in model."
    if book_id not in book_map: return f"Book {book_id} not in model."

    user_m_idx, book_m_idx = user_map[user_id], book_map[book_id]
    
    if (user_m_idx, book_m_idx) in existing_ratings_set: # Check if user already rated
        return "User has already rated this book."

    similar_users_data = find_top_n_similar_users_on_fly(user_m_idx, sp_matrix, n=top_k_similar)
    if not similar_users_data: return f"Not enough similar users for {user_id}."

    weighted_sum_centered_ratings, sim_sum = 0.0, 0.0
    for similarity_val, sim_m_idx in similar_users_data:
        if (sim_m_idx, book_m_idx) in existing_ratings_set: # If similar user rated the book
            centered_rating_by_similar_user = sp_matrix[sim_m_idx, book_m_idx]
            weighted_sum_centered_ratings += (similarity_val * centered_rating_by_similar_user)
            sim_sum += similarity_val
            
    if sim_sum > 0:
        predicted_centered_score = weighted_sum_centered_ratings / sim_sum
        target_user_avg_r = user_averages.get(user_id)
        
        if target_user_avg_r is None: return round(predicted_centered_score, 4) # Fallback if avg not found
            
        final_predicted_score = predicted_centered_score + target_user_avg_r # De-normalize
        final_predicted_score = np.clip(final_predicted_score, 1.0, 5.0) # Clip to 1-5 range
        return round(final_predicted_score, 4)
        
    return f"Could not predict for Book {book_id} for User {user_id}."


def get_user_based_collaborative_recommendations_on_fly(user_id, model_comps, books_info_df, 
                                                      top_n_recs=DEFAULT_N_RECS, 
                                                      top_k_similar_users=DEFAULT_K_SIMILAR):
    # Gets top N recommendations for a user.
    if model_comps is None: return "CF Model components missing."
    
    user_map = model_comps["user_to_idx"]
    idx_to_book = model_comps["idx_to_book"]
    existing_ratings_set = model_comps["existing_ratings_set"]
    
    if user_id not in user_map: return f"User {user_id} not in CF model."
    user_m_idx = user_map[user_id]
    
    print(f"Collaborative: Finding {top_k_similar_users} similar users for User ID {user_id}...")
    # Find similar users once for all candidate books.
    similar_users_with_scores = find_top_n_similar_users_on_fly(
        user_m_idx, model_comps["sparse_user_item_matrix"], n=top_k_similar_users
    )
    
    if not similar_users_with_scores:
        return f"No similar users found for User {user_id}."

    candidate_scores = {}
    # Iterate through all books known to the model.
    for original_book_id in model_comps["all_book_ids"]: 
        book_m_idx = model_comps["book_to_idx"].get(original_book_id)
        if book_m_idx is None: continue

        # Predict score if target user hasn't rated this book.
        if (user_m_idx, book_m_idx) not in existing_ratings_set:
            weighted_sum_centered_ratings, sim_sum = 0.0, 0.0
            for similarity_val, sim_m_idx in similar_users_with_scores:
                if (sim_m_idx, book_m_idx) in existing_ratings_set: # If this similar user rated the candidate book
                    centered_rating = model_comps["sparse_user_item_matrix"][sim_m_idx, book_m_idx]
                    weighted_sum_centered_ratings += (similarity_val * centered_rating)
                    sim_sum += similarity_val
            
            if sim_sum > 0:
                predicted_centered_score = weighted_sum_centered_ratings / sim_sum
                target_user_avg_r = model_comps["user_averages"].get(user_id)
                if target_user_avg_r is not None:
                    final_score = predicted_centered_score + target_user_avg_r
                    final_score = np.clip(final_score, 1.0, 5.0)
                    candidate_scores[original_book_id] = final_score
    
    if not candidate_scores: return f"No scores predicted for User {user_id}."

    sorted_recs = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
    
    final_list = []
    for orig_book_id, score in sorted_recs[:top_n_recs]:
        title_s = books_info_df.loc[books_info_df['book_id'] == orig_book_id, 'title']
        title = title_s.iloc[0] if not title_s.empty else f"Book ID {orig_book_id}"
        final_list.append({'book_id': orig_book_id, 'title': title, 'score': round(score, 4)})
    return final_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="User-Based CF Recommender")
    parser.add_argument("user_id", type=int, help="User ID for recommendations.")
    args = parser.parse_args()

    print(f"Running On-the-fly Collaborative Recommender for User ID: {args.user_id}...")
    
    all_ratings_data, books_information, _ = load_all_collaborative_data() 
    
    if all_ratings_data is None or books_information is None:
        exit("Collaborative: Failed to load data.")

    model_components = build_collaborative_model_components_on_all_data(all_ratings_data)
        
    if model_components:
        target_user = args.user_id
        if target_user not in model_components["all_user_ids"]: 
            print(f"User ID {target_user} not found in the dataset.")
        else:
            # Uses DEFAULT_N_RECS and DEFAULT_K_SIMILAR from function signatures.
            print(f"\nGenerating CF Recs for User ID: {target_user} (Top {DEFAULT_N_RECS}, using {DEFAULT_K_SIMILAR} similar users)...")
            print("This may take time...")
            
            recs = get_user_based_collaborative_recommendations_on_fly(
                target_user, model_components, books_information 
            )
            
            if isinstance(recs, str): print(recs)
            elif recs:
                for r in recs: print(f"- \"{r['title']}\" (ID: {r['book_id']}, Predicted Score: {r['score']})")
            else: print(f"No CF recs for User {target_user}")
    else: print("CF model components could not be built.")