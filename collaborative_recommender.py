import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.sparse import csr_matrix

DATA_DIR = 'data'
MIN_RATINGS_PER_USER = 150 # For managing memory
MIN_RATINGS_PER_BOOK = 50

def load_and_filter_collaborative_data(min_ratings_user, min_ratings_book):
    # Loads and filters ratings and books data.
    try:
        ratings_orig = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'))
        books_info = pd.read_csv(os.path.join(DATA_DIR, 'books.csv'))[['book_id', 'title']]
        # print("Collaborative: Data loaded.") # Optional
    except FileNotFoundError as e:
        print(f"Collaborative: Error loading: {e}")
        return None, None
    
    user_counts = ratings_orig['user_id'].value_counts()
    active_users = user_counts[user_counts >= min_ratings_user].index
    ratings_f_users = ratings_orig[ratings_orig['user_id'].isin(active_users)]

    book_counts = ratings_f_users['book_id'].value_counts()
    popular_books = book_counts[book_counts >= min_ratings_book].index
    final_ratings = ratings_f_users[ratings_f_users['book_id'].isin(popular_books)].copy()

    # print(f"Collaborative: Filtered ratings: {len(final_ratings)}") # Optional
    if final_ratings.empty:
        print("Collaborative: No data post-filtering.")
        return None, None
    return final_ratings, books_info

def build_collaborative_model_components(filtered_ratings_df):
    # Builds sparse matrix, similarity matrix, and ID maps.
    if filtered_ratings_df is None or filtered_ratings_df.empty: return None

    num_users = filtered_ratings_df['user_id'].nunique()
    num_books = filtered_ratings_df['book_id'].nunique()

    u_ids = filtered_ratings_df['user_id'].unique()
    user_map = {uid: i for i, uid in enumerate(u_ids)}
    idx_to_user = {i: uid for uid, i in user_map.items()}
    b_ids = filtered_ratings_df['book_id'].unique()
    book_map = {bid: i for i, bid in enumerate(b_ids)}
    idx_to_book = {i: bid for bid, i in book_map.items()}

    mapped_user_idx = filtered_ratings_df['user_id'].map(user_map)
    mapped_book_idx = filtered_ratings_df['book_id'].map(book_map)

    try:
        sp_matrix = csr_matrix(
            (filtered_ratings_df['rating'].values, (mapped_user_idx, mapped_book_idx)),
            shape=(num_users, num_books)
        )
        # print("Collaborative: Sparse Matrix shape:", sp_matrix.shape) # Optional
    except Exception as e:
        print(f"Collaborative: Sparse matrix error: {e}")
        return None

    try:
        u_sim_dense = cosine_similarity(sp_matrix)
    except MemoryError:
        print(f"Collaborative: MemoryError for user similarity (users: {num_users}). Increase filtering.")
        return None
    except Exception as e:
        print(f"Collaborative: User similarity error: {e}")
        return None

    u_sim_df = pd.DataFrame(u_sim_dense, index=u_ids, columns=u_ids)
    # print("Collaborative: User Similarity DF shape:", u_sim_df.shape) # Optional
    
    return {
        "user_similarity_df": u_sim_df, "sparse_user_item_matrix": sp_matrix,
        "user_to_idx": user_map, "idx_to_user": idx_to_user,
        "book_to_idx": book_map, "idx_to_book": idx_to_book,
        "unique_user_ids_filtered": u_ids, "unique_book_ids_filtered": b_ids,
        "filtered_ratings_df": filtered_ratings_df
    }

def get_collaborative_score_for_user_book_pair(user_id, book_id, model_comps, min_sim_users=5, min_sim_thresh=0.1):
    # Predicts score for one user-book pair.
    if model_comps is None: return "Model missing."
    
    user_sim_df = model_comps["user_similarity_df"]
    sp_matrix = model_comps["sparse_user_item_matrix"]
    user_map = model_comps["user_to_idx"]
    book_map = model_comps["book_to_idx"]

    if user_id not in user_map: return f"User {user_id} not in model."
    if book_id not in book_map: return f"Book {book_id} not in model."

    user_m_idx, book_m_idx = user_map[user_id], book_map[book_id]
    if sp_matrix[user_m_idx, book_m_idx] > 0: return "User rated this book."

    sim_users_s = user_sim_df[user_id].sort_values(ascending=False)
    sim_users_s = sim_users_s.drop(user_id, errors='ignore')[sim_users_s > min_sim_thresh]

    if len(sim_users_s) < min_sim_users: return f"Not enough similar users for {user_id}."

    sim_ids = sim_users_s.index.tolist()
    weighted_sum, sim_sum = 0, 0
    
    for sim_uid in sim_ids:
        if sim_uid not in user_map: continue
        sim_m_idx = user_map[sim_uid]
        rating = sp_matrix[sim_m_idx, book_m_idx]
        if rating > 0:
            sim_val = sim_users_s[sim_uid]
            weighted_sum += (sim_val * rating)
            sim_sum += sim_val
            
    if sim_sum > 0: return round(weighted_sum / sim_sum, 4)
    return f"Could not predict for Book {book_id}, User {user_id} (no similar users rated it)."

def get_user_based_collaborative_recommendations(user_id, model_comps, books_info, top_n=10, min_sim_users=5, min_sim_thresh=0.1):
    # Gets top N recommendations for a user.
    if model_comps is None: return "Model missing."
    
    user_map = model_comps["user_to_idx"]
    idx_to_book = model_comps["idx_to_book"]
    sp_matrix = model_comps["sparse_user_item_matrix"]

    if user_id not in user_map: return f"User {user_id} not in model."
    user_m_idx = user_map[user_id]
    rated_book_indices = sp_matrix[user_m_idx, :].indices
    
    cand_scores = {}
    for book_m_idx in range(sp_matrix.shape[1]): # Iterate all book model indices
        if book_m_idx not in rated_book_indices:
            orig_book_id = idx_to_book[book_m_idx]
            pred_score = get_collaborative_score_for_user_book_pair(
                user_id, orig_book_id, model_comps, min_sim_users, min_sim_thresh
            )
            if isinstance(pred_score, float): cand_scores[orig_book_id] = pred_score
    
    if not cand_scores: return f"No scores predicted for User {user_id}."

    sorted_recs = sorted(cand_scores.items(), key=lambda item: item[1], reverse=True)
    
    final_list = []
    for orig_book_id, score in sorted_recs[:top_n]:
        title_s = books_info.loc[books_info['book_id'] == orig_book_id, 'title']
        title = title_s.iloc[0] if not title_s.empty else f"Book ID {orig_book_id}"
        final_list.append({'book_id': orig_book_id, 'title': title, 'score': score})
    return final_list

if __name__ == "__main__":
    print("Running Collaborative Recommender Standalone...")
    
    ratings, books = load_and_filter_collaborative_data(MIN_RATINGS_PER_USER, MIN_RATINGS_PER_BOOK)
    if ratings is not None and books is not None:
        model_components = build_collaborative_model_components(ratings)
        
        if model_components:
            example_user = (75 if 75 in model_components["unique_user_ids_filtered"] 
                            else model_components["unique_user_ids_filtered"][0] 
                            if model_components["unique_user_ids_filtered"].size > 0 else None)
            
            if example_user:
                print(f"\nCollaborative Recs for User ID: {example_user} (Top 5)...")
                recs = get_user_based_collaborative_recommendations(example_user, model_components, books, 5)
                if isinstance(recs, str): print(recs)
                elif recs:
                    for r in recs: print(f"- \"{r['title']}\" (ID: {r['book_id']}, Score: {r['score']})")
                else: print(f"No recs for User {example_user}")

                if model_components["unique_book_ids_filtered"].size > 0:
                    bid_score_test = None
                    user_m_idx_ex = model_components["user_to_idx"][example_user]
                    for b_test_id in model_components["unique_book_ids_filtered"]:
                        b_m_idx_test = model_components["book_to_idx"][b_test_id]
                        if model_components["sparse_user_item_matrix"][user_m_idx_ex, b_m_idx_test] == 0: # Not rated
                            bid_score_test = b_test_id
                            break
                    if bid_score_test:
                        score = get_collaborative_score_for_user_book_pair(example_user, bid_score_test, model_components)
                        title = books.loc[books['book_id'] == bid_score_test, 'title'].iloc[0]
                        print(f"\nPredicted score for User {example_user}, Book {bid_score_test} ('{title}'): {score}")
            else: print("No example user in filtered data.")
            
            print(f"\nCollaborative Recs for User ID: 0 (Non-existent)...")
            print(get_user_based_collaborative_recommendations(0, model_components, books))
        else: print("Collaborative model building failed.")
    else: print("Collaborative data loading/filtering failed.")