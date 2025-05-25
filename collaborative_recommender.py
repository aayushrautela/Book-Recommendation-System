import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.sparse import csr_matrix # Import sparse matrix

# --- Configuration ---
DATA_DIR = 'data'
# Try these filtering values. If memory issues persist for user_similarity_matrix,
# you might need to increase these further (e.g., 75 or 100),
# which means fewer users and books in the model.
MIN_RATINGS_PER_USER = 150
MIN_RATINGS_PER_BOOK = 50

# --- Load Data ---
try:
    ratings_df_original = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'))
    books_df = pd.read_csv(os.path.join(DATA_DIR, 'books.csv'))[['book_id', 'title']]
    print("Original Ratings and Books data loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading datasets: {e}")
    exit()

print(f"Original number of ratings: {len(ratings_df_original)}")
print(f"Original number of unique users: {ratings_df_original['user_id'].nunique()}")
print(f"Original number of unique books: {ratings_df_original['book_id'].nunique()}")

# --- Filter Dataset ---
print(f"\nFiltering dataset: min_ratings_per_user={MIN_RATINGS_PER_USER}, min_ratings_per_book={MIN_RATINGS_PER_BOOK}")
user_counts = ratings_df_original['user_id'].value_counts()
active_users = user_counts[user_counts >= MIN_RATINGS_PER_USER].index
ratings_df_filtered_users = ratings_df_original[ratings_df_original['user_id'].isin(active_users)]

book_counts = ratings_df_filtered_users['book_id'].value_counts()
popular_books = book_counts[book_counts >= MIN_RATINGS_PER_BOOK].index
ratings_df = ratings_df_filtered_users[ratings_df_filtered_users['book_id'].isin(popular_books)].copy() # Use .copy() to avoid SettingWithCopyWarning

print(f"Number of ratings after filtering: {len(ratings_df)}")
if ratings_df.empty:
    print("Error: No data left after filtering. Try adjusting MIN_RATINGS_PER_USER and MIN_RATINGS_PER_BOOK.")
    exit()

num_users_filtered = ratings_df['user_id'].nunique()
num_books_filtered = ratings_df['book_id'].nunique()
print(f"Number of unique users after filtering: {num_users_filtered}")
print(f"Number of unique books after filtering: {num_books_filtered}")

# --- Create Mappings for Sparse Matrix ---
# User IDs to internal 0-based indices
unique_user_ids = ratings_df['user_id'].unique()
user_to_idx = {user_id: i for i, user_id in enumerate(unique_user_ids)}
idx_to_user = {i: user_id for user_id, i in user_to_idx.items()}

# Book IDs to internal 0-based indices
unique_book_ids = ratings_df['book_id'].unique()
book_to_idx = {book_id: i for i, book_id in enumerate(unique_book_ids)}
idx_to_book = {i: book_id for book_id, i in book_to_idx.items()}

# Map user_id and book_id in ratings_df to these new internal indices
ratings_df['user_idx'] = ratings_df['user_id'].map(user_to_idx)
ratings_df['book_idx'] = ratings_df['book_id'].map(book_to_idx)

# --- Create Sparse User-Item Matrix ---
print("\nCreating Sparse User-Item Matrix from filtered data...")
try:
    sparse_user_item_matrix = csr_matrix(
        (ratings_df['rating'].values, (ratings_df['user_idx'].values, ratings_df['book_idx'].values)),
        shape=(num_users_filtered, num_books_filtered)
    )
except Exception as e:
    print(f"Error creating sparse matrix: {e}")
    exit()
print("Sparse User-Item Matrix shape:", sparse_user_item_matrix.shape)
# This matrix stores only actual ratings, 0s are implied.

# --- Calculate User Similarity Matrix ---
# This will still be a dense matrix (num_users_filtered x num_users_filtered).
# If num_users_filtered is still too large (e.g., > 20k-30k), this step can cause MemoryError.
# If so, increasing MIN_RATINGS_PER_USER is the main way to reduce num_users_filtered.
print("\nCalculating User Similarity Matrix...")
try:
    user_similarity_matrix_dense = cosine_similarity(sparse_user_item_matrix)
except MemoryError:
    print("MemoryError during cosine_similarity for user_similarity_matrix_dense.")
    print(f"The number of unique users after filtering ({num_users_filtered}) might still be too high, leading to a large dense user-user similarity matrix.")
    print("Try increasing MIN_RATINGS_PER_USER significantly (e.g., 100, 150, or more) and re-run.")
    exit()
except Exception as e:
    print(f"Error calculating user similarity: {e}")
    exit()

# Convert to DataFrame, using original user_ids as index/columns
user_similarity_df = pd.DataFrame(user_similarity_matrix_dense, index=unique_user_ids, columns=unique_user_ids)
print("User Similarity Matrix (DataFrame) shape:", user_similarity_df.shape)


# --- Recommendation Function (Adapted for Sparse Matrix and Mappings) ---
def get_user_based_collaborative_recommendations(target_user_id, top_n=10, min_similar_users=5, min_similarity_threshold=0.1):
    if target_user_id not in user_to_idx: # Check if target user exists after filtering
        return f"User ID {target_user_id} not found (possibly filtered out or never existed)."

    # Get mapped index for the target user
    target_user_idx = user_to_idx[target_user_id]

    similar_users_scores = user_similarity_df[target_user_id].sort_values(ascending=False)
    similar_users_scores = similar_users_scores.drop(target_user_id, errors='ignore')
    similar_users_scores = similar_users_scores[similar_users_scores > min_similarity_threshold]

    if len(similar_users_scores) < min_similar_users:
        return f"Not enough similar users for User ID {target_user_id} (found {len(similar_users_scores)})."

    # Get original IDs of similar users
    similar_original_user_ids = similar_users_scores.index.tolist()

    # Get books rated by the target user (using sparse matrix and mapped index)
    target_user_ratings_sparse_row = sparse_user_item_matrix[target_user_idx, :]
    target_user_rated_book_indices = target_user_ratings_sparse_row.indices # .indices gives col indices of non-zero elements
    
    candidate_books_scores = {}
    # Iterate over all unique book indices in the sparse matrix
    for book_m_idx in range(sparse_user_item_matrix.shape[1]): # book_m_idx is the mapped book index
        if book_m_idx not in target_user_rated_book_indices: # If target user hasn't rated this book
            weighted_sum_ratings = 0
            sum_similarity_weights = 0
            
            # Find similar users who have rated this book (book_m_idx)
            for sim_user_original_id in similar_original_user_ids:
                if sim_user_original_id not in user_to_idx: continue # Should not happen if IDs from user_similarity_df
                
                sim_user_m_idx = user_to_idx[sim_user_original_id]
                rating_by_similar_user = sparse_user_item_matrix[sim_user_m_idx, book_m_idx]
                
                if rating_by_similar_user > 0: # If the similar user rated this book (non-zero rating)
                    similarity_score = similar_users_scores[sim_user_original_id]
                    weighted_sum_ratings += (similarity_score * rating_by_similar_user)
                    sum_similarity_weights += similarity_score
            
            if sum_similarity_weights > 0:
                predicted_score = weighted_sum_ratings / sum_similarity_weights
                original_book_id = idx_to_book[book_m_idx] # Convert mapped book index back to original book_id
                candidate_books_scores[original_book_id] = predicted_score

    if not candidate_books_scores:
        return f"Could not predict scores for User ID {target_user_id}."

    sorted_recommended_books = sorted(candidate_books_scores.items(), key=lambda item: item[1], reverse=True)
    
    final_recommendations = []
    for original_book_id, score in sorted_recommended_books[:top_n]:
        book_title_series = books_df.loc[books_df['book_id'] == original_book_id, 'title']
        if not book_title_series.empty:
            final_recommendations.append((book_title_series.iloc[0], round(score, 4)))
        else: # Fallback if title not found for the original_book_id
            final_recommendations.append((f"Book ID {original_book_id} (title not found)", round(score, 4)))
    return final_recommendations

# --- Example Usage ---
if 'user_similarity_df' in locals() and not ratings_df.empty:
    example_user_id = None
    # Try to pick a user that exists in the filtered dataset (using original IDs)
    if 1 in unique_user_ids: # Check if user 1 is in the list of users after filtering
        example_user_id = 1
    elif len(unique_user_ids) > 0:
        example_user_id = unique_user_ids[0] # Pick the first available user

    if example_user_id is not None:
        print(f"\n--- Collaborative Filtering Recommendations for User ID: {example_user_id} (Top 5) ---")
        recommendations_cf = get_user_based_collaborative_recommendations(example_user_id, top_n=5)

        if isinstance(recommendations_cf, str):
            print(recommendations_cf)
        elif recommendations_cf:
            for book_title, score in recommendations_cf:
                print(f"- \"{book_title}\" (Predicted Score: {score})")
        else:
            print(f"No recommendations generated for User ID: {example_user_id}")
    else:
        print("Could not select an example user from the filtered dataset.")
            
    non_existent_user_id = 0 # User IDs in goodbooks-10k generally start from 1
    if non_existent_user_id not in user_to_idx: # More robust check using the mapping
         print(f"\n--- Collaborative Filtering Recommendations for User ID: {non_existent_user_id} (Non-existent/Filtered Out) ---")
         recommendations_non_existent_cf = get_user_based_collaborative_recommendations(non_existent_user_id, top_n=5)
         print(recommendations_non_existent_cf)
else:
    print("\nSkipping Collaborative Filtering example due to unavailable dataframes or empty filtered dataset.")