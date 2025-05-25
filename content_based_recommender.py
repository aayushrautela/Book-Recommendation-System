import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Configuration ---
DATA_DIR = 'data'

# --- Load Data ---
try:
    books_df = pd.read_csv(os.path.join(DATA_DIR, 'books.csv'))
    tags_df = pd.read_csv(os.path.join(DATA_DIR, 'tags.csv'))
    book_tags_df = pd.read_csv(os.path.join(DATA_DIR, 'book_tags.csv'))
    print("Datasets loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading datasets: {e}")
    print(f"Please ensure 'books.csv', 'tags.csv', and 'book_tags.csv' are in the '{DATA_DIR}' directory.")
    exit() # Exit if essential data is missing

# --- Data Preprocessing for Content-Based Filtering ---

# Merge book_tags with tags to get tag names
book_tags_with_names_df = pd.merge(book_tags_df, tags_df, on='tag_id')

# Aggregate all tags for each book into a single string
book_all_tags_df = book_tags_with_names_df.groupby('goodreads_book_id')['tag_name'].apply(lambda x: ' '.join(x)).reset_index()
book_all_tags_df.rename(columns={'tag_name': 'book_tags_string'}, inplace=True)

# Merge aggregated tags back into the main books_df
# books_df uses 'goodreads_book_id' which matches 'goodreads_book_id' in book_all_tags_df
books_df_merged = pd.merge(books_df, book_all_tags_df, on='goodreads_book_id', how='left')

# Handle missing values
books_df_merged['authors'] = books_df_merged['authors'].fillna('')
books_df_merged['book_tags_string'] = books_df_merged['book_tags_string'].fillna('')
books_df_merged['title'] = books_df_merged['title'].fillna('') # Ensure title is not NaN

# Create a 'content' feature string
books_df_merged['content'] = (
    books_df_merged['title'] + ' ' +
    books_df_merged['authors'] + ' ' +
    books_df_merged['book_tags_string']
)

# Prepare DataFrame for TF-IDF
content_df = books_df_merged[['book_id', 'title', 'content']].copy()
content_df.dropna(subset=['content'], inplace=True) # Drop rows if content is still NaN
content_df.set_index('book_id', inplace=True) # Use book_id as index

if content_df.empty:
    print("Error: The content_df is empty after processing. Cannot proceed.")
    exit()

# --- TF-IDF Vectorization and Cosine Similarity ---
print("\n--- TF-IDF Vectorization and Cosine Similarity ---")

if content_df['content'].isnull().any(): # Should have been handled by dropna, but as a safeguard
    content_df['content'] = content_df['content'].fillna('')

tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=5)

try:
    tfidf_matrix = tfidf_vectorizer.fit_transform(content_df['content'])
    print("TF-IDF matrix shape:", tfidf_matrix.shape)
except ValueError as e:
    print(f"Error during TF-IDF vectorization: {e}")
    exit()

if tfidf_matrix.shape[0] == 0 or tfidf_matrix.shape[1] == 0:
    print("Error: TF-IDF matrix is empty. Check 'content_df' and TF-IDF parameters.")
    exit()

cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
print("Cosine similarity matrix shape:", cosine_sim_matrix.shape)

# Prepare for mapping titles to matrix indices
content_df_reset_for_indexing = content_df.reset_index()
title_to_idx_map = pd.Series(content_df_reset_for_indexing.index, index=content_df_reset_for_indexing['title']).drop_duplicates()

# --- Recommendation Function ---
def get_content_based_recommendations(input_title, top_n=10, cosine_sim_m=cosine_sim_matrix, df_with_titles=content_df_reset_for_indexing, title_idx_map=title_to_idx_map):
    matched_title = None
    if input_title in title_idx_map:
        matched_title = input_title
    else: # Try to find a match
        for title_in_db in title_idx_map.index: # Case-insensitive exact match
            if title_in_db.lower() == input_title.lower():
                matched_title = title_in_db
                print(f"Found title by case-insensitive match: '{matched_title}' for input '{input_title}'")
                break
        if not matched_title: # Partial match
            possible_matches = df_with_titles[df_with_titles['title'].str.contains(input_title, case=False, na=False)]
            if not possible_matches.empty:
                matched_title = possible_matches['title'].iloc[0]
                print(f"Found title by partial match: '{matched_title}' for input '{input_title}'")
            else:
                return f"Book with title '{input_title}' not found."

    if matched_title not in title_idx_map: # Should not happen if logic above is correct
        return f"Book with title '{input_title}' (matched as '{matched_title}') still not found in indices."

    try:
        book_idx = title_idx_map[matched_title]
    except KeyError:
         return f"Error finding index for '{matched_title}'." # Should be caught by earlier checks

    sim_scores_for_book = list(enumerate(cosine_sim_m[book_idx]))
    sim_scores_for_book_sorted = sorted(sim_scores_for_book, key=lambda x: x[1], reverse=True)
    top_similar_books_scores = sim_scores_for_book_sorted[1:(top_n + 1)]
    recommended_book_indices = [i[0] for i in top_similar_books_scores]

    recommended_books_with_scores = []
    for i in range(len(recommended_book_indices)):
        original_idx = recommended_book_indices[i]
        title = df_with_titles['title'].iloc[original_idx]
        score = round(top_similar_books_scores[i][1], 4)
        recommended_books_with_scores.append((title, score))
    
    return recommended_books_with_scores

# --- Example Usage (Content-Based) ---
if 'cosine_sim_matrix' in locals() and not content_df_reset_for_indexing.empty:
    example_titles = ["The Hobbit", "To Kill a Mockingbird", "A Non Existent Book Title XYZ123"]
    
    for ex_title in example_titles:
        print(f"\n--- Content-Based Recommendations for '{ex_title}' (Top 5) ---")
        recommendations = get_content_based_recommendations(ex_title, top_n=5)
        
        if isinstance(recommendations, str):
            print(recommendations)
        elif recommendations:
            for book_title, score in recommendations:
                print(f"- \"{book_title}\" (Similarity Score: {score})")
        else:
            print(f"No recommendations found or an issue occurred for '{ex_title}'.")
else:
    print("\nSkipping recommendation example: 'cosine_sim_matrix' or 'content_df_reset_for_indexing' not available.")