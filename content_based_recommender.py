import pandas as pd
import os

# --- Configuration ---
DATA_DIR = 'data' # Ensure your CSV files are in a 'data' subdirectory

# --- Load Data ---
try:
    books_df = pd.read_csv(os.path.join(DATA_DIR, 'books.csv'))
    tags_df = pd.read_csv(os.path.join(DATA_DIR, 'tags.csv'))
    book_tags_df = pd.read_csv(os.path.join(DATA_DIR, 'book_tags.csv'))
    print("Datasets loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading datasets: {e}")
    print(f"Please ensure 'books.csv', 'tags.csv', and 'book_tags.csv' are in the '{DATA_DIR}' directory.")
    exit()

# --- Display basic information ---
print("\n--- Books DataFrame ---")
print(books_df.info())
print(books_df.head())
# Note: books_df has 'book_id' and 'goodreads_book_id'. We'll likely use 'book_id' for internal consistency if merging.
# For content, 'title', 'authors' are directly useful. 'language_code' could also be a feature.

print("\n--- Tags DataFrame ---")
print(tags_df.info())
print(tags_df.head()) # 'tag_id', 'tag_name'

print("\n--- Book_Tags DataFrame ---")
print(book_tags_df.info())
print(book_tags_df.head()) # 'goodreads_book_id', 'tag_id', 'count'
# This links books (via goodreads_book_id) to tags. 'count' indicates tag relevance.

# --- Data Preprocessing for Content-Based Filtering ---

# 1. Merge book information with their tags
#    Need to merge book_tags_df with tags_df to get tag names
book_tags_with_names_df = pd.merge(book_tags_df, tags_df, on='tag_id')

#    Now, aggregate all tags for each book. We can sort by 'count' to get the most relevant tags.
#    Let's take top N tags for simplicity, or concatenate them.
#    For now, let's group tags for each book.
book_all_tags_df = book_tags_with_names_df.groupby('goodreads_book_id')['tag_name'].apply(lambda x: ' '.join(x)).reset_index()
book_all_tags_df.rename(columns={'tag_name': 'book_tags_string'}, inplace=True)

print("\n--- Aggregated Tags per Book ---")
print(book_all_tags_df.head())

# 2. Merge these tags back into the main books_df
#    books_df uses 'book_id', but book_tags_df uses 'goodreads_book_id'.
#    We need to ensure we are merging on the correct IDs. 'books.csv' contains 'goodreads_book_id'.
books_df_merged = pd.merge(books_df, book_all_tags_df, on='goodreads_book_id', how='left')

# Handle missing values that might have resulted from merges or were already present
books_df_merged['authors'] = books_df_merged['authors'].fillna('')
books_df_merged['book_tags_string'] = books_df_merged['book_tags_string'].fillna('')
# Consider 'original_title' as well, or just 'title'. 'title' might be more consistent.
books_df_merged['title'] = books_df_merged['title'].fillna('')


# 3. Create a 'content' feature string for each book
#    This will be used by TF-IDF. We combine title, authors, and the tags.
#    You could also add 'language_code' or parts of 'image_url' if relevant, but let's start simple.
books_df_merged['content'] = (
    books_df_merged['title'] + ' ' +
    books_df_merged['authors'] + ' ' +
    books_df_merged['book_tags_string']
)

print("\n--- Books DataFrame with Merged Tags and Content Feature ---")
print(books_df_merged[['book_id', 'goodreads_book_id', 'title', 'authors', 'book_tags_string', 'content']].head())
print(books_df_merged.info())

# Keep only necessary columns for content-based part to save memory
content_df = books_df_merged[['book_id', 'title', 'content']].copy()
content_df.dropna(subset=['content'], inplace=True) # Drop rows where content is still NaN
content_df.set_index('book_id', inplace=True) # Use book_id as index for easy lookup

print("\n--- Final Content DataFrame for TF-IDF ---")
print(content_df.head())

if content_df.empty:
    print("Error: The content_df is empty after processing. Check merge keys and data.")
    exit()