import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse 
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Local NLTK data setup.
NLTK_DATA_DIR = 'nltk_data'
if not os.path.exists(NLTK_DATA_DIR):
    os.mkdir(NLTK_DATA_DIR)
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)

# Download NLTK data packages if missing.
try:
    nltk.data.find('tokenizers/punkt', paths=[NLTK_DATA_DIR])
except LookupError:
    print(f"Downloading 'punkt' package to '{NLTK_DATA_DIR}'...")
    nltk.download('punkt', download_dir=NLTK_DATA_DIR)
try:
    nltk.data.find('corpora/wordnet', paths=[NLTK_DATA_DIR])
except LookupError:
    print(f"Downloading 'wordnet' package to '{NLTK_DATA_DIR}'...")
    nltk.download('wordnet', download_dir=NLTK_DATA_DIR)


DATA_DIR = 'data'

def lemmatize_text(text):
    # Reduces words to their root form.
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def load_and_prepare_content_data():
    # Loads and prepares all data needed for the content-based model.
    try:
        books_df_orig = pd.read_csv(os.path.join(DATA_DIR, 'books.csv'))
        tags_df = pd.read_csv(os.path.join(DATA_DIR, 'tags.csv'))
        book_tags_df = pd.read_csv(os.path.join(DATA_DIR, 'book_tags.csv'))
    except FileNotFoundError as e:
        print(f"Content-based: Error loading datasets: {e}")
        return None

    MIN_TAG_COUNT = 100 
    relevant_book_tags = book_tags_df[book_tags_df['count'] > MIN_TAG_COUNT]
    
    book_tags_with_names_df = pd.merge(relevant_book_tags, tags_df, on='tag_id')

    # Filter out common shelf tags like 'to-read', 'favorites', etc.
    non_content_tag_patterns = [
        'tbr', 'to-read', 'owned', 'reading-list', 'dnf', 
        'did-not-finish', 'favorite', 'currently-reading', 'library',
        'ebook', 'audiobook', 'kindle', 'books-i-own', 'to-buy',
        r'^\d{4}', # Also filter tags that are just years.
    ]
    regex_pattern = '|'.join(non_content_tag_patterns)
    
    filtered_tags_df = book_tags_with_names_df[
        ~book_tags_with_names_df['tag_name'].str.contains(regex_pattern, case=False, na=False)
    ]

    # Aggregate the cleaned tags for each book.
    book_all_tags_df = filtered_tags_df.groupby('goodreads_book_id')['tag_name'].apply(lambda x: ' '.join(x)).reset_index()
    book_all_tags_df.rename(columns={'tag_name': 'book_tags_string'}, inplace=True)

    books_df_merged = pd.merge(books_df_orig, book_all_tags_df, on='goodreads_book_id', how='left')
    books_df_merged['authors'] = books_df_merged['authors'].fillna('')
    books_df_merged['book_tags_string'] = books_df_merged['book_tags_string'].fillna('')
    books_df_merged['title'] = books_df_merged['title'].fillna('')
    
    # Repeat title and authors to give them more weight.
    books_df_merged['content'] = (
        books_df_merged['title'] + ' ' + books_df_merged['title'] + ' ' +
        books_df_merged['authors'] + ' ' + books_df_merged['authors'] + ' ' +
        books_df_merged['book_tags_string']
    )

    print("Lemmatizing content text... this may take some time.")
    books_df_merged['content'] = books_df_merged['content'].apply(lemmatize_text)
    
    content_df_intermediate = books_df_merged[['book_id', 'title', 'content', 'goodreads_book_id']].copy()
    content_df_intermediate.dropna(subset=['content'], inplace=True)
    
    if content_df_intermediate.empty:
        print("Content-based: The content_df is empty after processing.")
        return None
        
    content_df_final_for_indexing = content_df_intermediate.set_index('book_id', drop=False).copy()
    return content_df_final_for_indexing

def build_content_model(content_df_input):
    # Builds the TF-IDF vectorizer and cosine similarity matrix.
    if content_df_input is None or content_df_input.empty:
        print("Content-based: Input content_df is empty.")
        return None, None, None, None

    if content_df_input['content'].isnull().any():
        content_df_input['content'] = content_df_input['content'].fillna('')

    # Consider both single words and word pairs (ngrams).
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=5, ngram_range=(1, 2))
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform(content_df_input['content'])
    except ValueError as e:
        print(f"Content-based: Error during TF-IDF: {e}")
        return None, None, None, None

    if tfidf_matrix.shape[0] == 0 or tfidf_matrix.shape[1] == 0:
        print("Content-based: TF-IDF matrix is empty.")
        return None, None, None, None

    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    book_id_to_matrix_idx = pd.Series(range(len(content_df_input)), index=content_df_input['book_id'])
    
    return cosine_sim_matrix, content_df_input, book_id_to_matrix_idx, tfidf_vectorizer

def get_content_based_recommendations(input_book_id, top_n=10, cos_sim_matrix=None, books_data_df=None, book_id_to_idx_map=None):
    # Gets recommendations for a given book_id.
    if cos_sim_matrix is None or books_data_df is None or book_id_to_idx_map is None:
        return "Model components not provided."
    if input_book_id not in book_id_to_idx_map:
         return f"Book ID {input_book_id} not found in map."

    try:
        book_m_idx = book_id_to_idx_map[input_book_id]
    except KeyError:
         return f"Book ID {input_book_id} key error in map."

    sim_scores = list(enumerate(cos_sim_matrix[book_m_idx]))
    sim_scores_sorted = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_scores = sim_scores_sorted[1:(top_n + 1)] 

    recs = []
    for matrix_idx_rec, score_val in top_scores:
        try:
            rec_book_id = books_data_df['book_id'].iloc[matrix_idx_rec]
            rec_book_title = books_data_df['title'].iloc[matrix_idx_rec]
            recs.append({'book_id': rec_book_id, 'title': rec_book_title, 'score': round(score_val, 4)})
        except IndexError:
            continue 
    return recs

def get_similarity_score_for_book_pair(book_id1, book_id2, cos_sim_matrix=None, book_id_to_idx_map=None):
    # Gets content similarity between two book_ids.
    if cos_sim_matrix is None or book_id_to_idx_map is None: return "Model components missing."
    if book_id1 not in book_id_to_idx_map or book_id2 not in book_id_to_idx_map:
        return f"One or both Book IDs ({book_id1}, {book_id2}) not in map."
    
    idx1 = book_id_to_idx_map[book_id1]
    idx2 = book_id_to_idx_map[book_id2]
    return round(cos_sim_matrix[idx1, idx2], 4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Content-Based Book Recommender.")
    parser.add_argument("book_id", type=int, help="Book ID for which to generate recommendations.")
    args = parser.parse_args()
    
    print("Running Content-Based Recommender Standalone...")
    
    prepared_data = load_and_prepare_content_data()
    if prepared_data is not None and not prepared_data.empty:
        cos_sim, books_data, id_to_idx_map, _ = build_content_model(prepared_data)

        if cos_sim is not None and books_data is not None and id_to_idx_map is not None:
            target_book_id = args.book_id
            
            if target_book_id not in id_to_idx_map:
                print(f"\nError: Book ID {target_book_id} not found in the dataset.")
            else:
                target_book_title = books_data.loc[books_data['book_id'] == target_book_id, 'title'].iloc[0]
                print(f"\nContent-Based Recommendations for Book ID: {target_book_id} ('{target_book_title}')...")
                
                recommendations = get_content_based_recommendations(target_book_id, 5, cos_sim, books_data, id_to_idx_map)
                
                if isinstance(recommendations, str): 
                    print(recommendations)
                elif recommendations:
                    for rec in recommendations: 
                        print(f"- \"{rec['title']}\" (ID: {rec['book_id']}, Score: {rec['score']})")
                else: 
                    print("No recommendations found.")

        else: print("Content-based model building failed.")
    else: print("Content-based data preparation failed.")
