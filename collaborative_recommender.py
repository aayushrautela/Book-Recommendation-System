import pandas as pd
import os
import argparse
import pickle

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

DATA_DIR = 'data'
CF_MODEL_CHECKPOINT_PATH = 'cf_svd_model.joblib'

# Default training parameters
DEFAULT_N_FACTORS = 100 # Number of latent factors
DEFAULT_N_EPOCHS = 20   # Number of training epochs

def load_data_for_surprise():
    """Loads ratings data and prepares it for the surprise library."""
    try:
        ratings_df = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'))
        books_df = pd.read_csv(os.path.join(DATA_DIR, 'books.csv'))[['book_id', 'title']]
    except FileNotFoundError as e:
        print(f"Collaborative: Error loading data: {e}")
        return None, None

    reader = Reader(rating_scale=(1, 5))
    
    data = Dataset.load_from_df(ratings_df[['user_id', 'book_id', 'rating']], reader)
    
    print("Collaborative: Data loaded successfully for surprise.")
    return data, books_df

def train_svd_model(data, n_factors=DEFAULT_N_FACTORS, n_epochs=DEFAULT_N_EPOCHS):
    """Trains an SVD model on the provided dataset."""
    print(f"\nTraining SVD model with {n_factors} factors for {n_epochs} epochs...")
    print("This may take some time depending on the dataset size and number of epochs...")

    trainset = data.build_full_trainset()
    
    algo = SVD(n_factors=n_factors, n_epochs=n_epochs, verbose=True)
    
    # Train the algorithm on the trainset.
    algo.fit(trainset)
    
    print("\nTraining complete.")
    return algo

def save_model(model, filepath=CF_MODEL_CHECKPOINT_PATH):
    """Saves the trained model to a file."""
    print(f"Saving trained model to {filepath}...")
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully.")

def load_model(filepath=CF_MODEL_CHECKPOINT_PATH):
    """Loads a pretrained model from a file."""
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {filepath}.")
        return model
    except FileNotFoundError:
        print(f"Model file not found at {filepath}. Please train the model first.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def get_cf_score_for_user_book_pair(model, user_id, book_id):
    """Predicts a rating for a single user-book pair using the trained model."""
    if model is None:
        return "Model not available."
    
    # The predict method returns a prediction object.
    prediction = model.predict(uid=user_id, iid=book_id)
    
    # The 'est' attribute contains the estimated rating.
    return round(prediction.est, 4)

def get_top_n_cf_recommendations(model, user_id, books_df, n=10):
    """Generates top N recommendations for a user."""
    if model is None:
        return "Model not available."

    # First, get the list of all book IDs.
    all_book_ids = books_df['book_id'].unique()
    
   
    try:
        # Convert internal user id (inner_id) to raw user id and other way
        user_inner_id = model.trainset.to_inner_uid(user_id)
        rated_book_inner_ids = model.trainset.ur[user_inner_id]
        rated_book_raw_ids = {model.trainset.to_raw_iid(inner_id) for inner_id, _ in rated_book_inner_ids}
    except ValueError:
        # User was not in the training set
        rated_book_raw_ids = set()
        print(f"User {user_id} not in training set. Recommending based on general popularity (not implemented here) or all books.")

    # Predict ratings for all books the user hasn't rated yet.
    predictions = []
    for book_id in all_book_ids:
        if book_id not in rated_book_raw_ids:
            predicted_rating = model.predict(uid=user_id, iid=book_id).est
            predictions.append({'book_id': book_id, 'score': predicted_rating})
    
    # Sort the predictions by score.
    predictions.sort(key=lambda x: x['score'], reverse=True)
    
    # Get top N recommendations and add book titles.
    top_recs = []
    for pred in predictions[:n]:
        book_id = pred['book_id']
        book_title_series = books_df.loc[books_df['book_id'] == book_id, 'title']
        title = book_title_series.iloc[0] if not book_title_series.empty else f"Book ID {book_id}"
        top_recs.append({'book_id': book_id, 'title': title, 'score': round(pred['score'], 4)})
        
    return top_recs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model-Based Collaborative Filtering Recommender using SVD.")
    parser.add_argument("action", choices=['train', 'recommend'], help="Action to perform: 'train' or 'recommend'.")
    parser.add_argument("--user_id", type=int, help="User ID for whom to generate recommendations (required for 'recommend' action).")
    parser.add_argument("--epochs", type=int, default=DEFAULT_N_EPOCHS, help="Number of epochs for training.")
    parser.add_argument("--factors", type=int, default=DEFAULT_N_FACTORS, help="Number of latent factors for SVD.")
    
    args = parser.parse_args()

    if args.action == 'train':
        data, _ = load_data_for_surprise()
        if data:
            trained_model = train_svd_model(data, n_factors=args.factors, n_epochs=args.epochs)
            save_model(trained_model)
    
    elif args.action == 'recommend':
        if not args.user_id:
            print("Error: --user_id is required for the 'recommend' action.")
            exit()
            
        trained_model = load_model()
        if trained_model:
            _, books_df = load_data_for_surprise() # Load book titles
            if books_df is not None:
                print(f"\nGenerating Top 5 recommendations for User ID: {args.user_id}...")
                recommendations = get_top_n_cf_recommendations(trained_model, args.user_id, books_df, n=5)
                
                if isinstance(recommendations, str):
                    print(recommendations)
                elif recommendations:
                    for r in recommendations:
                        print(f"- \"{r['title']}\" (ID: {r['book_id']}, Predicted Score: {r['score']})")
                else:
                    print(f"No recommendations generated for User ID {args.user_id}.")
