# Book Recommendation System

## Project Overview

This project implements a hybrid book recommendation system combining Content-Based Filtering and a Model-Based Collaborative Filtering approach using **Singular Value Decomposition (SVD)**. This system provides book recommendations by leveraging both item content features and latent user/item taste profiles learned from rating patterns.

The solution is implemented in Python and utilizes the pandas, scikit-learn, and scikit-surprise libraries.

## Setup Instructions

1.  **Clone the Project Repository:**
    First, clone this project's repository to your local machine:
    ```bash
    git clone https://github.com/aayushrautela/Book-Recommendation-System.git
    cd Book-Recommendation-System
    ```

2.  **Prerequisites (Python Libraries):**
    * Python (version 3.10 or 3.11 recommended for library compatibility).
    * The required Python libraries are listed in `requirements.txt`.
    * It is recommended to use a virtual environment.
    * Install the dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Dataset (goodbooks-10k):**
    * **Download:** This project uses the **goodbooks-10k** dataset. You need to download it from its original GitHub repository:
        [https://github.com/zygmuntz/goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k)
    * **Placement:**
        * After downloading, create a subdirectory named `data/` in the root of this cloned project.
        * Extract the downloaded dataset and place the following essential CSV files into this `data/` subdirectory:
            * `books.csv`
            * `ratings.csv`
            * `tags.csv`
            * `book_tags.csv`

## Running the System (Two-Step Process)

This system requires a "training" step before recommendations can be made.

### Step 1: Train the Collaborative Filtering Model

The SVD model needs to be trained on the rating data first. This process will learn the latent factors and save a `cf_svd_model.joblib` file (a "pretrained checkpoint").

* To train the model with the optimized default parameters, run:
    ```bash
    python collaborative_recommender.py train
    ```
* This will run for a set number of epochs and only needs to be done once, or whenever you want to retrain the model.

### Step 2: Generate Recommendations

Once the model is trained, you can get recommendations using the following scripts.

1.  **Hybrid Recommender (`hybrid_recommender.py`) - Recommended**
    * This is the main script. It loads the pretrained CF model and the CB model to provide blended recommendations for a specific user.
    * To run, provide a `user_id`:
        ```bash
        python hybrid_recommender.py <user_id>
        ```
        *Example:* `python hybrid_recommender.py 100`

2.  **Standalone Content-Based Recommender (`content_based_recommender.py`)**
    * This script provides recommendations based only on content similarity to a specific book.
    * To run, provide a `book_id`:
        ```bash
        python content_based_recommender.py <book_id>
        ```
        *Example:* `python content_based_recommender.py 6`

3.  **Standalone Collaborative Filtering Recommender (`collaborative_recommender.py`)**
    * This script uses the pretrained SVD model to provide recommendations based only on collaborative filtering.
    * To run, use the `recommend` action and provide a `user_id`:
        ```bash
        python collaborative_recommender.py recommend --user_id <user_id>
        ```
        *Example:* `python collaborative_recommender.py recommend --user_id 100`

---
### Optional: Hyperparameter Tuning (`evaluate.py`) (WIP)

To find the best parameters for the SVD model, you can run the evaluation script.

* This script uses `GridSearchCV` to test multiple parameter combinations and will print the best ones based on RMSE. It saves the results to `tuning_results.csv` and generates a heatmap plot `tuning_heatmap.png`.
* **Warning:** This process is very time-consuming.
* To run:
    ```bash
    python evaluate.py
    ```

## Code Structure

* `content_based_recommender.py`: Contains all logic for the content-based filtering approach.
* `collaborative_recommender.py`: Implements the SVD model-based collaborative filtering, including training and inference.
* `hybrid_recommender.py`: Combines the two models to produce final recommendations.
* `evaluate.py`: Used for hyperparameter tuning and evaluation of the SVD model.
* `data/`: Directory where the dataset CSV files must be placed.
* `requirements.txt`: Lists all Python dependencies.
