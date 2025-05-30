# Book Recommendation System

## Project Overview

This project implements a hybrid book recommendation system combining Content-Based Filtering and User-Based Collaborative Filtering techniques. The system provides book recommendations based on item features and user rating patterns.

The solution is implemented in Python and utilizes common data science libraries.

## Setup Instructions

1.  **Clone the Project Repository:**
    First, clone this project's repository to your local machine:
    ```bash
    git clone https://github.com/aayushrautela/Book-Recommendation-System.git
    cd Book-Recommendation-System
    ```

2.  **Prerequisites (Python Libraries):**
    * Python (version 3.7 or newer recommended).
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
        * After downloading, create a subdirectory named `data/` in the root of folder.
        * Extract the downloaded dataset and place the following essential CSV files into this `data/` subdirectory:
            * `books.csv`
            * `ratings.csv`
            * `tags.csv`
            * `book_tags.csv`
    * The project scripts expect these files to be located at `data/<filename>.csv`.

## Running the Recommenders

The project consists of three main Python scripts. Each script can be run from the command line using Python. Ensure your terminal's current working directory is the root of the project (i.e., the `Book-Recommendation-System` directory after cloning).

1.  **Content-Based Recommender (`content_based_recommender.py`)**
    * This script implements the content-based filtering logic.
    * When run directly, it loads data, builds the content model (TF-IDF and cosine similarity matrix), and prints example recommendations for a predefined book and the similarity score between a pair of books.
    * To run:
        ```bash
        python content_based_recommender.py
        ```

2.  **User-Based Collaborative Filtering Recommender (`collaborative_recommender.py`)**
    * This script implements the user-based collaborative filtering logic.
    * It filters the dataset based on `MIN_RATINGS_PER_USER` and `MIN_RATINGS_PER_BOOK` parameters (defined within the script) to manage memory, then builds a sparse user-item matrix and a user-user similarity matrix.
    * When run directly, it prints example recommendations for a predefined user and a predicted score for a specific user-book pair.
    * To run:
        ```bash
        python collaborative_recommender.py <user>
        ```
    * **Note:** The filtering parameters within this script (`MIN_RATINGS_PER_USER`, `MIN_RATINGS_PER_BOOK`) may need adjustment based on available system memory. The current values (e.g., 150, 50) have been tested to work on systems with moderate RAM.

3.  **Hybrid Recommender (`hybrid_recommender.py`)**
    * This script combines the content-based and collaborative filtering approaches using a weighted strategy.
    * It imports and utilizes the functionalities from the other two scripts.
    * When run directly, it initializes both models, selects an example user, finds anchor books for that user (based on their highest ratings), and then generates and prints hybrid recommendations.
    * The `ALPHA` parameter (for weighting content vs. collaborative scores) and `TOP_N_ANCHOR_BOOKS` can be adjusted within this script.
    * To run:
        ```bash
        python hybrid_recommender.py <user>
        ```

## Code Structure

* `content_based_recommender.py`: Contains all logic for the content-based filtering approach.
* `collaborative_recommender.py`: Contains all logic for the user-based collaborative filtering approach, including data filtering and sparse matrix operations.
* `hybrid_recommender.py`: Implements the hybrid model by leveraging the other two modules. It combines their outputs to produce a final set of recommendations.
* `data/` (directory): This directory needs to be created by you and populated with the CSV files from the goodbooks-10k dataset as per the "Dataset" instructions.
* `requirements.txt`: Lists all Python dependencies.