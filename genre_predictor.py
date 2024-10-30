# import pandas as pd
# import numpy as np
# import argparse
# import sys
# import os
# import json
# import time

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import classification_report, f1_score, hamming_loss
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression

# import joblib

# # ==========================
# # Data Loading and Preprocessing
# # ==========================

# def load_data(csv_path: str) -> pd.DataFrame:
#     """
#     Loads the dataset from a CSV file.

#     Args:
#         csv_path (str): Path to the CSV file.

#     Returns:
#         pd.DataFrame: Loaded DataFrame.
#     """
#     if not os.path.exists(csv_path):
#         print(f"Error: CSV file '{csv_path}' not found.", file=sys.stderr)
#         sys.exit(1)
    
#     try:
#         df = pd.read_csv(csv_path)
#         print(f"Successfully loaded data from {csv_path}")
#         return df
#     except Exception as e:
#         print(f"Error loading CSV file: {e}", file=sys.stderr)
#         sys.exit(1)

# def preprocess_genres(df: pd.DataFrame, min_genre_count: int = 5) -> (pd.DataFrame, MultiLabelBinarizer):
#     """
#     Processes the 'genres' column into a binary matrix and filters out rare genres.

#     Args:
#         df (pd.DataFrame): DataFrame containing the 'genres' column.
#         min_genre_count (int): Minimum number of occurrences for a genre to be retained.

#     Returns:
#         pd.DataFrame: Original DataFrame with 'genres' column as lists.
#         MultiLabelBinarizer: Fitted MultiLabelBinarizer instance with filtered genres.
#     """
#     if 'genres' not in df.columns:
#         print("Error: 'genres' column not found in the dataset.", file=sys.stderr)
#         sys.exit(1)
    
#     # Handle missing genres
#     df['genres'] = df['genres'].fillna('')
    
#     # Split the comma-separated genres into lists
#     df['genres'] = df['genres'].apply(lambda x: [genre.strip() for genre in x.split(',')] if x else [])
    
#     # Count genre frequencies
#     genre_counts = df['genres'].explode().value_counts()
#     filtered_genres = genre_counts[genre_counts >= min_genre_count].index.tolist()
    
#     # Filter genres in each song
#     df['genres'] = df['genres'].apply(lambda genres: [genre for genre in genres if genre in filtered_genres])
    
#     # Initialize MultiLabelBinarizer with filtered genres
#     mlb = MultiLabelBinarizer()
#     genre_encoded = mlb.fit_transform(df['genres'])
    
#     # Create a DataFrame with genre labels
#     genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_, index=df.index)
    
#     # Concatenate the genre DataFrame with the original DataFrame
#     df = pd.concat([df, genre_df], axis=1)
    
#     print(f"Processed genres into {len(mlb.classes_)} binary columns after filtering out rare genres.")
    
#     return df, mlb

# def select_features_and_labels(df: pd.DataFrame, mlb: MultiLabelBinarizer) -> (pd.DataFrame, pd.DataFrame):
#     """
#     Selects feature columns and label columns.

#     Args:
#         df (pd.DataFrame): The preprocessed DataFrame.
#         mlb (MultiLabelBinarizer): Fitted MultiLabelBinarizer instance.

#     Returns:
#         pd.DataFrame: Feature DataFrame.
#         pd.DataFrame: Label DataFrame.
#     """
#     # Define feature columns (excluding identifiers and labels)
#     feature_cols = ['danceability', 'energy', 'key', 'loudness', 'mode',
#                    'speechiness', 'acousticness', 'instrumentalness',
#                    'liveness', 'valence', 'tempo', 'duration_ms', 
#                    'time_signature', 'popularity']
    
#     # Check if all feature columns exist
#     missing_features = set(feature_cols) - set(df.columns)
#     if missing_features:
#         print(f"Error: Missing feature columns: {missing_features}", file=sys.stderr)
#         sys.exit(1)
    
#     X = df[feature_cols].copy()
    
#     # Label columns
#     label_cols = mlb.classes_
#     y = df[label_cols].copy()
    
#     return X, y

# def prepare_data(csv_path: str) -> (pd.DataFrame, pd.DataFrame, MultiLabelBinarizer):
#     """
#     Loads and preprocesses the data.

#     Args:
#         csv_path (str): Path to the CSV file.

#     Returns:
#         pd.DataFrame: Feature DataFrame.
#         pd.DataFrame: Label DataFrame.
#         MultiLabelBinarizer: Fitted MultiLabelBinarizer instance.
#     """
#     df = load_data(csv_path)
    
#     # Drop duplicates if any
#     initial_count = len(df)
#     df = df.drop_duplicates(subset=['id'])
#     print(f"Dropped {initial_count - len(df)} duplicate tracks.")
    
#     # Drop rows with missing feature values
#     feature_cols = ['danceability', 'energy', 'key', 'loudness', 'mode',
#                    'speechiness', 'acousticness', 'instrumentalness',
#                    'liveness', 'valence', 'tempo', 'duration_ms', 
#                    'time_signature', 'popularity']
#     df = df.dropna(subset=feature_cols)
#     print(f"Data shape after dropping missing feature values: {df.shape}")
    
#     # Preprocess genres
#     df, mlb = preprocess_genres(df)
    
#     # Select features and labels
#     X, y = select_features_and_labels(df, mlb)
    
#     return X, y, mlb

# # ==========================
# # Model Building
# # ==========================

# def build_model():
#     """
#     Builds a machine learning pipeline for multi-label classification with class weights.

#     Returns:
#         sklearn.pipeline.Pipeline: The machine learning pipeline.
#     """
#     # Define the classifier with class weights
#     classifier = OneVsRestClassifier(
#         RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
#     )
    
#     # Create a pipeline with scaling and classification
#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('classifier', classifier)
#     ])
    
#     return pipeline

# # ==========================
# # Training and Evaluation
# # ==========================

# def split_data(X: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
#     """
#     Splits the data into training and testing sets.

#     Args:
#         X (pd.DataFrame): Feature DataFrame.
#         y (pd.DataFrame): Label DataFrame.
#         test_size (float): Proportion of the dataset to include in the test split.
#         random_state (int): Random seed.

#     Returns:
#         X_train, X_test, y_train, y_test
#     """
#     return train_test_split(X, y, test_size=test_size, random_state=random_state)

# def hit_rate(y_true, y_pred):
#     """
#     Calculates the Hit Rate: The proportion of samples where at least one predicted genre matches any actual genre.
    
#     Args:
#         y_true (np.ndarray): Ground truth binary labels.
#         y_pred (np.ndarray): Predicted binary labels.
    
#     Returns:
#         float: Hit Rate.
#     """
#     # For each sample, check if there's any intersection between true and predicted genres
#     hits = np.any(y_true & y_pred, axis=1)
#     return hits.mean()

# def train_model(pipeline, X_train: pd.DataFrame, y_train: pd.DataFrame):
#     """
#     Trains the machine learning pipeline.

#     Args:
#         pipeline (sklearn.pipeline.Pipeline): The machine learning pipeline.
#         X_train (pd.DataFrame): Training features.
#         y_train (pd.DataFrame): Training labels.

#     Returns:
#         sklearn.pipeline.Pipeline: The trained pipeline.
#     """
#     print("Training the model...")
#     pipeline.fit(X_train, y_train)
#     print("Model training completed.")
#     return pipeline

# def evaluate_model(pipeline, X_test: pd.DataFrame, y_test: pd.DataFrame):
#     """
#     Evaluates the model on the test set and prints classification metrics.

#     Args:
#         pipeline (sklearn.pipeline.Pipeline): The trained machine learning pipeline.
#         X_test (pd.DataFrame): Testing features.
#         y_test (pd.DataFrame): Testing labels.
#     """
#     print("Evaluating the model...")
#     y_pred = pipeline.predict(X_test)
    
#     # Calculate metrics
#     hamming = hamming_loss(y_test, y_pred)
#     f1_micro = f1_score(y_test, y_pred, average='micro')
#     f1_macro = f1_score(y_test, y_pred, average='macro')
#     hit = hit_rate(y_test.values, y_pred)
    
#     print(f"Hamming Loss: {hamming:.4f}")
#     print(f"F1 Score (Micro): {f1_micro:.4f}")
#     print(f"F1 Score (Macro): {f1_macro:.4f}")
#     print(f"Hit Rate: {hit:.4f}")
    
#     # Detailed classification report
#     print("\nClassification Report:")
#     print(classification_report(y_test, y_pred))

# # ==========================
# # Display Sample Predictions
# # ==========================

# def display_sample_predictions(pipeline, X_test: pd.DataFrame, y_test: pd.DataFrame, mlb: MultiLabelBinarizer, num_samples: int = 5):
#     """
#     Displays sample predictions alongside actual genres.

#     Args:
#         pipeline (sklearn.pipeline.Pipeline): The trained machine learning pipeline.
#         X_test (pd.DataFrame): Testing features.
#         y_test (pd.DataFrame): Testing labels.
#         mlb (MultiLabelBinarizer): Fitted MultiLabelBinarizer instance.
#         num_samples (int): Number of sample predictions to display.
#     """
#     # Ensure there are enough samples
#     if len(X_test) < num_samples:
#         num_samples = len(X_test)
    
#     # Select random samples
#     sample_indices = np.random.choice(X_test.index, size=num_samples, replace=False)
#     samples = X_test.loc[sample_indices]
#     actual_genres = y_test.loc[sample_indices]
    
#     # Predict genres
#     predicted_genres = pipeline.predict(samples)
    
#     # Convert binary labels back to genre names
#     # Ensure inputs are NumPy arrays with appropriate dtype
#     actual_genres_np = actual_genres.to_numpy().astype(int)
#     predicted_genres_np = predicted_genres.astype(int)
    
#     actual_genres_list = mlb.inverse_transform(actual_genres_np)
#     predicted_genres_list = mlb.inverse_transform(predicted_genres_np)
    
#     # Display
#     print("\nSample Predictions:")
#     for i, idx in enumerate(sample_indices):
#         print(f"\nSong {i + 1}:")
#         print(f"Features: {samples.loc[idx].to_dict()}")
#         print(f"Actual Genres: {', '.join(actual_genres_list[i]) if actual_genres_list[i] else 'No genres'}")
#         print(f"Predicted Genres: {', '.join(predicted_genres_list[i]) if predicted_genres_list[i] else 'No genres predicted'}")

# # ==========================
# # Model Persistence
# # ==========================

# def save_model(pipeline, mlb: MultiLabelBinarizer, model_path: str):
#     """
#     Saves the trained pipeline and MultiLabelBinarizer to disk.

#     Args:
#         pipeline (sklearn.pipeline.Pipeline): The trained machine learning pipeline.
#         mlb (MultiLabelBinarizer): Fitted MultiLabelBinarizer instance.
#         model_path (str): Path to save the model.
#     """
#     try:
#         joblib.dump({'model': pipeline, 'mlb': mlb}, model_path)
#         print(f"Model successfully saved to {model_path}")
#     except Exception as e:
#         print(f"Error saving model: {e}", file=sys.stderr)

# # ==========================
# # Command-Line Interface
# # ==========================

# def parse_arguments() -> argparse.Namespace:
#     """
#     Parses command-line arguments.

#     Returns:
#         argparse.Namespace: Parsed arguments.
#     """
#     parser = argparse.ArgumentParser(description="Train a model to predict song genres based on features.")
    
#     parser.add_argument(
#         '-i', '--input_csv',
#         type=str,
#         required=True,
#         help='Path to the input CSV file containing song data.'
#     )
    
#     parser.add_argument(
#         '-m', '--model_output',
#         type=str,
#         default='genre_prediction_model.joblib',
#         help='Path to save the trained model. Defaults to "genre_prediction_model.joblib".'
#     )
    
#     parser.add_argument(
#         '--test_size',
#         type=float,
#         default=0.2,
#         help='Proportion of the dataset to include in the test split. Defaults to 0.2.'
#     )
    
#     parser.add_argument(
#         '--random_state',
#         type=int,
#         default=42,
#         help='Random seed for data splitting. Defaults to 42.'
#     )

#     parser.add_argument(
#         '-s', '--save_model',
#         type=bool,
#         default=False,
#         help='Save predictive model?'
#     )
    
#     return parser.parse_args()

# # ==========================
# # Main Execution Function
# # ==========================

# def main():
#     # Parse command-line arguments
#     args = parse_arguments()
    
#     # Prepare the data
#     X, y, mlb = prepare_data(args.input_csv)
    
#     # Split the data
#     X_train, X_test, y_train, y_test = split_data(X, y, test_size=args.test_size, random_state=args.random_state)
#     print(f"Training set size: {X_train.shape[0]} samples")
#     print(f"Testing set size: {X_test.shape[0]} samples")
    
#     # Build the model pipeline
#     pipeline = build_model()
    
#     # Train the model
#     pipeline = train_model(pipeline, X_train, y_train)
    
#     # Evaluate the model
#     evaluate_model(pipeline, X_test, y_test)
    
#     # Display sample predictions
#     display_sample_predictions(pipeline, X_test, y_test, mlb, num_samples=5)
    
#     # Save the model
#     if (args.save_model):
#         save_model(pipeline, mlb, args.model_output)

# # ==========================
# # Entry Point
# # ==========================

# if __name__ == "__main__":
#     main()

import pandas as pd
import numpy as np
import argparse
import sys
import os
import json
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, f1_score, hamming_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import joblib

# ==========================
# Data Loading and Preprocessing
# ==========================

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Loads the dataset from a CSV file.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found.", file=sys.stderr)
        sys.exit(1)
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded data from {csv_path}")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}", file=sys.stderr)
        sys.exit(1)

def preprocess_genres(df: pd.DataFrame, min_genre_count: int = 5) -> (pd.DataFrame, MultiLabelBinarizer):
    """
    Processes the 'genres' column into a binary matrix and filters out rare genres.

    Args:
        df (pd.DataFrame): DataFrame containing the 'genres' column.
        min_genre_count (int): Minimum number of occurrences for a genre to be retained.

    Returns:
        pd.DataFrame: Original DataFrame with 'genres' column as lists.
        MultiLabelBinarizer: Fitted MultiLabelBinarizer instance with filtered genres.
    """
    if 'genres' not in df.columns:
        print("Error: 'genres' column not found in the dataset.", file=sys.stderr)
        sys.exit(1)
    
    # Handle missing genres
    df['genres'] = df['genres'].fillna('')
    
    # Split the comma-separated genres into lists
    df['genres'] = df['genres'].apply(lambda x: [genre.strip() for genre in x.split(',')] if x else [])
    
    # Count genre frequencies
    genre_counts = df['genres'].explode().value_counts()
    filtered_genres = genre_counts[genre_counts >= min_genre_count].index.tolist()
    
    # Filter genres in each song
    df['genres'] = df['genres'].apply(lambda genres: [genre for genre in genres if genre in filtered_genres])
    
    # Initialize MultiLabelBinarizer with filtered genres
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(df['genres'])
    
    # Create a DataFrame with genre labels
    genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_, index=df.index)
    
    # Concatenate the genre DataFrame with the original DataFrame
    df = pd.concat([df, genre_df], axis=1)
    
    print(f"Processed genres into {len(mlb.classes_)} binary columns after filtering out rare genres.")
    
    return df, mlb

def select_features_and_labels(df: pd.DataFrame, mlb: MultiLabelBinarizer) -> (pd.DataFrame, pd.DataFrame):
    """
    Selects feature columns and label columns.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        mlb (MultiLabelBinarizer): Fitted MultiLabelBinarizer instance.

    Returns:
        pd.DataFrame: Feature DataFrame.
        pd.DataFrame: Label DataFrame.
    """
    # Define feature columns (excluding identifiers and labels)
    feature_cols = ['danceability', 'energy', 'key', 'loudness', 'mode',
                   'speechiness', 'acousticness', 'instrumentalness',
                   'liveness', 'valence', 'tempo', 'duration_ms', 
                   'time_signature', 'popularity']
    
    # Check if all feature columns exist
    missing_features = set(feature_cols) - set(df.columns)
    if missing_features:
        print(f"Error: Missing feature columns: {missing_features}", file=sys.stderr)
        sys.exit(1)
    
    X = df[feature_cols].copy()
    
    # Label columns
    label_cols = mlb.classes_
    y = df[label_cols].copy()
    
    return X, y

def prepare_data(csv_path: str) -> (pd.DataFrame, pd.DataFrame, MultiLabelBinarizer, pd.Series):
    """
    Loads and preprocesses the data.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Feature DataFrame.
        pd.DataFrame: Label DataFrame.
        MultiLabelBinarizer: Fitted MultiLabelBinarizer instance.
        pd.Series: Series containing song names.
    """
    df = load_data(csv_path)
    
    # Drop duplicates if any
    initial_count = len(df)
    df = df.drop_duplicates(subset=['id'])
    print(f"Dropped {initial_count - len(df)} duplicate tracks.")
    
    # Drop rows with missing feature values
    feature_cols = ['danceability', 'energy', 'key', 'loudness', 'mode',
                   'speechiness', 'acousticness', 'instrumentalness',
                   'liveness', 'valence', 'tempo', 'duration_ms', 
                   'time_signature', 'popularity']
    df = df.dropna(subset=feature_cols)
    print(f"Data shape after dropping missing feature values: {df.shape}")
    
    # Extract song names
    names = df['name'].copy()
    
    # Preprocess genres
    df, mlb = preprocess_genres(df)
    
    # Select features and labels
    X, y = select_features_and_labels(df, mlb)
    
    return X, y, mlb, names

# ==========================
# Model Building
# ==========================

def build_model():
    """
    Builds a machine learning pipeline for multi-label classification with class weights.

    Returns:
        sklearn.pipeline.Pipeline: The machine learning pipeline.
    """
    # Define the classifier with class weights
    classifier = OneVsRestClassifier(
        RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    )
    
    # Create a pipeline with scaling and classification
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])
    
    return pipeline

# ==========================
# Training and Evaluation
# ==========================

def split_data(X: pd.DataFrame, y: pd.DataFrame, names: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the data into training and testing sets.

    Args:
        X (pd.DataFrame): Feature DataFrame.
        y (pd.DataFrame): Label DataFrame.
        names (pd.Series): Series containing song names.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed.

    Returns:
        X_train, X_test, y_train, y_test, names_train, names_test
    """
    return train_test_split(X, y, names, test_size=test_size, random_state=random_state)

def hit_rate(y_true, y_pred):
    """
    Calculates the Hit Rate: The proportion of samples where at least one predicted genre matches any actual genre.
    
    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_pred (np.ndarray): Predicted binary labels.
    
    Returns:
        float: Hit Rate.
    """
    # For each sample, check if there's any intersection between true and predicted genres
    hits = np.any(y_true & y_pred, axis=1)
    return hits.mean()

def train_model(pipeline, X_train: pd.DataFrame, y_train: pd.DataFrame):
    """
    Trains the machine learning pipeline.

    Args:
        pipeline (sklearn.pipeline.Pipeline): The machine learning pipeline.
        X_train (pd.DataFrame): Training features.
        y_train (pd.DataFrame): Training labels.

    Returns:
        sklearn.pipeline.Pipeline: The trained pipeline.
    """
    print("Training the model...")
    pipeline.fit(X_train, y_train)
    print("Model training completed.")
    return pipeline

def evaluate_model(pipeline, X_test: pd.DataFrame, y_test: pd.DataFrame):
    """
    Evaluates the model on the test set and prints classification metrics.

    Args:
        pipeline (sklearn.pipeline.Pipeline): The trained machine learning pipeline.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.DataFrame): Testing labels.
    """
    print("Evaluating the model...")
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    hamming = hamming_loss(y_test, y_pred)
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    hit = hit_rate(y_test.values, y_pred)
    
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"F1 Score (Micro): {f1_micro:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"Hit Rate: {hit:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# ==========================
# Display Sample Predictions
# ==========================

def display_sample_predictions(pipeline, X_test: pd.DataFrame, y_test: pd.DataFrame, mlb: MultiLabelBinarizer, names_test: pd.Series, num_samples: int = 5):
    """
    Displays sample predictions alongside actual genres and song names.

    Args:
        pipeline (sklearn.pipeline.Pipeline): The trained machine learning pipeline.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.DataFrame): Testing labels.
        mlb (MultiLabelBinarizer): Fitted MultiLabelBinarizer instance.
        names_test (pd.Series): Series containing song names for the test set.
        num_samples (int): Number of sample predictions to display.
    """
    # Ensure there are enough samples
    if len(X_test) < num_samples:
        num_samples = len(X_test)
    
    # Select random samples
    sample_indices = np.random.choice(X_test.index, size=num_samples, replace=False)
    samples = X_test.loc[sample_indices]
    actual_genres = y_test.loc[sample_indices]
    sample_names = names_test.loc[sample_indices]
    
    # Predict genres
    predicted_genres = pipeline.predict(samples)
    
    # Convert binary labels back to genre names
    # Ensure inputs are NumPy arrays with appropriate dtype
    actual_genres_np = actual_genres.to_numpy().astype(int)
    predicted_genres_np = predicted_genres.astype(int)
    
    actual_genres_list = mlb.inverse_transform(actual_genres_np)
    predicted_genres_list = mlb.inverse_transform(predicted_genres_np)
    
    # Display
    print("\nSample Predictions:")
    for i, idx in enumerate(sample_indices):
        print(f"\nSong {i + 1}:")
        print(f"Name: {sample_names.iloc[i]}")
        print(f"Features: {samples.loc[idx].to_dict()}")
        print(f"Actual Genres: {', '.join(actual_genres_list[i]) if actual_genres_list[i] else 'No genres'}")
        print(f"Predicted Genres: {', '.join(predicted_genres_list[i]) if predicted_genres_list[i] else 'No genres predicted'}")

# ==========================
# Model Persistence
# ==========================

def save_model(pipeline, mlb: MultiLabelBinarizer, model_path: str):
    """
    Saves the trained pipeline and MultiLabelBinarizer to disk.

    Args:
        pipeline (sklearn.pipeline.Pipeline): The trained machine learning pipeline.
        mlb (MultiLabelBinarizer): Fitted MultiLabelBinarizer instance.
        model_path (str): Path to save the model.
    """
    try:
        joblib.dump({'model': pipeline, 'mlb': mlb}, model_path)
        print(f"Model successfully saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}", file=sys.stderr)

# ==========================
# Command-Line Interface
# ==========================

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a model to predict song genres based on features.")
    
    parser.add_argument(
        '-i', '--input_csv',
        type=str,
        required=True,
        help='Path to the input CSV file containing song data.'
    )
    
    parser.add_argument(
        '-m', '--model_output',
        type=str,
        default='genre_prediction_model.joblib',
        help='Path to save the trained model. Defaults to "genre_prediction_model.joblib".'
    )
    
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Proportion of the dataset to include in the test split. Defaults to 0.2.'
    )
    
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for data splitting. Defaults to 42.'
    )
    
    return parser.parse_args()

# ==========================
# Main Execution Function
# ==========================

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Prepare the data
    X, y, mlb, names = prepare_data(args.input_csv)
    
    # Split the data
    X_train, X_test, y_train, y_test, names_train, names_test = split_data(
        X, y, names, test_size=args.test_size, random_state=args.random_state
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # Build the model pipeline
    pipeline = build_model()
    
    # Train the model
    pipeline = train_model(pipeline, X_train, y_train)
    
    # Evaluate the model
    evaluate_model(pipeline, X_test, y_test)
    
    # Display sample predictions with song names
    display_sample_predictions(pipeline, X_test, y_test, mlb, names_test, num_samples=5)
    
    # Save the model
    if (args.save_model):
        save_model(pipeline, mlb, args.model_output)

# ==========================
# Entry Point
# ==========================

if __name__ == "__main__":
    main()
