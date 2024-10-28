import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys

DEFAULT_CSV = 'songs_dataset_official.csv'

def load_data(file_path):
    """Load the dataset and handle missing values."""
    df = pd.read_csv(file_path)
    return df.dropna()

def select_features(df):
    """Select relevant features for clustering."""
    features = ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'duration_ms', 
                'time_signature', 'popularity']
    return df[features].copy(), features

def apply_weights(X, weights):
    """Apply weights to the selected features."""
    for feature in X.columns:
        X[feature] = X[feature] * weights.get(feature, 1.0)
    return X

def scale_features(X):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def determine_optimal_k(X_scaled):
    """Determine the optimal number of clusters using the elbow method."""
    wcss = []
    K = range(1, 50)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(K, wcss, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method for Optimal k')
    plt.show()

def perform_clustering(X_scaled, optimal_k):
    """Perform K-Means clustering and add cluster labels to the DataFrame."""
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    return kmeans.fit_predict(X_scaled)

def visualize_clusters(X_scaled, df):
    """Visualize the clusters using PCA."""
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    df['pc1'] = principal_components[:, 0]
    df['pc2'] = principal_components[:, 1]
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='pc1', y='pc2', hue='cluster', palette='Set1', alpha=0.7)
    plt.title('Song Clusters with Emphasized Features')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.show()

def display_cluster_songs(df):
    """Display songs in each cluster."""
    unique_clusters = df['cluster'].unique()
    
    for cluster in unique_clusters:
        print(f"\n--- Cluster {cluster} ---")
        cluster_df = df[df['cluster'] == cluster]
        print(cluster_df[['name', 'artist', 'album']].to_string(index=True))

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess and cluster songs")
    parser.add_argument(
        '-f', '--filename',
        type=str,
        default=DEFAULT_CSV,
        help='CSV file containing songs'
    )

    parser.add_argument(
        '-e', '--elbow_method',
        type=bool,
        default=False,
        help='Elbow method for finding optimal clusters'
    )

    parser.add_argument(
        '-v', '--visualize',
        type=bool,
        default=False,
        help='Visualize clusters with PCA (not useful yet)'
    )    

    parser.add_argument(
        '-k', '--k_clusters',
        type=int,
        default=5,
        help='Number of clusters, defaults to 5'
    )
    return parser.parse_args()

def main(**kwargs):
    """Main function to execute the clustering analysis."""
    file_path = kwargs.get('filename', DEFAULT_CSV)
    
    df = load_data(file_path)
    
    X, features = select_features(df)
    
    weights = {
        'danceability': 1.0,
        'energy': 1.0,
        'key': 1.0,
        'loudness': 1.0,
        'mode': 1.0,
        'speechiness': 1.0,
        'acousticness': 1.0,
        'instrumentalness': 1.0,
        'liveness': 1.0,
        'valence': 1.0,
        'tempo': 1.0,
        'duration_ms': 1.0,
        'time_signature': 1.0,
        'popularity': 1.0
    }
    
    X_weighted = apply_weights(X, weights)
    X_scaled = scale_features(X_weighted)
    
    if (kwargs.get('elbow_method')):
        determine_optimal_k(X_scaled)
    
    optimal_k = kwargs.get('k_clusters')  # Set this based on the elbow method result
    df['cluster'] = perform_clustering(X_scaled, optimal_k)
    
    if (kwargs.get('visualize')):
        visualize_clusters(X_scaled, df)
    display_cluster_songs(df)

if __name__ == "__main__":
    args = parse_arguments()
    main(filename=args.filename, elbow_method=args.elbow_method, visualize=args.visualize, k_clusters=args.k_clusters)
