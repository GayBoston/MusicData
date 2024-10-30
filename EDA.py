# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.neighbors import NearestNeighbors


# CSV_FILE = 'song_data.csv'

# def load_csv(filename):
#     df = pd.read_csv(filename)
#     return df

# def eda(df):
#     features = ['danceability', 'energy', 'tempo', 'valence', 'popularity']
#     for feature in features:
#         plt.figure(figsize=(10, 6))
#         sns.histplot(df[feature], kde=True)
#         plt.title(f'Distribution of {feature}')
#         plt.show()

# def correlation(df):
#     corr_matrix = df.corr()
#     plt.figure(figsize=(12, 10))
#     sns.heatmap(corr_matrix, annot=True, fmt=".2f")
#     plt.title('Correlation Matrix of Audio Features')
#     plt.show()

# def recommend(df):
#     features = ['danceability', 'energy', 'tempo', 'valence', 'popularity', 'mood',
#                 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'liveness']
#     X = df[features].values

#     nn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
#     nn.fit(X)
#     song_index = df[df['name'] == 'Vermilion, Pt. 2'].index[0]
#     distances, indices = nn.kneighbors([X[song_index]])
#     similar_songs = df.iloc[indices[0]]
#     print(similar_songs[['name', 'artist']])

# def main():
#     df = load_csv(CSV_FILE)
#     df['mood'] = df['valence'] * df['energy']
#     recommend(df)
#     # eda(df)
#     # correlation(df)

# if __name__ == '__main__':
#     main()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Constants
CSV_FILE = 'song_data.csv'  # Replace with your actual filenames

def load_and_merge_csv(filename):
    df = pd.read_csv(filename).drop_duplicates()
    return df

def exploratory_data_analysis(df):
    """
    Perform EDA by plotting distributions of selected features.
    """
    features = ['danceability', 'energy', 'tempo', 'valence', 'popularity', 'loudness']
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.show()

def correlation_analysis(df):
    """
    Perform correlation analysis and plot the correlation matrix.
    """
    features = ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'duration_ms', 
                'time_signature', 'popularity']
    corr_matrix = df[features].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix of Audio Features')
    plt.show()

def feature_engineering(df):
    """
    Create new features, such as a mood score.
    """
    df['mood'] = df['valence'] * df['energy']
    return df

def recommendation_system(df, song_name, n_neighbors=5):
    """
    Recommend similar songs based on audio features.
    """
    features = ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'duration_ms', 
                'time_signature']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    nn.fit(X_scaled)
    
    song_index = df[df['name'].str.lower() == song_name.lower()].index
    if song_index.empty:
        print(f"Song '{song_name}' not found in the dataset.")
        return
    song_index = song_index[0]
    distances, indices = nn.kneighbors([X_scaled[song_index]])
    similar_songs = df.iloc[indices[0]]
    print(f"\nSongs similar to '{df.iloc[song_index]['name']}':")
    print(similar_songs[['name', 'artist']])
    return similar_songs

def clustering_analysis(df, n_clusters=5):
    """
    Perform clustering on the dataset and visualize the clusters.
    """
    features = ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'duration_ms', 
                'time_signature', 'popularity']
    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # Visualize clusters using PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    df['pc1'] = principal_components[:, 0]
    df['pc2'] = principal_components[:, 1]

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='pc1', y='pc2', hue='cluster', palette='Set1', alpha=0.7)
    plt.title('Song Clusters')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.show()

def mood_analysis(df):
    """
    Categorize songs based on mood and visualize the distribution.
    """
    def mood_label(row):
        if row['valence'] > 0.5 and row['energy'] > 0.5:
            return 'Happy/Energetic'
        elif row['valence'] <= 0.5 and row['energy'] > 0.5:
            return 'Angry'
        elif row['valence'] <= 0.5 and row['energy'] <= 0.5:
            return 'Sad'
        else:
            return 'Calm'
    df['mood_category'] = df.apply(mood_label, axis=1)
    mood_counts = df['mood_category'].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=mood_counts.index, y=mood_counts.values)
    plt.title('Mood Distribution in Songs')
    plt.xlabel('Mood')
    plt.ylabel('Count')
    plt.show()
    return df

def tempo_distribution(df):
    """
    Analyze the distribution of tempo in the dataset.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['tempo'], bins=30, kde=True)
    plt.title('Tempo Distribution')
    plt.xlabel('Beats Per Minute (BPM)')
    plt.ylabel('Count')
    plt.show()

def duration_analysis(df):
    """
    Investigate song lengths in the dataset.
    """
    df['duration_min'] = df['duration_ms'] / 60000
    plt.figure(figsize=(10, 6))
    sns.histplot(df['duration_min'], bins=30, kde=True)
    plt.title('Song Duration Distribution')
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Count')
    plt.show()

def feature_importance_analysis(df):
    """
    Determine which audio features are most influential in predicting popularity.
    """
    features = ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'duration_ms', 
                'time_signature']
    X = df[features].dropna()
    y = df.loc[X.index, 'popularity']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"\nRoot Mean Squared Error: {rmse:.2f}")

    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance_df, x='importance', y='feature')
    plt.title('Feature Importances for Predicting Popularity')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

def main():
    # Load and merge CSV files
    df = load_and_merge_csv(CSV_FILE)

    # Perform Exploratory Data Analysis
    # exploratory_data_analysis(df)

    # # Feature Engineering
    df = feature_engineering(df)

    # # Perform Correlation Analysis
    correlation_analysis(df)

    # # Mood Analysis
    # df = mood_analysis(df)

    # # Tempo Distribution
    # tempo_distribution(df)

    # # Duration Analysis
    # duration_analysis(df)

    # Clustering Analysis
    # clustering_analysis(df, n_clusters=5)

    # # Feature Importance Analysis
    # feature_importance_analysis(df)

    # Recommendation System
    song_name = 'Toxicity'  # Replace with a song name from your dataset
    recommendation_system(df, song_name)

if __name__ == '__main__':
    main()
