import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('top_50_data.csv')

# Handle missing values
df = df.dropna()

# 1. Select Features for Clustering
features = ['danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature', 'popularity']

X = df[features].copy()

# 2. Define Weights for Features
# Assign higher weights to 'danceability' and 'energy'
weights = {
    'danceability': 1.0,  # Emphasize this feature
    'energy': 1.0,         # Emphasize this feature
    # Assign default weight of 1.0 to other features
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

# 3. Apply Weights to Features
for feature in features:
    X[feature] = X[feature] * weights.get(feature, 1.0)

# 4. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# wcss = []
# K = range(1, 50)

# for k in K:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(X_scaled)
#     wcss.append(kmeans.inertia_)

# # Plot the results
# plt.figure(figsize=(10,6))
# plt.plot(K, wcss, 'bx-')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Within-cluster Sum of Squares (WCSS)')
# plt.title('Elbow Method for Optimal k')
# plt.show()

# 5. Clustering with K-Means
optimal_k = 7
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# # 6. Visualization (Optional)
# pca = PCA(n_components=2)
# principal_components = pca.fit_transform(X_scaled)
# df['pc1'] = principal_components[:, 0]
# df['pc2'] = principal_components[:, 1]

# plt.figure(figsize=(10,8))
# sns.scatterplot(data=df, x='pc1', y='pc2', hue='cluster', palette='Set1', alpha=0.7)
# plt.title('Song Clusters with Emphasized Features')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.legend(title='Cluster')
# plt.show()

# Get the unique cluster labels
unique_clusters = df['cluster'].unique()

# Iterate through each cluster and display songs
for cluster in unique_clusters:
    print(f"\n--- Cluster {cluster} ---")
    cluster_df = df[df['cluster'] == cluster]
    print(cluster_df[['name', 'artist', 'album']].to_string(index=True))

# import pandas as pd

# # Load the dataset
# df = pd.read_csv('songs_dataset_official.csv')

# # Handle missing values
# df = df.dropna()

# # Select features for clustering
# features = ['danceability', 'energy', 'key', 'loudness', 'mode',
#             'speechiness', 'acousticness', 'instrumentalness',
#             'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature', 'popularity']

# X = df[features]

# # Feature Scaling
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans

# wcss = []
# K = range(1, 50)

# for k in K:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(X_scaled)
#     wcss.append(kmeans.inertia_)

# # Plot the results
# plt.figure(figsize=(10,6))
# plt.plot(K, wcss, 'bx-')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Within-cluster Sum of Squares (WCSS)')
# plt.title('Elbow Method for Optimal k')
# plt.show()

# # Assume optimal k is 5
# optimal_k = 20
# kmeans = KMeans(n_clusters=optimal_k, random_state=42)
# df['cluster'] = kmeans.fit_predict(X_scaled)


# # View cluster centers
# cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
# cluster_centers_df = pd.DataFrame(cluster_centers, columns=features)
# print(cluster_centers_df)

# # Get the unique cluster labels
# unique_clusters = df['cluster'].unique()

# # Iterate through each cluster and display songs
# for cluster in unique_clusters:
#     print(f"\n--- Cluster {cluster} ---")
#     cluster_df = df[df['cluster'] == cluster]
#     print(cluster_df[['name', 'artist', 'album']].to_string(index=True))



# # # Visualize clusters using PCA for dimensionality reduction
# # from sklearn.decomposition import PCA
# # import seaborn as sns

# # pca = PCA(n_components=2)
# # principal_components = pca.fit_transform(X_scaled)
# # df['pc1'] = principal_components[:, 0]
# # df['pc2'] = principal_components[:, 1]

# # plt.figure(figsize=(10,8))
# # sns.scatterplot(data=df, x='pc1', y='pc2', hue='cluster', palette='Set1')
# # plt.title('Song Clusters Visualized with PCA')
# # plt.show()

# # Select numerical features used for clustering
# features = ['danceability', 'energy', 'key', 'loudness', 'mode',
#             'speechiness', 'acousticness', 'instrumentalness',
#             'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature', 'popularity']

# # Group by cluster and compute mean for each feature
# cluster_summary = df.groupby('cluster')[features].mean().reset_index()

# print(cluster_summary)
