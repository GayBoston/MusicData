# import pandas as pd

# # Load the dataset
# df = pd.read_csv('songs_dataset.csv')

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

import pandas as pd

# Load the dataset
df = pd.read_csv('songs_dataset.csv')

# Display summary of missing values
print(df.isnull().sum())
