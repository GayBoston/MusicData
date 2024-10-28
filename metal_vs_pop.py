import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Example lists - replace with actual artist names from your dataset
metal_artists = [
    'Metallica',
    'Slipknot',
    'Black Sabbath',
    'Korn',
    'System of a Down'
]

pop_artists = [
    'Taylor Swift',
    'Eminem',
    '50 Cent',
    'JAY-Z',
    'Christina Aguilera'
]

# If your dataset is saved as a CSV
df = pd.read_csv('songs_dataset_official.csv')

# Ensure the 'artist' column exists
if 'artist' not in df.columns:
    raise ValueError("The dataset must contain an 'artist' column.")

# Convert artist names to a standard case (e.g., title case) to avoid case mismatches
df['artist'] = df['artist'].str.title()

# Similarly, ensure your metal and pop artist lists are in title case
metal_artists = [artist.title() for artist in metal_artists]
pop_artists = [artist.title() for artist in pop_artists]

# Filter for Metal artists
metal_df = df[df['artist'].isin(metal_artists)]

# Filter for Pop artists
pop_df = df[df['artist'].isin(pop_artists)]

# Verify the number of songs in each group
print(f"Number of Metal songs: {len(metal_df)}")
print(f"Number of Pop songs: {len(pop_df)}")




# Define the features to compare
features = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'duration_ms',
    'time_signature', 'popularity'
]

# Verify that all features exist in the dataset
missing_features = set(features) - set(df.columns)
if missing_features:
    raise ValueError(f"The following features are missing from the dataset: {missing_features}")

# Calculate mean values for Metal artists
metal_means = metal_df[features].mean().rename('Metal')

# Calculate mean values for Pop artists
pop_means = pop_df[features].mean().rename('Pop')

# Combine into a single DataFrame for comparison
comparison_df = pd.concat([metal_means, pop_means], axis=1)
print(comparison_df)

# Reset index for plotting
comparison_df = comparison_df.reset_index().rename(columns={'index': 'Feature'})

# Melt the DataFrame for seaborn compatibility
comparison_melted = comparison_df.melt(id_vars='Feature', var_name='Genre', value_name='Average')

# Plot
plt.figure(figsize=(15, 8))
sns.barplot(data=comparison_melted, x='Feature', y='Average', hue='Genre')
plt.title('Average Feature Values: Metal vs Pop')
plt.xticks(rotation=45)
plt.legend(title='Genre')
plt.tight_layout()
plt.show()


import numpy as np

# Function to create a radar chart
def create_radar_chart(categories, values, title, color):
    N = len(categories)
    
    # What will be the angle of each axis in the plot? (in radians)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    
    # The plot is made circular by completing the loop
    values += values[:1]
    angles += angles[:1]
    
    # Initialize the radar plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw the outline of the radar chart
    ax.plot(angles, values, color=color, linewidth=2)
    ax.fill(angles, values, color=color, alpha=0.25)
    
    # Fix the category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Set the range for radial axes
    ax.set_rlabel_position(30)
    ax.set_ylim(0, comparison_melted['Average'].max() + 0.1 * comparison_melted['Average'].max())
    
    plt.title(title, y=1.08)
    plt.show()

# Prepare data for Metal
metal_values = comparison_df[comparison_df['Genre'] == 'Metal']['Average'].tolist()
create_radar_chart(comparison_df['Feature'].tolist(), metal_values, 'Metal Artists Feature Averages', 'blue')

# Prepare data for Pop
pop_values = comparison_df[comparison_df['Genre'] == 'Pop']['Average'].tolist()
create_radar_chart(comparison_df['Feature'].tolist(), pop_values, 'Pop Artists Feature Averages', 'red')
