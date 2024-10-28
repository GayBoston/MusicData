import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

# Replace these with your actual credentials
CLIENT_ID = '82698dddd0a44119a3c3d4b98013e9c4'
CLIENT_SECRET = '97a866cc707e4dbd93f28bd052b2d481'

client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def get_song_features(track_id):
    # Get audio features
    features = sp.audio_features(track_id)[0]
    
    # Get track information
    track = sp.track(track_id)
    
    song_data = {
        'id': track_id,
        'name': track['name'],
        'artist': ', '.join([artist['name'] for artist in track['artists']]),
        'album': track['album']['name'],
        'release_date': track['album']['release_date'],
        'genre': track['artists'][0]['genres'] if 'genres' in track['artists'][0] else None,
        'danceability': features['danceability'],
        'energy': features['energy'],
        'key': features['key'],
        'loudness': features['loudness'],
        'mode': features['mode'],
        'speechiness': features['speechiness'],
        'acousticness': features['acousticness'],
        'instrumentalness': features['instrumentalness'],
        'liveness': features['liveness'],
        'valence': features['valence'],
        'tempo': features['tempo'],
        'duration_ms': features['duration_ms'],
        'time_signature': features['time_signature'],
        'popularity': track['popularity']
    }
    
    return song_data

def get_playlist_tracks(playlist_id):
    results = sp.playlist_tracks(playlist_id)
    # return results
    tracks = results['items']
    
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    
    track_ids = [item['track']['id'] for item in tracks if item['track']['id'] is not None]
    return track_ids

# Example: Fetching tracks from a popular playlist
PLAYLIST_ID = '4NDXWHwYWjFmgVPkNy4YlF'  # Replace with your desired playlist ID
track_ids = get_playlist_tracks(PLAYLIST_ID)

song_data_list = []

for idx, track_id in enumerate(track_ids):
    try:
        song_data = get_song_features(track_id)
        song_data_list.append(song_data)
        print(f"Processed {idx+1}/{len(track_ids)}: {song_data['name']} by {song_data['artist']}")
    except Exception as e:
        print(f"Error processing {track_id}: {e}")

# Create a DataFrame
df = pd.DataFrame(song_data_list)

# Save to CSV
df.to_csv('songs_dataset.csv', index=False)
