import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import argparse
import sys
import os
import json

# ==========================
# Configuration and Settings
# ==========================

def load_config(config_path: str = 'config.json') -> dict:
    """
    Loads configuration from a JSON file.

    Args:
        config_path (str): Path to the JSON configuration file.

    Returns:
        dict: A dictionary containing configuration data.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        json.JSONDecodeError: If the JSON is malformed.
        KeyError: If required keys are missing.
    """
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found.", file=sys.stderr)
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        
        # Validate that required keys exist
        if 'SPOTIFY_CLIENT_ID' not in config or 'SPOTIFY_CLIENT_SECRET' not in config:
            print("Error: 'SPOTIFY_CLIENT_ID' and 'SPOTIFY_CLIENT_SECRET' must be set in the configuration file.", file=sys.stderr)
            sys.exit(1)
        
        return config
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON configuration: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

# Load configuration from config.json
config = load_config()

# Extract Spotify API credentials
CLIENT_ID = config['SPOTIFY_CLIENT_ID']
CLIENT_SECRET = config['SPOTIFY_CLIENT_SECRET']

# Default Output CSV file name
DEFAULT_OUTPUT_CSV = 'songs_dataset_official.csv'

# ==========================
# Function Definitions
# ==========================

def initialize_spotify_client(client_id: str, client_secret: str) -> spotipy.Spotify:
    """
    Initializes and returns a Spotify client using client credentials.

    Args:
        client_id (str): Spotify API Client ID.
        client_secret (str): Spotify API Client Secret.

    Returns:
        spotipy.Spotify: Authenticated Spotify client.
    """
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    return sp

def get_playlist_track_ids(sp: spotipy.Spotify, playlist_id: str) -> list:
    """
    Retrieves all track IDs from the specified Spotify playlist.

    Args:
        sp (spotipy.Spotify): Authenticated Spotify client.
        playlist_id (str): Spotify Playlist ID.

    Returns:
        list: A list of track IDs.
    """
    try:
        results = sp.playlist_tracks(playlist_id)
        tracks = results['items']
        
        while results['next']:
            results = sp.next(results)
            tracks.extend(results['items'])
        
        track_ids = [item['track']['id'] for item in tracks if item['track']['id'] is not None]
        return track_ids
    except spotipy.SpotifyException as e:
        print(f"Spotify API error: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Unexpected error while fetching playlist tracks: {e}", file=sys.stderr)
        return []

def get_song_features(sp: spotipy.Spotify, track_id: str) -> dict:
    """
    Retrieves audio features and track information for a given track ID.

    Args:
        sp (spotipy.Spotify): Authenticated Spotify client.
        track_id (str): Spotify Track ID.

    Returns:
        dict: A dictionary containing song data.
    """
    try:
        # Get audio features
        features = sp.audio_features(track_id)[0]
        if features is None:
            raise ValueError("No audio features found.")
    
        # Get track information
        track = sp.track(track_id)
    
        song_data = {
            'id': track_id,
            'name': track['name'],
            'artist': ', '.join([artist['name'] for artist in track['artists']]),
            'album': track['album']['name'],
            'release_date': track['album']['release_date'],
            # 'genre': track['artists'][0]['genres'] if 'genres' in track['artists'][0] else None,
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
    except spotipy.SpotifyException as e:
        print(f"Spotify API error for track {track_id}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error processing track {track_id}: {e}", file=sys.stderr)
    return {}

def process_playlist(sp: spotipy.Spotify, playlist_id: str) -> list:
    """
    Processes all tracks in a playlist and retrieves their features.

    Args:
        sp (spotipy.Spotify): Authenticated Spotify client.
        playlist_id (str): Spotify Playlist ID.

    Returns:
        list: A list of dictionaries containing song data.
    """
    track_ids = get_playlist_track_ids(sp, playlist_id)
    if not track_ids:
        print("No tracks found in the playlist.", file=sys.stderr)
        return []
    
    song_data_list = []
    
    for idx, track_id in enumerate(track_ids):
        song_data = get_song_features(sp, track_id)
        if song_data:
            song_data_list.append(song_data)
            print(f"Processed {idx + 1}/{len(track_ids)}: {song_data['name']} by {song_data['artist']}")
        else:
            print(f"Skipped track {idx + 1}/{len(track_ids)}: ID {track_id}", file=sys.stderr)
    
    return song_data_list

def save_to_csv(data: list, filename: str):
    """
    Saves the song data to a CSV file.

    Args:
        data (list): List of dictionaries containing song data.
        filename (str): The output CSV file name.
    """
    if not data:
        print("No data to save.", file=sys.stderr)
        return
    
    try:
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Data successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving data to CSV: {e}", file=sys.stderr)

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Fetch Spotify playlist data and save to CSV.")
    
    parser.add_argument(
        '-p', '--playlist_id',
        type=str,
        required=True,
        help='Spotify Playlist ID to fetch tracks from.'
    )
    
    parser.add_argument(
        '-o', '--output_csv',
        type=str,
        default=DEFAULT_OUTPUT_CSV,
        help=f'Output CSV file name. Defaults to "{DEFAULT_OUTPUT_CSV}".'
    )
    
    return parser.parse_args()

# ==========================
# Main Execution Function
# ==========================

def main(**kwargs):
    """
    Main function to execute the Spotify playlist processing.

    Accepts keyword arguments for playlist_id and output_csv.
    
    Args:
        **kwargs: Arbitrary keyword arguments containing 'playlist_id' and 'output_csv'.
    """
    playlist_id = kwargs.get('playlist_id')
    output_csv = kwargs.get('output_csv', DEFAULT_OUTPUT_CSV)
    
    if not playlist_id:
        print("Error: Playlist ID is required.", file=sys.stderr)
        sys.exit(1)
    
    # Initialize Spotify client
    sp = initialize_spotify_client(CLIENT_ID, CLIENT_SECRET)
    
    # Process the playlist and get song data
    song_data = process_playlist(sp, playlist_id)
    
    # Save the data to a CSV file
    save_to_csv(song_data, output_csv)

# ==========================
# Entry Point
# ==========================

if __name__ == "__main__":
    args = parse_arguments()
    main(playlist_id=args.playlist_id, output_csv=args.output_csv)

