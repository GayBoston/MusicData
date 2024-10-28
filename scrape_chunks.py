import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import argparse
import sys
import os
import json
import time
from spotipy.exceptions import SpotifyException

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
    try:
        client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager, requests_timeout=10, retries=10, status_forcelist=[429, 500, 502, 503, 504])
        return sp
    except Exception as e:
        print(f"Failed to initialize Spotify client: {e}", file=sys.stderr)
        sys.exit(1)

def chunkify(lst, n):
    """
    Splits a list into chunks of size n.

    Args:
        lst (list): The list to split.
        n (int): The maximum size of each chunk.

    Returns:
        generator: A generator yielding chunks of the list.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_playlist_track_ids(sp: spotipy.Spotify, playlist_id: str, max_retries: int = 5) -> list:
    """
    Retrieves all track IDs from the specified Spotify playlist, handling 429 errors with retries.

    Args:
        sp (spotipy.Spotify): Authenticated Spotify client.
        playlist_id (str): Spotify Playlist ID.
        max_retries (int): Maximum number of retry attempts for 429 errors.

    Returns:
        list: A list of track IDs.
    """
    track_ids = []
    retries = 0
    while retries <= max_retries:
        try:
            results = sp.playlist_tracks(playlist_id, limit=100)
            tracks = results['items']
            track_ids.extend([item['track']['id'] for item in tracks if item['track']['id'] is not None])

            while results['next']:
                results = sp.next(results)
                tracks = results['items']
                track_ids.extend([item['track']['id'] for item in tracks if item['track']['id'] is not None])
            
            return track_ids

        except SpotifyException as e:
            if e.http_status == 429:
                retry_after = int(e.headers.get('Retry-After', 5))
                print(f"Rate limit exceeded. Retrying after {retry_after} seconds...", file=sys.stderr)
                time.sleep(retry_after)
                retries += 1
            else:
                print(f"Spotify API error: {e}", file=sys.stderr)
                return []
        except Exception as e:
            print(f"Unexpected error while fetching playlist tracks: {e}", file=sys.stderr)
            return []
    
    print("Max retries reached while fetching playlist tracks. Exiting.", file=sys.stderr)
    return []

def get_song_features_bulk(sp: spotipy.Spotify, track_ids: list, max_retries: int = 5) -> list:
    """
    Retrieves audio features and track information for a list of track IDs in bulk, handling 429 errors with retries.

    Args:
        sp (spotipy.Spotify): Authenticated Spotify client.
        track_ids (list): List of Spotify Track IDs.
        max_retries (int): Maximum number of retry attempts for 429 errors.

    Returns:
        list: A list of dictionaries containing song data.
    """
    retries = 0
    backoff_factor = 2  # Exponential backoff factor

    while retries <= max_retries:
        try:
            # Fetch audio features for the batch
            features_list = sp.audio_features(track_ids)
            # Fetch track information for the batch
            tracks_info = sp.tracks(track_ids)['tracks']
            
            song_data_list = []
            for track, features in zip(tracks_info, features_list):
                if features is None:
                    print(f"No audio features found for track ID {track['id']}. Skipping.", file=sys.stderr)
                    continue
                
                song_data = {
                    'id': track['id'],
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
                
                song_data_list.append(song_data)
                print(f"Processed: {song_data['name']} by {song_data['artist']}")
            
            return song_data_list

        except SpotifyException as e:
            if e.http_status == 429:
                retry_after = int(e.headers.get('Retry-After', 5))
                wait_time = retry_after * (backoff_factor ** retries)
                print(f"Rate limit exceeded. Retrying after {wait_time} seconds...", file=sys.stderr)
                time.sleep(wait_time)
                retries += 1
            else:
                print(f"Spotify API error: {e}", file=sys.stderr)
                return []
        except Exception as e:
            print(f"Unexpected error while fetching song features: {e}", file=sys.stderr)
            return []
    
    print("Max retries reached while fetching song features. Exiting.", file=sys.stderr)
    return []

def process_playlist(sp: spotipy.Spotify, playlist_id: str, batch_size: int = 100) -> list:
    """
    Processes all tracks in a playlist in batches and retrieves their features.

    Args:
        sp (spotipy.Spotify): Authenticated Spotify client.
        playlist_id (str): Spotify Playlist ID.
        batch_size (int): Number of tracks to process in each batch.

    Returns:
        list: A list of dictionaries containing song data.
    """
    track_ids = get_playlist_track_ids(sp, playlist_id)
    if not track_ids:
        print("No tracks found in the playlist.", file=sys.stderr)
        return []
    
    song_data_list = []
    total_tracks = len(track_ids)
    print(f"Total tracks to process: {total_tracks}")

    # Process in batches
    for i, batch in enumerate(chunkify(track_ids, batch_size), start=1):
        print(f"Processing batch {i}: Tracks {i*batch_size - batch_size + 1} to {min(i*batch_size, total_tracks)}")
        song_data = get_song_features_bulk(sp, batch)
        song_data_list.extend(song_data)
        # Optional: Sleep between batches to further reduce the risk of rate limiting
        time.sleep(0.1)  # Adjust as needed

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
        default=None,  # We'll handle the default dynamically
        help='Output CSV file name. Defaults to "{playlist_id}_data.csv" if not provided.'
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
    output_csv = kwargs.get('output_csv')
    
    if not playlist_id:
        print("Error: Playlist ID is required.", file=sys.stderr)
        sys.exit(1)
    
    if not output_csv:
        # Generate default filename based on playlist_id
        # Sanitize playlist_id if necessary
        safe_playlist_id = ''.join(c for c in playlist_id if c.isalnum() or c in ('-', '_')).rstrip()
        output_csv = f"{safe_playlist_id}_data.csv"
    
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
