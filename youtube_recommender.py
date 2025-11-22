import requests
from urllib.parse import quote
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple

class YouTubeRecommender:
    """
    Hand-coded YouTube video recommender using semantic similarity.
    Searches and ranks videos based on query and context relevance.
    """
    
    def __init__(self, api_key: str, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the recommender.
        
        Args:
            api_key: YouTube Data API v3 key
            embedding_model_name: Name of the sentence transformer model
        """
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3/search"
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
    def search_videos(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search YouTube videos using the Data API.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to fetch
            
        Returns:
            List of video dictionaries with metadata
        """

        params = {
            'part': 'snippet',
            'q': query,
            'type': 'video',
            'maxResults': max_results,
            'key': self.api_key,
            'order': 'relevance',
            'videoDefinition': 'any',
            'videoEmbeddable': 'true'
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            videos = []
            for item in data.get('items', []):
                video = {
                    'video_id': item['id']['videoId'],
                    'title': item['snippet']['title'],
                    'description': item['snippet']['description'],
                    'channel': item['snippet']['channelTitle'],
                    'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                    'thumbnail': item['snippet']['thumbnails']['high']['url']
                }
                videos.append(video)
            
            return videos
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching videos: {e}")
            return []
        