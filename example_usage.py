import requests
import os
from typing import Optional, Dict, Any
from pathlib import Path

class AudioTranscriptionAPI:
    def __init__(self, api_url: str, api_token: str):
        """
        Initialize the API client.
        
        Args:
            api_url: The base URL of your Modal API
            api_token: Your API token for authentication
        """
        self.api_url = api_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }

    def process_youtube_url(
        self,
        youtube_url: str,
        language: Optional[str] = None,
        model_name: str = "large-v2",
        compute_type: str = "float16",
        batch_size: int = 16,
        browser_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process audio from a YouTube URL.
        
        Args:
            youtube_url: URL of the YouTube video
            language: Optional language code (e.g., 'en', 'es')
            model_name: Whisper model name
            compute_type: Compute type ('float16', 'int8', 'float32')
            batch_size: Batch size for processing
            browser_name: Optional browser name for authentication
            
        Returns:
            Dict containing transcription and diarization results
        """
        payload = {
            "youtube_url": youtube_url,
            "language": language,
            "model_name": model_name,
            "compute_type": compute_type,
            "batch_size": batch_size,
            "browser_name": browser_name
        }
        
        response = requests.post(
            f"{self.api_url}/api/v1/process",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def process_local_file(
        self,
        file_path: str,
        language: Optional[str] = None,
        model_name: str = "large-v2",
        compute_type: str = "float16",
        batch_size: int = 16
    ) -> Dict[str, Any]:
        """
        Process a local audio file.
        
        Args:
            file_path: Path to the local audio file
            language: Optional language code (e.g., 'en', 'es')
            model_name: Whisper model name
            compute_type: Compute type ('float16', 'int8', 'float32')
            batch_size: Batch size for processing
            
        Returns:
            Dict containing transcription and diarization results
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        files = {
            "file": (Path(file_path).name, open(file_path, "rb"), "audio/mpeg")
        }
        
        data = {
            "language": language,
            "model_name": model_name,
            "compute_type": compute_type,
            "batch_size": batch_size
        }
        
        response = requests.post(
            f"{self.api_url}/api/v1/upload",
            headers={"Authorization": f"Bearer {self.headers['Authorization']}"},
            files=files,
            data=data
        )
        response.raise_for_status()
        return response.json()

# Example usage
if __name__ == "__main__":
    # Initialize the API client
    API_URL = "https://prajwal-vijendra--fastapi-app.modal.run"  # Replace with your Modal API URL
    API_TOKEN = "as-oOIOCpyyoYvFJAB0i468El"
    
    api = AudioTranscriptionAPI(API_URL, API_TOKEN)
    
    try:
        # Example 1: Process audio from YouTube URL
        # print("Processing audio from YouTube URL...")
        # result_url = api.process_youtube_url(
        #     youtube_url="https://www.youtube.com/watch?v=p_o4aY7xkXg",  # Example YouTube URL
        #     language="en",
        #     model_name="large-v2",  # Specify Whisper model 
        #     compute_type="float16",  # Computation precision
        #     batch_size=16,  # Processing batch size
        #     # browser_name="chrome"  # Optional: Use Chrome cookies for authentication
        # )
        # print("YouTube Processing Result:", result_url)
        
        # Example 2: Process local audio file
        print("\nProcessing local audio file...")
        result_local = api.process_local_file(
            file_path="/Users/prajwal/Downloads/videoplayback.mp3",
            language="en"
        )
        print("Local File Processing Result:", result_local)
        
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
    except Exception as e:
        print(f"Error: {e}")


#https://prajwal-vijendra--fastapi-app.modal.run/api/v1/process
#https://prajwal-vijendra--fastapi-app.modal.run/api/v1/process