import os
import json
import argparse
import requests
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()


class AudioTranscriptionAPI:
    def __init__(self, api_url: str, api_token: str):
        self.api_url = api_url.rstrip("/")
        self.auth_header = {"Authorization": f"Bearer {api_token}"}
        self.json_headers = {**self.auth_header, "Content-Type": "application/json"}

    def process_youtube_url(
        self,
        youtube_url: str,
        language: Optional[str] = None,
        model_name: str = "large-v2",
        compute_type: str = "float16",
        batch_size: int = 16,
        browser_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {
            "youtube_url": youtube_url,
            "language": language,
            "model_name": model_name,
            "compute_type": compute_type,
            "batch_size": batch_size,
            "browser_name": browser_name,
        }
        return self._post("/api/v1/process", payload, headers=self.json_headers)

    def process_local_file(
        self,
        file_path: str,
        language: Optional[str] = None,
        model_name: str = "large-v2",
        compute_type: str = "float16",
        batch_size: int = 16,
    ) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with path.open("rb") as file:
            files = {"file": (path.name, file, "audio/mpeg")}
            data = {
                "language": language,
                "model_name": model_name,
                "compute_type": compute_type,
                "batch_size": batch_size,
            }
            return self._post("/api/v1/upload", data, files=files, headers=self.auth_header)

    def _post(
        self,
        endpoint: str,
        data: Dict[str, Any],
        headers: Dict[str, str],
        files: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.api_url}{endpoint}"
        response = requests.post(
            url,
            headers=headers,
            json=None if files else data,
            data=data if files else None,
            files=files,
        )
        response.raise_for_status()
        return response.json()


def save_result_to_json(result: Dict[str, Any], folder: str = "output") -> Path:
    output_dir = Path(folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = Path(result.get("file_path", "result")).stem + ".json"
    output_file = output_dir / file_path
    with open(output_file, "w") as f:
        json.dump(result, f, indent=4)
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Transcribe YouTube or local audio files using Whisper.")
    parser.add_argument("--youtube_url", type=str, help="YouTube video URL")
    parser.add_argument("--file_path", type=str, help="Path to local audio file")
    parser.add_argument("--language", type=str, default=None, help="Language code (e.g., 'en')")
    parser.add_argument("--model_name", type=str, default="large-v2", help="Whisper model to use")
    parser.add_argument("--compute_type", type=str, default="float16", help="Compute type (float16/int8/float32)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")

    args = parser.parse_args()

    api_url = os.getenv("API_URL")
    api_token = os.getenv("API_TOKEN")

    if not api_url or not api_token:
        raise EnvironmentError("Missing API_URL or API_TOKEN in environment variables.")

    api = AudioTranscriptionAPI(api_url, api_token)

    try:
        if args.youtube_url:
            print("Processing YouTube URL...")
            yt_result = api.process_youtube_url(
                youtube_url=args.youtube_url,
                language=args.language,
                model_name=args.model_name,
                compute_type=args.compute_type,
                batch_size=args.batch_size,
            )
            yt_output = save_result_to_json(yt_result)
            print(f"YouTube result saved to {yt_output}")

        if args.file_path:
            print("Processing local file...")
            local_result = api.process_local_file(
                file_path=args.file_path,
                language=args.language,
                model_name=args.model_name,
                compute_type=args.compute_type,
                batch_size=args.batch_size,
            )
            local_output = save_result_to_json(local_result)
            print(f"Local file result saved to {local_output}")

        if not args.youtube_url and not args.file_path:
            print("Please provide at least one of --youtube_url or --file_path")

    except requests.exceptions.RequestException as e:
        print(f"API Request failed: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
