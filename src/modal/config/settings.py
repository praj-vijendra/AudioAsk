import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_TOKEN = os.environ["API_TOKEN"]
HF_TOKEN = os.environ["HF_TOKEN"]
# YouTube authentication options
YOUTUBE_TOKEN = os.environ.get("YOUTUBE_TOKEN")
YOUTUBE_COOKIES_PATH = os.environ.get("YOUTUBE_COOKIES_PATH")

# Model Configuration
DEFAULT_MODEL_NAME = "distil-whisper/distil-medium.en"
DEFAULT_COMPUTE_TYPE = "float16"
DEFAULT_BATCH_SIZE = 16

# Audio Processing Limits
MAX_AUDIO_DURATION = 3600  # 1 hour in seconds
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_AUDIO_TYPES = ["audio/mpeg", "audio/wav", "audio/x-wav", "audio/mp3"]

# Rate Limiting
RATE_LIMIT_REQUESTS = 100  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds

# Cache Configuration
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "ai_meeting")

# GPU Configuration
GPU_CONFIG = "A100"

# YouTube Configuration
YOUTUBE_DL_OPTIONS = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
    'outtmpl': '%(id)s.%(ext)s',
}