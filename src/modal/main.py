import modal
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from src.modal.api.routes import router
from src.modal.models.audio_processor import AudioProcessor
from src.modal.config.settings import GPU_CONFIG, CACHE_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

# Create FastAPI app
fastapi_application = FastAPI(
    title="Audio Transcription and Diarization API",
    description="API for transcribing audio using WhisperX and performing speaker diarization",
    version="0.1.0"
)

# Add CORS middleware
fastapi_application.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(fastapi_application).expose(fastapi_application)

fastapi_application.include_router(router, prefix="/api/v1")

# Create Modal app
app = modal.App("distil-whisper-diarization-api-prod")

# Create Modal image
whisper_image = (
    modal.Image.from_registry(f"nvidia/cuda:12.2.2-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "numpy<2.0.0",
        "torch==2.3.1",
        "torchaudio==2.3.1",
        "transformers",
        "whisperx",
        "pyannote.audio==3.3.2",
        "accelerate>=0.26.0",
        "hf_transfer",
        "requests",
        "ffmpeg-python",
        "fastapi",
        "uvicorn",
        "pydantic",
        "prometheus-client",
        "prometheus-fastapi-instrumentator",
        "tenacity",
        "python-multipart",
        "aiofiles",
        "python-dotenv",
        "yt-dlp",
        "browser_cookie3",
        "nvidia-pyindex",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # "TORCH_CUDA_ARCH_LIST": "7.5;8.0;8.6;9.0",
        "CUDA_VISIBLE_DEVICES": "0"
    })
    .add_local_python_source("src")
)

volume = modal.Volume.from_name(
    "distil-whisper-diarization-prod-cache",
    create_if_missing=True
)

# Mount the FastAPI app
@app.function(
    image=whisper_image,
    gpu=GPU_CONFIG,
    secrets=[
        modal.Secret.from_name("custom-secret-2")
    ],
    volumes={CACHE_DIR: volume},
    scaledown_window=600,
    timeout=1800,
)
@modal.asgi_app(label="fastapi-app")
def fastapi_app():
    return fastapi_application