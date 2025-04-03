# Filename: transcribe_diarize_api_prod.py

import modal
import tempfile
import os
import time
import logging
import gc
import torch
import traceback
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Request, Depends, HTTPException, Security, BackgroundTasks, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl, validator
import prometheus_client as prom
from prometheus_fastapi_instrumentator import Instrumentator
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torchaudio
import numpy as np
from pyannote.audio import Pipeline
from pyannote.core import Segment as PyannoteSegment
import torch.nn.functional as F

# Load environment variables
load_dotenv()

# --- Enhanced Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# --- Prometheus Metrics ---
PROCESSING_TIME = prom.Histogram(
    'audio_processing_seconds',
    'Time spent processing audio',
    ['stage']
)
ERROR_COUNTER = prom.Counter(
    'audio_processing_errors_total',
    'Total number of processing errors',
    ['error_type']
)
REQUEST_COUNTER = prom.Counter(
    'audio_processing_requests_total',
    'Total number of processing requests',
    ['status']
)

# --- Configuration Constants ---
DEFAULT_MODEL_NAME = "distil-whisper/distil-large-v3"
DEFAULT_COMPUTE_TYPE = "float16"
DEFAULT_BATCH_SIZE = 16
MAX_AUDIO_DURATION = 3600  # 1 hour in seconds
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
RATE_LIMIT_REQUESTS = 100  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds
ALLOWED_AUDIO_TYPES = ["audio/mpeg", "audio/wav", "audio/x-wav", "audio/mp3"]

# --- Stub Definition ---
stub = modal.Stub("distil-whisper-diarization-api-prod")

# --- Rate Limiting ---
class RateLimiter:
    def __init__(self, max_requests: int, window: int):
        self.max_requests = max_requests
        self.window = window
        self.requests = {}

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Clean old requests
        self.requests[client_id] = [req_time for req_time in self.requests[client_id] 
                                  if now - req_time < self.window]
        
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        self.requests[client_id].append(now)
        return True

rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW)

# --- Image Definition (Optimized) ---
whisper_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.1.0",
        "torchaudio==2.1.0",
        "transformers>=4.30.0",
        "pyannote.audio==2.1.1",
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
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TORCH_CUDA_ARCH_LIST": "7.5;8.0;8.6;9.0",
        "CUDA_VISIBLE_DEVICES": "0"
    })
)

# --- GPU Configuration ---
GPU_CONFIG = modal.gpu.A10G()

# --- Environment Variables ---
HF_TOKEN = os.getenv("HF_TOKEN")
API_TOKEN = os.getenv("API_TOKEN")

if not HF_TOKEN or not API_TOKEN:
    raise ValueError("Missing required environment variables: HF_TOKEN, API_TOKEN")

# --- Shared Volume for Model Cache ---
volume = modal.Volume.from_name("distil-whisper-diarization-prod-cache", create_if_missing=True)
CACHE_DIR = "/model_cache"

# --- Enhanced Audio Processing Class ---
@stub.cls(
    gpu=GPU_CONFIG,
    image=whisper_image,
    secrets=[HF_SECRET],
    volumes={CACHE_DIR: volume},
    container_idle_timeout=600,
    timeout=1800,
)
class AudioProcessor:
    def __init__(self):
        self.device = None
        self.model = None
        self.processor = None
        self.diarization_pipeline = None
        self.cache_dir = CACHE_DIR
        self.model_loaded = False

    @modal.enter(timeout=600)
    def startup(self):
        """Loads models when the container starts or wakes."""
        import os

        if self.model_loaded:
            logger.info("Models already loaded, skipping startup")
            return

        logger.info("Container starting, loading models...")
        start_time = time.time()
        os.makedirs(self.cache_dir, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        try:
            # Load models with retries
            self._load_models_with_retry()
            self.model_loaded = True
            end_time = time.time()
            logger.info(f"Models loaded successfully in {end_time - start_time:.2f} seconds.")
        except Exception as e:
            logger.exception(f"Failed to load models: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _load_models_with_retry(self):
        """Load models with retry logic."""
        # Load Distil-Whisper model and processor
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            DEFAULT_MODEL_NAME,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            cache_dir=self.cache_dir,
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            DEFAULT_MODEL_NAME,
            cache_dir=self.cache_dir,
        )

        # Load diarization pipeline
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN,
            cache_dir=self.cache_dir,
        ).to(self.device)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _download_audio(self, audio_url: str, temp_dir: str) -> str:
        """Downloads audio with retry logic and validation."""
        import requests
        import aiohttp
        import asyncio

        if not audio_url.startswith(("http://", "https://")):
            raise ValueError("Invalid audio source: Only HTTP/HTTPS URLs are supported.")

        filename = os.path.basename(audio_url.split("?")[0]) or f"audio_{int(time.time())}.tmp"
        filename = "".join(c for c in filename if c.isalnum() or c in ['.', '_', '-'])[:100]
        audio_path = os.path.join(temp_dir, filename)

        async def download():
            async with aiohttp.ClientSession() as session:
                async with session.get(audio_url, timeout=120) as response:
                    if response.status != 200:
                        raise ValueError(f"Failed to download audio: HTTP {response.status}")
                    
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > MAX_FILE_SIZE:
                        raise ValueError(f"Audio file exceeds maximum size of {MAX_FILE_SIZE/1024/1024}MB")

                    with open(audio_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192 * 4):
                            f.write(chunk)

        asyncio.run(download())
        return audio_path

    async def _save_upload_file(self, upload_file: UploadFile, temp_dir: str) -> str:
        """Saves uploaded file to temporary directory."""
        filename = f"upload_{int(time.time())}_{upload_file.filename}"
        filepath = os.path.join(temp_dir, filename)
        
        try:
            with open(filepath, "wb") as buffer:
                content = await upload_file.read()
                if len(content) > MAX_FILE_SIZE:
                    raise ValueError(f"File size exceeds maximum allowed size of {MAX_FILE_SIZE/1024/1024}MB")
                buffer.write(content)
            return filepath
        except Exception as e:
            logger.error(f"Error saving uploaded file: {e}")
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    def _load_audio(self, file_path: str) -> torch.Tensor:
        """Load and preprocess audio file."""
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        return waveform.squeeze().numpy()

    def _align_transcription_with_diarization(self, transcription: str, diarization: Any, duration: float) -> List[Dict[str, Any]]:
        """Align transcription with diarization segments."""
        # Create a single segment for the transcription
        segments = []
        current_speaker = None
        current_text = ""
        current_start = 0.0

        # Process each diarization turn
        for turn in diarization.get_timeline():
            if current_speaker != turn.label:
                # Save previous segment if exists
                if current_text:
                    segments.append({
                        "start": current_start,
                        "end": turn.start,
                        "text": current_text.strip(),
                        "speaker": current_speaker
                    })
                # Start new segment
                current_speaker = turn.label
                current_text = transcription  # Use full transcription for now
                current_start = turn.start

        # Add the last segment
        if current_text:
            segments.append({
                "start": current_start,
                "end": duration,
                "text": current_text.strip(),
                "speaker": current_speaker
            })

        return segments

    @modal.method()
    async def process_audio_file(self, file_path: str, language: Optional[str], 
                               model_name: str, compute_type: str, 
                               batch_size: int) -> Dict[str, Any]:
        """Processes audio file with enhanced error handling and monitoring."""
        processing_times = {}
        start_process_time = time.time()

        try:
            with PROCESSING_TIME.labels(stage='total').time():
                logger.info(f"Processing started for file: {file_path}")

                # Load audio
                with PROCESSING_TIME.labels(stage='load_audio').time():
                    audio_waveform = self._load_audio(file_path)
                    duration = len(audio_waveform)/16000
                    if duration > MAX_AUDIO_DURATION:
                        raise ValueError(f"Audio duration ({duration:.2f}s) exceeds maximum allowed duration ({MAX_AUDIO_DURATION}s)")

                # Transcription
                with PROCESSING_TIME.labels(stage='transcription').time():
                    inputs = self.processor(
                        audio_waveform,
                        sampling_rate=16000,
                        return_tensors="pt"
                    ).to(self.device)

                    # Generate transcription
                    generated_ids = self.model.generate(
                        **inputs,
                        max_length=225,
                        num_beams=5,
                        language=language if language else "en",
                        task="transcribe"
                    )

                    transcription = self.processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        language=language if language else "en"
                    )[0]

                # Diarization
                with PROCESSING_TIME.labels(stage='diarization').time():
                    try:
                        diarization = self.diarization_pipeline(
                            {"waveform": torch.from_numpy(audio_waveform).unsqueeze(0).to(self.device), "sample_rate": 16000},
                            num_speakers=None
                        )
                        
                        # Align transcription with diarization
                        segments = self._align_transcription_with_diarization(transcription, diarization, duration)
                    except Exception as diarize_e:
                        logger.warning(f"Diarization failed: {diarize_e}")
                        segments = [{
                            "start": 0.0,
                            "end": duration,
                            "text": transcription,
                            "speaker": None
                        }]

                total_time = time.time() - start_process_time
                processing_times = {
                    'total': total_time,
                    'load_audio': PROCESSING_TIME.labels(stage='load_audio')._sum.get(),
                    'transcription': PROCESSING_TIME.labels(stage='transcription')._sum.get(),
                    'diarization': PROCESSING_TIME.labels(stage='diarization')._sum.get()
                }

                REQUEST_COUNTER.labels(status='success').inc()
                return {
                    "language": language if language else "en",
                    "segments": segments,
                    "processing_times_seconds": processing_times
                }

        except Exception as e:
            ERROR_COUNTER.labels(error_type=type(e).__name__).inc()
            REQUEST_COUNTER.labels(status='error').inc()
            logger.exception(f"Critical error during processing: {e}")
            raise

    @modal.method()
    def process_audio_core(self, audio_url: str, language: Optional[str], 
                          model_name: str, compute_type: str, 
                          batch_size: int) -> Dict[str, Any]:
        """Processes audio from URL with enhanced error handling and monitoring."""
        temp_dir = None
        audio_path = None

        try:
            temp_dir = tempfile.mkdtemp()
            audio_path = self._download_audio(audio_url, temp_dir)
            return self.process_audio_file(audio_path, language, model_name, compute_type, batch_size)
        finally:
            if temp_dir and os.path.exists(temp_dir):
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary directory: {e}")

# --- FastAPI App Definition ---
web_app = FastAPI(
    title="Audio Transcription and Diarization API",
    description="API for transcribing audio using Distil-Whisper and performing speaker diarization",
    version="1.0.0"
)

# Add CORS middleware
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Prometheus metrics
Instrumentator().instrument(web_app).expose(web_app)

# --- Authentication ---
auth_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(auth_scheme)):
    """Verifies the provided bearer token."""
    token = credentials.credentials
    if token != API_TOKEN:
        logger.warning(f"Invalid token received: '{token[:5]}...'")
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return True

# --- Rate Limiting Middleware ---
@web_app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_id = request.client.host
    if not rate_limiter.is_allowed(client_id):
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Please try again later."}
        )
    response = await call_next(request)
    return response

# --- Pydantic Models ---
class ProcessRequest(BaseModel):
    audio_url: HttpUrl
    language: Optional[str] = Field(None, description="Optional: Language code (e.g., 'en', 'es')")
    model_name: Optional[str] = Field(DEFAULT_MODEL_NAME, description="Optional: Model name")
    compute_type: Optional[str] = Field(DEFAULT_COMPUTE_TYPE, description="Optional: Compute type")
    batch_size: Optional[int] = Field(DEFAULT_BATCH_SIZE, description="Optional: Batch size")

    @validator('compute_type')
    def validate_compute_type(cls, v):
        if v not in ['float16', 'int8', 'float32']:
            raise ValueError('compute_type must be one of: float16, int8, float32')
        return v

    @validator('batch_size')
    def validate_batch_size(cls, v):
        if v < 1 or v > 32:
            raise ValueError('batch_size must be between 1 and 32')
        return v

class Segment(BaseModel):
    start: Optional[float]
    end: Optional[float]
    text: str
    speaker: Optional[str] = None

class ProcessResponse(BaseModel):
    language: str
    segments: List[Segment]
    processing_times_seconds: Dict[str, float]
    warning: Optional[str] = None

# --- API Endpoints ---
processor = AudioProcessor()

@web_app.post("/process", response_model=ProcessResponse, dependencies=[Depends(verify_token)])
async def process_audio_endpoint(request: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Process audio file from URL for transcription and diarization.
    
    - Downloads audio from provided URL
    - Transcribes audio using Distil-Whisper
    - Performs speaker diarization
    - Returns transcription with speaker labels and timing information
    """
    logger.info(f"Received request for URL: {request.audio_url}")
    
    try:
        result = processor.process_audio_core.remote(
            audio_url=str(request.audio_url),
            language=request.language,
            model_name=request.model_name,
            compute_type=request.compute_type,
            batch_size=request.batch_size,
        )

        if isinstance(result, dict) and "error" in result:
            logger.error(f"Processing failed: {result.get('error')}")
            raise HTTPException(status_code=500, detail=f"Audio processing failed: {result.get('error')}")

        if not isinstance(result, dict) or "segments" not in result:
            logger.error(f"Unexpected result structure: {type(result)}")
            raise HTTPException(status_code=500, detail="Internal server error: Unexpected processing result format.")

        logger.info(f"Successfully processed URL: {request.audio_url}")
        return ProcessResponse(**result)

    except ValueError as e:
        logger.warning(f"Bad request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except TimeoutError as e:
        logger.error(f"Timeout: {e}")
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@web_app.post("/upload", response_model=ProcessResponse, dependencies=[Depends(verify_token)])
async def upload_audio_endpoint(
    file: UploadFile = File(...),
    language: Optional[str] = None,
    model_name: Optional[str] = DEFAULT_MODEL_NAME,
    compute_type: Optional[str] = DEFAULT_COMPUTE_TYPE,
    batch_size: Optional[int] = DEFAULT_BATCH_SIZE,
    background_tasks: BackgroundTasks = None
):
    """
    Upload and process audio file for transcription and diarization.
    
    - Accepts audio file upload
    - Transcribes audio using Distil-Whisper
    - Performs speaker diarization
    - Returns transcription with speaker labels and timing information
    """
    if file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_AUDIO_TYPES)}"
        )

    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        file_path = await processor._save_upload_file(file, temp_dir)
        
        result = await processor.process_audio_file.remote(
            file_path=file_path,
            language=language,
            model_name=model_name,
            compute_type=compute_type,
            batch_size=batch_size,
        )

        if isinstance(result, dict) and "error" in result:
            logger.error(f"Processing failed: {result.get('error')}")
            raise HTTPException(status_code=500, detail=f"Audio processing failed: {result.get('error')}")

        if not isinstance(result, dict) or "segments" not in result:
            logger.error(f"Unexpected result structure: {type(result)}")
            raise HTTPException(status_code=500, detail="Internal server error: Unexpected processing result format.")

        logger.info(f"Successfully processed uploaded file: {file.filename}")
        return ProcessResponse(**result)

    except ValueError as e:
        logger.warning(f"Bad request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {e}")

@web_app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

# Mount the FastAPI app
@stub.asgi_app()
def fastapi_app():
    return web_app

# --- Deployment Instructions ---
"""
To deploy this API:

1. Create a .env file with the following variables:
   ```
   HF_TOKEN=your_huggingface_token
   API_TOKEN=your_api_token
   ```

2. Deploy the API:
   ```bash
   modal deploy modal.py
   ```

3. Use the API:

   For URL-based processing:
   ```bash
   curl -X POST https://your-modal-url/process \
        -H "Authorization: Bearer your_api_token" \
        -H "Content-Type: application/json" \
        -d '{
              "audio_url": "https://example.com/audio.mp3",
              "language": "en"
            }'
   ```

   For file upload:
   ```bash
   curl -X POST https://your-modal-url/upload \
        -H "Authorization: Bearer your_api_token" \
        -F "file=@/path/to/your/audio.mp3" \
        -F "language=en"
   ```

4. Monitor the API:
   - Access Prometheus metrics at /metrics
   - Check logs in Modal dashboard
   - Monitor processing times and error rates
"""