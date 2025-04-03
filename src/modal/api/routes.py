import logging
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, List
from pydantic import BaseModel, Field, HttpUrl, field_validator
from ..config.settings import (
    DEFAULT_MODEL_NAME, DEFAULT_COMPUTE_TYPE,
    DEFAULT_BATCH_SIZE, ALLOWED_AUDIO_TYPES
)
from ..models.audio_processor import AudioProcessor
from ..utils.youtube import YouTubeExtractor
import os
import tempfile
import shutil
import hmac

logger = logging.getLogger(__name__)
router = APIRouter()
auth_scheme = HTTPBearer()
processor = AudioProcessor()
processor.startup()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    """Verifies the provided bearer token."""
    token = credentials.credentials
    if not token:
        logger.warning("Empty token received")
        raise HTTPException(status_code=401, detail="Missing authentication credentials")
    
    # Use constant-time comparison to prevent timing attacks
    if not hmac.compare_digest(token, os.environ["API_TOKEN"]):
        logger.warning("Invalid token received")
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return True

class ProcessRequest(BaseModel):
    youtube_url: HttpUrl
    language: Optional[str] = Field(None, description="Optional: Language code (e.g., 'en', 'es')")
    model_name: Optional[str] = Field(DEFAULT_MODEL_NAME, description="Optional: Model name")
    compute_type: Optional[str] = Field(DEFAULT_COMPUTE_TYPE, description="Optional: Compute type")
    batch_size: Optional[int] = Field(DEFAULT_BATCH_SIZE, description="Optional: Batch size")
    browser_name: Optional[str] = Field(None, description="Optional: Browser to extract cookies from (chrome, firefox, etc.)")

    @field_validator('compute_type')
    @classmethod
    def validate_compute_type(cls, v):
        if v not in ['float16', 'int8', 'float32']:
            raise ValueError('compute_type must be one of: float16, int8, float32')
        return v

    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v):
        if v < 1 or v > 32:
            raise ValueError('batch_size must be between 1 and 32')
        return v
        
    @field_validator('browser_name')
    @classmethod
    def validate_browser_name(cls, v):
        if v and v not in ['chrome', 'chromium', 'firefox', 'opera', 'edge', 'safari']:
            raise ValueError('browser_name must be one of: chrome, chromium, firefox, opera, edge, safari')
        return v

class Segment(BaseModel):
    start: Optional[float]
    end: Optional[float]
    text: str
    speaker: Optional[str] = None

class ProcessResponse(BaseModel):
    language: str
    segments: str
    duration: float
    warning: Optional[str] = None

@router.post("/process", response_model=ProcessResponse, dependencies=[Depends(verify_token)])
async def process_youtube_audio(request: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Process audio from YouTube URL for transcription and diarization.
    
    - Downloads audio from YouTube URL
    - Transcribes audio using Distil-Whisper
    - Performs speaker diarization
    - Returns transcription with speaker labels and timing information
    """
    logger.info(f"Received request for YouTube URL: {request.youtube_url}")
    
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        
        # Setup YouTube authentication
        youtube_token = os.environ.get("YOUTUBE_TOKEN")
        
        # Initialize YouTube extractor with all available authentication methods
        # The extractor will try them in order: cookies file, browser cookies, token
        youtube_extractor = YouTubeExtractor(
            temp_dir, 
            youtube_token=youtube_token,
        )
        
        # Extract audio from YouTube
        audio_path = youtube_extractor.extract_audio(str(request.youtube_url))
        
        # Process the audio
        result = await processor.process_audio_file(
            file_path=audio_path,
            language=request.language,
            model_name=request.model_name,
            compute_type=request.compute_type,
            batch_size=request.batch_size,
        )

        logger.info(f"Successfully processed YouTube URL: {request.youtube_url}")
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
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {e}")

@router.post("/upload", response_model=ProcessResponse)
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
        file_path = os.path.join(temp_dir, file.filename)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the audio
        result = await processor.process_audio_file(
            file_path=file_path,
            language=language,
            model_name=model_name,
            compute_type=compute_type,
            batch_size=batch_size,
        )

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
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {e}")

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}