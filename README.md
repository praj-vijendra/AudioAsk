# Audio Transcription and Diarization API

This API provides endpoints for transcribing audio from YouTube videos or uploaded files, with speaker diarization capabilities using Distil-Whisper and pyannote.audio.

## Features

- YouTube video audio extraction and processing
- Audio file upload and processing
- Speaker diarization
- Multi-language support
- Prometheus metrics monitoring
- Rate limiting
- Authentication

## Prerequisites

- Python 3.10+
- Modal account and CLI
- Hugging Face account with access to pyannote.audio
- FFmpeg installed (for audio processing)

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with the following variables:
```
HF_TOKEN=your_huggingface_token
API_TOKEN=your_api_token
```


## Deployment

Deploy the API to Modal:
```bash
modal deploy src/modal/main.py
```

## API Usage

### Process YouTube Video

```bash
curl -X POST https://your-modal-url/api/v1/process \
     -H "Authorization: Bearer your_api_token" \
     -H "Content-Type: application/json" \
     -d '{
           "youtube_url": "https://www.youtube.com/watch?v=VIDEO_ID",
           "language": "en"
         }'
```

### Upload Audio File

```bash
curl -X POST https://your-modal-url/api/v1/upload \
     -H "Authorization: Bearer your_api_token" \
     -F "file=@/path/to/your/audio.mp3" \
     -F "language=en"
```

### Health Check

```bash
curl https://your-modal-url/api/v1/health
```

## API Response Format

```json
{
    "language": "en",
    "segments": [
        {
            "start": 0.0,
            "end": 10.5,
            "text": "Transcribed text here",
            "speaker": "SPEAKER_00"
        }
    ],
    "duration": 120.5
}
```

## Monitoring

- Access Prometheus metrics at `/metrics`
- View logs in Modal dashboard
- Monitor processing times and error rates

## Rate Limits

- 100 requests per minute per IP address
- Maximum audio duration: 1 hour
- Maximum file size: 500MB

## Supported Audio Formats

- MP3
- WAV
- MPEG

## Error Handling

The API returns appropriate HTTP status codes and error messages:
- 400: Bad Request (invalid input)
- 401: Unauthorized (invalid token)
- 429: Too Many Requests (rate limit exceeded)
- 500: Internal Server Error
