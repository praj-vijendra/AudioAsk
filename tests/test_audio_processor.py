import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock, mock_open, AsyncMock, ANY
import os
import tempfile
import subprocess
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.modal.models.audio_processor import AudioProcessor, MAX_AUDIO_DURATION, DEFAULT_MODEL_NAME, DEFAULT_COMPUTE_TYPE, HF_TOKEN, CACHE_DIR


# Add the fixture definition
@pytest.fixture
def audio_processor():
    """Pytest fixture to provide an AudioProcessor instance for tests."""
    return AudioProcessor()

@pytest.mark.asyncio
@patch('torch.cuda.is_available', return_value=False) 
@patch('os.makedirs')
async def test_startup(mock_makedirs, mock_cuda_available, audio_processor):
    """Test the startup method initializes the device and creates cache dir."""
    assert not audio_processor.model_loaded
    await audio_processor.startup()
    assert audio_processor.model_loaded
    assert audio_processor.device == torch.device("cpu")
    mock_makedirs.assert_called_once_with(CACHE_DIR, exist_ok=True)

@pytest.mark.asyncio
@patch('torch.cuda.is_available', return_value=True)
@patch('os.makedirs')
async def test_startup_cuda(mock_makedirs, mock_cuda_available, audio_processor):
    """Test the startup method initializes the device correctly when CUDA is available."""
    assert not audio_processor.model_loaded
    await audio_processor.startup()
    assert audio_processor.model_loaded
    assert audio_processor.device == torch.device("cuda:0")
    mock_makedirs.assert_called_once_with(CACHE_DIR, exist_ok=True)

@patch('gc.collect')
@patch('torch.cuda.empty_cache')
@patch('torch.cuda.is_available', return_value=True)
def test_cleanup(mock_cuda_available, mock_empty_cache, mock_gc_collect, audio_processor):
    """Test the cleanup method calls gc.collect and cuda.empty_cache if available."""
    audio_processor.cleanup()
    mock_gc_collect.assert_called_once()
    mock_empty_cache.assert_called_once()

@patch('gc.collect')
@patch('torch.cuda.empty_cache')
@patch('torch.cuda.is_available', return_value=False)
def test_cleanup_no_cuda(mock_cuda_available, mock_empty_cache, mock_gc_collect, audio_processor):
    """Test the cleanup method calls gc.collect but not cuda.empty_cache if CUDA not available."""
    audio_processor.cleanup()
    mock_gc_collect.assert_called_once()
    mock_empty_cache.assert_not_called()


@patch('os.path.exists', return_value=True)
@patch('torchaudio.info')
@patch('whisperx.load_audio')
@patch('torchaudio.load')
def test_load_audio_success(mock_torchaudio_load, mock_whisperx_load, mock_torchaudio_info, mock_exists, audio_processor):
    """Test _load_audio successfully loads and returns audio data."""
    dummy_file = "dummy.wav"
    mock_sample_rate = 16000
    mock_duration_seconds = 10
    mock_waveform = torch.randn(1, mock_sample_rate * mock_duration_seconds)
    mock_whisperx_audio = np.random.rand(mock_sample_rate * mock_duration_seconds).astype(np.float32)

    mock_torchaudio_info.return_value = MagicMock(sample_rate=mock_sample_rate, num_frames=mock_sample_rate * mock_duration_seconds)
    mock_torchaudio_load.return_value = (mock_waveform, mock_sample_rate)
    mock_whisperx_load.return_value = mock_whisperx_audio

    audio_data, duration = audio_processor._load_audio(dummy_file)

    mock_exists.assert_called_once_with(dummy_file)
    mock_torchaudio_info.assert_called_once_with(dummy_file)
    mock_whisperx_load.assert_called_once_with(dummy_file, 16000)
    np.testing.assert_array_equal(audio_data, mock_whisperx_audio)
    assert duration == mock_duration_seconds

@patch('os.path.exists', return_value=False)
def test_load_audio_file_not_found(mock_exists, audio_processor):
    """Test _load_audio raises ValueError if file does not exist."""
    with pytest.raises(ValueError, match="Audio file not found: non_existent.wav"):
        audio_processor._load_audio("non_existent.wav")
    mock_exists.assert_called_once_with("non_existent.wav")

@patch('os.path.exists', return_value=True)
@patch('torchaudio.info')
@patch('whisperx.load_audio')
@patch('torchaudio.load')
def test_load_audio_duration_exceeded(mock_torchaudio_load, mock_whisperx_load, mock_torchaudio_info, mock_exists, audio_processor):
    """Test _load_audio raises ValueError if audio duration exceeds MAX_AUDIO_DURATION."""
    dummy_file = "long_audio.wav"
    mock_sample_rate = 16000
    mock_duration_seconds = MAX_AUDIO_DURATION + 1
    mock_waveform = torch.randn(1, mock_sample_rate * mock_duration_seconds)
    mock_whisperx_audio = np.random.rand(mock_sample_rate * mock_duration_seconds).astype(np.float32)

    mock_torchaudio_info.return_value = MagicMock(sample_rate=mock_sample_rate, num_frames=mock_sample_rate * mock_duration_seconds)
    mock_torchaudio_load.return_value = (mock_waveform, mock_sample_rate)
    mock_whisperx_load.return_value = mock_whisperx_audio

    with pytest.raises(ValueError, match=f"exceeds maximum allowed duration"):
        audio_processor._load_audio(dummy_file)

@patch('os.path.exists', return_value=True)
@patch('torchaudio.info', side_effect=Exception("Torchaudio error"))
def test_load_audio_torchaudio_error(mock_torchaudio_info, mock_exists, audio_processor):
    """Test _load_audio raises RuntimeError if torchaudio fails."""
    with pytest.raises(RuntimeError, match="Failed to get audio metadata"):
        audio_processor._load_audio("dummy.wav")


@patch('tempfile.NamedTemporaryFile')
@patch('subprocess.run')
def test_convert_to_mono_16khz_ffmpeg_success(mock_subprocess_run, mock_temp_file, audio_processor):
    """Test successful ffmpeg conversion."""
    mock_temp_file_obj = MagicMock()
    mock_temp_file_obj.name = "/tmp/temp_output.wav"
    mock_temp_file.return_value.__enter__.return_value = mock_temp_file_obj

    mock_result = MagicMock()
    mock_result.stdout = "ffmpeg output"
    mock_result.stderr = ""
    mock_result.check_returncode.return_value = None # Indicates success
    mock_subprocess_run.return_value = mock_result

    input_path = "input.mp3"
    target_sr = 16000
    output_path = audio_processor._convert_to_mono_16khz_ffmpeg(input_path, target_sr)

    assert output_path == "/tmp/temp_output.wav"
    mock_temp_file.assert_called_once_with(suffix='.wav', delete=False)
    expected_command = [
        'ffmpeg', '-i', input_path, '-ac', '1', '-ar', str(target_sr),
        '-acodec', 'pcm_s16le', '-y', output_path
    ]
    mock_subprocess_run.assert_called_once_with(
        expected_command, capture_output=True, text=True, check=True
    )

@patch('tempfile.NamedTemporaryFile')
@patch('subprocess.run')
def test_convert_to_mono_16khz_ffmpeg_failure(mock_subprocess_run, mock_temp_file, audio_processor):
    """Test ffmpeg conversion failure raises RuntimeError."""
    mock_temp_file_obj = MagicMock()
    mock_temp_file_obj.name = "/tmp/temp_output.wav"
    mock_temp_file.return_value.__enter__.return_value = mock_temp_file_obj

    error_message = "ffmpeg error stderr"
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd="ffmpeg ...", stderr=error_message
    )

    input_path = "input.mp3"
    target_sr = 16000

    with pytest.raises(RuntimeError, match=f"Failed to convert audio.*{error_message}"):
        audio_processor._convert_to_mono_16khz_ffmpeg(input_path, target_sr)


def test_group_word_segments_empty(audio_processor):
    """Test grouping with empty input."""
    assert audio_processor._group_word_segments_by_speaker([]) == []

def test_group_word_segments_single_speaker(audio_processor):
    """Test grouping segments from a single speaker."""
    word_segments = [
        {'start': 0.1, 'end': 0.5, 'word': 'Hello', 'speaker': 'SPEAKER_00'},
        {'start': 0.6, 'end': 1.0, 'word': 'world', 'speaker': 'SPEAKER_00'},
    ]
    expected = [{
        'start': 0.1, 'end': 1.0, 'text': 'Hello world', 'speaker': 'SPEAKER_00'
    }]
    assert audio_processor._group_word_segments_by_speaker(word_segments) == expected

def test_group_word_segments_multiple_speakers(audio_processor):
    """Test grouping segments from multiple speakers."""
    word_segments = [
        {'start': 0.1, 'end': 0.5, 'word': 'Hello', 'speaker': 'SPEAKER_00'},
        {'start': 0.6, 'end': 1.0, 'word': 'there', 'speaker': 'SPEAKER_01'},
        {'start': 1.1, 'end': 1.5, 'word': 'General', 'speaker': 'SPEAKER_01'},
        {'start': 1.6, 'end': 2.0, 'word': 'Kenobi', 'speaker': 'SPEAKER_00'},
    ]
    expected = [
        {'start': 0.1, 'end': 0.5, 'text': 'Hello', 'speaker': 'SPEAKER_00'},
        {'start': 0.6, 'end': 1.5, 'text': 'there General', 'speaker': 'SPEAKER_01'},
        {'start': 1.6, 'end': 2.0, 'text': 'Kenobi', 'speaker': 'SPEAKER_00'},
    ]
    assert audio_processor._group_word_segments_by_speaker(word_segments) == expected

def test_group_word_segments_unknown_speaker(audio_processor):
    """Test grouping segments with missing or None speaker labels."""
    word_segments = [
        {'start': 0.1, 'end': 0.5, 'word': 'Hello'}, # Missing speaker
        {'start': 0.6, 'end': 1.0, 'word': 'world', 'speaker': None}, # None speaker
        {'start': 1.1, 'end': 1.5, 'word': 'again', 'speaker': 'SPEAKER_00'},
        {'start': 1.6, 'end': 2.0, 'word': '!', 'speaker': None}, # None speaker again
    ]
    expected = [
        {'start': 0.1, 'end': 1.0, 'text': 'Hello world', 'speaker': 'UNKNOWN'},
        {'start': 1.1, 'end': 1.5, 'text': 'again', 'speaker': 'SPEAKER_00'},
        {'start': 1.6, 'end': 2.0, 'text': '!', 'speaker': 'UNKNOWN'},
    ]
    assert audio_processor._group_word_segments_by_speaker(word_segments) == expected

def test_group_word_segments_missing_data(audio_processor):
    """Test grouping skips segments with missing time or text data."""
    word_segments = [
        {'start': 0.1, 'end': 0.5, 'word': 'Valid1', 'speaker': 'SPK_0'},
        {'end': 1.0, 'word': 'MissingStart', 'speaker': 'SPK_0'}, # Missing start
        {'start': 1.1, 'end': 1.5, 'word': '  ', 'speaker': 'SPK_1'}, # Empty word
        {'start': 1.6, 'end': 2.0, 'word': 'Valid2', 'speaker': 'SPK_1'},
        {'start': 2.1, 'end': None, 'word': 'MissingEnd', 'speaker': 'SPK_1'}, # Missing end
        {'start': 2.5, 'end': 3.0, 'word': 'Valid3', 'speaker': 'SPK_0'},
    ]
    expected = [
        {'start': 0.1, 'end': 0.5, 'text': 'Valid1', 'speaker': 'SPK_0'},
        {'start': 1.6, 'end': 2.0, 'text': 'Valid2', 'speaker': 'SPK_1'},
        {'start': 2.5, 'end': 3.0, 'text': 'Valid3', 'speaker': 'SPK_0'},
    ]
    assert audio_processor._group_word_segments_by_speaker(word_segments) == expected

def test_group_word_segments_from_text_key(audio_processor):
    """Test grouping when input segments use 'text' instead of 'word' key (like from alignment)."""
    word_segments = [
        {'start': 0.1, 'end': 0.5, 'text': 'Segment 1', 'speaker': 'SPEAKER_00'},
        {'start': 0.6, 'end': 1.0, 'text': 'Segment 2', 'speaker': 'SPEAKER_00'},
        {'start': 1.1, 'end': 1.5, 'text': 'Different', 'speaker': 'SPEAKER_01'},
    ]
    expected = [
        {'start': 0.1, 'end': 1.0, 'text': 'Segment 1 Segment 2', 'speaker': 'SPEAKER_00'},
        {'start': 1.1, 'end': 1.5, 'text': 'Different', 'speaker': 'SPEAKER_01'},
    ]
    assert audio_processor._group_word_segments_by_speaker(word_segments) == expected


def test_fallback_segments_basic(audio_processor):
    """Test creating fallback segments from a basic whisper result."""
    result = {
        "segments": [
            {"start": 0.0, "end": 5.0, "text": " This is segment one. "},
            {"start": 5.5, "end": 10.0, "text": "Segment two here. "},
        ]
    }
    expected = [
        {"start": 0.0, "end": 5.0, "text": "This is segment one.", "speaker": "UNKNOWN"},
        {"start": 5.5, "end": 10.0, "text": "Segment two here.", "speaker": "UNKNOWN"},
    ]
    assert audio_processor._fallback_segments(result) == expected

def test_fallback_segments_empty_result(audio_processor):
    """Test fallback with an empty segments list."""
    result = {"segments": []}
    assert audio_processor._fallback_segments(result) == []

def test_fallback_segments_missing_segments_key(audio_processor):
    """Test fallback when the 'segments' key is missing."""
    result = {"language": "en"}
    assert audio_processor._fallback_segments(result) == []

def test_fallback_segments_missing_time_data(audio_processor):
    """Test fallback skips segments missing start or end times."""
    result = {
        "segments": [
            {"start": 0.0, "text": "Missing end"},
            {"end": 5.0, "text": "Missing start"},
            {"start": 6.0, "end": 7.0, "text": "Valid segment"},
            {"start": None, "end": 8.0, "text": "None start"},
        ]
    }
    expected = [
        {"start": 6.0, "end": 7.0, "text": "Valid segment", "speaker": "UNKNOWN"},
    ]
    assert audio_processor._fallback_segments(result) == expected

def test_fallback_segments_missing_text(audio_processor):
    """Test fallback handles segments missing the 'text' key."""
    result = {
        "segments": [
            {"start": 0.0, "end": 5.0},
            {"start": 6.0, "end": 7.0, "text": "Valid segment"},
        ]
    }
    expected = [
        {"start": 0.0, "end": 5.0, "text": "", "speaker": "UNKNOWN"},
        {"start": 6.0, "end": 7.0, "text": "Valid segment", "speaker": "UNKNOWN"},
    ]
    assert audio_processor._fallback_segments(result) == expected
