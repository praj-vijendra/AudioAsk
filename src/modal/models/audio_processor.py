import torch
import torchaudio
import numpy as np
import logging
import os
import tempfile
import subprocess
from typing import Dict, Any, List, Optional, Tuple
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pyannote.audio import Pipeline
from collections import Counter
import gc
from ..config.settings import (
    DEFAULT_MODEL_NAME, DEFAULT_COMPUTE_TYPE, MAX_AUDIO_DURATION,
    HF_TOKEN, CACHE_DIR
)

logger = logging.getLogger(__name__)

class AudioProcessor:
    """A class for processing audio files with speech recognition and speaker diarization.
    
    This class handles loading and processing audio files to identify who said what in an audio recording,
    using speech recognition and speaker diarization models.
    """
    
    def __init__(self):
        """Initialize the AudioProcessor with default settings."""
        self.device = None
        self.model = None
        self.processor = None
        self.diarization_pipeline = None
        self.asr_pipeline = None
        self.cache_dir = CACHE_DIR
        self.model_loaded = False

    async def startup(self) -> None:
        """Loads required models for speech recognition and diarization."""
        if self.model_loaded:
            logger.info("Models already loaded")
            return

        logger.info("Loading models...")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Load speech recognition model
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                DEFAULT_MODEL_NAME,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                cache_dir=self.cache_dir,
                token=HF_TOKEN,
            ).to(self.device)
            
            # Configure for word-level timestamps
            self.model.generation_config.alignment_heads = [[0, 0], [0, 2], [0, 4], [0, 8], [1, 0], [1, 2], [1, 4]]
            self.model.generation_config.return_timestamps = True
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                DEFAULT_MODEL_NAME,
                cache_dir=self.cache_dir,
                token=HF_TOKEN,
            )
            
            # Initialize ASR pipeline
            self.asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=25,
                batch_size=16,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device=self.device,
                return_timestamps=True,
                generate_kwargs={"return_timestamps": True}
            )
            
            # Load diarization pipeline
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=HF_TOKEN,
                cache_dir=self.cache_dir,
            ).to(self.device)
            
            self.model_loaded = True
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.exception(f"Failed to load models: {e}")
            self.model = None
            self.processor = None
            self.diarization_pipeline = None
            self.asr_pipeline = None
            self.model_loaded = False
            raise RuntimeError(f"Failed to initialize models: {e}")

    def cleanup(self) -> None:
        """Clean up temporary resources and free memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Temporary resources cleaned up")

    def _load_audio(self, file_path: str) -> Tuple[torch.Tensor, float]:
        """Load and preprocess audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (waveform tensor, duration in seconds)
        """
        if not os.path.exists(file_path):
            raise ValueError(f"Audio file not found: {file_path}")
            
        try:
            # Check if we need to convert the audio to mono
            info = torchaudio.info(file_path)
            if info.num_channels > 1:
                logger.info(f"Converting {info.num_channels}-channel audio to mono")
                temp_file = self._convert_to_mono_ffmpeg(file_path)
                waveform, sample_rate = torchaudio.load(temp_file)
                os.remove(temp_file)
            else:
                waveform, sample_rate = torchaudio.load(file_path)
            
            # Ensure mono audio
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            duration = len(waveform.squeeze()) / 16000
            if duration > MAX_AUDIO_DURATION:
                raise ValueError(f"Audio duration ({duration:.2f}s) exceeds maximum allowed duration ({MAX_AUDIO_DURATION}s)")
                
            return waveform, duration
            
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {e}")

    def _convert_to_mono_ffmpeg(self, file_path: str) -> str:
        """Convert multi-channel audio to mono using ffmpeg."""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                mono_file_path = temp_file.name
            
            command = [
                'ffmpeg',
                '-i', file_path,
                '-ac', '1',  # Set to 1 audio channel (mono)
                '-ar', '16000',  # Set sample rate to 16kHz
                '-y',  # Overwrite output file if it exists
                mono_file_path
            ]
            
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return mono_file_path
            
        except Exception as e:
            logger.warning(f"Failed to convert audio to mono using ffmpeg: {e}")
            raise RuntimeError(f"Failed to convert audio to mono: {e}")

    def _align_transcription_with_diarization(self, transcription: Any, diarization: Any, duration: float) -> List[Dict[str, Any]]:
        """Align transcription with diarization segments."""
        # Check if we have chunks with timestamps
        is_pipeline_output = isinstance(transcription, dict) and "chunks" in transcription
        
        # Get diarization segments
        diarization_segments = list(diarization.itertracks())
        if not diarization_segments:
            logger.warning("No diarization segments found, returning single segment")
            text = transcription if isinstance(transcription, str) else transcription.get("text", "")
            return [{
                "start": 0.0,
                "end": duration,
                "text": text,
                "speaker": "SPEAKER_0"
            }]
        
        # Process based on available information
        if is_pipeline_output:
            return self._align_with_timestamps(transcription["chunks"], diarization)
        else:
            text = transcription if isinstance(transcription, str) else transcription.get("text", "")
            return self._align_without_timestamps(text, diarization, duration)

    def _align_with_timestamps(self, chunks, diarization):
        """Align when we have word-level timestamps from ASR."""
        segments = []
        
        # Create a mapping of speaker segments by time
        speaker_map = {}
        for segment, track, speaker in diarization.itertracks(yield_label=True):
            # Create time slices for mapping
            for t in np.arange(segment.start, segment.end, 0.1):
                speaker_map[round(t, 1)] = speaker
        
        current_segment = None
        
        for chunk in chunks:
            chunk_start = chunk["timestamp"][0]
            chunk_end = chunk["timestamp"][1]
            chunk_text = chunk["text"]
            
            # Find the most common speaker during this chunk
            speakers_in_chunk = []
            for t in np.arange(chunk_start, chunk_end, 0.1):
                t = round(t, 1)
                if t in speaker_map:
                    speakers_in_chunk.append(speaker_map[t])
            
            # Get most frequent speaker or use unknown
            chunk_speaker = "UNKNOWN"
            if speakers_in_chunk:
                chunk_speaker = Counter(speakers_in_chunk).most_common(1)[0][0]
            
            # Group consecutive chunks from the same speaker
            if current_segment and current_segment["speaker"] == chunk_speaker:
                current_segment["end"] = chunk_end
                current_segment["text"] += " " + chunk_text.strip()
            else:
                if current_segment:
                    segments.append(current_segment)
                
                current_segment = {
                    "start": chunk_start,
                    "end": chunk_end,
                    "text": chunk_text.strip(),
                    "speaker": chunk_speaker
                }
        
        # Add the last segment
        if current_segment:
            segments.append(current_segment)
        
        # Clean up the segments
        for segment in segments:
            segment["text"] = segment["text"].strip()
        
        return segments

    def _align_without_timestamps(self, transcription, diarization, duration):
        """Create segments when we only have the full transcription without timestamps."""
        # Get all diarization segments with speaker labels
        segments = []
        diarization_segments = []
        
        for segment, track, speaker in diarization.itertracks(yield_label=True):
            diarization_segments.append({
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker
            })
        
        # Merge adjacent segments from the same speaker
        merged_segments = []
        current_segment = None
        
        for segment in sorted(diarization_segments, key=lambda x: x["start"]):
            if current_segment is None:
                current_segment = segment.copy()
            elif current_segment["speaker"] == segment["speaker"] and segment["start"] - current_segment["end"] < 0.5:
                # Merge if same speaker and gap is less than 0.5 seconds
                current_segment["end"] = segment["end"]
            else:
                merged_segments.append(current_segment)
                current_segment = segment.copy()
        
        if current_segment:
            merged_segments.append(current_segment)
        
        # Fall back to single segment if no diarization segments
        if not merged_segments:
            return [{
                "start": 0.0,
                "end": duration,
                "text": transcription,
                "speaker": "SPEAKER_0"
            }]
        
        # Distribute text across segments
        words = transcription.split()
        total_speech_duration = sum([s["end"] - s["start"] for s in merged_segments])
        words_per_second = len(words) / max(total_speech_duration, 1.0)
        
        word_index = 0
        for segment in merged_segments:
            segment_duration = segment["end"] - segment["start"]
            word_count = int(round(segment_duration * words_per_second))
            word_count = min(word_count, len(words) - word_index)
            
            if word_count > 0:
                segment_words = words[word_index:word_index + word_count]
                segment["text"] = " ".join(segment_words)
                word_index += word_count
            else:
                segment["text"] = ""
            
            segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "speaker": segment["speaker"]
            })
        
        # Handle any remaining words
        if word_index < len(words):
            remaining_text = " ".join(words[word_index:])
            if segments:
                segments[-1]["text"] += " " + remaining_text
            else:
                segments.append({
                    "start": 0.0,
                    "end": duration,
                    "text": remaining_text,
                    "speaker": "SPEAKER_0"
                })
        
        # Remove empty segments
        segments = [s for s in segments if s["text"].strip()]
        
        return segments

    def _create_fallback_segments(self, result, duration):
        """Create fallback segments when diarization fails."""
        # Check if we have chunks with timestamps
        if isinstance(result, dict) and "chunks" in result and len(result["chunks"]) > 1:
            # Create segments based on pauses in speech
            segments = []
            current_segment = None
            pause_threshold = 1.0  # seconds
            
            for chunk in result["chunks"]:
                if not isinstance(chunk.get("timestamp"), (list, tuple)) or len(chunk["timestamp"]) != 2:
                    continue
                    
                chunk_start = chunk["timestamp"][0]
                chunk_end = chunk["timestamp"][1]
                chunk_text = chunk["text"]
                
                # Skip invalid timestamps
                if not isinstance(chunk_start, (int, float)) or not isinstance(chunk_end, (int, float)):
                    continue
                    
                # Alternate speakers based on pauses
                if current_segment is None:
                    current_segment = {
                        "start": chunk_start,
                        "end": chunk_end,
                        "text": chunk_text.strip(),
                        "speaker": "SPEAKER_0"
                    }
                elif chunk_start - current_segment["end"] > pause_threshold:
                    # Significant pause - switch speaker
                    segments.append(current_segment)
                    next_speaker = "SPEAKER_1" if current_segment["speaker"] == "SPEAKER_0" else "SPEAKER_0"
                    current_segment = {
                        "start": chunk_start,
                        "end": chunk_end,
                        "text": chunk_text.strip(),
                        "speaker": next_speaker
                    }
                else:
                    # Continue current segment
                    current_segment["end"] = chunk_end
                    current_segment["text"] += " " + chunk_text.strip()
            
            # Add the last segment
            if current_segment:
                segments.append(current_segment)
            
            return segments
        else:
            # Extract text from result
            if isinstance(result, dict) and "text" in result:
                text = result["text"]
            elif isinstance(result, str):
                text = result
            else:
                text = str(result)
            
            # Split text by sentence-ending punctuation
            sentences = []
            current = ""
            for char in text:
                current += char
                if char in ['.', '!', '?']:
                    sentences.append(current.strip())
                    current = ""
            
            if current.strip():
                sentences.append(current.strip())
                
            # Create segments with alternating speakers
            if len(sentences) > 1:
                segment_duration = duration / len(sentences)
                segments = []
                
                for i, sentence in enumerate(sentences):
                    if not sentence.strip():
                        continue
                        
                    segments.append({
                        "start": i * segment_duration,
                        "end": (i + 1) * segment_duration,
                        "text": sentence,
                        "speaker": "SPEAKER_0" if i % 2 == 0 else "SPEAKER_1"
                    })
                
                return segments
            else:
                # Single segment
                return [{
                    "start": 0.0,
                    "end": duration,
                    "text": text,
                    "speaker": "SPEAKER_0"
                }]

    async def process_audio_file(self, file_path: str, language: Optional[str] = None, 
                               model_name: str = DEFAULT_MODEL_NAME, 
                               compute_type: str = DEFAULT_COMPUTE_TYPE, 
                               batch_size: int = 16) -> Dict[str, Any]:
        """Process audio file to identify speakers and transcribe speech.
        
        Args:
            file_path: Path to the audio file
            language: Optional language code
            model_name: Name of the model to use (default: from settings)
            compute_type: Type of compute device (default: from settings)
            batch_size: Batch size for processing (default: 16)
            
        Returns:
            Dictionary containing transcription results with speaker segments
        """
        try:
            logger.info(f"Processing audio file: {file_path}")
            
            # Ensure models are loaded
            if not self.model_loaded:
                await self.startup()
            
            # Load and preprocess audio
            audio_waveform, duration = self._load_audio(file_path)
            audio_numpy = audio_waveform.squeeze().numpy()
            # Handle case where the array might still be 2D after squeeze
            if len(audio_numpy.shape) > 1:
                audio_numpy = audio_numpy[0]  # Take first channel
            
            # Perform speech recognition with timestamps
            try:
                result = self.asr_pipeline(
                    audio_numpy,
                    # return_timestamps="word",
                    chunk_length_s=15,
                    batch_size=batch_size
                )
            except Exception as e:
                logger.warning(f"Word-level timestamps failed: {e}. Falling back to regular transcription.")
                result = self.asr_pipeline(
                    audio_numpy,
                    return_timestamps=False,
                    chunk_length_s=15,
                    batch_size=batch_size
                )
            
            # Perform speaker diarization
            try:
                # Ensure audio is properly formatted for diarization
                if isinstance(audio_waveform, np.ndarray):
                    audio_tensor = torch.from_numpy(audio_waveform)
                else:
                    audio_tensor = audio_waveform.clone()
                
                # Ensure it's a 2D tensor with shape (channel, time)
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                elif audio_tensor.dim() > 2:
                    audio_tensor = audio_tensor[0].unsqueeze(0)
                
                diarization = self.diarization_pipeline(
                    {"waveform": audio_tensor, "sample_rate": 16000},
                    num_speakers=None
                )
                
            #     # Align transcription with diarization
            #     segments = self._align_transcription_with_diarization(result, diarization, duration)
            except Exception as e:
                logger.warning(f"Diarization failed: {e}. Using fallback segmentation.")
            #     segments = self._create_fallback_segments(result, duration)
            
            # Clean up resources
            del audio_waveform
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                "language": language if language else "en",
                "segments": result["text"],
                # "diarization": diarization,
                "duration": duration
            }

        except Exception as e:
            logger.exception(f"Error processing audio file: {e}")
            raise RuntimeError(f"Failed to process audio file: {e}")