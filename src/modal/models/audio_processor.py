import torch
import torchaudio
import numpy as np
import logging
import os
import tempfile
import subprocess
from typing import Dict, Any, List, Optional, Tuple
import whisperx
import gc
from ..config.settings import (
    DEFAULT_MODEL_NAME, DEFAULT_COMPUTE_TYPE, MAX_AUDIO_DURATION,
    HF_TOKEN, CACHE_DIR
)

logger = logging.getLogger(__name__)

class AudioProcessor:
    """A class for processing audio files with speech recognition and speaker diarization using WhisperX.
    
    This class handles loading and processing audio files to identify who said what in an audio recording,
    leveraging WhisperX for transcription, alignment, and diarization.
    """
    
    def __init__(self):
        """Initialize the AudioProcessor with default settings."""
        self.device = None
        self.cache_dir = CACHE_DIR
        self.model_loaded = False

    async def startup(self) -> None:
        """Sets up the processing environment (e.g., device). Models are loaded on demand."""
        if self.model_loaded:
            logger.info("AudioProcessor already initialized.")
            return

        logger.info("Initializing AudioProcessor...")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.model_loaded = True
        logger.info("AudioProcessor initialized successfully.")

    def cleanup(self) -> None:
        """Clean up temporary resources and free memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Resources potentially released (models are managed in process_audio_file).")

    def _load_audio(self, file_path: str) -> Tuple[np.ndarray, float]:
        """Load and preprocess audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (waveform numpy array, duration in seconds)
        """
        if not os.path.exists(file_path):
            raise ValueError(f"Audio file not found: {file_path}")
            
        try:
            info = torchaudio.info(file_path)
            target_sr = 16000
            
            # Use WhisperX's load_audio for consistency
            audio = whisperx.load_audio(file_path, target_sr)
            
            # Get original waveform for duration calculation
            waveform, sample_rate = torchaudio.load(file_path)
            duration = waveform.shape[1] / sample_rate
            
            if duration > MAX_AUDIO_DURATION:
                raise ValueError(f"Audio duration ({duration:.2f}s) exceeds maximum allowed duration ({MAX_AUDIO_DURATION}s)")
            
            return audio, duration
            
        except Exception as e:
            logger.exception(f"Failed to load audio file: {file_path}")
            raise RuntimeError(f"Failed to load audio file: {e}")

    def _convert_to_mono_16khz_ffmpeg(self, file_path: str, target_sr: int) -> str:
        """Convert audio to mono 16kHz WAV using ffmpeg."""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                output_path = temp_file.name
            
            command = [
                'ffmpeg',
                '-i', file_path,
                '-ac', '1',          # Set to 1 audio channel (mono)
                '-ar', str(target_sr), # sample rate
                '-acodec', 'pcm_s16le', # WAV format
                '-y',                 # Overwrite output file if it exists
                output_path
            ]
            
            logger.info(f"Running ffmpeg command: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            logger.debug(f"FFmpeg stdout: {result.stdout}")
            logger.debug(f"FFmpeg stderr: {result.stderr}")
            return output_path
            
        except subprocess.CalledProcessError as e:
             logger.error(f"FFmpeg conversion failed for {file_path}: {e}")
             logger.error(f"FFmpeg stderr: {e.stderr}")
             raise RuntimeError(f"Failed to convert audio to mono 16kHz WAV: {e.stderr}")
        except Exception as e:
            logger.error(f"Error during ffmpeg conversion setup for {file_path}: {e}")
            raise RuntimeError(f"Failed to convert audio: {e}")

    def _group_word_segments_by_speaker(self, word_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Groups whisperx word segments into speaker segments."""
        if not word_segments:
            return []

        segments = []
        current_segment = None

        # Process each word segment
        for word_data in word_segments:
            logger.debug(f"Processing word segment: {word_data}")
            
            if "word" in word_data:
                speaker = word_data.get("speaker", "UNKNOWN")
                start_time = word_data.get("start")
                end_time = word_data.get("end")
                text = word_data.get("word", "").strip()
            else:
                speaker = word_data.get("speaker", "UNKNOWN")
                start_time = word_data.get("start")
                end_time = word_data.get("end")
                text = word_data.get("text", "").strip()

            if start_time is None or end_time is None or not text:
                continue

            if speaker is None:
                speaker = "UNKNOWN"

            if current_segment and current_segment["speaker"] == speaker:
                current_segment["end"] = end_time
                current_segment["text"] += " " + text
            else:
                if current_segment:
                    current_segment["text"] = current_segment["text"].strip()
                    segments.append(current_segment)
                
                current_segment = {
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                    "speaker": speaker
                }

        if current_segment:
            current_segment["text"] = current_segment["text"].strip()
            segments.append(current_segment)
            
        return segments

    def _fallback_segments(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create fallback segments when diarization fails."""
        segments = []
        
        for seg in result.get("segments", []):
            if seg.get("start") is not None and seg.get("end") is not None:
                segments.append({
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": seg.get("text", "").strip(),
                    "speaker": "UNKNOWN"
                })
        
        return segments

    async def process_audio_file(self, file_path: str, language: Optional[str] = 'en', 
                               model_name: str = DEFAULT_MODEL_NAME, 
                               compute_type: str = DEFAULT_COMPUTE_TYPE, 
                               batch_size: int = 16) -> Dict[str, Any]:
        """Process audio file using WhisperX for transcription, alignment, and diarization.
        
        Args:
            file_path: Path to the audio file
            language: Optional language code (e.g., "en", "es"). If None, WhisperX detects.
            model_name: Name of the Whisper model (e.g., "large-v2", "base")
            compute_type: Compute type for Whisper model ("float16", "int8", "float32")
            batch_size: Batch size for transcription
            
        Returns:
            Dictionary containing transcription results with speaker segments
        """
        whisper_model = None
        align_model = None
        diarize_model = None
        
        try:
            logger.info(f"Processing audio file with WhisperX: {file_path}")
            logger.info(f"Model: {model_name}, Compute: {compute_type}, Batch: {batch_size}, Language: {language or 'auto'}")

            if not self.model_loaded:
                await self.startup()
            
            audio_waveform, duration = self._load_audio(file_path)
            logger.info(f"Audio loaded successfully. Duration: {duration:.2f}s")

            logger.info("Loading Whisper model...")

            device_str = "cuda" if self.device.type == "cuda" else "cpu"
            whisper_model = whisperx.load_model(
                model_name, 
                device_str,
                compute_type=compute_type, 
                language=language,
                asr_options={"word_timestamps": True},
                download_root=self.cache_dir
             )
            logger.info("Transcribing audio...")
            result = whisper_model.transcribe(audio_waveform, batch_size=batch_size)

            logger.debug(f"Transcribing audio result: {result}")
            
            # Save the raw transcript before any processing
            detected_language = result.get("language", language or "en")
            full_transcript = " ".join(segment.get("text", "").strip() for segment in result.get("segments", [])).strip()
            logger.info(f"Transcription complete. Detected language: {detected_language}")
            logger.info(f"Full transcript: {full_transcript}")
            
            # Create a default fallback result in case alignment or diarization fails
            final_output = {
                "language": detected_language,
                "segments": self._fallback_segments(result),
                "transcript": full_transcript,
                "duration": duration,
                "warning": None
            }

            del whisper_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if "segments" not in result or not result["segments"]:
                 logger.warning("Transcription did not produce any segments.")
                 return final_output

            # Proceed with alignment
            aligned_result = None
            try:
                logger.info(f"Loading alignment model for language: {detected_language}...")
                align_model, metadata = whisperx.load_align_model(
                    language_code=detected_language, 
                    device=self.device,
                    model_name=None,
                    model_dir=os.path.join(self.cache_dir, "alignment")
                )
                logger.info("Aligning transcription...")
                segments_to_align = result["segments"]
                aligned_result = whisperx.align(
                    segments_to_align, 
                    align_model, 
                    metadata, 
                    audio_waveform, 
                    self.device, 
                    return_char_alignments=False
                )
                logger.debug(f"Alignment result: {aligned_result}")
                logger.info("Alignment complete.")
                
                # Update segments with aligned ones
                if aligned_result and "segments" in aligned_result and aligned_result["segments"]:
                    final_output["segments"] = [{
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "text": seg.get("text", "").strip(),
                        "speaker": "UNKNOWN"
                    } for seg in aligned_result.get("segments", []) if seg.get("start") is not None and seg.get("end") is not None]
                
                del align_model
                del metadata
                gc.collect()
                if torch.cuda.is_available():
                     torch.cuda.empty_cache()
            except Exception as align_error:
                 logger.warning(f"Alignment failed: {align_error}. Proceeding without alignment.")
                 final_output["warning"] = f"Alignment failed: {align_error}"

            # Proceed with diarization if alignment was successful
            try:
                if aligned_result and "segments" in aligned_result and aligned_result["segments"]:
                    logger.info("Loading diarization model...")
                    if not HF_TOKEN:
                        logger.warning("Hugging Face token (HF_TOKEN) not set. Diarization might fail if model requires authentication.")
                    
                    diarize_model = whisperx.DiarizationPipeline(
                        use_auth_token=HF_TOKEN, 
                        device=self.device,
                    )
                    logger.info("Performing speaker diarization...")
                    diarize_segments = diarize_model(audio_waveform) 
                    logger.debug(f"Diarization segments: {diarize_segments}")
                    logger.info("Diarization complete.")
                    
                    # Try to assign speakers to words
                    try:
                        logger.info("Assigning speaker labels to words...")
                        result_with_speakers = whisperx.assign_word_speakers(diarize_segments, aligned_result)
                        logger.debug(f"Result with speakers: {result_with_speakers}")
                        logger.info("Speaker assignment complete.")
                        
                        # Group by speaker
                        if result_with_speakers and "segments" in result_with_speakers:
                            final_segments = self._group_word_segments_by_speaker(result_with_speakers.get("segments", []))
                            if final_segments:
                                final_output["segments"] = final_segments
                    except Exception as assign_error:
                        logger.warning(f"Speaker assignment failed: {assign_error}. Using segments without speaker information.")
                        final_output["warning"] = f"Speaker assignment failed: {assign_error}"
                
                del diarize_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as diarize_error:
                logger.warning(f"Diarization failed: {diarize_error}. Using segments without speaker information.")
                final_output["warning"] = f"Diarization failed: {diarize_error}"
            
            # Ensure we have at least something in the segments
            if not final_output["segments"]:
                logger.warning("No segments were produced after processing. Using fallback segments.")
                final_output["segments"] = self._fallback_segments(result)
                final_output["warning"] = "No segments were produced after processing."

            # Clean up resources
            del audio_waveform
            if 'result_with_speakers' in locals():
                 del result_with_speakers
            if 'diarize_segments' in locals():
                 del diarize_segments
            if aligned_result:
                del aligned_result
            del result

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return final_output

        except Exception as e:
            logger.exception(f"Error processing audio file with WhisperX: {e}")
            if 'whisper_model' in locals() and whisper_model:
                del whisper_model
            if 'align_model' in locals() and align_model:
                del align_model
            if 'diarize_model' in locals() and diarize_model:
                del diarize_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(f"Failed to process audio file with WhisperX: {e}")