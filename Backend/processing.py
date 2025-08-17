"""
{
  "module": "processing",
  "description": "Core media processing utilities for SumIT. Responsible for extracting audio from "
                 "video files (using ffmpeg) and converting audio to text (using OpenAI's Whisper). "
                 "This module intentionally keeps functions synchronous because they perform blocking "
                 "I/O/CPU work; call them from async FastAPI endpoints using run_in_executor to avoid "
                 "blocking the event loop.",
  "notes": [
    "Requires ffmpeg installed and available on PATH.",
    "Requires `openai-whisper` (pip package name: openai-whisper) to be available for transcription.",
    "In production you should reuse loaded Whisper models between requests for performance.",
    "Functions return paths to temporary files or transcription strings. Caller is responsible for cleanup "
    "of returned temporary files when appropriate (this module will attempt to clean up intermediate "
    "temporary files it creates)."
  ]
}
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("sumit.processing")
logger.setLevel(logging.INFO)


def _run_ffmpeg(cmd: list) -> None:
    """
    Internal helper to run an ffmpeg command and raise a helpful error on failure.

    :param cmd: List[str] - the ffmpeg command and arguments
    :raises RuntimeError: if ffmpeg returns a non-zero exit code
    """
    try:
        # We capture stderr to surface a helpful message on failure, but we hide stdout
        completed = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace") if completed.stderr else ""
            raise RuntimeError(f"ffmpeg failed (exit {completed.returncode}): {stderr.strip()[:400]}")
    except FileNotFoundError as e:
        # ffmpeg binary not found
        logger.exception("ffmpeg not found on PATH.")
        raise RuntimeError("ffmpeg is not installed or not found on PATH. Install ffmpeg and try again.") from e
    except Exception:
        logger.exception("Unexpected error while running ffmpeg.")
        raise


def extract_audio_from_video(video_path: str, sample_rate: int = 16000) -> str:
    """
    Extract audio from a video file and write a normalized mono WAV file.

    This function uses ffmpeg (CLI). It will create a temporary .wav file and return its path.
    The caller is responsible for deleting the returned file when finished.

    JSONDOC:
    {
      "name": "extract_audio_from_video",
      "description": "Extract audio track from a video file and normalize it to mono WAV.",
      "parameters": {
        "video_path": "str - path to input video",
        "sample_rate": "int - desired sample rate in Hz (default 16000)"
      },
      "returns": "str - path to temporary WAV file (caller must delete)",
      "raises": ["FileNotFoundError", "RuntimeError"]
    }

    :param video_path: Path to the input video file.
    :param sample_rate: Desired audio sample rate for the output WAV.
    :return: Path to the created temporary WAV file.
    :raises FileNotFoundError: If video_path does not exist.
    :raises RuntimeError: If ffmpeg fails to extract or convert audio.
    """
    logger.info("extract_audio_from_video: video_path=%s sample_rate=%d", video_path, sample_rate)

    if not os.path.exists(video_path):
        logger.error("Video file not found: %s", video_path)
        raise FileNotFoundError(f"Video file not found: {video_path}")

    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_wav_path = tmp_wav.name
    tmp_wav.close()

    cmd = [
        "ffmpeg",
        "-y",                 # overwrite
        "-i", str(video_path),# input
        "-ac", "1",           # mono
        "-ar", str(sample_rate),  # sample rate
        "-vn",                # no video
        "-hide_banner",
        "-loglevel", "error",
        str(tmp_wav_path)
    ]

    try:
        _run_ffmpeg(cmd)
        logger.info("Audio extracted to %s", tmp_wav_path)
        return tmp_wav_path
    except Exception as e:
        # Clean up partial file if it exists
        try:
            if os.path.exists(tmp_wav_path):
                os.remove(tmp_wav_path)
        except Exception:
            logger.debug("Failed to remove temporary audio after ffmpeg failure: %s", tmp_wav_path)
        logger.exception("Failed to extract audio from video: %s", e)
        raise


def _ensure_wav(input_path: str, sample_rate: int = 16000) -> Tuple[str, bool]:
    """
    Ensure the provided audio (or media) file is a mono WAV with the requested sample rate.

    If the input is already a WAV file, this function will still convert it to ensure
    sample rate and channel layout are correct (so the returned file is always normalized).
    A temporary WAV file is created and the returned boolean indicates whether the caller
    should delete the returned path (True if temporary).

    JSONDOC:
    {
      "name": "_ensure_wav",
      "description": "Normalize any audio file to mono WAV with desired sample rate using ffmpeg.",
      "parameters": {
        "input_path": "str - path to input audio file",
        "sample_rate": "int - sample rate in Hz"
      },
      "returns": ["str - path to WAV file", "bool - True if path is temporary and should be deleted by caller"]
    }

    :param input_path: Path to the input audio file.
    :param sample_rate: Target sample rate in Hz.
    :return: (wav_path, is_temporary)
    """
    logger.debug("_ensure_wav: input=%s sample_rate=%d", input_path, sample_rate)

    if not os.path.exists(input_path):
        logger.error("Input audio file not found: %s", input_path)
        raise FileNotFoundError(f"Input audio file not found: {input_path}")

    # Always create a new temp WAV to ensure normalization (keeps behavior predictable)
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_wav_path = tmp_wav.name
    tmp_wav.close()

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", str(sample_rate),
        "-hide_banner",
        "-loglevel", "error",
        str(tmp_wav_path)
    ]

    try:
        _run_ffmpeg(cmd)
        logger.debug("Normalized audio saved to %s", tmp_wav_path)
        return tmp_wav_path, True
    except Exception as e:
        # Remove created file on failure
        try:
            if os.path.exists(tmp_wav_path):
                os.remove(tmp_wav_path)
        except Exception:
            logger.debug("Failed to remove tmp wav after conversion failure: %s", tmp_wav_path)
        logger.exception("Failed to normalize/convert audio: %s", e)
        raise


def audio_to_transcript(audio_path: str, model_size: str = "tiny", language: Optional[str] = None) -> str:
    """
    Transcribe an audio file into text using Whisper (openai-whisper).

    This function will:
     - Normalize the input to mono WAV at 16kHz using ffmpeg (_ensure_wav).
     - Load a Whisper model (blocking and potentially expensive).
     - Run transcription and return the resulting text.
     - Clean up temporary WAV created during normalization.

    Notes:
    - Loading the Whisper model is expensive. In production, load the model once and reuse it across requests.
    - This function is synchronous and may be CPU/GPU heavy. Call it via run_in_executor in async contexts.

    JSONDOC:
    {
      "name": "audio_to_transcript",
      "description": "Convert audio to transcript text using Whisper.",
      "parameters": {
        "audio_path": "str - path to audio file",
        "model_size": "str - whisper model size (tiny|base|small|medium|large). Default 'small'.",
        "language": "Optional[str] - language hint e.g. 'en' or 'en-US' (optional)"
      },
      "returns": "str - transcript text (may be empty if no speech detected)",
      "raises": ["FileNotFoundError", "RuntimeError", "ImportError"]
    }

    :param audio_path: Path to the audio file (any common audio format).
    :param model_size: Whisper model size to load (see openai-whisper docs).
    :param language: Optional language hint for transcription.
    :return: Transcript string (may be empty).
    """
    logger.info("audio_to_transcript: audio_path=%s model_size=%s language=%s", audio_path, model_size, language)

    if not os.path.exists(audio_path):
        logger.error("Audio file not found: %s", audio_path)
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    wav_path = None
    delete_wav = False
    try:
        # Normalize to WAV suitable for ASR
        wav_path, delete_wav = _ensure_wav(audio_path, sample_rate=16000)

        # Import whisper lazily so the package is optional at module import time
        try:
            import whisper  # type: ignore
        except Exception as e:
            logger.exception("Whisper library not available.")
            raise ImportError("The `openai-whisper` package is required for transcription. Install it via `pip install openai-whisper`.") from e

        # Load model (blocking). Consider caching this model in production to avoid repeated loads.
        try:
            logger.debug("Loading Whisper model: %s", model_size)
            model = whisper.load_model(model_size)
        except Exception as e:
            logger.exception("Failed to load Whisper model %s", model_size)
            raise RuntimeError(f"Failed to load Whisper model '{model_size}': {e}") from e

        # Transcribe (blocking). If language is provided, pass it as `language` param.
        try:
            logger.debug("Starting transcription of %s", wav_path)
            if language:
                result = model.transcribe(wav_path, language=language)
            else:
                result = model.transcribe(wav_path)
            transcript = result.get("text", "").strip() if isinstance(result, dict) else str(result)
            logger.info("Transcription complete (chars=%d)", len(transcript))
            return transcript
        except Exception as e:
            logger.exception("Whisper transcription failed.")
            raise RuntimeError(f"Transcription failed: {e}") from e

    finally:
        # Clean up temporary WAV created for normalization if we created one
        try:
            if delete_wav and wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
                logger.debug("Deleted temporary normalized WAV: %s", wav_path)
        except Exception:
            logger.warning("Failed to delete temporary WAV file: %s", wav_path)

