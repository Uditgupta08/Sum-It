"""
{
  "module": "main",
  "description": "FastAPI entrypoint for the Meeting Summarizer service (SumIT). "
                 "Provides a single async endpoint POST /summarize/ that accepts an uploaded "
                 "file (video/audio/text), saves it to a temporary file, delegates processing "
                 "and summarization to other modules, and returns a JSON with transcript and summary.",
  "notes": [
    "This file expects `processing.py` and `summarizer.py` to expose the following functions:",
    "- processing.extract_audio_from_video(video_path: str) -> str",
    "- processing.audio_to_transcript(audio_path: str) -> str",
    "- summarizer.generate_summary(transcript: str) -> str (async)",
    "Blocking operations (heavy CPU / I/O) are offloaded to a threadpool using asyncio.run_in_executor."
  ]
}
"""

import asyncio
import logging
import os
import tempfile
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel, HttpUrl 

# Import the processing and summarizer modules. These modules are expected to be
# implemented separately. Import errors will surface at import time; that is acceptable
# because main.py relies on them to function.
from processing import audio_to_transcript, extract_audio_from_video  # type: ignore
from summarizer import generate_summary  # type: ignore
from youtube_processor import get_youtube_transcript, TranscriptNotFound


# -----------------------
# JSONDOC: configuration
# -----------------------
JSONDOC: dict = {
    "api": {
        "paths": {
            "/summarize/": {
                "method": "POST",
                "request": {
                    "multipart": {
                        "file": "UploadFile (video/* | audio/* | text/plain)"
                    }
                },
                "response": {
                    "200": {"application/json": {"transcript": "string", "summary": "string"}},
                    "400": "Bad Request - unsupported media type or missing content-type",
                    "500": "Internal Server Error - processing/transcription/summarization failure"
                }
            }
        }
    }
}


# -----------------------
# App & logger setup
# -----------------------
app = FastAPI(title="SumIT — Meeting Summarizer")
logger = logging.getLogger("sumit.main")
logging.basicConfig(level=logging.INFO)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST)
    allow_headers=["*"], # Allows all headers
)

# -----------------------
# Response models
# -----------------------
class SummarizeResponse(BaseModel):
    transcript: str
    summary: str

class YouTubeLinkPayload(BaseModel):
    youtube_url: HttpUrl

# -----------------------
# Utility helpers
# -----------------------
async def _save_upload_to_temp(upload_file: UploadFile) -> str:
    """
    {
      "name": "_save_upload_to_temp",
      "description": "Save an UploadFile to a temporary file and return its path.",
      "parameters": {
        "upload_file": "fastapi.UploadFile - the uploaded file object"
      },
      "returns": "str - path to the saved temporary file (caller must delete)."
    }
    """
    # Preserve extension if available
    suffix = ""
    if upload_file.filename and "." in upload_file.filename:
        suffix = os.path.splitext(upload_file.filename)[1]

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    # Need to close so we can write bytes to it safely; we'll write using Python file IO
    tmp.close()

    # Read contents asynchronously from UploadFile and write to temp file synchronously
    content = await upload_file.read()
    with open(tmp_path, "wb") as f:
        f.write(content)

    logger.debug("Saved uploaded file to temporary path: %s", tmp_path)
    return tmp_path


# -----------------------
# Main endpoint
# -----------------------
@app.post("/summarize/", response_model=SummarizeResponse)
async def summarize(file: UploadFile = File(...)) -> JSONResponse:
    """
    {
      "name": "summarize",
      "description": "Asynchronous endpoint that accepts a single uploaded file (video/audio/text), "
                     "processes it into a transcript, generates a summary, and returns both.",
      "request": {
        "content-type": "multipart/form-data",
        "field": "file"
      },
      "behavior": [
        "Saves uploaded file to a temp file",
        "For text files: read text -> transcript",
        "For audio files: transcribe audio -> transcript",
        "For video files: extract audio -> transcribe -> transcript",
        "Call async generate_summary(transcript) to produce summary",
        "Clean up all temporary files even on error"
      ],
      "responses": {
        "200": {"transcript": "string", "summary": "string"},
        "400": "Unsupported media type or missing content type",
        "500": "Processing/transcription/summarization failure"
      }
    }
    """
    logger.info("Received upload: filename=%s content_type=%s", file.filename, file.content_type)

    # Basic validation of content type
    if not file.content_type:
        logger.warning("Uploaded file missing content type.")
        raise HTTPException(status_code=400, detail="Missing content type for uploaded file.")

    ctype = file.content_type.lower()
    if not (ctype.startswith("video/") or ctype.startswith("audio/") or ctype.startswith("text/")):
        logger.warning("Unsupported content type: %s", ctype)
        raise HTTPException(status_code=400, detail=f"Unsupported media type: {ctype}")

    temp_paths: List[str] = []
    transcript: Optional[str] = None

    try:
        # Save uploaded file
        uploaded_path = await _save_upload_to_temp(file)
        temp_paths.append(uploaded_path)

        loop = asyncio.get_event_loop()

        # Handle text files directly
        if ctype.startswith("text/"):
            logger.debug("Processing uploaded file as text.")
            def _read_text(path: str) -> str:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()

            try:
                transcript = await loop.run_in_executor(None, _read_text, uploaded_path)
            except Exception as e:
                logger.exception("Failed to read text file: %s", e)
                raise HTTPException(status_code=500, detail=f"Failed to read text file: {e}")

        else:
            # For video: extract audio first, then transcribe
            if ctype.startswith("video/"):
                logger.debug("Processing uploaded file as video. Extracting audio...")
                try:
                    # extract_audio_from_video is expected to be a synchronous function (blocking I/O)
                    audio_path = await loop.run_in_executor(None, extract_audio_from_video, uploaded_path)
                    temp_paths.append(audio_path)
                except HTTPException:
                    # Re-raise FastAPI HTTPExceptions unchanged
                    raise
                except Exception as e:
                    logger.exception("Video audio extraction failed: %s", e)
                    raise HTTPException(status_code=500, detail=f"Video processing failed: {e}")

                # Transcribe the extracted audio (blocking) in executor
                try:
                    transcript = await loop.run_in_executor(None, audio_to_transcript, audio_path)
                except HTTPException:
                    raise
                except Exception as e:
                    logger.exception("Audio transcription failed: %s", e)
                    raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

            # For audio files: transcribe directly
            elif ctype.startswith("audio/"):
                logger.debug("Processing uploaded file as audio. Transcribing...")
                try:
                    transcript = await loop.run_in_executor(None, audio_to_transcript, uploaded_path)
                except HTTPException:
                    raise
                except Exception as e:
                    logger.exception("Audio transcription failed: %s", e)
                    raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

        # At this point we should have a transcript (even empty string is accepted)
        if transcript is None:
            logger.error("Transcript generation returned None unexpectedly.")
            raise HTTPException(status_code=500, detail="Internal processing error: transcript is None")

        # Generate summary via the async summarizer (assumed to be non-blocking)
        try:
            summary = await generate_summary(transcript)
        except Exception as e:
            logger.exception("Summary generation failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Summary generation failed: {e}")

        logger.info("Successfully processed upload: transcript_length=%d summary_length=%d",
                    len(transcript), len(summary) if summary else 0)
        return JSONResponse(status_code=200, content={"transcript": transcript, "summary": summary})

    finally:
        # Ensure temporary files are always removed
        for path in temp_paths:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
                    logger.debug("Deleted temporary file: %s", path)
            except Exception as e:
                # Log but do not raise - don't mask original exceptions
                logger.warning("Failed to delete temp file %s: %s", path, e)


@app.post("/summarize-youtube/", response_model=SummarizeResponse)
async def summarize_youtube(payload: YouTubeLinkPayload):
    """Summarizes a YouTube video from its URL."""
    logger.info("Received YouTube URL: %s", payload.youtube_url)
    loop = asyncio.get_event_loop()
    
    try:
        transcript = await loop.run_in_executor(
            None, get_youtube_transcript, str(payload.youtube_url)
        )
        summary = await generate_summary(transcript)
        logger.info("Successfully processed YouTube link: transcript_len=%d, summary_len=%d", len(transcript), len(summary))
        return JSONResponse(status_code=200, content={"transcript": transcript, "summary": summary})
    
    except (TranscriptNotFound, ValueError) as e:
        logger.warning("Failed to process YouTube URL: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.exception("An unexpected error occurred during YouTube summarization.")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")