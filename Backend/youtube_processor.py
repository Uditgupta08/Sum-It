"""
{
  "module": "youtube_processor",
  "description": "Fetches YouTube video transcripts, handling various URL formats and errors.",
  "dependencies": ["youtube-transcript-api"],
  "notes": [
    "This module provides a synchronous function to get a transcript from a YouTube URL.",
    "Includes a robust function to parse video IDs from various YouTube URL formats.",
  ]
}
"""
import logging
from typing import Dict, List, Optional
from urllib.parse import urlparse, parse_qs

# youtube_transcript_api must be installed and up-to-date
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled, VideoUnavailable

logger = logging.getLogger("sumit.youtube")
logger.setLevel(logging.INFO)


class TranscriptNotFound(Exception):
    """Custom exception raised when a transcript cannot be fetched."""
    pass


def _extract_video_id(url: str) -> Optional[str]:
    """
    Robustly extracts the YouTube video ID from various URL formats.
    
    Handles:
    - youtube.com/watch?v=...
    - youtu.be/...
    - youtube.com/embed/...
    - youtube.com/shorts/...
    """
    if not url:
        return None
        
    # urlparse to handle various URL structures
    parsed_url = urlparse(url)
    
    # Standard /watch URL
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com'] and parsed_url.path == '/watch':
        query_params = parse_qs(parsed_url.query)
        return query_params.get('v', [None])[0]

    # Shortened youtu.be URL
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]

    # Embed or Shorts URL
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        path_parts = parsed_url.path.strip('/').split('/')
        if path_parts[0] in ['embed', 'shorts']:
            return path_parts[1]
            
    return None


def get_youtube_transcript(video_url: str) -> str:
    """
    Fetches the transcript for a given YouTube video URL.

    This function is synchronous and should be called from an async context
    using `run_in_executor` to avoid blocking the event loop.

    :param video_url: The full URL of the YouTube video.
    :return: A string containing the full transcript.
    :raises TranscriptNotFound: If no transcript is available for any reason.
    :raises ValueError: If the URL is invalid or a video ID cannot be extracted.
    """
    video_id = _extract_video_id(video_url)
    
    if not video_id:
        raise ValueError("Could not extract a valid YouTube video ID from the URL.")
        
    logger.info("Extracted YouTube video ID: %s", video_id)

    try:
        # Create YouTubeTranscriptApi instance
        ytt_api = YouTubeTranscriptApi()
        
        # Try multiple language codes for English, prioritizing manual over auto-generated
        english_codes = ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU']
        fetched_transcript = None
        
        # First try to get English transcripts (manual or auto-generated)
        for lang_code in english_codes:
            try:
                fetched_transcript = ytt_api.fetch(video_id, languages=[lang_code])
                logger.info("Successfully fetched transcript in language: %s", lang_code)
                break
            except (NoTranscriptFound, TranscriptsDisabled):
                continue
        
        # If no English transcript found, try to get any available transcript and use translation
        if fetched_transcript is None:
            try:
                # Get list of available transcripts
                transcript_list = ytt_api.list(video_id)
                
                # Try to find any transcript that can be translated to English
                for transcript in transcript_list:
                    if transcript.is_translatable and 'en' in [lang.language_code for lang in transcript.translation_languages]:
                        # Translate to English
                        translated_transcript = transcript.translate('en')
                        fetched_transcript = translated_transcript.fetch()
                        logger.info("Successfully fetched and translated transcript from %s to English", transcript.language_code)
                        break
                    elif transcript.language_code in english_codes:
                        # Direct fetch if it's already English
                        fetched_transcript = transcript.fetch()
                        logger.info("Successfully fetched direct English transcript: %s", transcript.language_code)
                        break
                
                # If still no transcript, just get the first available one
                if fetched_transcript is None and len(transcript_list) > 0:
                    first_transcript = transcript_list[0]
                    fetched_transcript = first_transcript.fetch()
                    logger.info("Using first available transcript in language: %s", first_transcript.language_code)
                    
            except Exception as e:
                logger.warning("Failed to get transcript list or translate: %s", e)
        
        if fetched_transcript is None:
            raise TranscriptNotFound("No transcripts could be retrieved for this video")
        
        # Combine the transcript segments into a single string
        # The new API returns a FetchedTranscript object that is iterable
        transcript = " ".join([snippet.text for snippet in fetched_transcript])
        logger.info("Successfully processed transcript for video ID %s (length: %d)", video_id, len(transcript))
        return transcript

    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable) as e:
        logger.warning("Could not retrieve transcript for video %s: %s", video_id, e)
        raise TranscriptNotFound(f"A transcript is not available for this video. It may be disabled or the video is unavailable.")
    except Exception as e:
        logger.exception("An unexpected error occurred while fetching YouTube transcript for video %s.", video_id)
        # Re-raise as our custom exception to simplify error handling upstream
        raise TranscriptNotFound(f"An unexpected error occurred: {e}")