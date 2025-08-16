

"""
{
  "module": "summarizer",
  "description": "AI abstraction layer for generating summaries from transcripts using Llama 3 via Groq.",
  "notes": [
    "This module provides an asynchronous implementation of generate_summary that calls a real LLM.",
    "It requires the `groq` and `python-dotenv` packages.",
    "It expects a GROQ_API_KEY to be set in a .env file or as an environment variable."
  ]
}
"""

import asyncio
import logging
import os
from typing import Optional

# Use python-dotenv to load environment variables from a .env file
from dotenv import load_dotenv
from groq import AsyncGroq, GroqError

# Load environment variables
load_dotenv()

logger = logging.getLogger("sumit.summarizer")
logger.setLevel(logging.INFO)

# --- Initialize the Groq Client ---
# It's best practice to initialize the client once and reuse it.
# The client will automatically pick up the GROQ_API_KEY from the environment.
try:
    groq_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
    if not os.environ.get("GROQ_API_KEY"):
        logger.warning("GROQ_API_KEY environment variable not found. The summarizer will not work.")
        groq_client = None
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {e}")
    groq_client = None


def _build_summarization_prompt(transcript: str) -> str:
    """Creates a structured prompt for the summarization task."""
    return f"""
    **Act as an expert executive assistant responsible for creating concise and actionable meeting summaries.**

    Analyze the following meeting transcript and produce a summary in a clear, professional format.

    **Transcript:**
    \"\"\"
    {transcript}
    \"\"\"

    **Please structure your summary with the following sections:**

    1.  **Overall Summary:** A brief paragraph (2-3 sentences) capturing the main purpose and key outcomes of the meeting.
    2.  **Key Decisions Made:** A bulleted list of all significant decisions that were finalized.
    3.  **Action Items:** A clear, bulleted list of all tasks assigned. Specify **who** is responsible. Use the format: "- [Task Description] (Owner: [Name])".

    Ensure the final output is clear, concise, and focuses only on the information present in the transcript.
    """


async def generate_summary(transcript: str) -> str:
    """
    Asynchronously generate a short summary for a meeting transcript using Llama 3 via Groq.

    :param transcript: Full transcript text to summarize.
    :return: A summary string, or an error message if generation fails.
    :raises TypeError: If transcript is not a string.
    """
    if not isinstance(transcript, str):
        logger.error("generate_summary called with non-str transcript: %s", type(transcript))
        raise TypeError("transcript must be a string")

    text = transcript.strip()
    if not text:
        logger.info("Empty transcript received by generate_summary.")
        return "No spoken content detected to summarize."

    if not groq_client:
        logger.error("Groq client is not initialized. Cannot generate summary.")
        return "Error: Summarizer service is not configured."

    prompt = _build_summarization_prompt(text)

    try:
        logger.info("Making API call to Groq for summarization...")
        chat_completion = await groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192", # A fast and capable model for this task
            temperature=0.2, # Lower temperature for more factual summaries
            max_tokens=1024,
        )
        
        summary = chat_completion.choices[0].message.content
        logger.info("Successfully received summary from Groq.")
        return summary.strip()

    except GroqError as e:
        logger.exception(f"Groq API error during summarization: {e}")
        return f"Error: Could not generate summary due to an API issue. (Status: {e.status_code})"
    except Exception as e:
        logger.exception(f"An unexpected error occurred during summarization: {e}")
        return "Error: An unexpected issue occurred while generating the summary."

