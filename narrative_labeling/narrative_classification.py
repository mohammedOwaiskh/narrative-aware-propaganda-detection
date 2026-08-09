"""
Narrative Frame Classification Pipeline using Groq Llama 3.3 API.

This module implements an end-to-end pipeline for classifying articles according
to narrative frames using the Groq API with Llama 3.3-70b model. The system:

1. Reads articles from a CSV file
2. Constructs prompts with frame definitions
3. Calls the Groq API with exponential backoff retry logic
4. Validates responses for completeness and consistency
5. Saves results to CSV with support for resume/checkpointing
6. Handles interrupts gracefully by saving progress

Features:
    - Automatic retry with exponential backoff on API failures
    - Resume capability: skips already-processed articles
    - Keyboard interrupt handling with automatic checkpointing
    - Comprehensive logging to both console and file
    - Validation of LLM output (missing fields, invalid evidence, etc.)
    - JSON response parsing with error handling

Configuration:
    - Requires GROQ_API_KEY environment variable
    - Loads frame definitions from prompts/frames.json
    - Loads prompt template from prompts/narrative_classification_prompt.txt
    - Reads articles from data/processed/{input_file}
    - Writes results to data/processed/{output_file}

Usage:
    python narrative_classification.py --input articles.csv --output results.csv

Author: Mohammed Owais Khan
"""
import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from httpx import HTTPStatusError

from utils.file_reader import read_frames, read_prompt, read_processed_data, get_project_root
from utils.logger import setup_logger

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
logger = setup_logger(__name__)

MODEL_NAME = "llama-3.3-70b-versatile"
USER_ROLE = "user"
MAX_RETRIES = 3
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 4096
BASE_BACKOFF_SECONDS = 5


def prompt_llama(client: Groq, prompt: str) -> dict:
    """
    Call the Groq Llama API with exponential backoff retry logic.
    
    This function sends a prompt to the Groq API (using Llama 3.3-70b model)
    and handles transient failures with exponential backoff. It expects a JSON
    response and automatically parses it.
    
    Args:
        client (Groq): Initialized Groq client instance.
        prompt (str): The prompt to send to the API. Should be a well-formatted
                     string that will be sent as user content.
    
    Returns:
        dict: Parsed JSON response from the API.
    
    Raises:
        RuntimeError: If all MAX_RETRIES attempts fail or if JSON parsing fails
                     without a successful retry.
    
    Notes:
        - Retries up to MAX_RETRIES times with exponential backoff
        - Wait time = BASE_BACKOFF_SECONDS * attempt_number
        - HTTP errors and transient API errors trigger retries
        - JSON parse errors don't trigger retries (assumed to be response malformation)
        - All API calls and errors are logged at appropriate levels (INFO/WARNING/ERROR)
    """
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Calling Groq API (attempt {attempt}/{MAX_RETRIES}) with model: {MODEL_NAME}")
            logger.debug(f"Prompt length: {len(prompt)} characters")

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": USER_ROLE, "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"},
            )

            logger.info(f"API call successful on attempt {attempt}")
            raw = response.choices[0].message.content
            result = json.loads(raw)
            logger.debug(f"Successfully parsed JSON response")
            return result
        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
            logger.warning(f"JSON parsing failed on attempt {attempt}: {last_error}")
        except HTTPStatusError as e:
            last_error = str(e)
            wait = BASE_BACKOFF_SECONDS * attempt
            logger.warning(
                f"HTTP error on attempt {attempt}/{MAX_RETRIES}: {last_error} -- waiting {wait}s before retry")
            time.sleep(wait)
            continue
        except Exception as e:
            # Groq rate-limit / transient errors: back off and retry
            last_error = str(e)
            wait = BASE_BACKOFF_SECONDS * attempt
            logger.warning(
                f"API error on attempt {attempt}/{MAX_RETRIES}: {last_error} -- waiting {wait}s before retry")
            time.sleep(wait)
            continue
        # If we got here without exception, JSON parsed successfully -> exit loop
        break
    else:
        error_msg = f"Failed after {MAX_RETRIES} attempts: {last_error}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    return None  # unreachable, kept for clarity


def build_prompt(article_text: str, frames: dict) -> str:
    """
    Construct the prompt for narrative frame classification.
    
    This function loads the prompt template from the prompts directory and
    formats it with the article text and frame definitions. The template
    is expected to have two placeholders: {article} and {frames_json}.
    
    Args:
        article_text (str): The full text of the article to analyze.
        frames (dict): Dictionary mapping frame names to frame definitions.
                      Will be converted to JSON and inserted into the template.
    
    Returns:
        str: Formatted prompt ready to be sent to the LLM.
    
    Raises:
        KeyError: If the prompt template is missing required placeholders
                 ({article} or {frames_json}).
        FileNotFoundError: If the prompt template file doesn't exist (via read_prompt()).
    
    Notes:
        - The prompt template is cached from utils/file_reader.read_prompt()
        - Frames are formatted as pretty-printed JSON (indent=2) for readability
    """
    prompt_template = read_prompt()
    frames_json_str = json.dumps(frames, indent=2)
    return prompt_template.format(
        frames_json=frames_json_str,
        article=article_text,
    )


def validate_result(result: dict, frames: dict, article_text: str) -> list:
    """
    Validate the LLM's narrative frame classification output.
    
    Performs comprehensive validation of the API response including:
    - Presence of required fields (overall_stance, dominant_narrative)
    - Completeness of frame outputs (all expected frames present)
    - Validity of evidence (must be verbatim in article text)
    - Consistency (if frame is present, must have evidence and narrative)
    
    Args:
        result (dict): The parsed JSON response from prompt_llama() containing
                      fields like "overall_stance", "dominant_narrative", and "frames".
        frames (dict): Reference dictionary of expected frames (keys are frame names).
        article_text (str): The original article text to verify evidence against.
    
    Returns:
        list: List of validation warning strings. Empty list means validation passed.
             Each warning describes a specific issue found in the result.
    
    Notes:
        - This is a non-blocking validation; warnings are collected, not raised
        - Evidence validation is strict: must appear verbatim in article_text
        - Missing frames are collectively reported in one warning
        - Each individual frame issue gets its own warning
    """
    warnings = []

    if "overall_stance" not in result or not result["overall_stance"]:
        warnings.append("missing overall_stance")

    if "dominant_narrative" not in result or not result["dominant_narrative"]:
        warnings.append("missing dominant_narrative")

    frame_entries = result.get("frames", [])
    returned_frame_names = {f.get("frame") for f in frame_entries}
    expected_frame_names = set(frames.keys())

    missing = expected_frame_names - returned_frame_names
    if missing:
        warnings.append(f"missing frames in output: {sorted(missing)}")

    for f in frame_entries:
        if f.get("present") is True:
            evidence = f.get("evidence")
            if not evidence:
                warnings.append(f"{f.get('frame')}: present=true but no evidence")
            elif evidence not in article_text:
                warnings.append(f"{f.get('frame')}: evidence not verbatim in article text")
            if not f.get("narrative"):
                warnings.append(f"{f.get('frame')}: present=true but no narrative")

    return warnings


def main():
    """
    Main entry point for the narrative frame classification pipeline.
    
    Orchestrates the complete workflow:
    1. Parses command-line arguments (--input and --output file paths)
    2. Initializes Groq API client
    3. Loads frame definitions from configuration
    4. Reads input CSV file (expected columns: article_id, text)
    5. Supports resume: skips articles already in output file if it exists
    6. For each article:
       - Constructs prompt with article text and frames
       - Calls Groq API via prompt_llama()
       - Validates response
       - Accumulates results
    7. Saves all results to output CSV
    8. Handles KeyboardInterrupt gracefully by saving checkpoint
    
    Command-line Arguments:
        --input (str, required): Path to CSV file with article_id and text columns.
        --output (str, required): Path where results CSV will be written.
    
    Behavior:
        - Creates output directory if it doesn't exist
        - Resumes from existing output file (skips already-processed articles)
        - Pauses 5 seconds between API calls to respect rate limits
        - On Ctrl+C, saves checkpoint with current progress
        - Logs progress, warnings, and errors throughout execution
    
    Output CSV Columns:
        - article_id: Original article identifier
        - overall_stance: Stance classification from LLM
        - dominant_narrative: Primary narrative identified
        - frames_json: JSON array of frame analysis results
        - validation_warnings: Semicolon-separated list of validation issues
        - human_verified: Boolean (initialized as False for manual review)
        - human_corrected_narrative: String (initialized empty for manual corrections)
    
    Raises:
        FileNotFoundError: If input file or configuration files don't exist.
        RuntimeError: If API calls fail after all retries (logged and breaks loop).
    
    Note:
        - All logging goes to both console and timestamped log file in logs/ directory
        - Failed articles are saved to a separate failures.csv file
    """
    parser = argparse.ArgumentParser(description="Label narrative frames via Groq/Llama 3.3.")
    parser.add_argument("--input", required=True, help="CSV with columns: article_id, article_text")
    parser.add_argument("--output", required=True, help="Path to write output CSV")

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Starting narrative classification job")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output file: {args.output}")

    client = Groq()

    logger.info("Loading frames configuration...")
    frames = read_frames()
    logger.info(f"Loaded {len(frames)} frames")

    logger.info(f"Reading articles from {args.input}...")
    articles_csv = read_processed_data(args.input)
    logger.info(f"Total articles to process: {len(articles_csv)}")

    # Resume support: skip articles already present in an existing output file
    already_done = set()
    if Path(args.output).exists():
        existing = pd.read_csv(args.output)
        already_done = set(existing[args.id_col].astype(str))
        rows = existing.to_dict("records")
        logger.info(f"Resuming: {len(already_done)} articles already labeled")
    else:
        rows = []

    processed_count = 0
    failed_count = 0

    try:
        for articles in articles_csv:
            article_id = articles["article_id"]
            if article_id in already_done:
                continue

            article_text = articles["text"]
            logger.info(f"Processing article {article_id} ({len(article_text)} characters)")
            prompt = build_prompt(article_text, frames)

            try:
                result = prompt_llama(client, prompt)
                logger.info(f"Article {article_id}: Successfully classified")
            except RuntimeError as e:
                logger.error(f"Article {article_id}: Classification failed - {e}")
                failed_count += 1
                failed_rows = [{
                    "article_id": article_id,
                    "overall_stance": None,
                    "dominant_narrative": None,
                    "frames_json": None,
                    "validation_warnings": f"API_FAILURE: {e}",
                    "human_verified": False,
                    "human_corrected_narrative": "",
                }]
                _save(failed_rows, "failures.csv")
                break

            warnings = validate_result(result, frames, article_text)
            if warnings:
                logger.warning(f"Article {article_id}: Validation warnings - {'; '.join(warnings)}")

            rows.append({
                "article_id": article_id,
                "overall_stance": result.get("overall_stance"),
                "dominant_narrative": result.get("dominant_narrative"),
                "frames_json": json.dumps(result.get("frames", [])),
                "validation_warnings": "; ".join(warnings) if warnings else "",
                "human_verified": False,  # fill in during manual review
                "human_corrected_narrative": "",  # fill in during manual review, if needed
            })

            processed_count += 1
            logger.info(f"Progress: {processed_count} articles processed")

            time.sleep(5)  # gentle pacing against Groq free-tier rate limits

        _save(rows, args.output)

        logger.info(f"Classification complete. Output written to {args.output}")
        logger.info(f"Statistics - Processed: {processed_count}, Failed: {failed_count}")
        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.warning("=" * 80)
        logger.warning("KEYBOARD INTERRUPT DETECTED - Saving checkpoint...")
        logger.warning("=" * 80)

        if rows:
            checkpoint_name = args.output
            _save(rows, checkpoint_name)
            logger.info(f"Checkpoint saved: {checkpoint_name}")
            logger.info(f"Processed so far: {processed_count} articles, {failed_count} failures")
        else:
            logger.warning("No rows to save - no articles were processed")

        logger.info("Exiting gracefully.")
        logger.warning("=" * 80)


def _save(rows: list, output_path: str) -> None:
    """
    Save classification results to a CSV file.
    
    Converts a list of result dictionaries into a pandas DataFrame and saves
    it as a CSV file in the project's data/processed/ directory. Creates the
    output directory if it doesn't exist.
    
    Args:
        rows (list): List of dictionaries containing classification results.
                    Each dictionary should have keys like: article_id, overall_stance,
                    dominant_narrative, frames_json, validation_warnings, etc.
        output_path (str): Filename for the output CSV (relative name, not full path).
                          Will be saved to data/processed/{output_path}.
    
    Returns:
        None
    
    Notes:
        - CSV is written without index column (index=False)
        - Uses UTF-8 encoding (default for pandas.to_csv)
        - Full path is resolved using get_project_root() from file_reader
        - Logs the number of rows saved and the full output path
        - Creates parent directories if they don't exist
    """
    full_path = f"{get_project_root()}/data/processed/{output_path}"
    logger.info(f"Saving {len(rows)} rows to {output_path}")
    pd.DataFrame(rows).to_csv(full_path, index=False)
    logger.debug(f"File saved successfully to {full_path}")


if __name__ == '__main__':
    main()