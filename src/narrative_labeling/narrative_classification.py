"""
Narrative Frame Classification Pipeline using Llama 3.1 8B.

This module implements an end-to-end pipeline for classifying articles according
to narrative frames using the Llama 3.1-8B-Instruct model with 4-bit quantization.
The system:

1. Reads articles from a CSV file
2. Constructs prompts with frame definitions
3. Calls the local Llama model with proper tokenization
4. Validates responses for completeness and consistency
5. Saves results to CSV with support for resume/checkpointing
6. Handles interrupts gracefully by saving progress

Features:
    - Local model inference with 4-bit quantization (BitsAndBytes)
    - Resume capability: skips already-processed articles
    - Keyboard interrupt handling with automatic checkpointing
    - Comprehensive logging to both console and file
    - Validation of LLM output (missing fields, invalid evidence, etc.)
    - JSON response parsing with automatic repair on decode errors

Configuration:
    - Loads frame definitions from prompts/frames.json
    - Loads prompt template from prompts/narrative_classification_prompt.txt
    - Reads articles from data/processed/{input_file}
    - Writes results to data/processed/{output_file}
    - Uses meta-llama/Llama-3.1-8B-Instruct model

Usage:
    python narrative_classification.py --input articles.csv --output results.csv

Author: Mohammed Owais Khan
"""
import argparse
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from json_repair import repair_json

from utils.file_reader import read_frames, read_prompt, read_processed_data, get_project_root, get_processed_data_path
from utils.logger import setup_logger, DEBUG

load_dotenv()
logger = setup_logger(__name__)

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

_model = None
_tokenizer = None


def load_model():
    """
    Load 4-bit quantized Llama model with HuggingFace authentication.
    
    Requires:
    - HuggingFace account with access to meta-llama/Llama-3.1-8B-Instruct
    - Either HF_TOKEN environment variable or huggingface login via CLI
    - At least 10GB free disk space for model cache
    
    To set up authentication:
    1. Accept license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
    2. Create HF token at https://huggingface.co/settings/tokens
    3. Set: export HF_TOKEN="your_token" or huggingface-cli login
    
    Raises:
        OSError: If authentication fails or model cannot be downloaded
    """
    global _model, _tokenizer
    
    try:
        from huggingface_hub import login
        
        # Try to login using token if available
        token = os.getenv("HF_TOKEN")
        if token:
            login(token=token, add_to_git_credential=False)
            logger.info("Authenticated with HuggingFace using HF_TOKEN")
        else:
            logger.warning("No HF_TOKEN found. Ensure you've run 'huggingface-cli login'")
    except Exception as e:
        logger.warning(f"HuggingFace login skipped: {e}")

    logger.info(f"Loading model: {MODEL_NAME}")
    logger.info("This may take 2-5 minutes on first run (model is ~16GB)")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    return _model, _tokenizer


def generate(prompt: str, max_new_tokens: int = 1024) -> dict:
    """
    Generate a response using the local Llama model.
    
    Tokenizes the prompt using the chat template, generates tokens with the model,
    and parses the output as JSON. Attempts automatic JSON repair if parsing fails.
    
    Args:
        prompt (str): The prompt to send to the model.
        max_new_tokens (int): Maximum number of new tokens to generate (default: 1024).
    
    Returns:
        dict: Parsed JSON response from the model.
    
    Raises:
        json.JSONDecodeError: If JSON parsing fails even after repair attempt.
    """
    messages = [{"role": "user", "content": prompt}]
    input_ids = _tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(_model.device)

    with torch.no_grad():
        output_ids = _model.generate(
            **input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=_tokenizer.eos_token_id,
        )

    generated = output_ids[0][input_ids["input_ids"].shape[-1]:]
    raw_result = _tokenizer.decode(generated, skip_special_tokens=True)
    logger.debug(raw_result)
    try:
        parsed = json.loads(raw_result)
    except json.JSONDecodeError:
        parsed = json.loads(repair_json(raw_result))
    return parsed


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


def parse_arguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Label narrative frames via local Llama 3.1-8B model.")
    parser.add_argument("--input", required=True, help="CSV with columns: article_id, article_text")
    parser.add_argument("--output", required=True, help="Path to write output CSV")
    return parser.parse_args()


def load_resources(input_file: str) -> tuple:
    """
    Load frames and articles from input file.
    
    Args:
        input_file (str): Path to input CSV file with articles.
    
    Returns:
        tuple: (frames dict, articles list)
    """
    logger.info("Loading frames configuration...")
    frames = read_frames()
    logger.info(f"Loaded {len(frames)} frames")

    logger.info(f"Reading articles from {input_file}...")
    articles_csv = read_processed_data(input_file)
    logger.info(f"Total articles to process: {len(articles_csv)}")

    return frames, articles_csv


def load_existing_results(output_path: str) -> tuple:
    """
    Load existing results for resume capability.
    
    Args:
        output_path (str): Path to output CSV file.
    
    Returns:
        tuple: (already_done set of article_ids, existing rows list)
    """
    already_done = set()
    rows = []

    full_path = get_processed_data_path() / output_path
    if Path(full_path).exists():
        existing = pd.read_csv(full_path)
        already_done = set(existing["article_id"].astype(str))
        rows = existing.to_dict("records")
        logger.info(f"Resuming: {len(already_done)} articles already labeled")

    return already_done, rows


def process_single_article(article: dict, frames: dict) -> dict:
    """
    Process a single article: classify and validate results.
    
    Args:
        article (dict): Article data with 'article_id' and 'text'.
        frames (dict): Frame definitions.
    
    Returns:
        dict: Result row with classification data, or None if failed.
    """
    article_id = article["article_id"]
    article_text = article["text"]

    logger.info(f"Processing article {article_id} ({len(article_text)} characters)")
    prompt = build_prompt(article_text, frames)

    try:
        result = generate(prompt)
        logger.info(f"Article {article_id}: Successfully classified")
    except RuntimeError as e:
        logger.error(f"Article {article_id}: Classification failed - {e}")
        failed_row = {
            "article_id": article_id,
            "overall_stance": None,
            "dominant_narrative": None,
            "frames_json": None,
            "validation_warnings": f"MODEL_FAILURE: {e}",
            "human_verified": False,
            "human_corrected_narrative": "",
        }
        _append_failure(failed_row)
        return None

    warnings = validate_result(result, frames, article_text)
    if warnings:
        logger.warning(f"Article {article_id}: Validation warnings - {'; '.join(warnings)}")

    return {
        "article_id": article_id,
        "overall_stance": result.get("overall_stance"),
        "dominant_narrative": result.get("dominant_narrative"),
        "frames_json": json.dumps(result.get("frames", [])),
        "validation_warnings": "; ".join(warnings) if warnings else "",
        "human_verified": False,
        "human_corrected_narrative": "",
    }


def process_articles(articles_csv: list, frames: dict, already_done: set, rows: list, output_path: str) -> tuple:
    """
    Process all articles with incremental saving every 5 articles.
    
    Args:
        articles_csv (list): List of articles to process.
        frames (dict): Frame definitions.
        already_done (set): Set of already-processed article IDs.
        rows (list): Existing result rows (for resume).
        output_path (str): Path to save results.
    
    Returns:
        tuple: (processed_count, failed_count)
    """
    processed_count = 0
    failed_count = 0
    SAVE_INTERVAL = 5
    batch_rows = []

    for article in articles_csv:
        article_id = article["article_id"]
        if article_id in already_done:
            continue

        try:
            result_row = process_single_article(article, frames)
            if result_row:
                batch_rows.append(result_row)
                rows.append(result_row)
                processed_count += 1
            else:
                failed_count += 1
        except Exception as e:
            logger.error(f"Unexpected error processing article {article_id}: {e}")
            failed_count += 1
            continue

        logger.info(f"Progress: {processed_count} articles processed")

        if processed_count % SAVE_INTERVAL == 0:
            logger.info(f"Incremental save: {processed_count} articles processed so far")
            _save(batch_rows, output_path, incremental=True)
            batch_rows = []

        time.sleep(5)  # gentle pacing to manage resource usage

    return processed_count, failed_count


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
    1. Parses command-line arguments
    2. Loads the local Llama 3.1-8B-Instruct model with 4-bit quantization
    3. Loads frame definitions and articles from configuration
    4. Supports resume: skips articles already in output file if it exists
    5. Processes articles with validation and incremental saving
    6. Handles KeyboardInterrupt gracefully by saving checkpoint
    
    Command-line Arguments:
        --input (str, required): Path to CSV file with article_id and text columns.
        --output (str, required): Path where results CSV will be written.
    
    Behavior:
        - Loads resources (model, frames, articles)
        - Resumes from existing output file (skips already-processed articles)
        - Incremental save after every 5 processed articles
        - Pauses 5 seconds between model calls to manage resource usage
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
    
    Note:
        - All logging goes to both console and timestamped log file in logs/ directory
        - Failed articles are saved to a separate failures.csv file
    """
    args = parse_arguments()

    logger.info("=" * 80)
    logger.info("Starting narrative classification job")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output file: {args.output}")

    load_model()

    frames, articles_csv = load_resources(args.input)
    already_done, rows = load_existing_results(args.output)

    try:
        processed_count, failed_count = process_articles(
            articles_csv, frames, already_done, rows, args.output
        )

        _save(rows, args.output)

        logger.info(f"Classification complete. Output written to {args.output}")
        logger.info(f"Statistics - Processed: {processed_count}, Failed: {failed_count}")
        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.warning("=" * 80)
        logger.warning("KEYBOARD INTERRUPT DETECTED - Saving checkpoint...")
        logger.warning("=" * 80)

        if rows:
            _save(rows, args.output)
            logger.info(f"Checkpoint saved: {args.output}")
            logger.info(f"Processed so far: {len(rows)} articles")
        else:
            logger.warning("No rows to save - no articles were processed")

        logger.info("Exiting gracefully.")
        logger.warning("=" * 80)


def _save(rows: list, output_path: str, incremental: bool = False) -> None:
    """
    Save classification results to a CSV file.
    
    Supports both full overwrites and incremental appends. When incremental is True,
    appends new rows to the existing file (if it exists). When False, overwrites the file.
    
    Args:
        rows (list): List of dictionaries containing classification results.
                    Each dictionary should have keys like: article_id, overall_stance,
                    dominant_narrative, frames_json, validation_warnings, etc.
        output_path (str): Filename for the output CSV (relative name, not full path).
                          Will be saved to data/processed/{output_path}.
        incremental (bool): If True, append new rows; if False, overwrite file (default: False).
    
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

    if incremental and Path(full_path).exists():
        logger.info(f"Appending {len(rows)} rows to {output_path} (incremental save)")
        df = pd.DataFrame(rows)
        df.to_csv(full_path, mode="a", header=False, index=False)
    else:
        logger.info(f"Saving {len(rows)} rows to {output_path}")
        pd.DataFrame(rows).to_csv(full_path, index=False)

    logger.debug(f"File saved successfully to {full_path}")


def _append_failure(row: dict) -> None:
    """Append a single failed article's record to failures.csv without overwriting prior failures."""
    failures_path = f"{get_project_root()}/data/processed/failures.csv"
    file_exists = Path(failures_path).exists()

    existing_ids = set()
    if file_exists:
        existing = pd.read_csv(failures_path)
        existing_ids = set(existing["article_id"].astype(str))

    if str(row["article_id"]) in existing_ids:
        return

    df_row = pd.DataFrame([row])
    df_row.to_csv(
        failures_path,
        mode="a",
        header=not file_exists,
        index=False,
    )
    logger.info(f"Failure logged to failures.csv (article_id={row['article_id']})")


if __name__ == '__main__':
    main()