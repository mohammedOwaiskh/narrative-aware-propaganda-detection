import csv
import json
from pathlib import Path
import os

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def read_prompt() -> str:
    prompt_path =   os.path.join(get_project_root(), 'prompts', 'narrative_classification_prompt.txt')
    with open(prompt_path, 'r') as prompt_file:
        prompt = prompt_file.read()
        return prompt

def read_frames() -> dict:
    frames_path = os.path.join(get_project_root(), 'prompts', 'frames.json')
    with open(frames_path) as frames_file:
        frames = json.loads(frames_file.read())
        return frames

def read_processed_data(data_file: str) -> list[dict]:
    if not data_file.endswith('.csv'):
        raise ValueError('File must end with .csv')
    data_path = os.path.join(get_project_root(), 'data', 'processed', data_file)
    data = csv.DictReader(open(data_path, 'r', encoding='utf-8'))
    return [row for row in data]

if __name__ == "__main__":
    print(get_project_root())