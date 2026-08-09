import csv
import json


def read_prompt():
    with open('../prompts/narrative_classification_prompt.txt', 'r') as prompt_file:
        prompt = prompt_file.read()
        return prompt

def read_frames():
    with open('../prompts/frames.json') as frames_file:
        frames = json.loads(frames_file.read())
        return frames

def read_processed_data():
    data = csv.DictReader(open('../data/processed/immigration_subset.csv', 'r',encoding='utf-8'))
    return [row for row in data]
