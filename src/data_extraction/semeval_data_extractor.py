"""
Step 1 setup script — Narrative-Aware Propaganda Detection term paper.

What this does:
1. Clones the SemEval-2020 Task 11 data (mirrored, no registration needed).
2. Extracts the "Immigration" topic cluster (71 train articles) used as the
   narrative-annotation subset, along with their gold propaganda-technique
   labels (span-level, task2-TC format).
3. Writes a single tidy CSV: article_id, text, techniques (list), narrative (blank, to fill in)

Run this first, in Colab or locally. No GPU needed for this step.
"""

import collections
import csv
import os
import re
from utils.file_reader import get_project_root

ROOT_DIR = get_project_root()
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw", "datasets-v2", "datasets")

TOPIC_KEYWORDS = {
    "immigration": ["immigra", "border wall", "daca", "deport"],
    "israel_middle_east": ["israel", "palestin", "gaza", "jerusalem"],
    "kavanaugh_scotus": ["kavanaugh", "supreme court nomin"],
    "trump_russia_probe": ["mueller", "russia probe", "collusion", "manafort"],
    "las_vegas_shooting": ["las vegas", "paddock", "mandalay bay"],
}

SELECTED_TOPIC = "immigration"

def load_articles():
    articles = {}
    art_dir = os.path.join(RAW_DATA_DIR, "train-articles")
    for fname in os.listdir(art_dir):
        aid = re.search(r"\d+", fname).group()
        with open(os.path.join(art_dir, fname), encoding="utf-8", errors="ignore") as fh:
            articles[aid] = fh.read()
    return articles


def load_technique_labels():
    labels = collections.defaultdict(list)  # aid -> list of (technique, start, end)
    label_dir = os.path.join(RAW_DATA_DIR, "train-labels-task2-technique-classification")
    for fname in os.listdir(label_dir):
        aid = re.search(r"\d+", fname).group()
        with open(os.path.join(label_dir, fname)) as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) != 4:
                    continue
                _, technique, start, end = parts
                labels[aid].append((technique, int(start), int(end)))
    return labels


def main():
    articles = load_articles()
    labels = load_technique_labels()

    kws = TOPIC_KEYWORDS[SELECTED_TOPIC]
    subset_ids = sorted(
        aid for aid, text in articles.items() if any(k in text.lower() for k in kws)
    )
    print(f"Selected topic: {SELECTED_TOPIC} -> {len(subset_ids)} articles")

    out_path = os.path.join(DATA_DIR, "processed", f"{SELECTED_TOPIC}_subset.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["article_id", "text", "techniques", "narrative_label"])
        for aid in articles.keys():
            techs = sorted(set(t for t, s, e in labels.get(aid, [])))
            writer.writerow([aid, articles[aid], "|".join(techs), ""])  # narrative_label left blank for step 2

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()