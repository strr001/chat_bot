import json
import os
from rapidfuzz import fuzz


# ⬇ абсолютний шлях на основі поточної директорії
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "cv_index_semantic.json")

with open(INDEX_PATH, "r", encoding="utf-8") as f:
    EXAMPLES = json.load(f)

def find_best_resume_example(user_message: str) -> dict | None:
    user_query = user_message.lower()
    best = {"score": 0, "match": None}

    for item in EXAMPLES:
        score = 0

        if item.get("role"):
            score += fuzz.partial_ratio(user_query, item["role"].lower()) * 3

        for tech in item.get("tech", []):
            score += fuzz.partial_ratio(user_query, tech.lower())

        if item.get("level"):
            score += fuzz.partial_ratio(user_query, item["level"].lower()) * 0.5

        if item.get("domain"):
            for domain_tag in item["domain"]:
                score += fuzz.partial_ratio(user_query, domain_tag.lower())

        if score > best["score"]:
            best = {"score": score, "match": item}

    return best["match"] if best["score"] >= 300 else None

