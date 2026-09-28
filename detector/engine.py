import time
from pathlib import Path

import joblib

from .adaptive import AdaptiveAgent
from .email_parser import parse_email
from .text_analyzer import TextAnalyzer, detect_intents
from .url_analyzer import UrlAnalyzer, explain_url

MODEL_FILE = "models.joblib"
AGENT_FILE = "agent.json"


class PhishingDetector:
    THRESHOLD = 0.5

    def __init__(self, text_model: str = "tfidf"):
        if text_model == "distilbert":
            from .distilbert_analyzer import DistilBertTextAnalyzer
            self.text_path = DistilBertTextAnalyzer()
        else:
            self.text_path = TextAnalyzer()
        self.url_path = UrlAnalyzer()
        self.agent = AdaptiveAgent()

    @property
    def text_model(self) -> str:
        return "distilbert" if type(self.text_path).__name__ == "DistilBertTextAnalyzer" else "tfidf"

    def train(self, emails, labels) -> dict:
        parsed = [parse_email(e) for e in emails]
        self.text_path.fit([p["text"] for p in parsed], labels)

        urls, url_labels = [], []
        for p, label in zip(parsed, labels):
            for u in p["urls"]:
                urls.append(u)
                url_labels.append(label)
        self.url_path.fit(urls, url_labels)
        return {"emails": len(emails), "urls": len(urls), "url_path_active": self.url_path.is_trained}

    def scan(self, raw_email: str, explore: bool = False) -> dict:
        start = time.perf_counter()

        parsed = parse_email(raw_email)
        text_score = self.text_path.score(parsed["text"])
        url_score, worst_url = self.url_path.score(parsed["urls"])
        weight = self.agent.choose(explore=explore)

        if url_score is None:
            final = text_score
        else:
            final = weight * text_score + (1 - weight) * url_score

        is_phishing = final >= self.THRESHOLD
        elapsed = time.perf_counter() - start

        return {
            "verdict": "PHISHING" if is_phishing else "SAFE",
            "label": int(is_phishing),
            "final_score": final,
            "text_score": text_score,
            "url_score": url_score,
            "worst_url": worst_url,
            "url_red_flags": explain_url(worst_url) if worst_url else [],
            "intents": detect_intents(parsed["text"]),
            "num_links": len(parsed["urls"]),
            "text_weight": weight if url_score is not None else 1.0,
            "arm": weight,
            "used_url_path": url_score is not None,
            "seconds": elapsed,
            "text_model": self.text_model,
        }

    def learn_from_feedback(self, raw_email: str, true_label: int) -> dict:
        result = self.scan(raw_email, explore=True)
        correct = result["label"] == true_label
        reward = 1.0 if correct else -1.0

        if result["used_url_path"]:
            self.agent.update(result["arm"], reward)

        parsed = parse_email(raw_email)
        self.text_path.update(parsed["text"], true_label)
        self.url_path.update(parsed["urls"], true_label)
        return {"was_correct": correct, "reward": reward, "prediction": result}

    def save(self, folder="models"):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump({"text": self.text_path, "url": self.url_path}, folder / MODEL_FILE)
        self.agent.save(folder / AGENT_FILE)

    @classmethod
    def load(cls, folder="models"):
        folder = Path(folder)
        if not (folder / MODEL_FILE).exists():
            raise FileNotFoundError(f"No trained model in '{folder}'. Run: python main.py train --data <file.csv>")
        detector = cls()
        parts = joblib.load(folder / MODEL_FILE)
        detector.text_path, detector.url_path = parts["text"], parts["url"]
        detector.agent = AdaptiveAgent.load(folder / AGENT_FILE)
        return detector
