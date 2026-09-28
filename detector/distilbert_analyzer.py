import sys

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from .text_analyzer import _intent_features

MODEL_NAME = "distilbert-base-uncased"


class DistilBertTextAnalyzer:
    def __init__(self, max_length: int = 256, batch_size: int = 16):
        self.max_length = max_length
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.model = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=30,
                                   tol=None, random_state=42)
        self._tokenizer = None
        self._bert = None

    def _load_bert(self):
        if self._bert is None:
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as error:
                raise ImportError("DistilBERT mode needs extra libraries. Run: "
                                  "python -m pip install torch transformers") from error
            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self._bert = AutoModel.from_pretrained(MODEL_NAME)
            self._bert.eval()

    def embed(self, texts, show_progress=False) -> np.ndarray:
        self._load_bert()
        torch = self._torch
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t if t else "empty" for t in texts[start:start + self.batch_size]]
            encoded = self._tokenizer(batch, padding=True, truncation=True,
                                      max_length=self.max_length, return_tensors="pt")
            with torch.no_grad():
                hidden = self._bert(input_ids=encoded["input_ids"],
                                    attention_mask=encoded["attention_mask"]).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            vectors.append(pooled.numpy())
            if show_progress:
                done = min(start + self.batch_size, len(texts))
                sys.stdout.write(f"\r  DistilBERT is reading emails: {done}/{len(texts)}")
                sys.stdout.flush()
        if show_progress:
            print()
        return np.vstack(vectors)

    def _features(self, texts, fit=False):
        meaning = self.embed(texts, show_progress=fit)
        meaning = self.scaler.fit_transform(meaning) if fit else self.scaler.transform(meaning)
        return np.hstack([meaning, _intent_features(texts).toarray()])

    def fit(self, texts, labels):
        self.model.fit(self._features(texts, fit=True), labels)

    def score(self, text: str) -> float:
        return float(self.model.predict_proba(self._features([text]))[0, 1])

    def update(self, text: str, label: int, passes: int = 3):
        X = self._features([text])
        for _ in range(passes):
            self.model.partial_fit(X, [label])

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ("_tokenizer", "_bert", "_torch"):
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tokenizer = None
        self._bert = None
