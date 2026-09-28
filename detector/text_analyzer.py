import re

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier

INTENT_CUES = {
    "urgency": ["urgent", "immediately", "within 24 hours", "act now", "right away",
                "final notice", "expires today", "as soon as possible", "last chance"],
    "credential_theft": ["verify your account", "confirm your password", "password",
                         "log in", "login", "sign in", "bank details", "banking details",
                         "card details", "billing details", "pin", "otp", "id number"],
    "fear_or_threat": ["suspended", "locked", "unusual activity", "unauthorized",
                       "legal action", "closed", "terminated", "blocked", "security alert"],
    "reward_or_greed": ["congratulations", "winner", "you have won", "prize", "reward",
                        "claim", "free", "gift card", "refund", "entitled to"],
}

_CUE_PATTERNS = {
    intent: [(phrase, re.compile(r"\b" + re.escape(phrase) + r"\b")) for phrase in phrases]
    for intent, phrases in INTENT_CUES.items()
}
INTENT_NAMES = list(INTENT_CUES.keys())


def detect_intents(clean_text: str) -> dict:
    found = {}
    for intent, patterns in _CUE_PATTERNS.items():
        hits = [phrase for phrase, pattern in patterns if pattern.search(clean_text)]
        if hits:
            found[intent] = hits
    return found


def _intent_features(texts) -> csr_matrix:
    rows = []
    for t in texts:
        intents = detect_intents(t)
        rows.append([min(len(intents.get(name, [])), 3) / 3 for name in INTENT_NAMES])
    return csr_matrix(np.array(rows, dtype=float))


class TextAnalyzer:
    def __init__(self):
        self.hasher = HashingVectorizer(ngram_range=(1, 2), n_features=2 ** 18,
                                        alternate_sign=False, norm=None)
        self.tfidf = TfidfTransformer(sublinear_tf=True)
        self.model = SGDClassifier(loss="log_loss",
                                   alpha=1e-5, max_iter=30, tol=None, random_state=42)

    def _features(self, texts, fit=False):
        counts = self.hasher.transform(texts)
        weighted = self.tfidf.fit_transform(counts) if fit else self.tfidf.transform(counts)
        return hstack([weighted, _intent_features(texts)]).tocsr()

    def fit(self, texts, labels):
        self.model.fit(self._features(texts, fit=True), labels)

    def score(self, text: str) -> float:
        return float(self.model.predict_proba(self._features([text]))[0, 1])

    def update(self, text: str, label: int, passes: int = 3):
        X = self._features([text])
        for _ in range(passes):
            self.model.partial_fit(X, [label])
