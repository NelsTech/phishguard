import math
import re
from collections import Counter
from urllib.parse import urlparse

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

SUSPICIOUS_TLDS = (".xyz", ".top", ".tk", ".ml", ".ga", ".cf", ".gq", ".click",
                   ".link", ".ru", ".cn", ".zip", ".work", ".support", ".live")
SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "cutt.ly", "rb.gy"}
BAIT_WORDS = ("login", "log-in", "signin", "verify", "secure", "account", "update",
              "confirm", "password", "banking", "wallet", "billing")
IP_ADDRESS = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")

FEATURE_NAMES = ["url_length", "host_length", "dots_in_host", "hyphens", "has_at_sign",
                 "host_is_ip", "uses_https", "digit_count", "suspicious_tld",
                 "is_shortener", "bait_words", "entropy", "path_depth"]


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def structural_features(url: str) -> list:
    parsed = urlparse(url if "://" in url else "http://" + url)
    host = (parsed.hostname or "").lower()
    bare_host = host[4:] if host.startswith("www.") else host
    lower = url.lower()
    return [
        len(url),
        len(host),
        host.count("."),
        url.count("-"),
        1.0 if "@" in url else 0.0,
        1.0 if IP_ADDRESS.match(host) else 0.0,
        1.0 if lower.startswith("https://") else 0.0,
        sum(ch.isdigit() for ch in url),
        1.0 if host.endswith(SUSPICIOUS_TLDS) else 0.0,
        1.0 if bare_host in SHORTENERS else 0.0,
        sum(word in lower for word in BAIT_WORDS),
        shannon_entropy(lower),
        parsed.path.count("/"),
    ]


def explain_url(url: str) -> list:
    f = dict(zip(FEATURE_NAMES, structural_features(url)))
    flags = []
    if f["host_is_ip"]:
        flags.append("uses a raw IP address instead of a domain name")
    if f["has_at_sign"]:
        flags.append("contains '@' (can hide the real destination)")
    if f["suspicious_tld"]:
        flags.append("uses a domain ending often abused by scammers")
    if f["is_shortener"]:
        flags.append("uses a link shortener that hides the destination")
    if not f["uses_https"]:
        flags.append("not encrypted (http, not https)")
    if f["bait_words"] >= 2:
        flags.append("stuffed with bait words like 'verify', 'login', 'secure'")
    if f["dots_in_host"] >= 3:
        flags.append("unusually many sub-domains")
    return flags


class UrlAnalyzer:
    def __init__(self):
        self.hasher = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                        n_features=2 ** 18, alternate_sign=False, norm=None)
        self.tfidf = TfidfTransformer(sublinear_tf=True)
        self.scaler = StandardScaler()
        self.model = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=30,
                                   tol=None, random_state=42)
        self.is_trained = False

    def _features(self, urls, fit=False):
        counts = self.hasher.transform(urls)
        struct = np.array([structural_features(u) for u in urls], dtype=float)
        if fit:
            chars = self.tfidf.fit_transform(counts)
            struct = self.scaler.fit_transform(struct)
        else:
            chars = self.tfidf.transform(counts)
            struct = self.scaler.transform(struct)
        struct = np.clip(struct, -5, 5)
        return hstack([chars, csr_matrix(struct)]).tocsr()

    def fit(self, urls, labels):
        if len(urls) == 0 or len(set(labels)) < 2:
            self.is_trained = False
            return
        self.model.fit(self._features(urls, fit=True), labels)
        self.is_trained = True

    def score(self, urls):
        if not urls or not self.is_trained:
            return None, None
        probs = self.model.predict_proba(self._features(urls))[:, 1]
        worst = int(np.argmax(probs))
        return float(probs[worst]), urls[worst]

    def update(self, urls, label: int, passes: int = 3):
        if urls and self.is_trained:
            X = self._features(urls)
            for _ in range(passes):
                self.model.partial_fit(X, [label] * len(urls))
