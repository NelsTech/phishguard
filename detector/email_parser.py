import html
import re

URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s<>\"'()\[\]]+", re.IGNORECASE)
HREF_PATTERN = re.compile(r"href\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
HTML_TAG = re.compile(r"<[^>]+>")


def extract_urls(raw_email: str) -> list:
    raw = html.unescape(raw_email)
    candidates = HREF_PATTERN.findall(raw) + URL_PATTERN.findall(raw)

    urls = []
    seen = set()
    for url in candidates:
        url = url.strip().rstrip(".,;:!?")
        if url.lower().startswith(("http", "www.")) and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def clean_text(raw_email: str) -> str:
    text = html.unescape(raw_email)
    text = HTML_TAG.sub(" ", text)
    text = URL_PATTERN.sub(" linktoken ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_email(raw_email: str) -> dict:
    return {"text": clean_text(raw_email), "urls": extract_urls(raw_email)}
