import pandas as pd

TEXT_COLUMNS = ["text", "body", "email text", "email", "message", "content", "text_combined"]
LABEL_COLUMNS = ["label", "email type", "class", "type", "is_phishing", "target"]
PHISHING_WORDS = {"phishing", "phishing email", "phish", "malicious", "fraud", "bad"}
SAFE_WORDS = {"legitimate", "safe email", "safe", "ham", "benign", "legit", "good"}


def to_label(value):
    s = str(value).strip().lower()
    if s in PHISHING_WORDS:
        return 1
    if s in SAFE_WORDS:
        return 0
    try:
        number = float(s)
        if number in (0.0, 1.0):
            return int(number)
    except ValueError:
        pass
    return None


def load_dataset(path, verbose=True):
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    columns = {c.lower().strip(): c for c in df.columns}
    text_col = next((columns[c] for c in TEXT_COLUMNS if c in columns), None)
    label_col = next((columns[c] for c in LABEL_COLUMNS if c in columns), None)
    if text_col is None or label_col is None:
        raise ValueError(f"Could not find text/label columns in {list(df.columns)}. "
                         "Rename them to 'text' and 'label'.")

    df = pd.DataFrame({"text": df[text_col], "label": df[label_col].map(to_label)})
    start = len(df)

    df = df.dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[(df["text"].str.len() > 3) & (df["text"].str.lower() != "empty")]
    after_empty = len(df)

    key = df["text"].str.lower().str.split().str.join(" ")
    conflicting = key.map(df.groupby(key)["label"].nunique()) > 1
    df, key = df[~conflicting], key[~conflicting]
    after_conflict = len(df)

    df = df[~key.duplicated()]
    if verbose:
        print(f"Data cleaning: {start} rows -> {len(df)} usable emails "
              f"(removed {start - after_empty} empty/unlabelled, "
              f"{after_empty - after_conflict} contradictory, "
              f"{after_conflict - len(df)} duplicates).")
    return df["text"].tolist(), df["label"].astype(int).tolist()
