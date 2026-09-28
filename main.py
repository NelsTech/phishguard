import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from detector import PhishingDetector
from detector.data_loader import load_dataset
from detector.evaluation import evaluate, print_report


def cmd_train(args):
    emails, labels = load_dataset(args.data)
    print(f"Loaded {len(emails)} emails ({sum(labels)} phishing, {len(labels) - sum(labels)} safe).")

    if args.test_size > 0:
        train_e, test_e, train_y, test_y = train_test_split(
            emails, labels, test_size=args.test_size, stratify=labels, random_state=42)
    else:
        train_e, train_y, test_e, test_y = emails, labels, [], []

    detector = PhishingDetector(text_model=args.text_model)
    print(f"Text path model: {'DistilBERT (meaning-based)' if args.text_model == 'distilbert' else 'TF-IDF (word-based)'}")
    start = time.perf_counter()
    info = detector.train(train_e, train_y)
    print(f"Trained on {info['emails']} emails and {info['urls']} links "
          f"in {time.perf_counter() - start:.1f}s (URL path active: {info['url_path_active']}).")
    detector.save(args.model_dir)
    print(f"Model saved to '{args.model_dir}/'.")

    if test_e:
        holdout = Path(args.model_dir) / "holdout_test.csv"
        pd.DataFrame({"text": test_e, "label": test_y}).to_csv(holdout, index=False)
        print_report(evaluate(detector, test_e, test_y), "HOLD-OUT TEST (emails never seen in training)")
        print(f"Hold-out set saved to {holdout}")


def cmd_evaluate(args):
    detector = PhishingDetector.load(args.model_dir)
    emails, labels = load_dataset(args.data)
    print_report(evaluate(detector, emails, labels), f"EVALUATION on {args.data}")


def read_email(args):
    if args.text:
        return args.text
    return Path(args.file).read_text(encoding="utf-8", errors="ignore")


def cmd_scan(args):
    detector = PhishingDetector.load(args.model_dir)
    r = detector.scan(read_email(args))

    print("\n" + "=" * 60)
    print(f" VERDICT: {r['verdict']}   (phishing probability {r['final_score']:.1%})")
    print("=" * 60)
    print(f" Text path score : {r['text_score']:.1%}   (model: {'DistilBERT' if r['text_model'] == 'distilbert' else 'TF-IDF'})")
    if r["url_score"] is None:
        print(" URL path score  : n/a (no links found - decision made on text only)")
    else:
        print(f" URL path score  : {r['url_score']:.1%}   ({r['num_links']} link(s) checked)")
        print(f" Most suspicious : {r['worst_url']}")
        for flag in r["url_red_flags"]:
            print(f"     - {flag}")
        print(f" Trust split     : {r['text_weight']:.0%} text / {1 - r['text_weight']:.0%} URL (learned by RL agent)")
    if r["intents"]:
        print(" Scam intents detected:")
        for intent, phrases in r["intents"].items():
            print(f"     - {intent}: {', '.join(phrases)}")
    print(f" Scan time       : {r['seconds'] * 1000:.2f} ms")
    print("=" * 60)


def parse_label(text):
    t = text.strip().lower()
    if t in ("phishing", "phish", "1"):
        return 1
    if t in ("safe", "legitimate", "0"):
        return 0
    sys.exit("--label must be 'phishing' or 'safe'")


def cmd_feedback(args):
    detector = PhishingDetector.load(args.model_dir)
    outcome = detector.learn_from_feedback(read_email(args), parse_label(args.label))
    before = outcome["prediction"]
    print(f"Detector had said {before['verdict']} ({before['final_score']:.1%}); "
          f"you said {args.label.upper()}. Reward = {outcome['reward']:+.0f}")
    after = detector.scan(read_email(args))
    print(f"After learning, it now says {after['verdict']} ({after['final_score']:.1%}).")
    detector.save(args.model_dir)


def cmd_adapt(args):
    detector = PhishingDetector.load(args.model_dir)
    emails, labels = load_dataset(args.data)
    correct = []
    for email, label in zip(emails, labels):
        correct.append(detector.learn_from_feedback(email, label)["was_correct"])

    half = len(correct) // 2
    print(f"Learned from {len(correct)} new emails.")
    print(f"  Accuracy on the FIRST half of the stream : {sum(correct[:half]) / max(half, 1):.1%}")
    print(f"  Accuracy on the SECOND half of the stream: {sum(correct[half:]) / max(len(correct) - half, 1):.1%}")
    print("  RL agent values (text-weight -> average reward):")
    for arm, q in detector.agent.q.items():
        print(f"     {arm}: {q:+.3f}  (tried {detector.agent.n[arm]} times)")
    print(f"  Current best text weight: {detector.agent.best_arm()}")
    detector.save(args.model_dir)


def cmd_eda(args):
    import re
    from collections import Counter
    from statistics import median
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    from detector.email_parser import parse_email
    from detector.text_analyzer import INTENT_NAMES, detect_intents

    emails, labels = load_dataset(args.data)
    groups = {1: "Phishing", 0: "Safe"}
    stats = {g: {"n": 0, "words": [], "links": [], "with_link": 0, "intents": Counter(), "df": Counter()}
             for g in groups}
    for email, label in zip(emails, labels):
        parsed = parse_email(email)
        words = parsed["text"].split()
        st = stats[label]
        st["n"] += 1
        st["words"].append(len(words))
        st["links"].append(len(parsed["urls"]))
        st["with_link"] += bool(parsed["urls"])
        st["intents"].update(detect_intents(parsed["text"]).keys())
        st["df"].update({w for w in words if len(w) > 2 and w not in ENGLISH_STOP_WORDS
                         and w != "linktoken" and not re.fullmatch(r"\d+", w)})

    total = len(emails)
    line = "=" * 64
    print(f"\n{line}\n EXPLORATORY DATA ANALYSIS ({total} clean emails)\n{line}")
    print(f" {'':<34}{'Phishing':>14}{'Safe':>14}")
    row = lambda name, a, b: print(f" {name:<34}{a:>14}{b:>14}")
    P, S = stats[1], stats[0]
    row("Number of emails", f"{P['n']} ({P['n'] / total:.1%})", f"{S['n']} ({S['n'] / total:.1%})")
    row("Median length (words)", f"{median(P['words']):.0f}", f"{median(S['words']):.0f}")
    row("Emails with at least one link", f"{P['with_link'] / P['n']:.1%}", f"{S['with_link'] / S['n']:.1%}")
    row("Average links per email", f"{sum(P['links']) / P['n']:.2f}", f"{sum(S['links']) / S['n']:.2f}")
    for name in INTENT_NAMES:
        row(f"Contains {name.replace('_', ' ')} words", f"{P['intents'][name] / P['n']:.1%}",
            f"{S['intents'][name] / S['n']:.1%}")

    def top(a, b, k=10):
        scores = {w: (c / a["n"]) / ((b["df"][w] + 1) / b["n"])
                  for w, c in a["df"].items() if c >= max(20, 0.01 * a["n"])}
        return sorted(scores, key=lambda w: (-scores[w], w))[:k]

    print(f"\n Words most typical of PHISHING: {', '.join(top(P, S))}")
    print(f" Words most typical of SAFE:     {', '.join(top(S, P))}")
    bigger = "SAFE" if S["n"] >= P["n"] else "PHISHING"
    print(f" Majority-class baseline (always say {bigger}): {max(S['n'], P['n']) / total:.2%} accuracy")
    print(line)


def cmd_baselines(args):
    from detector.email_parser import parse_email
    from detector.text_analyzer import detect_intents
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    emails, labels = load_dataset(args.data)
    train_e, test_e, train_y, test_y = train_test_split(
        emails, labels, test_size=args.test_size, stratify=labels, random_state=42)
    print(f"Training on {len(train_e)} emails, testing on {len(test_e)}... (this takes a minute or two)")
    detector = PhishingDetector()
    detector.train(train_e, train_y)

    names = ["Always say SAFE (majority class)", "Keyword filter (any scam word)",
             "Text inspector only", "Link inspector only", "Full system (both + RL agent)"]
    preds = {name: [] for name in names}
    for email in test_e:
        parsed = parse_email(email)
        text_score = detector.text_path.score(parsed["text"])
        url_score, _ = detector.url_path.score(parsed["urls"])
        preds[names[0]].append(0)
        preds[names[1]].append(int(bool(detect_intents(parsed["text"]))))
        preds[names[2]].append(int(text_score >= 0.5))
        preds[names[3]].append(int(url_score is not None and url_score >= 0.5))
        preds[names[4]].append(detector.scan(email)["label"])

    print("\n" + "=" * 78)
    print(f" BASELINE COMPARISON on {len(test_e)} hold-out emails")
    print("=" * 78)
    print(f" {'Approach':<34}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}")
    for name in names:
        y = preds[name]
        print(f" {name:<34}{accuracy_score(test_y, y):>10.2%}"
              f"{precision_score(test_y, y, zero_division=0):>11.2%}"
              f"{recall_score(test_y, y, zero_division=0):>9.2%}{f1_score(test_y, y, zero_division=0):>9.2%}")
    print("=" * 78)
    print(" Link inspector only: emails with no links are counted as SAFE.")


def main():
    parser = argparse.ArgumentParser(description="PhishGuard: NLP-driven adaptive phishing detection")
    parser.add_argument("--model-dir", default="models", help="where the trained model is stored")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train on a labelled CSV")
    p.add_argument("--data", required=True)
    p.add_argument("--test-size", type=float, default=0.2, help="fraction kept aside for testing")
    p.add_argument("--text-model", choices=["tfidf", "distilbert"], default="tfidf",
                   help="tfidf (default, fast) or distilbert (reads meaning; needs torch + transformers)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="measure accuracy and speed on a labelled CSV")
    p.add_argument("--data", required=True)
    p.set_defaults(func=cmd_evaluate)

    for name, func, text in [("scan", cmd_scan, "check one email"),
                             ("feedback", cmd_feedback, "teach the detector the true label of one email")]:
        p = sub.add_parser(name, help=text)
        group = p.add_mutually_exclusive_group(required=True)
        group.add_argument("--file", help="path to a .txt file containing the email")
        group.add_argument("--text", help="the email text itself")
        if name == "feedback":
            p.add_argument("--label", required=True, help="phishing or safe")
        p.set_defaults(func=func)

    p = sub.add_parser("eda", help="explore the dataset before training (reads only)")
    p.add_argument("--data", required=True)
    p.set_defaults(func=cmd_eda)

    p = sub.add_parser("baselines", help="compare the full system with simpler baselines")
    p.add_argument("--data", required=True)
    p.add_argument("--test-size", type=float, default=0.2)
    p.set_defaults(func=cmd_baselines)

    p = sub.add_parser("adapt", help="learn from a CSV stream of newly labelled emails")
    p.add_argument("--data", required=True)
    p.set_defaults(func=cmd_adapt)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
