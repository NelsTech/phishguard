from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

ACCURACY_TARGET = 0.85
SPEED_TARGET_SECONDS = 1.5


def evaluate(detector, emails, labels) -> dict:
    predictions, times = [], []
    for email in emails:
        result = detector.scan(email)
        predictions.append(result["label"])
        times.append(result["seconds"])

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "n": len(emails),
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "avg_seconds": sum(times) / len(times),
        "max_seconds": max(times),
    }


def print_report(r: dict, title="EVALUATION REPORT"):
    ok = lambda passed: "PASS" if passed else "FAIL"
    print("\n" + "=" * 60)
    print(f" {title}  ({r['n']} emails)")
    print("=" * 60)
    print(f" Accuracy   : {r['accuracy']:.2%}   target >= 85%  -> {ok(r['accuracy'] >= ACCURACY_TARGET)}")
    print(f" Precision  : {r['precision']:.2%}   (of emails flagged, how many were really phishing)")
    print(f" Recall     : {r['recall']:.2%}   (of real phishing, how many we caught)")
    print(f" F1-score   : {r['f1']:.2%}")
    print(f" Avg time   : {r['avg_seconds'] * 1000:.2f} ms per email")
    print(f" Worst time : {r['max_seconds'] * 1000:.2f} ms   target < 1500 ms -> "
          f"{ok(r['max_seconds'] < SPEED_TARGET_SECONDS)}")
    print("\n Confusion matrix:")
    print("                     predicted SAFE   predicted PHISHING")
    print(f"   actually SAFE     {r['tn']:>14}   {r['fp']:>18}")
    print(f"   actually PHISHING {r['fn']:>14}   {r['tp']:>18}")
    print("=" * 60)
