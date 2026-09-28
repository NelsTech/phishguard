import os
import runpy
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detector import PhishingDetector
from detector.adaptive import AdaptiveAgent
from detector.data_loader import load_dataset
from detector.email_parser import clean_text, extract_urls
from detector.url_analyzer import shannon_entropy

SAMPLES = ROOT / "data" / "samples"


class TestParser(unittest.TestCase):
    def test_TC01_hidden_link_behind_button_is_found(self):
        email = '<a href="http://capitec-login.xyz/verify">www.capitecbank.co.za</a>'
        self.assertIn("http://capitec-login.xyz/verify", extract_urls(email))

    def test_TC02_repeated_link_is_counted_once(self):
        email = "Go to http://a.com and again http://a.com"
        self.assertEqual(extract_urls(email), ["http://a.com"])

    def test_TC03_text_is_normalised(self):
        self.assertEqual(clean_text("URGENT!!! <b>Act</b> NOW"), "urgent act now")


class TestUrlFeatures(unittest.TestCase):
    def test_TC04_random_link_has_higher_entropy(self):
        self.assertGreater(shannon_entropy("x7q9zk2m4w8v"), shannon_entropy("aaaabbbb"))


class TestAgent(unittest.TestCase):
    def test_TC05_reward_average_is_correct(self):
        agent = AdaptiveAgent()
        agent.update(0.5, 1.0)
        agent.update(0.5, -1.0)
        self.assertAlmostEqual(agent.q["0.5"], 0.0)
        self.assertEqual(agent.n["0.5"], 2)

    def test_TC06_agent_picks_best_rewarded_setting(self):
        agent = AdaptiveAgent()
        agent.update(0.7, 1.0)
        self.assertEqual(agent.best_arm(), 0.7)


class TestDataCleaning(unittest.TestCase):
    def test_TC07_empty_duplicate_and_contradictory_rows_removed(self):
        rows = ("Email Text,Email Type\n"
                "Verify your account now http://1.2.3.4/login,Phishing Email\n"
                "verify your   ACCOUNT now http://1.2.3.4/login,Phishing Email\n"
                "empty,Safe Email\n"
                "Lunch on Friday?,Safe Email\n"
                "Lunch on Friday?,Phishing Email\n"
                "Meeting moved to 10am,Safe Email\n")
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write(rows)
        try:
            emails, labels = load_dataset(f.name, verbose=False)
        finally:
            os.remove(f.name)
        self.assertEqual(len(emails), 2)
        self.assertEqual(sorted(labels), [0, 1])


class TestDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT / "data" / "train.csv").exists():
            runpy.run_path(str(ROOT / "data" / "make_sample_data.py"), run_name="__main__")
        emails, labels = load_dataset(ROOT / "data" / "train.csv", verbose=False)
        cls.detector = PhishingDetector()
        cls.detector.train(emails, labels)

    def test_TC08_phishing_sample_is_flagged(self):
        result = self.detector.scan((SAMPLES / "phishing_example.txt").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "PHISHING")

    def test_TC09_safe_sample_passes(self):
        result = self.detector.scan((SAMPLES / "safe_example.txt").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "SAFE")

    def test_TC10_email_without_links_uses_text_only(self):
        result = self.detector.scan("Hi, are we still meeting for lunch on Friday?")
        self.assertFalse(result["used_url_path"])
        self.assertEqual(result["text_weight"], 1.0)

    def test_TC11_scan_is_under_1_5_seconds(self):
        start = time.perf_counter()
        self.detector.scan((SAMPLES / "phishing_example.txt").read_text(encoding="utf-8"))
        self.assertLess(time.perf_counter() - start, 1.5)

    def test_TC12_feedback_raises_score_of_new_scam(self):
        email = (SAMPLES / "zero_day_example.txt").read_text(encoding="utf-8")
        before = self.detector.scan(email)["final_score"]
        self.detector.learn_from_feedback(email, 1)
        after = self.detector.scan(email)["final_score"]
        self.assertGreater(after, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
