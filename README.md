# PhishGuard: NLP-Driven Adaptive Phishing Detection

A lightweight Python command-line prototype that detects phishing emails with two
analysis paths (email text + embedded links) and a self-updating reinforcement-learning module.

## 1. Install (once)

    pip install -r requirements.txt

## 2. Create practice data (once)

    python data/make_sample_data.py

## 3. Run the demo (in this order)

    python main.py train    --data data/train.csv
    python main.py evaluate --data data/test.csv
    python main.py scan     --file data/samples/phishing_example.txt
    python main.py scan     --file data/samples/safe_example.txt

Zero-day adaptation demo (shows the system learning a NEW scam style):

    python main.py evaluate --data data/zero_day_test.csv      # before: accuracy drops
    python main.py adapt    --data data/zero_day_stream.csv    # admin feedback stream
    python main.py evaluate --data data/zero_day_test.csv      # after: accuracy recovers

Teach it one email live:

    python main.py feedback --file data/samples/zero_day_example.txt --label phishing

## 4. Using a real dataset (for the performance report)

The sample data is generated from templates, so it is easy and scores near 100%.
For the official results, download a real labelled email dataset, for example:

* Kaggle "Phishing Emails" (subhajournal/phishingemails, about 18,000 emails)
* Zenodo "Phishing validation emails dataset" (2,000 emails)

Both use the columns `Email Text` and `Email Type`, which this program reads automatically. Then:

    python main.py train --data Phishing_Email.csv

This trains on 80% and tests on the unseen 20%, printing accuracy and speed against the targets.

## 5. Data exploration, test cases and baseline comparison

Exploratory data analysis (reads the data only, changes nothing):

    python main.py eda --data phishing_Email.csv


Automated test cases (12 tests, about 1 second, never touch the saved model):

    python -m unittest discover tests -v

Baseline comparison on the real dataset (about 2 minutes). Compares the full
system with: always saying SAFE, a keyword filter, the text inspector alone and
the link inspector alone, on the same hold-out emails:

    python main.py baselines --data phishing_Email.csv

## 6. Dataset

The real dataset is not stored in this repository. Download "Phishing Emails"
(subhajournal) from Kaggle: https://www.kaggle.com/datasets/subhajournal/phishingemails
and put the CSV in the project folder.

## Project structure

    main.py                  command-line interface (train / evaluate / scan / feedback / adapt)
    detector/email_parser.py Step 1: split email into clean text + links
    detector/text_analyzer.py Step 2a: Text Path (NLP + intent recognition)
    detector/url_analyzer.py Step 2b: URL Path (character n-grams + structural features)
    detector/adaptive.py     Step 3: reinforcement-learning agent (epsilon-greedy bandit)
    detector/engine.py       runs the pipeline, fuses the scores, handles feedback
    detector/evaluation.py   accuracy, precision, recall, F1, confusion matrix, timing
    detector/data_loader.py  reads CSV datasets
    data/make_sample_data.py synthetic practice data generator
    tests/test_detector.py   12 automated test cases
