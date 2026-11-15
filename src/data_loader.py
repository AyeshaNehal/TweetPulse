# Owner: Mohsina Zeeshan (Lab 06)

import os
import csv
import json
from datetime import datetime

import pandas as pd
import requests

from exceptions import DataLoadError

# project root, so paths work from any folder
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results")

TRAIN_PATH = os.path.join(RAW_DIR, "train.csv")
TEST_PATH = os.path.join(RAW_DIR, "test.csv")
CACHE_PATH = os.path.join(RAW_DIR, "api_cache.json")
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")

REQUIRED_COLUMNS = ("text", "sentiment")

DATASET_URL = ("https://www.kaggle.com/datasets/abhi8923shriv/"
               "sentiment-analysis-dataset?select=train.csv")

HN_TOP_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{}.json"


def write_log(message, path=LOG_PATH):
    """Append a timestamped line to the run log."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {message}\n")


def load_tweets(path=TRAIN_PATH, required_columns=REQUIRED_COLUMNS):
    """Load a tweet CSV into a DataFrame, raising DataLoadError on problems."""
    try:
        # latin-1 because column names like 'Km²' are not valid utf-8
        df = pd.read_csv(path, encoding="latin-1")
    except FileNotFoundError:
        raise DataLoadError(
            f"File not found: {path}\n"
            f"Download train.csv and test.csv from {DATASET_URL} "
            f"and put them in data/raw/.") from None
    except pd.errors.EmptyDataError:
        raise DataLoadError(f"The file is empty: {path}") from None
    except pd.errors.ParserError as e:
        raise DataLoadError(f"The file is malformed and cannot be read: "
                            f"{path}\nDetails: {e}") from None

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise DataLoadError(f"{os.path.basename(path)} is missing the "
                            f"column(s): {', '.join(missing)}")
    if df.empty:
        raise DataLoadError(f"{os.path.basename(path)} has a header "
                            f"but no rows.")

    write_log(f"Loaded {os.path.basename(path)}: "
              f"{df.shape[0]} rows x {df.shape[1]} columns")
    return df


def save_predictions_csv(rows, path=None):
    """Save (text, true_label, predicted_label) rows to a CSV file."""
    if path is None:
        path = os.path.join(RESULTS_DIR, "predictions.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "true_label", "predicted_label"])
        writer.writerows(rows)
    write_log(f"Saved {len(rows)} predictions to {os.path.basename(path)}")
    return path


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise DataLoadError(f"JSON file not found: {path}") from None
    except json.JSONDecodeError as e:
        raise DataLoadError(f"JSON file is corrupted: {path} ({e})") from None


def fetch_headlines(n=10, timeout=5, cache_path=CACHE_PATH,
                    top_url=HN_TOP_URL, item_url=HN_ITEM_URL):
    """Fetch top Hacker News headlines, or use the cache if offline.

    Returns (headlines, source) where source is "live" or "cache".
    """
    try:
        reply = requests.get(top_url, timeout=timeout)
        if reply.status_code != 200:
            raise ValueError(f"status code {reply.status_code}")
        story_ids = reply.json()[:n]

        headlines = []
        for story_id in story_ids:
            item_reply = requests.get(item_url.format(story_id),
                                      timeout=timeout)
            if item_reply.status_code != 200:
                continue
            item = item_reply.json()
            if isinstance(item, dict) and item.get("title"):
                headlines.append(item["title"])

        if not headlines:
            raise ValueError("the API returned no headlines")

        save_json({"fetched_at": datetime.now().isoformat(timespec="seconds"),
                   "source": top_url,
                   "headlines": headlines}, cache_path)
        write_log(f"Fetched {len(headlines)} live headlines; cache updated")
        return headlines, "live"

    except requests.exceptions.ConnectionError:
        reason = "no network connection"
    except requests.exceptions.Timeout:
        reason = f"the request timed out after {timeout} seconds"
    except ValueError as e:
        reason = str(e)

    # live fetch failed, fall back to the cache
    write_log(f"API failed ({reason}); trying the cache")
    try:
        cached = load_json(cache_path)
    except DataLoadError:
        raise DataLoadError(f"Could not reach the API ({reason}) and there "
                            f"is no cache at {cache_path}. Connect to the "
                            f"internet and try again.") from None

    headlines = cached.get("headlines") if isinstance(cached, dict) else None
    if not headlines:
        raise DataLoadError(f"Could not reach the API ({reason}) and the "
                            f"cache has no headlines.")
    print(f"API unavailable ({reason}). Using cached headlines from "
          f"{cached.get('fetched_at', 'an earlier run')}.")
    return headlines[:n], "cache"


if __name__ == "__main__":
    tweets = load_tweets()
    print("train.csv shape:", tweets.shape)
    print(tweets[["text", "sentiment"]].head())
    titles, source = fetch_headlines(5)
    print(f"\nHeadlines ({source}):")
    for title in titles:
        print(" -", title)
