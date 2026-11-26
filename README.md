# TweetPulse: Reading the Mood of Twitter with NLP

TweetPulse is a console application that classifies tweets as **positive**, **negative** or **neutral**. It cleans raw tweets, turns them into numbers with NumPy, trains and compares five sentiment models on one shared train/test split, saves report figures, and can classify your own text or live Hacker News headlines.

## Team

| Name | Roll number | Role |
|---|---|---|
| Ayesha Nehal (Leader) | `25K-0002` | Models, evaluation, menu, integration |
| M. Hamdan | `25K-0136` | Text processing, NumPy, visualisations |
| Mohsina Zeeshan | `25k-0095` | Data layer, API and cache, Pandas |

## Features

- **11-option menu** with validated input and clear messages when an option is run before its prerequisite.
- **Text cleaning:** removes URLs, @mentions and `#` signs, expands contractions (`didn't` -> `did not`), shortens stretched words (`soooo` -> `so`), and removes stopwords while keeping negations (`not`, `never`).
- **NumPy Bag-of-Words matrix** (about 2,000 tweets x 500 words), row normalisation by broadcasting, and top words per sentiment class.
- **Five model configurations** compared in one table on a single stratified split: VADER, Naive Bayes (Bag-of-Words / TF-IDF) and Logistic Regression (unigrams / unigrams + bigrams). Metrics: accuracy, macro precision, recall and F1, plus the confusion matrix on the test set only.
- **Five figures** saved to `outputs/figures/`: class counts, tweet-length histogram, VADER score box plot, confusion matrix, model comparison.
- **Disagreement analysis** between models, and sentiment prediction for your own text.
- **Live Hacker News headlines** through a web API, cached to `data/raw/api_cache.json` and reused offline.
- **Clear error handling** with a custom `DataLoadError` (missing file, empty file, malformed file, missing columns, no network).
- **Final check:** the best model is scored once on the unseen `test.csv` and is never fitted or tuned on it.

## Project structure

```
TweetPulse/
├── src/
│   ├── main.py            menu and integration
│   ├── models.py          Tweet, SentimentClassifier and the classifiers
│   ├── evaluate.py        split, metrics, model comparison
│   ├── preprocessing.py   text cleaning and word counts
│   ├── numpy_ops.py       Bag-of-Words matrix and NumPy operations
│   ├── visualize.py       the five figures
│   ├── data_loader.py     CSV, JSON, log file and API with cache
│   ├── analysis.py        Pandas cleaning and summaries
│   └── exceptions.py      DataLoadError
├── data/
│   ├── raw/               train.csv, test.csv, api_cache.json
│   └── processed/         cleaned data created by Option 1
├── outputs/
│   ├── figures/           the five PNG figures
│   └── results/           predictions and run log
├── docs/                  proposal and report (PDF and MD)
├── requirements.txt
└── README.md
```

## Setup

Python **3.12** is recommended (minimum 3.10).

```bash
# 1. Clone the repository
git clone <repository-url>
cd TweetPulse

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux

# 3. Install the libraries
pip install -r requirements.txt
```

NLTK data (stopwords, tokenizer, POS tagger, VADER lexicon) is downloaded automatically the first time it is needed, so an internet connection is required on the first run.

## Data

Download `train.csv` and `test.csv` from Kaggle and place both in `data/raw/`:

https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset

The CSV files are not stored in the repository (they are listed in `.gitignore`). The file names must be exactly `train.csv` and `test.csv`.

## Run

From the project root, with the virtual environment active:

```bash
python src/main.py
```

Suggested order: Option 1 (load and clean), then 4, 5, 6, 7 and 10. Options 2, 3, 8 and 9 can be used at any time, but Option 8 and 9 use the VADER baseline until models are trained in Option 5.

## Limitations

- English-only text. Slang, sarcasm, emojis and mixed-language tweets are often misclassified.
- The Country, Age and Time columns of the dataset look synthetic (see Option 2), so they are not used as model features.
- Models are classical (VADER, Naive Bayes, Logistic Regression) and use word counts, so they miss word order and context beyond bigrams.
- Hacker News headlines are mostly neutral technology news, which differs from tweets, so predictions on them are only a demonstration.
- The live headline feature needs internet on first use. Without it, the program falls back to the cached file.
- Results depend on the dataset and the fixed random split (`random_state=42`).

## Ethics notes

- **Misuse and limits:** sentiment labels are estimates and should not be used to judge or profile individuals.
- **Slang and language bias:** informal English, dialects and non-English text can be scored unfairly.
- **Privacy and dataset bias:** tweets are personal text, and the dataset may not represent all users or topics.

## Documents

`docs/` holds the proposal and the final report, each as PDF and Markdown.