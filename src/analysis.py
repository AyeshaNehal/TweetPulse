# Owner: Mohsina Zeeshan (Lab 08)

import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

VALID_LABELS = {"positive", "negative", "neutral"}


def clean_dataframe(df, text_col="text", label_col="sentiment"):
    """Drop missing/duplicate tweets, lowercase labels, add word_count."""
    report = {"rows_before": len(df)}
    df = df.copy()

    df = df.dropna(subset=[text_col, label_col])
    report["missing_removed"] = report["rows_before"] - len(df)

    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""]

    before = len(df)
    df = df.drop_duplicates(subset=[text_col])
    report["duplicates_removed"] = before - len(df)

    df[label_col] = df[label_col].astype(str).str.strip().str.lower()
    before = len(df)
    df = df[df[label_col].isin(VALID_LABELS)]
    report["bad_labels_removed"] = before - len(df)

    df["word_count"] = df[text_col].str.split().str.len()

    df = df.reset_index(drop=True)
    report["rows_after"] = len(df)
    df.attrs["cleaning_report"] = report
    return df


def summarise_by_sentiment(df, label_col="sentiment"):
    """Count, share and average length of tweets per sentiment."""
    summary = df.groupby(label_col).agg(
        tweets=("text", "count"),
        avg_words=("word_count", "mean"),
        max_words=("word_count", "max"),
    )
    summary["share_%"] = summary["tweets"] / summary["tweets"].sum() * 100
    if "vader_compound" in df.columns:
        summary["avg_vader"] = df.groupby(label_col)["vader_compound"].mean()
    return summary.round(2)


def sentiment_by_group(df, group_col="Age of User", label_col="sentiment",
                       percent=True):
    """Crosstab of sentiment against a column like age, time or country."""
    if group_col not in df.columns:
        raise KeyError(f"Column '{group_col}' is not in the DataFrame. "
                       f"Choose from: {', '.join(df.columns)}")
    if percent:
        table = pd.crosstab(df[group_col], df[label_col], normalize="index")
        return (table * 100).round(1)
    return pd.crosstab(df[group_col], df[label_col])


def filter_rows(df, column, value):
    """Rows where column equals value, e.g. filter_rows(df, "Country", "Pakistan")."""
    if column not in df.columns:
        raise KeyError(f"Column '{column}' is not in the DataFrame.")
    if isinstance(value, str):
        mask = df[column].astype(str).str.lower() == value.strip().lower()
    else:
        mask = df[column] == value
    return df[mask]


def _get_vader():
    """Create the VADER analyser, downloading its word list if missing."""
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


def add_vader_scores(df, text_col="text"):
    """Add a vader_compound column from -1 (negative) to +1 (positive)."""
    sia = _get_vader()
    df = df.copy()
    df["vader_compound"] = df[text_col].map(
        lambda t: sia.polarity_scores(str(t))["compound"])
    return df


def check_synthetic_columns(df, cols=("Country", "Age of User",
                                      "Time of Tweet")):
    """Print group sizes and order to see if these columns are generated."""
    for col in cols:
        if col not in df.columns:
            continue
        counts = df[col].value_counts()
        print(f"\n{col}: {counts.size} distinct values, group sizes "
              f"{counts.min()}-{counts.max()} (mean {counts.mean():.0f})")
        print("  first 6 values in file order:", df[col].head(6).tolist())


if __name__ == "__main__":
    import data_loader

    raw = data_loader.load_tweets()
    clean = clean_dataframe(raw)
    print("Cleaning report:", clean.attrs["cleaning_report"])
    clean = add_vader_scores(clean)
    print("\nSummary by sentiment:\n", summarise_by_sentiment(clean))
    print("\nSentiment by age group (%):\n", sentiment_by_group(clean))
    print("\nSentiment by time of tweet (%):\n",
          sentiment_by_group(clean, "Time of Tweet"))
    print("\nTweets from Pakistan:", len(filter_rows(clean, "Country",
                                                     "Pakistan")))
    check_synthetic_columns(clean)
