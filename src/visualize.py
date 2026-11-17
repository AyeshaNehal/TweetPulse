# Owner: M. Hamdan (Lab 09)

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "figures")
SENTIMENT_ORDER = ["negative", "neutral", "positive"]

def _save(fig, filename):
    """Save a figure into outputs/figures/ and close it. Returns the path."""
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.normpath(os.path.join(FIG_DIR, filename))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _ordered(labels):
    """Sort labels as negative, neutral, positive (others after)."""
    return sorted(labels, key=lambda l: (SENTIMENT_ORDER.index(l) if l in SENTIMENT_ORDER else 99, l))


def plot_class_counts(df, label_col="sentiment", filename="class_counts.png"):
    """Figure 1: number of tweets per sentiment class."""
    counts = df[label_col].value_counts()
    counts = counts.reindex(_ordered(counts.index))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color=["tab:red", "tab:gray", "tab:green"][:len(counts)])
    ax.set_title("Tweets per sentiment class")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of tweets")
    for x, v in zip(counts.index, counts.values):
        ax.text(x, v, str(v), ha="center", va="bottom")
    return _save(fig, filename)


def plot_length_histogram(df, text_col="text", label_col="sentiment",
                          filename="tweet_length_hist.png"):
    """Figure 2: histogram of tweet length (words), one colour per class."""
    if "wordcount" in df.columns:
        lengths = df["wordcount"]
    else:
        lengths = df[text_col].astype(str).str.split().str.len()
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(0, lengths.max() + 2, 2)
    for label in _ordered(df[label_col].unique()):
        ax.hist(lengths[df[label_col] == label], bins=bins, alpha=0.5, label=label)
    ax.set_title("Tweet length distribution")
    ax.set_xlabel("Words per tweet")
    ax.set_ylabel("Number of tweets")
    ax.legend(title="Sentiment")
    return _save(fig, filename)


def plot_vader_boxplot(df, score_col="vader_compound", label_col="sentiment",
                       filename="vader_boxplot.png"):
    """Figure 3: VADER compound score per true sentiment class."""
    labels = _ordered(df[label_col].unique())
    data = [df.loc[df[label_col] == l, score_col].dropna() for l in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", label="score = 0")
    ax.set_title("VADER compound score by true sentiment")
    ax.set_xlabel("True sentiment")
    ax.set_ylabel("VADER compound score")
    ax.legend()
    return _save(fig, filename)


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion matrix",
                          filename="confusion_matrix.png"):
    """Figure 4: confusion matrix (rows = true, columns = predicted)."""
    labels = labels or _ordered(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, label="Number of tweets")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    return _save(fig, filename)


def plot_model_comparison(results, filename="model_comparison.png"):
    """Figure 5: grouped bars comparing models on each metric.

    results: {"Model name": {"accuracy": 0.7, "f1": 0.68, ...}, ...}
    """
    table = pd.DataFrame(results).T
    fig, ax = plt.subplots(figsize=(8, 4.5))
    table.plot(kind="bar", ax=ax, rot=20)
    ax.set_title("Model comparison on the test set")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(title="Metric")
    fig.tight_layout()
    return _save(fig, filename)


def save_all_figures(df, y_true, y_pred, results, labels=None):
    """Create all five figures and return their file paths."""
    return [
        plot_class_counts(df),
        plot_length_histogram(df),
        plot_vader_boxplot(df),
        plot_confusion_matrix(y_true, y_pred, labels),
        plot_model_comparison(results),
    ]
