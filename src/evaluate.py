# Owner: Ayesha Nehal (Labs 10-11)
"""
evaluate.py
-----------
Evaluation suite for TweetPulse sentiment models:
- Stratified train/test splitting (fixed random_state=42).
- Calculation of Accuracy, Macro Precision, Recall, and Macro F1.
- Polymorphic comparison table across all 5 configurations.
- Unseen generalization scoring on test.csv.
- Model disagreement and error analysis.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from models import ( SentimentClassifier, VaderClassifier, NaiveBayesClassifier, LogisticClassifier)
import data_loader

SENTIMENT_CLASSES = ["negative", "neutral", "positive"]


def split_data(df, test_size=0.2, random_state=42, text_col="text", label_col="sentiment"):
    """Perform a stratified train/test split shared by all models.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe containing tweets and labels.
        test_size (float): Fraction of dataset reserved for testing (default 0.2).
        random_state (int): Seed ensuring reproducible splits (default 42).
        text_col (str): Column name for tweet text.
        label_col (str): Column name for sentiment labels.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) as pandas Series.
    """
    X = df[text_col].astype(str)
    y = df[label_col].astype(str).str.lower()

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=test_size, random_state=random_state,stratify=y )
    return X_train, X_test, y_train, y_test


def evaluate_predictions(y_true, y_pred, labels=SENTIMENT_CLASSES):
    """Computing accuracy, macro precision, recall, macro F1, and confusion matrix.
    
    Args:
        y_true (list or array): Ground truth sentiment labels.
        y_pred (list or array): Predicted sentiment labels.
        labels (list): Class label ordering.
        
    Returns:
        dict: Metric dictionary with rounded float scores and confusion matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": cm
    }


def evaluate_classifier(classifier, X_test, y_test):
    """Evaluate a single SentimentClassifier instance on test data.
    
    Args:
        classifier (SentimentClassifier): Classifier instance.
        X_test (iterable): Test tweet texts.
        y_test (iterable): True labels.
        
    Returns:
        dict: Evaluation metrics and predicted label list.
    """
    preds_with_conf = classifier.predict_many(X_test)
    y_pred = [label for label, _ in preds_with_conf]
    confidences = [conf for _, conf in preds_with_conf]

    metrics = evaluate_predictions(y_test, y_pred)
    metrics["y_pred"] = y_pred
    metrics["confidences"] = confidences
    return metrics


def build_all_models():
    """Instantiate the five configurations defined in Section 6 of the proposal.
    
    Returns:
        list of SentimentClassifier:
            1. VADER baseline
            2. Naive Bayes (Bag-of-Words)
            3. Naive Bayes (TF-IDF)
            4. Logistic Regression (TF-IDF Unigrams)
            5. Logistic Regression (TF-IDF Unigrams + Bigrams)
    """
    return [
        VaderClassifier(),
        NaiveBayesClassifier(feature_type="bow"),
        NaiveBayesClassifier(feature_type="tfidf"),
        LogisticClassifier(ngram_range=(1, 1)),
        LogisticClassifier(ngram_range=(1, 2)),
    ]


def compare_models(models, X_train, y_train, X_test, y_test):
    """Train all models on the shared training set and compare them on the test set.
    
    Args:
        models (list): List of SentimentClassifier instances.
        X_train (iterable): Training texts.
        y_train (iterable): Training labels.
        X_test (iterable): Test texts.
        y_test (iterable): Test labels.
        
    Returns:
        tuple: (results_dict, best_model, best_y_pred)
            - results_dict: Nested dict matching visualize.plot_model_comparison format.
            - best_model: Classifier with the highest Macro F1 score.
            - best_y_pred: Prediction list from the highest scoring model.
    """
    results = {}
    best_model = None
    best_f1 = -1.0
    best_y_pred = None

    print("\n" + "=" * 80)
    print(f"{'MODEL COMPARISON ON SHARED TEST SPLIT':^80}")
    print("=" * 80)
    print(f"{'Model Name':<38} | {'Accuracy':^10} | {'Precision':^10} | {'Recall':^10} | {'Macro F1':^10}")
    print("-" * 88)

    y_test_list = list(y_test)

    for clf in models:
        # Train if model requires fitting
        if not clf.is_trained:
            clf.train(X_train, y_train)

        metrics = evaluate_classifier(clf, X_test, y_test_list)
        results[clf.name] = {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"]
        }

        print(f"{clf.name:<38} | "
              f"{metrics['accuracy']:^10.4f} | "
              f"{metrics['precision']:^10.4f} | "
              f"{metrics['recall']:^10.4f} | "
              f"{metrics['f1']:^10.4f}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model = clf
            best_y_pred = metrics["y_pred"]

    print("=" * 88)
    print(f"Top Performer: {best_model.name} with Macro F1: {best_f1:.4f}\n")
    data_loader.write_log(f"Model comparison completed. Best model: {best_model.name} (F1={best_f1:.4f})")
    return results, best_model, best_y_pred


def score_best_model_on_test_csv(best_model, path=None):
    """Score the best-performing model once on the unseen test.csv.
    
    Args:
        best_model (SentimentClassifier): Best model identified in comparison.
        path (str, optional): Custom path to test.csv (default: data_loader.TEST_PATH).
        
    Returns:
        dict: Metrics on test.csv (if labeled), and path to saved predictions.
    """
    if path is None:
        path = data_loader.TEST_PATH

    print(f"\n[Test Set Verification] Loading unseen test data from: {os.path.basename(path)}")
    test_df = data_loader.load_tweets(path=path)

    # Clean text column
    test_df = test_df.dropna(subset=["text"])
    texts = test_df["text"].astype(str).tolist()

    preds_with_conf = best_model.predict_many(texts)
    y_pred = [label for label, _ in preds_with_conf]

    has_labels = "sentiment" in test_df.columns and test_df["sentiment"].dropna().shape[0] > 0
    metrics = None

    if has_labels:
        y_true = test_df["sentiment"].astype(str).str.lower().tolist()
        metrics = evaluate_predictions(y_true, y_pred)
        print("\n" + "=" * 50)
        print(f"Unseen test.csv Evaluation: {best_model.name}")
        print("=" * 50)
        print(f"Accuracy  : {metrics['accuracy']:.4f}")
        print(f"Precision : {metrics['precision']:.4f}")
        print(f"Recall    : {metrics['recall']:.4f}")
        print(f"Macro F1  : {metrics['f1']:.4f}")
        print("=" * 50)
    else:
        y_true = ["unknown"] * len(y_pred)
        print(f"test.csv does not contain true sentiment labels. Generated predictions for {len(y_pred)} rows.")

    rows = list(zip(texts, y_true, y_pred))
    saved_path = data_loader.save_predictions_csv(rows)
    print(f"Predictions successfully saved to: {saved_path}")
    data_loader.write_log(f"Scored {best_model.name} on test.csv ({len(rows)} rows)")
    return metrics, saved_path


def find_disagreements(models, X, y_true, n=5):
    """Find and display sample tweets where models produce conflicting predictions.
    
    Args:
        models (list): List of trained SentimentClassifier models.
        X (iterable): Tweet texts.
        y_true (iterable): Ground truth labels.
        n (int): Maximum number of disagreement cases to display.
    """
    texts = list(X)
    labels = list(y_true)

    # Collect predictions for all models
    model_preds = {m.name: [label for label, _ in m.predict_many(texts)] for m in models}

    disagreements = []
    for i, (text, true_lbl) in enumerate(zip(texts, labels)):
        preds_for_text = {name: preds[i] for name, preds in model_preds.items()}
        unique_preds = set(preds_for_text.values())

        # Check if models disagreed amongst themselves or with true label
        if len(unique_preds) > 1 or (unique_preds and list(unique_preds)[0] != true_lbl):
            disagreements.append((text, true_lbl, preds_for_text))
            if len(disagreements) >= n:
                break

    print("\n" + "=" * 80)
    print(f"{'DISAGREEMENT & ERROR CASE ANALYSIS':^80}")
    print("=" * 80)
    for idx, (t, actual, preds) in enumerate(disagreements, start=1):
        print(f"\nCase #{idx}:")
        print(f"  Tweet text : \"{t}\"")
        print(f"  True Label : {actual.upper()}")
        print("  Model Predictions:")
        for m_name, p in preds.items():
            mark = "✓" if p == actual else "✗"
            print(f"    - {m_name:<36}: {p.upper():<8} [{mark}]")
    print("=" * 80 + "\n")