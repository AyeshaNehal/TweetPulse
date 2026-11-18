# Owner: Ayesha Nehal (Labs 01-02) - integration
"""
main.py
-------
Main entry point for TweetPulse: Reading the Mood of Twitter with NLP.
Featuring  an interactive 11-option console interface with validated user input,
state tracking between stages, and end-to-end integration of all lab manual and project proposal components.
"""

import sys
import os
import pandas as pd

import data_loader
import preprocessing
import numpy_ops
import analysis
import evaluate
import visualize
import models
from exceptions import DataLoadError


def get_int(prompt, min_val=1, max_val=11):
    """Prompt the user for an integer choice within [min_val, max_val].
    
    Repeats the prompt until a valid integer within bounds is entered.
    
    Args:
        prompt (str): Text prompt to display.
        min_val (int): Minimum acceptable option value.
        max_val (int): Maximum acceptable option value.
        
    Returns:
        int: Validated integer choice.
    """
    while True:
        try:
            raw = input(prompt).strip()
            val = int(raw)
            if min_val <= val <= max_val:
                return val
            print(f"Invalid choice. Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("Invalid input. Please enter a valid whole number.")


def print_menu():
    """Print the numbered console menu."""
    print("\n" + "=" * 65)
    print("       TWEETPULSE: READING THE MOOD OF TWITTER WITH NLP!       ")
    print("=" * 65)
    print(" 1.  Load & clean dataset (train.csv)")
    print(" 2.  Explore dataset summary & synthetic columns check")
    print(" 3.  Demonstrate text cleaner & NLTK POS tagging")
    print(" 4.  Build NumPy Bag-of-Words matrix & top words per class")
    print(" 5.  Train & compare 5 model configurations (shared split)")
    print(" 6.  Generate & save 5 report visualisations (outputs/figures/)")
    print(" 7.  Analyze model disagreements & failure cases")
    print(" 8.  Predict sentiment for your own text")
    print(" 9.  Fetch & classify live Hacker News headlines (Web API)")
    print(" 10. Evaluate best model on unseen test.csv")
    print(" 11. Exit")
    print("=" * 65)


def run_pipeline():
    """Execute the interactive stateful application loop."""
    # Central application state dictionary
    state = {
        "df_raw": None,
        "df_clean": None,
        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None,
        "model_suite": None,
        "results": None,
        "best_model": None,
        "best_y_pred": None,
    }

    print("\nWelcome to TweetPulse!")
    data_loader.write_log("Application started.")

    while True:
        print_menu()
        choice = get_int("Select an option (1-11): ", 1, 11)

        # -------------------------------------------------------------
        # OPTION 1: Load and clean dataset
        # -------------------------------------------------------------
        if choice == 1:
            print("\n--- [Option 1] Loading Dataset ---")
            try:
                state["df_raw"] = data_loader.load_tweets()
                print(f"Loaded raw dataset with {len(state['df_raw'])} rows.")
                state["df_clean"] = analysis.clean_dataframe(state["df_raw"])
                rep = state["df_clean"].attrs.get("cleaning_report", {})
                print(f"Cleaned dataset: {rep.get('rows_after', len(state['df_clean']))} rows retained.")
                print(f"  - Missing removed    : {rep.get('missing_removed', 0)}")
                print(f"  - Duplicates removed : {rep.get('duplicates_removed', 0)}")
                print(f"  - Bad labels removed : {rep.get('bad_labels_removed', 0)}")
            except DataLoadError as e:
                print(f"\n[Error] {e}")

        # -------------------------------------------------------------
        # OPTION 2: Explore dataset summary
        # -------------------------------------------------------------
        elif choice == 2:
            print("\n--- [Option 2] Dataset Exploration ---")
            if state["df_clean"] is None:
                print("[!] Please load the dataset first using Option 1.")
                continue

            # Add VADER compound scores to the DataFrame if not yet present
            if "vader_compound" not in state["df_clean"].columns:
                print("Calculating VADER compound scores for dataset summary...")
                state["df_clean"] = analysis.add_vader_scores(state["df_clean"])

            print("\nSummary by Sentiment Class:")
            print(analysis.summarise_by_sentiment(state["df_clean"]))

            if "Age of User" in state["df_clean"].columns:
                print("\nSentiment Distribution by Age of User (%):")
                print(analysis.sentiment_by_group(state["df_clean"], "Age of User"))

            print("\nChecking Synthetic Characteristics of Metadata Columns:")
            analysis.check_synthetic_columns(state["df_clean"])

        # -------------------------------------------------------------
        # OPTION 3: Text cleaner demo & NLTK POS tagging
        # -------------------------------------------------------------
        elif choice == 3:
            print("\n--- [Option 3] Text Preprocessing & POS Demo ---")
            preprocessing.demo_cleaning()
            print("Running NLTK Tokenizer and POS Tagging Demonstration:")
            sample_sentence = "The actors didn`t perform well, but the visuals were soooo good!"
            print(f"Sentence: \"{sample_sentence}\"")
            preprocessing.nltk_demo(sample_sentence)

        # -------------------------------------------------------------
        # OPTION 4: NumPy Bag-of-Words matrix & top words per class
        # -------------------------------------------------------------
        elif choice == 4:
            print("\n--- [Option 4] NumPy Bag-of-Words Matrix Operations ---")
            if state["df_clean"] is None:
                print("[!] Please load the dataset first using Option 1.")
                continue

            print("Extracting tokens from cleaned dataframe sample (2,000 tweets)...")
            sample_df = state["df_clean"].sample(min(2000, len(state["df_clean"])), random_state=42)
            sample_df = preprocessing.add_clean_column(sample_df)
            tokens_list = sample_df["tokens"].tolist()

            vocab = numpy_ops.select_vocabulary(tokens_list, size=500)
            print(f"Selected vocabulary size: {len(vocab)} words.")

            bow = numpy_ops.build_bow_matrix(tokens_list, vocab)
            norm_bow = numpy_ops.row_normalise(bow)
            numpy_ops.describe_shapes(bow, norm_bow)

            labels = sample_df["sentiment"].tolist()
            print("\nTop words per sentiment class (computed via NumPy vectorization & boolean masking):")
            for label in ("positive", "negative", "neutral"):
                top_class_words = numpy_ops.top_words_for_class(bow, labels, vocab, label, n=5)
                words_fmt = ", ".join([f"{w} ({c})" for w, c in top_class_words])
                print(f"  - {label.upper():<8}: {words_fmt}")

        # -------------------------------------------------------------
        # OPTION 5: Train & compare 5 model configurations
        # -------------------------------------------------------------
        elif choice == 5:
            print("\n--- [Option 5] Model Training & Evaluation ---")
            if state["df_clean"] is None:
                print("[!] Please load the dataset first using Option 1.")
                continue

            print("Splitting dataset into stratified 80% train / 20% test sets (random_state=42)...")
            X_tr, X_te, y_tr, y_te = evaluate.split_data(state["df_clean"])
            state["X_train"] = X_tr
            state["X_test"] = X_te
            state["y_train"] = y_tr
            state["y_test"] = y_te

            state["model_suite"] = evaluate.build_all_models()
            res, best_m, best_pred = evaluate.compare_models(
                state["model_suite"], X_tr, y_tr, X_te, y_te
            )
            state["results"] = res
            state["best_model"] = best_m
            state["best_y_pred"] = best_pred

        # -------------------------------------------------------------
        # OPTION 6: Save 5 visualization figures
        # -------------------------------------------------------------
        elif choice == 6:
            print("\n--- [Option 6] Generating Visualisations ---")
            if state["df_clean"] is None:
                print("[!] Please load the dataset first using Option 1.")
                continue
            if state["results"] is None or state["best_y_pred"] is None:
                print("[!] Please run model training first using Option 5.")
                continue

            if "vader_compound" not in state["df_clean"].columns:
                state["df_clean"] = analysis.add_vader_scores(state["df_clean"])

            saved = visualize.save_all_figures(
                df=state["df_clean"],
                y_true=list(state["y_test"]),
                y_pred=state["best_y_pred"],
                results=state["results"]
            )
            print("Successfully generated and saved all five figures:")
            for p in saved:
                print(f"  -> {p}")

        # -------------------------------------------------------------
        # OPTION 7: Disagreement & error analysis
        # -------------------------------------------------------------
        elif choice == 7:
            print("\n--- [Option 7] Disagreement Analysis ---")
            if state["model_suite"] is None or state["X_test"] is None:
                print("[!] Please train models first using Option 5.")
                continue

            evaluate.find_disagreements(
                state["model_suite"],
                state["X_test"],
                state["y_test"],
                n=5
            )

        # -------------------------------------------------------------
        # OPTION 8: Predict sentiment for custom text
        # -------------------------------------------------------------
        elif choice == 8:
            print("\n--- [Option 8] Interactive Sentiment Prediction ---")
            active_model = state["best_model"] if state["best_model"] else models.VaderClassifier()
            if state["best_model"] is None:
                print("(Note: Models have not been trained yet. Using VADER rule-based baseline.)")
            else:
                print(f"Using trained best model: {active_model.name}")

            user_text = input("\nEnter tweet or sentence to predict: ").strip()
            if not user_text:
                print("Empty input. Returning to menu.")
                continue

            pred_label, conf = active_model.predict(user_text)
            print(f"\nResult: Sentiment = {pred_label.upper()} | Confidence = {conf:.4f}")

        # -------------------------------------------------------------
        # OPTION 9: Live Hacker News headlines fetch & classify
        # -------------------------------------------------------------
        elif choice == 9:
            print("\n--- [Option 9] Web API Live Headline Classification ---")
            active_model = state["best_model"] if state["best_model"] else models.VaderClassifier()
            if state["best_model"] is None:
                print("(Using VADER rule-based baseline for headline predictions.)")

            try:
                headlines, source = data_loader.fetch_headlines(n=5)
                print(f"\nFetched {len(headlines)} headlines (Source: {source.upper()}):")
                for h in headlines:
                    pred_label, conf = active_model.predict(h)
                    print(f"  • \"{h}\"")
                    print(f"    --> {pred_label.upper()} (Confidence: {conf:.4f})")
            except DataLoadError as e:
                print(f"\n[API Error] {e}")

        # -------------------------------------------------------------
        # OPTION 10: Evaluate best model on unseen test.csv
        # -------------------------------------------------------------
        elif choice == 10:
            print("\n--- [Option 10] Score on Unseen test.csv ---")
            if state["best_model"] is None:
                print("[!] Please train and evaluate models first using Option 5.")
                continue

            try:
                evaluate.score_best_model_on_test_csv(state["best_model"])
            except DataLoadError as e:
                print(f"\n[Error] {e}")

        # -------------------------------------------------------------
        # OPTION 11: Exit
        # -------------------------------------------------------------
        elif choice == 11:
            print("\nExiting TweetPulse. Goodbye!")
            data_loader.write_log("Application exited normally.")
            sys.exit(0)


if __name__ == "__main__":
    run_pipeline()