
# Owner: Ayesha Nehal (Labs 05, 10-11)
"""
models.py
---------
Defines the Tweet data class and the object-oriented classifier hierarchy:
- Tweet: Encapsulated tweet object with input validation and tokenization.
- SentimentClassifier: Abstract base class enforcing train() and predict().
- VaderClassifier: Rule-based sentiment analysis baseline using NLTK VADER.
- NaiveBayesClassifier: Multinomial Naive Bayes supporting BoW or TF-IDF.
- LogisticClassifier: Logistic Regression supporting unigrams or unigram+bigram TF-IDF.
"""

from abc import ABC, abstractmethod
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from preprocessing import clean_text


class Tweet:
    """Represents a single tweet with encapsulated attributes and validation.
    
    Attributes:
        _text (str): The raw text of the tweet (non-empty).
        _label (str or None): Sentiment label ('positive', 'negative', 'neutral').
        _tokens (list or None): Cleaned list of token strings.
    """

    def __init__(self, text, label=None):
        """Initialize a Tweet instance.
        
        Args:
            text (str): Raw tweet text. Must be a non-empty string.
            label (str, optional): Target sentiment label.
            
        Raises:
            ValueError: If text is not a string or contains only whitespace.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Tweet text must be a non-empty string.")
            
        self._text = text.strip()
        self._label = label.strip().lower() if isinstance(label, str) and label.strip() else None
        self._tokens = None

    @property
    def text(self):
        """Getter for the raw tweet text."""
        return self._text

    @property
    def label(self):
        """Getter for the tweet sentiment label."""
        return self._label

    @property
    def tokens(self):
        """Getter for cleaned tokens. Automatically tokenizes if not yet cleaned."""
        if self._tokens is None:
            self.clean()
        return self._tokens

    def clean(self):
        """Clean the raw text into filtered tokens using the preprocessing module.
        
        Returns:
            list: List of cleaned token strings.
        """
        self._tokens = clean_text(self._text)
        return self._tokens

    def __str__(self):
        """Return a readable string representation of the tweet."""
        label_str = f"[{self._label.upper()}]" if self._label else "[UNLABELED]"
        preview = self._text if len(self._text) <= 50 else f"{self._text[:47]}..."
        return f"Tweet({label_str} {preview})"


class SentimentClassifier(ABC):
    """Abstract base class for all sentiment classification models.
    
    Attributes:
        _name (str): Identifier name for the classifier.
        _is_trained (bool): Tracks whether the classifier is fitted and ready.
    """

    def __init__(self, name):
        """Initialize the base classifier.
        
        Args:
            name (str): Human-readable model name.
        """
        self._name = name
        self._is_trained = False

    @property
    def name(self):
        """Getter for the model name."""
        return self._name

    @property
    def is_trained(self):
        """Getter for training status flag."""
        return self._is_trained

    @abstractmethod
    def train(self, texts, labels):
        """Train the classifier on a sequence of texts and labels.
        
        Args:
            texts (iterable of str): Training tweet texts.
            labels (iterable of str): Target sentiment labels.
        """
        pass

    @abstractmethod
    def predict(self, text):
        """Predict the sentiment and confidence for a single text.
        
        Args:
            text (str): Raw or preprocessed tweet text.
            
        Returns:
            tuple: (predicted_label: str, confidence: float)
        """
        pass

    def predict_many(self, texts):
        """Polymorphic batch prediction over multiple texts.
        
        Args:
            texts (iterable of str): Input texts to classify.
            
        Returns:
            list of tuple: List of (predicted_label, confidence) tuples.
        """
        return [self.predict(t) for t in texts]


class VaderClassifier(SentimentClassifier):
    """Rule-based sentiment baseline using NLTK's SentimentIntensityAnalyzer."""

    def __init__(self):
        """Initialize VADER classifier."""
        super().__init__("VADER (Rule-based Baseline)")
        self._analyzer = SentimentIntensityAnalyzer()
        # VADER is a rule-based lexicon and does not require training
        self._is_trained = True

    def train(self, texts=None, labels=None):
        """No-op training step to satisfy the abstract interface."""
        self._is_trained = True
        return self

    def predict(self, text):
        """Classify sentiment from compound polarity score.
        
        Rules:
            compound >= 0.05  -> positive
            compound <= -0.05 -> negative
            otherwise         -> neutral
            
        Args:
            text (str): Input text string.
            
        Returns:
            tuple: (predicted_label: str, confidence: float)
        """
        if not isinstance(text, str):
            text = str(text)

        scores = self._analyzer.polarity_scores(text)
        compound = scores["compound"]

        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        # Confidence mapped from compound intensity (or neutral score)
        confidence = abs(compound) if label != "neutral" else scores["neu"]
        return label, round(float(confidence), 4)


class NaiveBayesClassifier(SentimentClassifier):
    """Multinomial Naive Bayes with either Bag-of-Words or TF-IDF features."""

    def __init__(self, feature_type="bow"):
        """Initialize Naive Bayes with specified vectorization type.
        
        Args:
            feature_type (str): 'bow' for CountVectorizer, 'tfidf' for TfidfVectorizer.
            
        Raises:
            ValueError: If feature_type is not 'bow' or 'tfidf'.
        """
        feature_type = feature_type.lower()
        if feature_type not in ("bow", "tfidf"):
            raise ValueError("feature_type must be either 'bow' or 'tfidf'.")

        name = f"Naive Bayes ({feature_type.upper()})"
        super().__init__(name)
        self._feature_type = feature_type

        if feature_type == "bow":
            self._vectorizer = CountVectorizer(min_df=2, stop_words="english")
        else:
            self._vectorizer = TfidfVectorizer(min_df=2, stop_words="english")

        self._model = MultinomialNB()

    def train(self, texts, labels):
        """Fit vectorizer and train MultinomialNB."""
        text_list = [str(t) for t in texts]
        X = self._vectorizer.fit_transform(text_list)
        self._model.fit(X, labels)
        self._is_trained = True
        return self

    def predict(self, text):
        """Predict label and posterior probability confidence."""
        if not self._is_trained:
            raise RuntimeError(f"{self._name} has not been trained yet.")

        X = self._vectorizer.transform([str(text)])
        probs = self._model.predict_proba(X)[0]
        max_idx = int(np.argmax(probs))
        label = self._model.classes_[max_idx]
        confidence = probs[max_idx]
        return str(label), round(float(confidence), 4)


class LogisticClassifier(SentimentClassifier):
    """Logistic Regression classifier with TF-IDF n-gram features."""

    def __init__(self, ngram_range=(1, 1)):
        """Initialize Logistic Regression with specified n-gram range.
        
        Args:
            ngram_range (tuple): (min_n, max_n), e.g., (1, 1) or (1, 2).
        """
        ngram_desc = "Unigrams" if ngram_range == (1, 1) else "Unigrams+Bigrams"
        name = f"Logistic Regression (TF-IDF {ngram_desc})"
        super().__init__(name)
        self._ngram_range = ngram_range
        self._vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=2,
            stop_words="english",
            max_features=10000
        )
        self._model = LogisticRegression(max_iter=1000, random_state=42)

    def train(self, texts, labels):
        """Fit TF-IDF vectorizer and train Logistic Regression model."""
        text_list = [str(t) for t in texts]
        X = self._vectorizer.fit_transform(text_list)
        self._model.fit(X, labels)
        self._is_trained = True
        return self

    def predict(self, text):
        """Predict label and softmax probability confidence."""
        if not self._is_trained:
            raise RuntimeError(f"{self._name} has not been trained yet.")

        X = self._vectorizer.transform([str(text)])
        probs = self._model.predict_proba(X)[0]
        max_idx = int(np.argmax(probs))
        label = self._model.classes_[max_idx]
        confidence = probs[max_idx]
        return str(label), round(float(confidence), 4)
