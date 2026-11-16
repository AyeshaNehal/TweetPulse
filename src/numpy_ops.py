# Owner: M. Hamdan (Lab 07)

import numpy as np
from preprocessing import word_frequencies, top_words

def select_vocabulary(token_lists, size=500):
    """Return the `size` most frequent words as a list (column order)."""
    freq = word_frequencies(token_lists)
    return [word for word, _ in top_words(freq, size)]


def build_bow_matrix(token_lists, vocab):
    """Build a bag-of-words count matrix.

    Args:
        token_lists: list of token lists, one per tweet.
        vocab: list of words; column j counts vocab[j].

    Returns:
        np.ndarray of shape (n_tweets, len(vocab)).
    """
    index = {word: j for j, word in enumerate(vocab)}
    matrix = np.zeros((len(token_lists), len(vocab)), dtype=np.int32)
    pairs = [(i, index[w]) for i, tokens in enumerate(token_lists)
             for w in tokens if w in index]
    if pairs:
        pairs = np.array(pairs)
        np.add.at(matrix, (pairs[:, 0], pairs[:, 1]), 1)
    return matrix


def row_normalise(matrix):
    """Divide each row by its sum using broadcasting (rows sum to 1).

    Shapes: matrix (n, m) / sums (n, 1) -> (n, m). Empty rows stay zero.
    """
    sums = matrix.sum(axis=1, keepdims=True)
    sums = np.where(sums == 0, 1, sums)
    return matrix / sums


def top_words_for_class(matrix, labels, vocab, target_label, n=10):
    """Return the n most used words for one sentiment as (word, count) tuples.

    Uses a boolean mask to pick that class's rows, then sum(axis=0).
    """
    mask = np.asarray(labels) == target_label        # shape (n_tweets,)
    totals = matrix[mask].sum(axis=0)                # shape (n_words,)
    top_idx = np.argsort(totals)[::-1][:n]
    words = np.asarray(vocab)[top_idx]
    return list(zip(words.tolist(), totals[top_idx].tolist()))


def describe_shapes(matrix, normalised=None):
    """Print array shapes for the report."""
    print("BoW matrix shape      :", matrix.shape, "(tweets x words)")
    print("Row sums shape        :", matrix.sum(axis=1, keepdims=True).shape)
    if normalised is not None:
        print("Normalised matrix     :", normalised.shape)
        print("Normalised row sum ~1 :", np.allclose(normalised.sum(axis=1)[matrix.sum(axis=1) > 0], 1))


if __name__ == "__main__":
    docs = [["love", "great", "day"], ["bad", "not", "good"], ["love", "love", "sad"]]
    labels = ["positive", "negative", "positive"]
    vocab = select_vocabulary(docs, size=6)
    bow = build_bow_matrix(docs, vocab)
    norm = row_normalise(bow)
    describe_shapes(bow, norm)
    print("vocab:", vocab)
    print(bow)
    print(top_words_for_class(bow, labels, vocab, "positive", n=3))