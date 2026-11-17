import sys; sys.path.insert(0, "src")
from data_loader import load_tweets
from preprocessing import add_clean_column, word_frequencies, top_words
from numpy_ops import select_vocabulary, build_bow_matrix, row_normalise, describe_shapes

df = load_tweets().dropna(subset=["text"]).sample(2000, random_state=42)
df = add_clean_column(df)

for before, after in zip(df["text"].head(5), df["tokens"].head(5)):
    print("BEFORE:", before)
    print("AFTER :", after, "\n")

tokens = df["tokens"].tolist()
print("Top words:", top_words(word_frequencies(tokens), 10))

vocab = select_vocabulary(tokens, 500)
bow = build_bow_matrix(tokens, vocab)
describe_shapes(bow, row_normalise(bow))