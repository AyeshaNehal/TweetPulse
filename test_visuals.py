import sys; sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import visualize as v

rng = np.random.default_rng(0)
labels = ["negative", "neutral", "positive"]

df = pd.DataFrame({
    "text": ["a b c"] * 300,
    "sentiment": rng.choice(labels, 300),
    "wordcount": rng.integers(1, 30, 300),
    "vader_compound": rng.uniform(-1, 1, 300),
})
y_true = df["sentiment"].tolist()
y_pred = rng.choice(labels, 300).tolist()
results = {"VADER": {"accuracy": .6, "f1": .58},
           "Naive Bayes": {"accuracy": .7, "f1": .69}}

print(v.save_all_figures(df, y_true, y_pred, results))