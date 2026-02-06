import json, pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

with open("data/transcripts.json", "r", encoding="utf-8") as f:
    transcripts = json.load(f)["transcripts"]

texts, labels = [], []

for t in transcripts:
    texts.append(" ".join(turn["text"] for turn in t["conversation"]))
    labels.append(t["intent"])

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=5000,
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Macro-F1:", f1_score(y_test, y_pred, average="macro"))

with open("models/intent_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Saved intent_model.pkl and vectorizer.pkl")
