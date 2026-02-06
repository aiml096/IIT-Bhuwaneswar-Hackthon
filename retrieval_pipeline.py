import json
import joblib
import re

# ===============================
# 1. LOAD DATA
# ===============================

with open("Conversational_Transcript_Dataset.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

transcripts = raw_data["transcripts"]

print(f"Loaded {len(transcripts)} transcripts")


# ===============================
# 2. LOAD TRAINED MODEL
# ===============================

intent_model = joblib.load("intent_classifier_tuned.pkl")

def predict_intent(query: str) -> str:
    return intent_model.predict([query])[0]


# ===============================
# 3. KEYWORD EXTRACTION
# ===============================

KEYWORDS = [
    "delivered", "not received", "missing",
    "refund", "return", "payment",
    "technical", "login", "error",
    "delay", "cancel", "escalated"
]

def extract_keywords(query: str):
    query = query.lower()
    return [kw for kw in KEYWORDS if kw in query]


# ===============================
# 4. FILTER BY INTENT
# ===============================

def filter_by_intent(transcripts, intent):
    return [t for t in transcripts if t["intent"].lower() == intent.lower()]


# ===============================
# 5. KEYWORD MATCH SCORING
# ===============================

def keyword_score(transcript, keywords):
    text = transcript["reason_for_call"].lower()

    for turn in transcript["conversation"]:
        if turn["speaker"].lower() == "customer":
            text += " " + turn["text"].lower()

    score = 0
    for kw in keywords:
        if kw in text:
            score += 1

    return score


# ===============================
# 6. MAIN RETRIEVAL FUNCTION
# ===============================

def retrieve_call_ids(query, top_k=3):
    intent = predict_intent(query)
    keywords = extract_keywords(query)

    candidates = filter_by_intent(transcripts, intent)

    scored = []
    for t in candidates:
        score = keyword_score(t, keywords)
        if score > 0:
            scored.append((t["transcript_id"], score, t))

    scored.sort(key=lambda x: x[1], reverse=True)

    retrieved_ids = [x[0] for x in scored[:top_k]]
    retrieved_context = [x[2] for x in scored[:top_k]]

    return retrieved_ids, retrieved_context


# ===============================
# 7. TEST RUN
# ===============================

if __name__ == "__main__":
    test_query = "Why was the delivery escalated even though it shows delivered?"

    ids, context = retrieve_call_ids(test_query)

    print("\nQuery:", test_query)
    print("Retrieved Call IDs:", ids)

    for c in context:
        print("\n--- Transcript ---")
        print("ID:", c["transcript_id"])
        print("Reason:", c["reason_for_call"])
