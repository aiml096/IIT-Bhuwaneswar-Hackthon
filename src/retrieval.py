import re

def extract_keywords(query):
    stopwords = {"the","is","and","to","of","a","in","for","on","with","my","i"}
    query = re.sub(r"[^a-zA-Z0-9 ]", "", query.lower())
    return [w for w in query.split() if w not in stopwords]

def keyword_score(transcript, keywords):
    text = " ".join(turn["text"].lower() for turn in transcript["conversation"])
    return sum(1 for k in keywords if k in text)

def retrieve_calls(query, intent, transcripts, top_k=3):
    keywords = extract_keywords(query)
    scored = []

    for t in transcripts:
        if t["intent"] == intent:
            score = keyword_score(t, keywords)
            if score > 0:
                scored.append({
                    "call_id": t["transcript_id"],
                    "score": score,
                    "confidence": round(score / max(len(keywords),1), 2)
                })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
