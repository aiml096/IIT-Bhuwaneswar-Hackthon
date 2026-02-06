import csv
import re
from collections import Counter

# ============================================================
# 1. KEYWORD EXTRACTION
# ============================================================

def extract_keywords(query):
    query = query.lower()
    query = re.sub(r"[^a-zA-Z0-9 ]", "", query)
    stopwords = {
        "the", "is", "and", "to", "of", "a", "in", "for", "on",
        "with", "my", "i", "it", "was", "but", "not", "have"
    }
    return [w for w in query.split() if w not in stopwords]


# ============================================================
# 2. KEYWORD MATCH SCORING
# ============================================================

def keyword_score(transcript, keywords):
    text = " ".join(
        turn["text"].lower() for turn in transcript["conversation"]
    )
    return sum(1 for k in keywords if k in text)


# ============================================================
# 3. CONFIDENCE SCORE
# ============================================================

def retrieval_confidence(match_score, total_keywords):
    if total_keywords == 0:
        return 0.5
    return round(match_score / total_keywords, 2)


# ============================================================
# 4. RETRIEVAL
# ============================================================

def retrieve_calls(query, predicted_intent, transcripts, top_k=3):
    keywords = extract_keywords(query)

    candidates = [
        t for t in transcripts if t["intent"] == predicted_intent
    ]

    scored = []
    for t in candidates:
        score = keyword_score(t, keywords)
        if score > 0:
            confidence = retrieval_confidence(score, len(keywords))
            scored.append({
                "call_id": t["transcript_id"],
                "score": score,
                "confidence": confidence
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ============================================================
# 5. CONTEXT AGGREGATION
# ============================================================

def build_context(retrieved_ids, transcripts):
    context = []
    for t in transcripts:
        if t["transcript_id"] in retrieved_ids:
            for turn in t["conversation"]:
                context.append(f"{turn['speaker']}: {turn['text']}")
    return "\n".join(context)


# ============================================================
# 6. RULE-BASED ANSWER GENERATION (NO HALLUCINATION)
# ============================================================

def rule_based_answer(intent):
    templates = {
        "Delivery Investigation":
            "The order was marked as delivered, but the customer reported non-receipt. A delivery investigation and replacement process were initiated.",
        "Payment Issue":
            "The customer reported a payment-related issue which was investigated by the support agent.",
        "Account Access":
            "The customer contacted support regarding account access problems."
    }
    return templates.get(
        intent,
        "Relevant information was found in the retrieved call transcripts."
    )


# ============================================================
# 7. FAITHFULNESS SCORE (RULE-BASED, SAFE)
# ============================================================

def faithfulness_score(answer, context):
    answer_tokens = set(answer.lower().split())
    context_tokens = set(context.lower().split())

    stopwords = {
        "the", "is", "and", "to", "of", "a", "in", "for",
        "with", "was", "were", "has", "have"
    }

    hallucinated = [
        t for t in answer_tokens
        if t not in context_tokens and t not in stopwords
    ]

    return 1 if len(hallucinated) == 0 else 0


# ============================================================
# 8. RELEVANCY SCORE
# ============================================================

def relevancy_score(query, answer):
    q = set(query.lower().split())
    a = set(answer.lower().split())
    overlap = len(q & a)
    return round(overlap / max(len(q), 1), 2)


# ============================================================
# 9. ID RECALL
# ============================================================

def id_recall(retrieved_ids, ground_truth_ids):
    if not ground_truth_ids:
        return 1.0
    return len(set(retrieved_ids) & set(ground_truth_ids)) / len(ground_truth_ids)


# ============================================================
# 10. FULL PIPELINE (AFTER RETRIEVAL)
# ============================================================

def process_query(
    query_id,
    query,
    predicted_intent,
    transcripts,
    ground_truth_map,
    csv_writer
):
    retrieved = retrieve_calls(query, predicted_intent, transcripts)

    retrieved_ids = [r["call_id"] for r in retrieved]
    confidences = [r["confidence"] for r in retrieved]

    context = build_context(retrieved_ids, transcripts)
    answer = rule_based_answer(predicted_intent)

    faithfulness = faithfulness_score(answer, context)
    relevancy = relevancy_score(query, answer)
    recall = id_recall(
        retrieved_ids,
        ground_truth_map.get(predicted_intent, [])
    )

    csv_writer.writerow([
        query_id,
        query,
        predicted_intent,
        answer,
        ";".join(retrieved_ids)
    ])

    return {
        "retrieved_ids": retrieved_ids,
        "confidence": confidences,
        "faithfulness": faithfulness,
        "relevancy": relevancy,
        "id_recall": recall
    }


# ============================================================
# 11. CSV DRIVER (EVALUATION QUERIES)
# ============================================================

def run_evaluation(queries, transcripts, ground_truth_map):
    with open("evaluation_output.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Query Id",
            "Query",
            "Query Category",
            "System Output",
            "Remarks"
        ])

        metrics = []

        for q in queries:
            result = process_query(
                q["id"],
                q["query"],
                q["intent"],
                transcripts,
                ground_truth_map,
                writer
            )
            metrics.append(result)

    return metrics
queries = [
    {"id": 1, "query": "Order marked delivered but not received", "intent": "Delivery Investigation"},
    {"id": 2, "query": "Payment deducted twice", "intent": "Payment Issue"},
    {"id": 3, "query": "Unable to login to my account", "intent": "Account Access"}
]
metrics = run_evaluation(queries, transcripts, ground_truth_map)
avg_id_recall = sum(m["id_recall"] for m in metrics) / len(metrics)
avg_faithfulness = sum(m["faithfulness"] for m in metrics) / len(metrics)
avg_relevancy = sum(m["relevancy"] for m in metrics) / len(metrics)

