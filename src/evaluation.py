# main evaluation script
from answer_generator import BERTCausalAnalyzer, rule_based_answer

# Load BERT analyzer
bert_analyzer = BERTCausalAnalyzer("./models/bert_causal")

# In your query loop:
for q in queries:
    intent = predict_intent(q["query"])
    retrieved = retrieve_calls(q["query"], intent, transcripts)
    call_ids = [r["call_id"] for r in retrieved]
    context = build_context(call_ids, transcripts)
    
    # Use BERT-enhanced answer generation
    answer = rule_based_answer(intent, call_ids, context, q["query"], bert_analyzer)
    
    # Rest remains same...

def id_recall(retrieved_ids, ground_truth_ids):
    if not ground_truth_ids:
        return 1.0
    return len(set(retrieved_ids) & set(ground_truth_ids)) / len(ground_truth_ids)

def relevancy_score(query, answer):
    q = set(query.lower().split())
    a = set(answer.lower().split())
    return round(len(q & a) / max(len(q),1), 2)
