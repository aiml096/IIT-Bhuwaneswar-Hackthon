def id_recall(retrieved_ids, ground_truth_ids):
    if not ground_truth_ids:
        return 1.0
    return len(set(retrieved_ids) & set(ground_truth_ids)) / len(ground_truth_ids)

def relevancy_score(query, answer):
    q = set(query.lower().split())
    a = set(answer.lower().split())
    return round(len(q & a) / max(len(q),1), 2)
