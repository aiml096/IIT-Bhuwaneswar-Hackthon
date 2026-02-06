import pickle, csv
from src.data_loader import load_transcripts      # Fixed import
from src.retrieval import retrieve_calls
from src.context_builder import build_context
from src.answer_generator import rule_based_answer  # Updated version
from src.faithfulness import faithfulness_score
from src.evaluation import id_recall, relevancy_score

# Load data
transcripts = load_transcripts("data/Conversational_Transcript_Dataset.json")

# Load model + vectorizer
with open("models/intent_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def predict_intent(query):
    return model.predict(vectorizer.transform([query]))[0]

# Ground truth mapping
ground_truth = {}
for t in transcripts:
    ground_truth.setdefault(t["intent"], []).append(t["transcript_id"])

queries = [
    {"id":1, "query":"Order delivered but not received"},
    {"id":2, "query":"Payment deducted twice"},
]

# Create results folder if missing
import os
os.makedirs("results", exist_ok=True)

with open("results/evaluation_output.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Query Id", "Query", "Query Category", "System Output", "Remarks"])

    for q in queries:
        intent = predict_intent(q["query"])
        retrieved = retrieve_calls(q["query"], intent, transcripts)
        call_ids = [r["call_id"] for r in retrieved]
        context = build_context(call_ids, transcripts)

        # FIXED: Pass all required args
        answer = rule_based_answer(intent, call_ids, context, q["query"])
        
        faith = faithfulness_score(answer, context)
        relevancy = relevancy_score(q["query"], answer)
        recall = id_recall(call_ids, ground_truth.get(intent, []))

        # FIXED: Match hackathon CSV format
        writer.writerow([
            q["id"], 
            q["query"], 
            intent, 
            answer, 
            f"Intent:{intent}|CallIDs:{','.join(call_ids)}|Recall:{recall}"
        ])

        print(f"Query: {q['query']}")
        print(f"Intent: {intent}")
        print(f"Retrieved: {call_ids}")
        print(f"Faithfulness: {faith}")
        print(f"Relevancy: {relevancy}")
        print(f"ID Recall: {recall}")
        print("-" * 50)
