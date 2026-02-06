# answer_generator.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class BERTCausalAnalyzer:
    def __init__(self, model_path="./models/bert_causal"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
    
    def score_turn(self, text):
        """Score if turn contains causal trigger (0-1 probability)"""
        inputs = self.tokenizer(text, return_tensors="pt", 
                              padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            causal_prob = probs[0][1].item()  # Probability of causal trigger
        return causal_prob

def rule_based_answer(intent, call_ids, context_text, query, bert_analyzer=None):
    """Enhanced causal explanation using BERT + rules"""
    
    # Causal factors (same as before)
    causal_factors = {
        "Delivery Investigation": [
            "Customer reports non-delivery despite system confirmation",
            "Multiple delivery attempts mentioned",
            "Frustration about tracking discrepancies"
        ],
        "Payment Issue": [
            "Payment failure notifications received", 
            "Multiple payment attempts documented",
            "Billing cycle confusion expressed"
        ]
    }
    
    factors = causal_factors.get(intent, ["Behavioral patterns identified"])
    
    # BERT-powered evidence extraction
    evidence_spans = []
    if bert_analyzer:
        lines = context_text.split('\n')
        for i, line in enumerate(lines[:20]):  # Top 20 lines
            if ':' in line:  # Valid turn format
                text = line.split(':', 1)[1].strip()
                score = bert_analyzer.score_turn(text)
                if score > 0.6:  # High-confidence causal triggers
                    evidence_spans.append({
                        'call_id': call_ids[0],  # Simplify for demo
                        'turn': i+1,
                        'text': text[:100],
                        'causal_score': score
                    })
    
    # Structured causal explanation
    explanation = f"## Causal Analysis: {intent}\n\n"
    explanation += "**Primary Behavioral Factors:**\n"
    
    for i, factor in enumerate(factors[:2], 1):
        explanation += f"{i}. {factor}\n"
    
    explanation += f"\n**Key Evidence (BERT Causal Scores > 0.6):**\n"
    for span in sorted(evidence_spans, key=lambda x: x['causal_score'], reverse=True)[:3]:
        explanation += f"- `{span['call_id']}` (Turn {span['turn']}): {span['text']}... (score: {span['causal_score']:.2f})\n"
    
    explanation += f"\n**Analyzed Calls:** `{', '.join(call_ids[:3])}`"
    
    return explanation.strip()
