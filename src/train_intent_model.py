import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import os

class CausalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, 
            max_length=max_length, return_tensors='pt'
        )
        self.labels = torch.tensor(labels)
    
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

def load_transcripts(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('transcripts', [])

def create_weak_labels(transcripts):
    causal_keywords = ['supervisor', 'manager', 'weeks', 'not received', 
                      'multiple times', 'complaint', 'cancel', 'failed']
    
    labeled_turns = []
    for transcript in transcripts[:500]:  # Fast training
        for turn in transcript.get('conversation', []):
            text = turn['text'].lower()
            label = 1 if any(kw in text for kw in causal_keywords) else 0
            labeled_turns.append({'text': turn['text'], 'label': label})
    
    return pd.DataFrame(labeled_turns)

print("🔄 Loading data...")
transcripts = load_transcripts("Conversational_Transcript_Dataset.json")
df = create_weak_labels(transcripts)

print(f"📊 {len(df)} turns, {df['label'].sum()} causal")

# Train/val split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

# Datasets
train_dataset = CausalDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer)
val_dataset = CausalDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Training (NO Trainer class - pure PyTorch)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
model.train()

print("🚀 Training BERT (2 epochs, ~2min)...")
for epoch in range(2):
    total_loss = 0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if i % 20 == 0:
            print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.3f}")
    
    print(f"✅ Epoch {epoch+1} complete: avg_loss={total_loss/len(train_loader):.3f}")

# Save
os.makedirs("models/bert_causal", exist_ok=True)
model.save_pretrained("models/bert_causal")
tokenizer.save_pretrained("models/bert_causal")
print("🎉 BERT causal classifier SAVED to models/bert_causal/")
