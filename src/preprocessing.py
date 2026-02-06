# save this as preprocess_data.py (or adjust to your structure)

import json
import pickle

def main():
    # 1. Load the raw JSON dataset
    with open("data/Conversational_Transcript_Dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. Extract the transcripts list
    transcripts = data["transcripts"]  # keep structure as-is

    # 3. Save as pickle for faster loading later
    with open("data/preprocessed_transcripts.pkl", "wb") as f:
        pickle.dump(transcripts, f)

if __name__ == "__main__":
    main()
