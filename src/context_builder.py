def build_context(call_ids, transcripts):
    context = []
    for t in transcripts:
        if t["transcript_id"] in call_ids:
            for turn in t["conversation"]:
                context.append(f"{turn['speaker']}: {turn['text']}")
    return "\n".join(context)
