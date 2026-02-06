def faithfulness_score(answer, context):
    a = set(answer.lower().split())
    c = set(context.lower().split())
    stopwords = {"the","is","and","to","of","a","in","for","with","was","were"}

    hallucinated = [w for w in a if w not in c and w not in stopwords]
    return 1 if len(hallucinated) == 0 else 0
