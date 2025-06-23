import random

def generate_mcqs(text):
    sentences = text.split(".")
    mcqs = []
    for i in range(min(5, len(sentences))):
        sent = sentences[i].strip()
        if len(sent) < 20:
            continue
        options = [sent + f" ({j})" for j in range(4)]
        correct = random.choice(options)
        mcqs.append({
            "question": f"What is the meaning of: '{sent[:30]}...?'",
            "options": options,
            "answer": correct
        })
    return mcqs
