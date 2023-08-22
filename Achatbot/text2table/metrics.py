from nltk.translate.bleu_score import sentence_bleu
import numpy as np

# import evaluate
from datasets import load_metric

ALL_METRICS = [
    "BLEU",
    "ROUGE",
    # "GLEU",
]


def compute_metric(arg, target_sentences, generated_sentences):
    avg_score = None
    if arg.metric == "BLEU":
        avg_score = compute_bleu_scores(target_sentences, generated_sentences)
    elif arg.metric == "ROUGE":
        avg_score = compute_rogue_scores(target_sentences, generated_sentences)
    elif arg.metric == "GLEU":
        avg_score = compute_google_bleu(target_sentences, generated_sentences)
    else:
        assert False, f"{arg.metric} not defined"
    return avg_score


def compute_bleu_scores(target_sentences, generated_sentences):
    bleu_scores = [
        sentence_bleu([target_sentences[i].split()], generated_sentences[i].split())
        for i, sen in enumerate(generated_sentences)
    ]
    return np.mean(bleu_scores)


def compute_rogue_scores(target_sentences, generated_sentences):
    metric = load_metric("rouge")
    # metric = evaluate.load("rouge")
    metric.add_batch(predictions=generated_sentences, references=target_sentences)
    score = metric.compute()
    rougeL_f = score["rougeL"].mid.fmeasure
    return rougeL_f


def compute_google_bleu(target_sentences, generated_sentences):
    # metric = evaluate.load("google_bleu")
    metric = load_metric("google_bleu")
    metric.add_batch(predictions=generated_sentences, references=target_sentences)
    result = metric.compute()
    return result["google_bleu"].mid.fmeasure