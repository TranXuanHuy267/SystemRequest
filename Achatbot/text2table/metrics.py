from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import pandas as pd
# import evaluate
from datasets import load_metric
import json
from sklearn.metrics import recall_score

ALL_METRICS = [
    "BLEU",
    "ROUGE",
    # "GLEU",
]

def check_form(text):
    if ('"LOẠI BIỂU ĐỒ": "' in text) \
       and ('", "ĐƠN VỊ": "' in text) \
       and ('", "CHU KỲ THỜI GIAN": "' in text) \
       and ('", "THỨ": "' in text) \
       and ('", "NGÀY": "' in text) \
       and ('", "TUẦN": "' in text) \
       and ('", "THÁNG": "' in text) \
       and ('", "QUÝ": "' in text) \
       and ('", "NĂM": "' in text) \
       and ('", "CHỈ TIÊU": "' in text) \
       and ('", "TIÊU ĐỀ": "' in text):
        pass
    else:
        return False
     
    text = "{"+text+"}"
    try:
        json_object = json.loads(text)
        return True
    except json.JSONDecodeError:
        return False

def calculate_macro_recall(predicted_labels, ground_truth_labels):
    """
    Calculate macro recall given the lists of predicted labels and ground truth labels.
    
    Args:
    predicted_labels (list): List of predicted labels.
    ground_truth_labels (list): List of actual ground truth labels.
    
    Returns:
    float: Macro recall value.
    """
    labels = set(ground_truth_labels)  # Extract unique labels from ground truth
    
    macro_recall = recall_score(ground_truth_labels, predicted_labels, labels=list(labels), average='macro')
    return macro_recall


def component_metrics(target_sentences, generated_sentences):
    results = pd.DataFrame({
        'Output': generated_sentences,
        'Target': target_sentences
    })
    results['Check form'] = results['Output'].apply(check_form)
    print("Number of instances: ", len(results))
    results = results[results['Check form']==True]
    print('Number of correct format instances:', len(results))
    results['Target dict'] = results['Target'].apply(lambda x: json.loads("{"+x+"}"))
    results['Output dict'] = results['Output'].apply(lambda x: json.loads("{"+x+"}"))
    for sth in ['LOẠI BIỂU ĐỒ', 'ĐƠN VỊ', 'CHU KỲ THỜI GIAN', 'THỨ', 'NGÀY', 'TUẦN', 'THÁNG', 'QUÝ', 'NĂM', 'CHỈ TIÊU', 'TIÊU ĐỀ']:
        results['Output '+sth] = results['Output dict'].apply(lambda x: x[sth])
        results['Target '+sth] = results['Target dict'].apply(lambda x: x[sth])
    for sth in ['LOẠI BIỂU ĐỒ', 'ĐƠN VỊ', 'CHU KỲ THỜI GIAN', 'THỨ', 'NGÀY', 'TUẦN', 'THÁNG', 'QUÝ', 'NĂM', 'CHỈ TIÊU', 'TIÊU ĐỀ']:
        print("The macro recall of ", sth, ":", calculate_macro_recall(results['Output '+sth], results['Target '+sth]))


def compute_metric(arg, target_sentences, generated_sentences, type_eval='each'):
    avg_score = None
    if arg.metric == "BLEU":
        avg_score = compute_bleu_scores(target_sentences, generated_sentences)
    elif arg.metric == "ROUGE":
        avg_score = compute_rogue_scores(target_sentences, generated_sentences)
    elif arg.metric == "GLEU":
        avg_score = compute_google_bleu(target_sentences, generated_sentences)
    else:
        assert False, f"{arg.metric} not defined"
    if type_eval=='all':
        component_metrics(target_sentences, generated_sentences)
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