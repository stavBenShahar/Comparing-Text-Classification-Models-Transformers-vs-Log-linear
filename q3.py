import numpy as np
from .utils import get_data, plot, announce, threadify, split_list, multiproc
from .consts import category_dict

from transformers import pipeline
from sklearn.metrics import accuracy_score
import torch
from tqdm import tqdm
import threading
import multiprocessing


# Q3
def zeroshot_classification(portion=1.):
    """
    Perform zero-shot classification
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    _, _, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)
    clf = pipeline("zero-shot-classification", model='cross-encoder/nli-MiniLM2-L6-H768')
    candidate_labels = list(category_dict.values())
    predictions = []

    for text in tqdm(x_test):
        result = clf(text, candidate_labels)
        predicted_label = result['labels'][0]
        predictions.append(predicted_label)

    label_to_index = {label: i for i, label in enumerate(candidate_labels)}
    predicted_indices = [label_to_index[label] for label in predictions]

    accuracy = accuracy_score(y_test, predicted_indices)
    return accuracy


@announce
def q3() -> None:
    print("Zero-shot result:")
    print(zeroshot_classification())


__all__ = [
    "zeroshot_classification",
    "q3"
]
