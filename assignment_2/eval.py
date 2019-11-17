import numpy as np


def precision(retrieved, relevant):
    """Precision at cutoff 10 (only top 10 retrieved documents in the list are considered for each query)
    Args:
        retrieved (dict): The retrieved documents
        relevant (dict): The relevant documents
    Returns:
        precision (float): The precision value
    """
    retrieved_and_relevant = list(set(retrieved).intersection(relevant))
    precision = len(retrieved_and_relevant) / len(retrieved)
    return precision


def recall(retrieved, relevant):
    """Recall at cutoff 50
    Args:
        retrieved (dict): The retrieved documents
        relevant (dict): The relevant documents
    Returns:
        recall (float): The recall value
    """
    retrieved_and_relevant = list(set(retrieved).intersection(relevant))
    recall = len(retrieved_and_relevant) / len(relevant)
    return recall


def avg_precision(all_retrieved, relevant):
    """Average precision
    Args:
        retrieved (dict): The retrieved documents
        relevant (dict): The relevant documents
    Returns:
        recall (float): The average precision value
    """
    ap = 0

    for k in range(1, len(all_retrieved)):
        if list(all_retrieved.keys())[k - 1] in relevant:
            k_retrieved = {c: all_retrieved[c] for c in list(all_retrieved.keys())[:k]}
            precision_k = precision(list(k_retrieved.keys()), relevant)
            ap += precision_k * 1
        else:
            ap += 0

    return ap / len(relevant)


def nDCG(retrieved, relevant):
    '''Normalized discount cumulative gain at cutoff k
    Args:
        retrieved (dict): The retrieved documents
        relevant (dict): The relevant documents
    Returns:
        nDCG (float): The nDCG value
    '''
    first_doc = retrieved[0][0]
    rel1 = int(dict(relevant)[first_doc]) if first_doc in dict(relevant) else 0
    DCG = rel1
    # Calculate DCG@k
    for doc, rank in retrieved[1:]:
        if doc in dict(relevant):
            DCG += int(dict(relevant)[doc]) / np.log2(int(rank))

    # Calculate iDCG@k
    irel1 = int(relevant[0][1])
    iDCG = irel1

    for idx, (doc, rel) in enumerate(relevant[1:len(retrieved)]):
        iDCG += int(rel) / np.log2(int(idx) + 2)

    return DCG / iDCG
