import numpy as np


def precision(retrieved, relevant):
    """precision at cutoff 10 (only top 10 retrieved documents in the list are considered for each query)"""
    retrieved_and_relevant = list(set(retrieved).intersection(relevant))
    precision = len(retrieved_and_relevant) / len(retrieved)
    return precision


def recall(retrieved, relevant):
    """recall at cutoff 50"""
    retrieved_and_relevant = list(set(retrieved).intersection(relevant))
    recall = len(retrieved_and_relevant) / len(relevant)
    return recall


def avg_precision(all_retrieved, relevant):
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
    '''normalized discount cumulative gain at cutoff k'''
    first_doc = retrieved[0][0]
    rel1 = int(dict(relevant)[first_doc]) if first_doc in dict(relevant) else 0
    DCG = rel1
    # Calculate DCG@k
    for doc, rank in retrieved[1:]:
        if doc in dict(relevant):
            DCG += int(dict(relevant)[doc]) / np.log2(int(rank))

    # print('\n')
    # Calculate iDCG@k
    irel1 = int(relevant[0][1])
    iDCG = irel1
    # print('irel1', iDCG)
    for idx, (doc, rel) in enumerate(relevant[1:len(retrieved)]):
        iDCG += int(rel) / np.log2(int(idx) + 2)

    return DCG / iDCG
