

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

        if list(all_retrieved.keys())[k] in relevant:
            k_retrieved = {c: all_retrieved[c] for c in list(all_retrieved.keys())[:k]}
            precision_k = precision(list(k_retrieved.keys()), relevant)
            ap += precision_k * 1
        else:
            ap += 0

    return ap / len(relevant)


def nDCG(k, retrieved, relevant):
    '''normalized discount cumulative gain at cutoff k'''

    print(retrieved)
