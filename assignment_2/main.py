import re
import numpy as np
from eval import precision, recall, avg_precision, nDCG
from scipy import stats

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def get_retrieved_docs_for_system_file(filename):
    """Gets the retrieved docs for the provided system file
    Args:
        filename (string): The path of the system file
    Returns:
        retrieved_docs (dict): A dictionaty of the retrieved docs
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
        retrieved_docs = dict()

        for line in lines:
            query_id, _, doc_id, rank_of_doc, doc_score, _ = line.split(' ')
            doc_details = tuple([rank_of_doc, doc_score])

            if query_id in retrieved_docs:
                retrieved_docs[query_id].update({doc_id: doc_details})
            else:
                retrieved_docs.setdefault(query_id, {doc_id: doc_details})
    return retrieved_docs


def get_relative_docs(filename):
    """Gets the relative docs for each query
    Args:
        filename (string): The path of the relative docs file
    Returns:
        relevant_docs_dict (dict): A dictionary of the relevant docs
    """
    with open(filename, 'r') as qrels_f:
        lines = qrels_f.readlines()
        relevant_docs_dict = dict()

        for line in lines:
            query_id, docs = line.split(':')
            # Extract document number and rank from string
            relev_docs = list(filter(None, re.split(r'\((.*?)\)', docs.replace(' ', ''))))

            relev_docs_list = []
            for doc in relev_docs:
                if ',' in doc:
                    relev_docs_list.append(tuple(doc.split(',')))

            if query_id in relevant_docs_dict:
                relevant_docs_dict[query_id].update({relev_docs_list})
            else:
                relevant_docs_dict.setdefault(query_id, relev_docs_list)
    return relevant_docs_dict


def first_n_retrieved(all_retrieved, n):
    """Return the first n retrieved documents
    Args:
        all_retrieved (dict): The dictionary of retrieved docs
        n (int): The number of documents to be returned
    Returns:
        (dict)
    """
    return {k: all_retrieved[k] for k in list(all_retrieved.keys())[:n]}


def write_scores_to_file(filename, scores, all_systems):
    """Writes the evaluation scores to the provided file
    Args:
        filename (string): The files to write the scores
        scores (list): List of scores
        all_systems (bool): If true, the scores files for all systems will be created
    """
    with open('./eval_results/' + filename + '.eval', 'w') as f:
        column_names = ['P@10', 'R@50', 'r-Precision', 'AP', 'nDCG@10', 'nDCG@20']
        f.write('\t' + '\t'.join(column_names) + '\n')

        for idx, score in enumerate(scores):
            ids_col = 'S' + str(idx + 1) if all_systems is True else str(idx + 1)
            f.write(ids_col + '\t')
            f.write('\t'.join(format(x, ".3f") for x in score))
            f.write('\n')

        if all_systems is False:
            f.write('mean\t' + '\t'.join(format(x, ".3f") for x in np.mean(scores, axis=0)))


def get_metric_column(filename, column_index):
    with open(filename) as f1:
        lines = f1.readlines()
        lines = lines[1: len(lines) - 1]
        scores_vector = []

        for line in lines:
            scores_vector.append(float(line.split('\t')[column_index]))
    return scores_vector


def t_test(system_1, system_2, column_index, column_name):
    """Perform t-test to compare two different systems
    Args:
        system_1 (string): System 1
        system_2 (string): System 2
        column_index (int): The index of the column to be retrieved
        column_name (string): The name of the column for printing purposes
    """
    scores_1 = get_metric_column('./eval_results/' + system_1 + '.eval', column_index)
    scores_2 = get_metric_column('./eval_results/' + system_2 + '.eval', column_index)
    _, pvalue = stats.ttest_ind(scores_1, scores_2)
    print('%.3f' % (pvalue))

    if pvalue > 0.05:
        print('Systems {} and {} are not significantly different'.format(system_1, system_2))
    else:
        print('Systems {} and {} are significantly different\n'.format(system_1, system_2))


if __name__ == '__main__':
    system_files = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
    relevant_docs_dict = get_relative_docs('./systems/qrels.txt')
    avg_scores_for_systems = []
    system_scores = []

    # Loop through all files in the system_files array
    for system_file in system_files:
        retrieved_docs = get_retrieved_docs_for_system_file('./systems/' + system_file + '.results')
        scores = []
        system_metrics = []

        # Loop through all the quries for each system file
        for query in retrieved_docs:
            first_10_retrieved = first_n_retrieved(retrieved_docs[query], 10)
            first_20_retrieved = first_n_retrieved(retrieved_docs[query], 20)
            first_50_retrieved = first_n_retrieved(retrieved_docs[query], 50)
            relevant_docs = [x[0] for x in relevant_docs_dict[query]]

            precision_10 = precision(first_10_retrieved.keys(), relevant_docs)

            recall_50 = recall(first_50_retrieved.keys(), relevant_docs)

            rank_r_documents = first_n_retrieved(retrieved_docs[query], len(relevant_docs))
            r_precision = precision(rank_r_documents.keys(), relevant_docs)

            ap = avg_precision(retrieved_docs[query], relevant_docs)

            first_10_retrieved_with_rank = [(k, v[0]) for k, v in first_10_retrieved.items()]
            nDCG_10 = nDCG(first_10_retrieved_with_rank, relevant_docs_dict[query])

            first_20_retrieved_with_rank = [(k, v[0]) for k, v in first_20_retrieved.items()]
            nDCG_20 = nDCG(first_20_retrieved_with_rank, relevant_docs_dict[query])

            scores.append([precision_10, recall_50, r_precision, ap, nDCG_10, nDCG_20])
            system_metrics.append([precision_10, recall_50, r_precision, ap, nDCG_10, nDCG_20])

        scores = np.array(scores)
        # Create eval.txt file for each system
        write_scores_to_file(system_file, scores, False)
        avg_scores_for_systems.append(np.mean(scores, axis=0))
        system_scores.append(system_metrics)

    # Create All.eval file
    write_scores_to_file('All', avg_scores_for_systems, True)

    # Perform t-test for the metrics that only one system performed the best,
    # because there is no statistically significant difference between the systems if both systems achieve the same score
    t_test('S2', 'S1', 2, 'precision')
    t_test('S3', 'S6', 4, 'AP')
    t_test('S3', 'S6', 5, 'nDCG@10')
    t_test('S3', 'S6', 6, 'nDCG@20')
