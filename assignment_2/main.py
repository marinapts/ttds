import re
import numpy as np
from eval import precision, recall, avg_precision, nDCG

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def get_system_file(filename):
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
            # doc_id,
            if query_id in relevant_docs_dict:
                relevant_docs_dict[query_id].update({relev_docs_list})
            else:
                relevant_docs_dict.setdefault(query_id, relev_docs_list)
    return relevant_docs_dict


def first_n_retrieved(all_retrieved, n):
    first_10_docs_retrieved = {k: all_retrieved[k] for k in list(all_retrieved.keys())[:n]}
    return first_10_docs_retrieved


def write_scores_to_file(filename, scores):
    with open('./eval_results/' + filename + '.eval', 'w') as f:
        column_names = ['P@10', 'R@50', 'r-Precision', 'AP', 'nDCG@10', 'nDCG@20']
        f.write('\t' + '\t'.join(column_names) + '\n')

        for idx, score in enumerate(scores):
            f.write(str(idx + 1) + '\t')
            f.write('\t'.join(format(x, ".3f") for x in score))
            f.write('\n')
        f.write('mean\t' + '\t'.join(format(x, ".3f") for x in np.mean(scores, axis=0)))


if __name__ == '__main__':
    system_files = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']

    for system_file in system_files:
        retrieved_docs = get_system_file('./systems/' + system_file + '.results')
        relevant_docs_dict = get_relative_docs('./systems/qrels.txt')

        avg_precisions = []
        scores = []

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
            avg_precisions.append(ap)

            first_10_retrieved_with_rank = [(k, v[0]) for k, v in first_10_retrieved.items()]
            nDCG_10 = nDCG(first_10_retrieved_with_rank, relevant_docs_dict[query])

            first_20_retrieved_with_rank = [(k, v[0]) for k, v in first_20_retrieved.items()]
            nDCG_20 = nDCG(first_20_retrieved_with_rank, relevant_docs_dict[query])

            scores.append([precision_10, recall_50, r_precision, ap, nDCG_10, nDCG_20])

        print('MAP:', sum(avg_precisions) / len(retrieved_docs))

        scores = np.array(scores)
        write_scores_to_file(system_file, scores)
