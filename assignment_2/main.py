import re
from eval import precision, recall, avg_precision, nDCG


def first_n_retrieved(all_retrieved, n):
    first_10_docs_retrieved = {k: all_retrieved[k] for k in list(all_retrieved.keys())[:n]}
    return first_10_docs_retrieved.keys()


if __name__ == '__main__':
    with open('./systems/S1.results', 'r') as f:
        lines = f.readlines()
        retrieved_docs = dict()

        for line in lines:
            query_id, _, doc_id, rank_of_doc, doc_score, _ = line.split(' ')
            # doc_details = {'rank': rank_of_doc, 'score': doc_score}
            doc_details = tuple([rank_of_doc, doc_score])

            if query_id in retrieved_docs:
                retrieved_docs[query_id].update({doc_id: doc_details})
            else:
                retrieved_docs.setdefault(query_id, {doc_id: doc_details})

    with open('./systems/qrels.txt', 'r') as qrels_f:
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

    print('Precision - Recall - r_precision - AP')
    avg_precisions = []
    for query in retrieved_docs:
        first_10_retrieved = first_n_retrieved(retrieved_docs[query], 10)
        first_50_retrieved = first_n_retrieved(retrieved_docs[query], 50)
        relevant_docs = [x[0] for x in relevant_docs_dict[query]]

        precision_10 = precision(first_10_retrieved, relevant_docs)
        recall_50 = recall(first_50_retrieved, relevant_docs)

        rank_r_documents = first_n_retrieved(retrieved_docs[query], len(relevant_docs))
        r_precision = precision(rank_r_documents, relevant_docs)

        ap = avg_precision(retrieved_docs[query], relevant_docs)
        avg_precisions.append(ap)

        nDCG_10 = nDCG(10, first_10_retrieved, relevant_docs)

        print('%.3f - %.3f - %.3f - %.3f' % (precision_10, recall_50, r_precision, ap))

    print('MAP:', sum(avg_precisions)/len(retrieved_docs))
